import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import pickle
import os
import numpy as np
import random
import csv
from Utils.process_smiles import espf_feature_extraction
from model_hyper_graph import H_GCL, BioEncoder, HgnnEncoder, HeteroEncoder, Decoder
from Utils.Tools2 import write_type_1234, parameters_set, build_hetero_graph, build_hypergraph, hit_ndcg_value, \
    get_metrics
from Utils.right_negative_sample_generate import neg_data_generate
import torch.nn.functional as F

def embedding_margin(embeddings, labels):
    pos_emb = embeddings[labels == 1]
    neg_emb = embeddings[labels == 0]
    if len(pos_emb) < 2 or len(neg_emb) < 2:
        return None
    # 正样本内部相似度
    pos_sim = F.cosine_similarity(pos_emb.unsqueeze(1), pos_emb.unsqueeze(0), dim=-1)
    pos_sim = pos_sim.triu(1)
    pos_sim = pos_sim[pos_sim != 0]

    # 正负相似度
    neg_sim = F.cosine_similarity(pos_emb.unsqueeze(1), neg_emb.unsqueeze(0), dim=-1).view(-1)
    return pos_sim.mean().item(), neg_sim.mean().item()
def series_num(data, drug_num, target_num, disease_num):
    """调整索引以匹配节点范围"""
    data = data.copy().astype(int)
    invalid_rows = []
    for i, line in enumerate(data):
        if not (0 <= line[0] < drug_num and 0 <= line[1] < target_num and 0 <= line[2] < disease_num):
            invalid_rows.append((i, line))
    if invalid_rows:
        print(f"series_num: Found {len(invalid_rows)} invalid rows: {invalid_rows}")
        data = np.array([line for i, line in enumerate(data) if i not in [r[0] for r in invalid_rows]])
    print(f"series_num: data shape={data.shape}, min={data.min(axis=0)}, max={data.max(axis=0)}")
    return data

def random_seed(sd):
    random.seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def train(drug_fea, target_fea, disease_fea, hetero_data, hg_pos, hg_neg_ls, train_data, model, optimizer, my_loss, args, device):
    model.train()
    print('--- Start training ---')
    optimizer.zero_grad()
    emb, pred, train_embed_pos1, train_embed_pos2, graph_embed_neg_ls, embed_pred_hetero_to_hyper, embed_pred_hyper_to_hetero, hyper_embed_true, hetero_embed_true = model(
        drug_fea, target_fea, disease_fea, hetero_data, hg_pos,
        train_data[:, 0], train_data[:, 1], train_data[:, 2], hg_neg_ls, device)
    loss_1 = my_loss(pred.view(-1, 1), train_data[:, 3].view(-1, 1).float().to(device))
    loss_2 = model.info_nce_loss(train_embed_pos1, train_embed_pos2, graph_embed_neg_ls)
    # Cross-view reconstruction loss
    loss_3 = model.diffusion_loss(embed_pred_hetero_to_hyper, hyper_embed_true) + \
             model.diffusion_loss(embed_pred_hyper_to_hetero, hetero_embed_true)
    loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2 + args.diffusion_loss_ratio * loss_3
    loss.backward()
    optimizer.step()
    print(f'epoch:{e+1:02d}, loss_train:{loss.item():.6f}, bce:{loss_1.item():.6f}, contrast:{loss_2.item():.6f}, diffusion:{loss_3.item():.6f}')
    return loss.item(), emb.detach(), train_data[:, 3].detach()


def evaluate(drug_fea, target_fea, disease_fea, hetero_data, hg_pos, hg_neg_ls, val_data_1, val_data_2, val_data_3, val_data_4, model, args, device):
    with torch.no_grad():
        model.eval()
        print('--- Start valuating ---')
        val = 0
        val_metrics = []
        for val_data in [val_data_1, val_data_2, val_data_3, val_data_4]:
            val += 1
            _, pred_val, _, _, _, _, _, _, _ = model(
                drug_fea, target_fea, disease_fea, hetero_data, hg_pos,
                val_data[:, 0], val_data[:, 1], val_data[:, 2], hg_neg_ls, device
            )
            # Compute Hits@3 and NDCG@3
            hits_1, ndcg_1 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_1)
            hits_3, ndcg_3 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_3)
            hits_5, ndcg_5 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_5)
            # Compute AUPR and AUC
            real_score = val_data[:, 3].cpu().numpy()  # Ground truth labels
            predict_score = pred_val.cpu().numpy()     # Predicted scores
            metrics = get_metrics(real_score, predict_score)

            aupr = np.asarray(metrics[0]).item()
            auc = np.asarray(metrics[1]).item()
            f1 = np.asarray(metrics[2]).item()
            val_metrics.append([hits_3, ndcg_3, aupr, auc, f1])
            early_stop(val, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5)
        avg_hits_3, avg_ndcg_3, avg_aupr, avg_auc, avg_f1 = np.mean(val_metrics, axis=0)
        print(
            f'val:avg, '
            f'hits_3:{avg_hits_3:.6f}, '
            f'ndcg_3:{avg_ndcg_3:.6f}, '
            f'aupr:{avg_aupr:.6f}, '
            f'auc:{avg_auc:.6f}, '
            f'f1:{avg_f1:.6f}'
        )

def early_stop(val, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5):
    global hits_max_matrix, ndcg_max_matrix, epoch_max_matrix, patience_num_matrix
    if hits_1 >= hits_max_matrix[val-1][0]:
        hits_max_matrix[val-1][0] = hits_1
        ndcg_max_matrix[val-1][0] = ndcg_1
        hits_max_matrix[val-1][1] = hits_3
        ndcg_max_matrix[val-1][1] = ndcg_3
        hits_max_matrix[val-1][2] = hits_5
        ndcg_max_matrix[val-1][2] = ndcg_5
        epoch_max_matrix[0][val-1] = e + 1
        patience_num_matrix[0][val-1] = 0
    else:
        patience_num_matrix[0][val-1] += 1

if __name__ == '__main__':
    # 清理 GPU 内存
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = parameters_set()
    args.diffusion_loss_ratio = 0.1
    random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    patience_num_matrix = np.zeros((1, 4))
    epoch_max_matrix = np.zeros((1, 4))
    hits_max_matrix = np.zeros((4, 3))
    ndcg_max_matrix = np.zeros((4, 3))
    patience = 300

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    data_path = 'D:\\研究生阶段学习\\论文+专利\\小论文相关\\HHCL_DTD_Base Model\\Data'
    dtd_file = os.path.join(data_path, 'dtd_interaction_adjusted_modified.txt')  #三元关联数据集
    smiles_file = f'{data_path}\\drug_smiles_124.csv'   #药物的smiles序列
    target_seq_file = f'{data_path}\\target_sequence.txt'    #靶标的氨基酸序列
    disease_file = f'{data_path}\\node_num.csv'           #疾病的mesh主题词表，格式为：MESH:D006966,105
    drug_target_file = f'{data_path}\\drug_target_num.csv'  #药物-靶标关联数据
    drug_disease_file = f'{data_path}\\drug_disease_num.csv' #药物-疾病关联数据
    cache_path = os.path.join(data_path, 'cached_features')

    drug_num = 124
    target_num = 104
    disease_num = 177

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    drug_cache_file = os.path.join(cache_path, 'drug_features.pkl')
    target_cache_file = os.path.join(cache_path, 'target_features.pkl')

    if os.path.exists(drug_cache_file):
        with open(drug_cache_file, 'rb') as f:
            drug_features = pickle.load(f)
        print(f"Loaded cached drug features: {drug_features.shape}")
    else:
        drug_features = []
        with open(smiles_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                smile = row[1]
                feature = espf_feature_extraction(smile, is_smiles=True)
                drug_features.append(feature)
        drug_features = torch.tensor(drug_features, dtype=torch.float)
        with open(drug_cache_file, 'wb') as f:
            pickle.dump(drug_features, f)
        print(f"Saved drug features to {drug_cache_file}")

    if os.path.exists(target_cache_file):
        with open(target_cache_file, 'rb') as f:
            target_features = pickle.load(f)
        print(f"Loaded cached target features: {target_features.shape}")
    else:
        target_features = []
        with open(target_seq_file, 'r', encoding='utf-8') as f:
            for line in f:
                seq = line.strip()
                feature = espf_feature_extraction(seq, is_smiles=False)
                target_features.append(feature)
        target_features = torch.tensor(target_features, dtype=torch.float)
        with open(target_cache_file, 'wb') as f:
            pickle.dump(target_features, f)
        print(f"Saved target features to {target_cache_file}")

    disease_features = torch.eye(disease_num)

    drug_features = drug_features.to(device)
    target_features = target_features.to(device)
    disease_features = disease_features.to(device)

    assert drug_features.shape[0] == drug_num, f"Drug features shape {drug_features.shape[0]} != {drug_num}"
    assert target_features.shape[0] == target_num, f"Target features shape {target_features.shape[0]} != {target_num}"
    assert disease_features.shape[0] == disease_num, f"Disease features shape {disease_features.shape[0]} != {disease_num}"

    hetero_data = build_hetero_graph(drug_target_file, drug_disease_file, drug_num, target_num, disease_num).to(device)


    # ================== Load real DTD triples ==================
    adj_data = np.loadtxt(dtd_file, dtype=int)

    assert adj_data.shape[1] == 4, "DTD file must be [drug, target, disease, label]"
    assert adj_data[:, 0].max() < drug_num
    assert adj_data[:, 1].max() < target_num
    assert adj_data[:, 2].max() < disease_num

    print(f"Loaded real DTD triples: {adj_data.shape}")
    np.random.shuffle(adj_data)
    # ============================================================

    print(f"adj_data: shape={adj_data.shape}, min={adj_data.min(axis=0)}, max={adj_data.max(axis=0)}")
    assert adj_data[:, 0].max() < drug_num, f"adj_data drug_idx max={adj_data[:, 0].max()}"
    assert adj_data[:, 1].max() < target_num, f"adj_data target_idx max={adj_data[:, 1].max()}"
    assert adj_data[:, 2].max() < disease_num, f"adj_data disease_idx max={adj_data[:, 2].max()}"
    np.random.shuffle(adj_data)
    test_data = adj_data[:int(0.1 * len(adj_data)), :]
    train_cv_data = adj_data[int(0.1 * len(adj_data)):, :]

    resultFileName = './indep_results.txt'
    train_data_all, tn_1, tn_2, tn_3, tn_4, te_1, te_2, te_3, te_4 = neg_data_generate(
        adj_data, train_cv_data, test_data, args.seed
    )
    for name, d in [('train_data_all', train_data_all), ('tn_1', tn_1), ('te_1', te_1)]:
        d = np.array(d)
        print(f"{name} before series_num: shape={d.shape}, min={d.min(axis=0)}, max={d.max(axis=0)}")
        d = np.clip(d, [0, 0, 0, 0], [drug_num-1, target_num-1, disease_num-1, 1])
        if name == 'train_data_all':
            train_data_all = d
        elif name == 'tn_1':
            tn_1 = d
        elif name == 'te_1':
            te_1 = d

    train_data_pos = series_num(train_cv_data, drug_num, target_num, disease_num)
    train_data_neg_1 = series_num(np.array(tn_1), drug_num, target_num, disease_num)
    train_data_all = series_num(np.array(train_data_all), drug_num, target_num, disease_num)
    val_data_1 = series_num(np.array(te_1), drug_num, target_num, disease_num)
    val_data_2 = series_num(np.array(te_2), drug_num, target_num, disease_num)
    val_data_3 = series_num(np.array(te_3), drug_num, target_num, disease_num)
    val_data_4 = series_num(np.array(te_4), drug_num, target_num, disease_num)

    print(f"train_data_pos: shape={train_data_pos.shape}, min={train_data_pos.min(axis=0)}, max={train_data_pos.max(axis=0)}")
    hypergraph_pos = build_hypergraph(train_data_pos).to(device)
    hypergraph_neg_1 = build_hypergraph(train_data_neg_1).to(device)
    hypergraph_neg_ls = [hypergraph_neg_1]

    train_data_all = torch.tensor(train_data_all, dtype=torch.long).to(device)
    val_data_1 = torch.tensor(val_data_1, dtype=torch.long).to(device)
    val_data_2 = torch.tensor(val_data_2, dtype=torch.long).to(device)
    val_data_3 = torch.tensor(val_data_3, dtype=torch.long).to(device)
    val_data_4 = torch.tensor(val_data_4, dtype=torch.long).to(device)

    model = H_GCL(
        BioEncoder(128, 128, disease_num, args.bio_out_dim),
        HeteroEncoder(args.bio_out_dim, args.hgnn_dim_1),
        HgnnEncoder(args.bio_out_dim, args.hgnn_dim_1),
        Decoder(args.hgnn_dim_1 * 3),
        args.hgnn_dim_1,
        args.bio_out_dim
    ).to(device)
    my_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = 1024
    margin_log = []

    for e in range(args.epochs):

        loss, emb_train, labels_train = train(
            drug_features, target_features, disease_features,
            hetero_data, hypergraph_pos, hypergraph_neg_ls,
            train_data_all, model, optimizer, my_loss, args, device
        )

        # ====== ⭐ 每 20 个 epoch 计算一次判别间隔 ======
        # if (e + 1) % 20 == 0:
        #     with torch.no_grad():
        #         margin = embedding_margin(
        #             emb_train.cpu(),
        #             labels_train.cpu()
        #         )
        #         if margin is not None:
        #             pos_sim, neg_sim = margin
        #             margin_log.append([e + 1, pos_sim, neg_sim])
        #             print(
        #                 f"[Margin@{e + 1:03d}] "
        #                 f"pos_sim={pos_sim:.4f} "
        #                 f"neg_sim={neg_sim:.4f} "
        #                 f"gap={pos_sim - neg_sim:.4f}"
        #             )

        evaluate(
            drug_features, target_features, disease_features,
            hetero_data, hypergraph_pos, hypergraph_neg_ls,
            val_data_1, val_data_2, val_data_3, val_data_4,
            model, args, device
        )

        if all(patience_num_matrix[0] >= patience):
            break

    # figure_dir = r"D:\研究生阶段学习\学习\HHCL_DTD(version01)-main 测试版本\figure"
    # os.makedirs(figure_dir, exist_ok=True)
    #
    # margin_log = np.array(margin_log)
    #
    # plt.figure(figsize=(8, 5))
    # plt.plot(margin_log[:, 0], margin_log[:, 1], label="Positive Similarity")
    # plt.plot(margin_log[:, 0], margin_log[:, 2], label="Negative Similarity")
    # plt.plot(
    #     margin_log[:, 0],
    #     margin_log[:, 1] - margin_log[:, 2],
    #     label="Margin (Gap)",
    #     linewidth=2
    # )
    #
    # plt.xlabel("Epoch")
    # plt.ylabel("Cosine Similarity")
    # plt.title("Embedding Discriminative Margin vs Epoch")
    # plt.legend()
    # plt.grid(True)
    #
    # save_path = os.path.join(figure_dir, "embedding_margin_curve.png")
    # plt.savefig(save_path, dpi=300)
    # plt.close()
    #
    # print(f"[Saved] Embedding margin curve → {save_path}")
    #
    # print('success')
    for top, idx in zip(['top1', 'top3', 'top5'], [0, 1, 2]):
        write_type_1234(
            resultFileName, 'indep',
            hits_max_matrix[0][idx], ndcg_max_matrix[0][idx],
            hits_max_matrix[1][idx], ndcg_max_matrix[1][idx],
            hits_max_matrix[2][idx], ndcg_max_matrix[2][idx],
            hits_max_matrix[3][idx], ndcg_max_matrix[3][idx],
            epoch_max_matrix[0][0], epoch_max_matrix[0][1],
            epoch_max_matrix[0][2], epoch_max_matrix[0][3],
            top=top
        )



