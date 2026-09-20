import warnings
import torch
import torch.backends.cudnn
import pickle
import os
import numpy as np
import random
import csv
from sklearn.model_selection import KFold
from Utils.Tools2 import write_type_1234, parameters_set, build_hetero_graph, build_hypergraph, hit_ndcg_value, \
    get_metrics
from Utils.process_smiles import espf_feature_extraction
from model_hyper_graph import H_GCL, BioEncoder, HgnnEncoder, HeteroEncoder, Decoder
from Utils.right_negative_sample_generate import neg_data_generate
# 90%的交叉验证集（五折cv，一份训练，四份验证），10%的独立测试集
warnings.filterwarnings("ignore")


def series_num(data, drug_num, target_num, disease_num):
    """Adjust indices to match node ranges."""
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

def train(drug_fea, target_fea, disease_fea, hetero_data, hg_pos, hg_neg_ls, train_data, model, optimizer, my_loss,
          args, device):
    global fold_num

    model.train()
    print('--- Start training ---')
    optimizer.zero_grad()
    emb, pred, train_embed_pos1, train_embed_pos2, graph_embed_neg_ls, embed_pred_hetero_to_hyper, embed_pred_hyper_to_hetero, hyper_embed_true, hetero_embed_true = model(
        drug_fea, target_fea, disease_fea, hetero_data, hg_pos,
        train_data[:, 0], train_data[:, 1], train_data[:, 2], hg_neg_ls, device
    )
    loss_1 = my_loss(pred.view(-1, 1), train_data[:, 3].view(-1, 1).float().to(device))
    loss_2 = model.info_nce_loss(train_embed_pos1, train_embed_pos2, graph_embed_neg_ls)
    # Cross-view reconstruction loss (hypergraph_loss_ratio=0.8,diffusion_loss_ratio=0.1)
    loss_3 = model.diffusion_loss(embed_pred_hetero_to_hyper, hyper_embed_true) + \
             model.diffusion_loss(embed_pred_hyper_to_hetero, hetero_embed_true)
    loss = args.hypergraph_loss_ratio * loss_1 + \
           (1 - args.hypergraph_loss_ratio) * loss_2 + args.diffusion_loss_ratio * loss_3
    loss.backward()
    # 损失1 (loss_1)：监督的二元交叉熵（BCE） —— 用于直接训练解码器预测三元组的存在/标签（有/无）
    # 损失2 (loss_2)：对比学习 / InfoNCE（无监督） —— 通过把两个增强视图（正样本对）拉近并把负样本推远，学习区分性更强、鲁棒的节点/图表示
    # 损失3 (loss_3)：扩散/重构（MSE）损失 —— 用于跨视图的“去噪重构”目标（hetero ↔ hyper），增强异构图视图和超图视图之间的对齐与可逆性
    optimizer.step()
    print(
        f'fold_num:{fold_num:02d},'
        f' epoch:{e + 1:02d}, loss_train:{loss.item():.6f}, bce:{loss_1.item():.6f}, contrast:{loss_2.item():.6f}, diffusion:{loss_3.item():.6f}')
    return loss.item()

def evaluate(drug_fea, target_fea, disease_fea, hetero_data, hg_pos, hg_neg_ls, val_data_1, val_data_2, val_data_3,
         val_data_4, model, args, device):
    with torch.no_grad():
        model.eval()
        print('--- Start validating ---')
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
            f'val:avg, hits_3:{avg_hits_3:.6f}, ndcg_3:{avg_ndcg_3:.6f}, '
            f'aupr:{avg_aupr:.6f}, auc:{avg_auc:.6f}, f1:{avg_f1:.6f}'
        )

def get_train_val_data(all_data, train_ind, val_ind, adj_data, drug_num, target_num, disease_num):
    train_data_pos, val_data_pos = all_data[train_ind], all_data[val_ind]
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, tr_neg_4_ls, te_neg_1_ls, te_neg_2_ls, te_neg_3_ls, te_neg_4_ls = neg_data_generate(
        adj_data, train_data_pos, val_data_pos, args.seed
    )
    print(
        f"train_data_all before series_num: shape={train_data_all.shape}, min={train_data_all.min(axis=0)}, max={train_data_all.max(axis=0)}")
    np.random.shuffle(train_data_all)
    train_data_pos = series_num(train_data_pos, drug_num, target_num, disease_num)
    train_data_neg_1 = series_num(np.array(tr_neg_1_ls), drug_num, target_num, disease_num)
    train_data_all = series_num(np.array(train_data_all), drug_num, target_num, disease_num)
    val_data_1 = series_num(np.array(te_neg_1_ls), drug_num, target_num, disease_num)
    val_data_2 = series_num(np.array(te_neg_2_ls), drug_num, target_num, disease_num)
    val_data_3 = series_num(np.array(te_neg_3_ls), drug_num, target_num, disease_num)
    val_data_4 = series_num(np.array(te_neg_4_ls), drug_num, target_num, disease_num)

    print(
        f"train_data_pos: shape={train_data_pos.shape}, min={train_data_pos.min(axis=0)}, max={train_data_pos.max(axis=0)}")
    hypergraph_positive = build_hypergraph(train_data_pos)
    hypergraph_neg_1 = build_hypergraph(train_data_neg_1)
    hypergraph_negative_ls = [hypergraph_neg_1]
    return hypergraph_positive, hypergraph_negative_ls, train_data_all, val_data_1, val_data_2, val_data_3, val_data_4

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
    # Clear GPU memory
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    args = parameters_set()
    args.diffusion_loss_ratio = 0.1
    random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    patience = 300
    data_path = r'D:\研究生阶段学习\论文+专利\小论文相关\HHCL_DTD_Base Model\Data'
    smiles_file = os.path.join(data_path, 'drug_smiles_124.csv')
    target_seq_file = os.path.join(data_path, 'target_sequence.txt')
    disease_file = os.path.join(data_path, 'node_num.csv')
    drug_target_file = os.path.join(data_path, 'drug_target_num.csv')
    drug_disease_file = os.path.join(data_path, 'drug_disease_num.csv')
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
        with open(smiles_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, row in enumerate(reader):
                if len(row) < 2:
                    print(f"Warning: Invalid row {i + 2}: {row}")
                    continue
                smile = row[1].strip()
                if not smile:
                    print(f"Warning: Empty SMILES at row {i + 2}")
                    continue
                try:
                    feature = espf_feature_extraction(smile, is_smiles=True)
                    drug_features.append(feature)
                except Exception as e:
                    print(f"Error processing SMILES at row {i + 2}: {e}")
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
        with open(target_seq_file, 'r', encoding='utf-8', newline='') as f:
            seq_count = 0
            for line in f:
                seq = line.strip()
                if seq and seq_count < target_num:
                    try:
                        feature = espf_feature_extraction(seq, is_smiles=False)
                        target_features.append(feature)
                        seq_count += 1
                    except Exception as e:
                        print(f"Error processing sequence {seq_count} at line {line}: {e}")
        target_features = torch.tensor(target_features, dtype=torch.float)
        with open(target_cache_file, 'wb') as f:
            pickle.dump(target_features, f)
        print(f"Saved target features to {target_cache_file}")

    disease_features = torch.eye(disease_num)
    # -------------------------------------------------------打印drug、target、disease--features------------------------------------------------
    drug_features = drug_features.to(device)
    print("Drug feature dim:", drug_features.shape)
    target_features = target_features.to(device)
    print("Target feature dim:", target_features.shape)
    disease_features = disease_features.to(device)
    print(f"Disease feature dim: {disease_features.shape[1]}")

    assert drug_features.shape[0] == drug_num, f"Drug features shape {drug_features.shape[0]} != {drug_num}"
    assert target_features.shape[0] == target_num, f"Target features shape {target_features.shape[0]} != {target_num}"
    assert disease_features.shape[0] == disease_num, f"Disease features shape {disease_features.shape[0]} != {disease_num}"

    hetero_data = build_hetero_graph(drug_target_file, drug_disease_file, drug_num, target_num, disease_num).to(device)


    # adj_data = []
    # drug_target = np.loadtxt(drug_target_file, delimiter=',')
    # drug_disease = np.loadtxt(drug_disease_file, delimiter=',')
    # print(f"drug_target: shape={drug_target.shape}, min={drug_target.min(axis=0)}, max={drug_target.max(axis=0)}")
    # print(f"drug_disease: shape={drug_disease.shape}, min={drug_disease.min(axis=0)}, max={drug_disease.max(axis=0)}")
    # drug_target = drug_target[(drug_target[:, 0] >= 0) & (drug_target[:, 0] < drug_num) &
    #                           (drug_target[:, 1] >= 0) & (drug_target[:, 1] < target_num)]
    # drug_disease = drug_disease[(drug_disease[:, 0] >= 0) & (drug_disease[:, 0] < drug_num) &
    #                             (drug_disease[:, 1] >= 0) & (drug_disease[:, 1] < disease_num)]
    # print(
    #     f"Filtered drug_target: shape={drug_target.shape}, min={drug_target.min(axis=0)}, max={drug_target.max(axis=0)}")
    # print(
    #     f"Filtered drug_disease: shape={drug_disease.shape}, min={drug_disease.min(axis=0)}, max={drug_disease.max(axis=0)}")
    # for dt in drug_target:
    #     for dd in drug_disease:
    #         if dt[0] == dd[0]:
    #             adj_data.append([dt[0], dt[1], dd[1], 1])
    # adj_data = np.array(adj_data)



    # ================== Load real DTD triples ==================
    dtd_file = os.path.join(data_path, 'dtd_interaction_adjusted_modified.txt')
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
    cv_data = adj_data[int(0.1 * len(adj_data)):, :]

    resultFileName = './cv_results.txt'
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold_num = 0
    for train_index, val_index in kf.split(cv_data):
        patience_num_matrix = np.zeros((1, 4))
        epoch_max_matrix = np.zeros((1, 4))
        hits_max_matrix = np.zeros((4, 3))
        ndcg_max_matrix = np.zeros((4, 3))
        fold_num += 1

        hypergraph_pos, hypergraph_neg_ls, tra_data, val_1, val_2, val_3, val_4 = get_train_val_data(
            cv_data, train_index, val_index, adj_data, drug_num, target_num, disease_num
        )
        hypergraph_pos = hypergraph_pos.to(device)
        hypergraph_neg_ls = [hg.to(device) for hg in hypergraph_neg_ls]
        tra_data = torch.tensor(tra_data, dtype=torch.long).to(device)
        val_1 = torch.tensor(val_1, dtype=torch.long).to(device)
        val_2 = torch.tensor(val_2, dtype=torch.long).to(device)
        val_3 = torch.tensor(val_3, dtype=torch.long).to(device)
        val_4 = torch.tensor(val_4, dtype=torch.long).to(device)

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

        for e in range(args.epochs):
            train(drug_features, target_features, disease_features, hetero_data, hypergraph_pos, hypergraph_neg_ls,
                  tra_data, model, optimizer, my_loss, args, device)
            evaluate(drug_features, target_features, disease_features, hetero_data, hypergraph_pos, hypergraph_neg_ls,
                 val_1, val_2, val_3, val_4, model, args, device)

            if patience_num_matrix[0][0] >= patience and patience_num_matrix[0][1] >= patience and \
                    patience_num_matrix[0][2] >= patience and patience_num_matrix[0][3] >= patience:
                break

        print(f'Fold {fold_num} completed successfully')
        for top, idx in zip(['top1', 'top3', 'top5'], [0, 1, 2]):
            write_type_1234(
                resultFileName, fold_num,
                hits_max_matrix[0][idx], ndcg_max_matrix[0][idx],
                hits_max_matrix[1][idx], ndcg_max_matrix[1][idx],
                hits_max_matrix[2][idx], ndcg_max_matrix[2][idx],
                hits_max_matrix[3][idx], ndcg_max_matrix[3][idx],
                epoch_max_matrix[0][0], epoch_max_matrix[0][1],
                epoch_max_matrix[0][2], epoch_max_matrix[0][3],
                top=top
            )
