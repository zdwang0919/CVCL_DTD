# wo_Hetero.py
# 消融实验：w/o Hetero（去除异构图分支，仅使用超图 + BioEncoder 进行预测）
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
import random
import csv
from sklearn.model_selection import KFold
from Utils.Tools2 import (
    write_type_1234, parameters_set, build_hypergraph,
    hit_ndcg_value, get_metrics
)
from Utils.process_smiles import espf_feature_extraction
from model_hyper_graph import BioEncoder, HgnnEncoder, Decoder
from Utils.right_negative_sample_generate import neg_data_generate

warnings.filterwarnings("ignore")

# ============================ 新增：仅超图的模型（无异构图） ============================
class H_GCL_wo_Hetero(nn.Module):
    def __init__(self, bio_encoder, hyper_encoder, decoder, embed_dim):
        super(H_GCL_wo_Hetero, self).__init__()
        self.bio_encoder = bio_encoder
        self.hyper_encoder = hyper_encoder
        self.decoder = decoder
        self.embed_dim = embed_dim

        # 超图增强参数
        self.dropout_ratio = 0.15  # 节点特征掩码比例
        self.hyperedge_dropout_ratio = 0.15

        # 对比学习投影头
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.temperature = 0.2

    def node_mask(self, embed, mask_ratio):
        mask = torch.rand(embed.size(0), device=embed.device) < mask_ratio
        embed_masked = embed.clone()
        embed_masked[mask] = 0
        return embed_masked

    def hyperedge_dropout(self, edge_index, drop_ratio):
        if edge_index.size(1) == 0:
            return edge_index
        num_hyperedges = edge_index[1].max().item() + 1
        mask = torch.rand(num_hyperedges, device=edge_index.device) > drop_ratio
        kept_edges = mask[edge_index[1]]
        return edge_index[:, kept_edges]

    def info_nce_loss(self, z1, z2, neg_z_list):
        z1 = F.normalize(self.projector(z1), dim=-1)
        z2 = F.normalize(self.projector(z2), dim=-1)
        neg_z_list = [F.normalize(self.projector(neg_z), dim=-1) for neg_z in neg_z_list]

        pos_score = (z1 * z2).sum(dim=-1) / self.temperature
        pos_score = torch.exp(pos_score)

        neg_scores = [torch.exp((z1 * neg_z).sum(dim=-1) / self.temperature) for neg_z in neg_z_list]
        neg_score = sum(neg_scores) + 1e-15

        loss = -torch.log(pos_score / (pos_score + neg_score)).mean()
        return loss

    def forward(self, drug_feature, target_fea, disease_fea,
                hyper_pos, hyper_neg_list,  # hyper_pos 是正样本超图，hyper_neg_list 是负样本超图列表
                drug_id, target_id, disease_id, device):
        # 1. 生物特征编码 → 拼接成统一节点特征
        x_drug, x_target, x_disease = self.bio_encoder(drug_feature, target_fea, disease_fea)
        embed = torch.cat([x_drug, x_target, x_disease], dim=0)  # [N_total, dim]

        # 2. 生成两个增强视图（节点掩码 + 超边丢弃）
        embed_view1 = self.node_mask(embed, self.dropout_ratio)
        embed_view2 = self.node_mask(embed, self.dropout_ratio)

        hyper_pos_view1 = self.hyperedge_dropout(hyper_pos, self.hyperedge_dropout_ratio)
        hyper_pos_view2 = self.hyperedge_dropout(hyper_pos, self.hyperedge_dropout_ratio)

        # 3. 超图编码
        hyper_embed1 = self.hyper_encoder(embed_view1, hyper_pos_view1.long())
        hyper_embed2 = self.hyper_encoder(embed_view2, hyper_pos_view2.long())

        # 负样本超图编码（用于对比学习）
        hyper_embed_neg_list = [self.hyper_encoder(embed, neg_edge.long()) for neg_edge in hyper_neg_list]

        # 4. 解码器输入：使用 view1 的超图嵌入（也可取平均）
        total_drug = x_drug.size(0)
        total_target = x_target.size(0)

        graph_embed = {
            'drug': hyper_embed1[:total_drug],
            'target': hyper_embed1[total_drug:total_drug + total_target],
            'disease': hyper_embed1[total_drug + total_target:]
        }

        # 5. 解码器预测三元组得分
        _, res = self.decoder(graph_embed, drug_id, target_id, disease_id)

        return None, res, hyper_embed1, hyper_embed2, hyper_embed_neg_list
# ==============================================================================

def series_num(data, drug_num, target_num, disease_num):
    data = data.copy().astype(int)
    invalid_rows = []
    for i, line in enumerate(data):
        if not (0 <= line[0] < drug_num and 0 <= line[1] < target_num and 0 <= line[2] < disease_num):
            invalid_rows.append((i, line))
    if invalid_rows:
        print(f"series_num: Found {len(invalid_rows)} invalid rows")
        data = np.delete(data, [r[0] for r in invalid_rows], axis=0)
    return data

def random_seed(sd):
    random.seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def train(drug_fea, target_fea, disease_fea, hyper_pos, hyper_neg_ls, train_data,
          model, optimizer, my_loss, args, device):
    model.train()
    optimizer.zero_grad()

    _, pred, embed_pos1, embed_pos2, neg_list = model(
        drug_fea, target_fea, disease_fea, hyper_pos, hyper_neg_ls,
        train_data[:, 0], train_data[:, 1], train_data[:, 2], device
    )

    # 监督损失（BCE）
    loss_1 = my_loss(pred.view(-1, 1), train_data[:, 3].view(-1, 1).float().to(device))

    # 超图视图对比损失
    loss_2 = model.info_nce_loss(embed_pos1, embed_pos2, neg_list)

    # 总损失（使用原模型相同的权重设置）
    loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2

    loss.backward()
    optimizer.step()

    print(f'epoch:{e+1:03d}, loss:{loss.item():.6f}, bce:{loss_1.item():.6f}, contrast:{loss_2.item():.6f}')
    return loss.item()

def test(drug_fea, target_fea, disease_fea, hyper_pos, hyper_neg_ls, val_data_list,
         model, args, device):
    model.eval()
    with torch.no_grad():
        for val_idx, val_data in enumerate(val_data_list):
            _, pred_val, _, _, _ = model(
                drug_fea, target_fea, disease_fea, hyper_pos, hyper_neg_ls,
                val_data[:, 0], val_data[:, 1], val_data[:, 2], device
            )
            hits_1, ndcg_1 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_1)
            hits_3, ndcg_3 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_3)
            hits_5, ndcg_5 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_5)

            real_score = val_data[:, 3].cpu().numpy()
            predict_score = pred_val.cpu().numpy()
            metrics = get_metrics(real_score, predict_score)
            aupr, auc = metrics[0], metrics[1]

            print(f'val:{val_idx+1}, hits@3:{hits_3:.6f}, ndcg@3:{ndcg_3:.6f}, aupr:{aupr:.6f}, auc:{auc:.6f}')

            early_stop(val_idx, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5)

# early_stop 相关全局变量
hits_max_matrix = np.zeros((4, 3))
ndcg_max_matrix = np.zeros((4, 3))
epoch_max_matrix = np.zeros((1, 4))
patience_num_matrix = np.zeros((1, 4))
patience = 50

def early_stop(val_idx, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5):
    global hits_max_matrix, ndcg_max_matrix, epoch_max_matrix, patience_num_matrix
    if hits_1 > hits_max_matrix[val_idx][0]:
        hits_max_matrix[val_idx][0] = hits_1
        hits_max_matrix[val_idx][1] = hits_3
        hits_max_matrix[val_idx][2] = hits_5
        ndcg_max_matrix[val_idx][0] = ndcg_1
        ndcg_max_matrix[val_idx][1] = ndcg_3
        ndcg_max_matrix[val_idx][2] = ndcg_5
        epoch_max_matrix[0][val_idx] = e + 1
        patience_num_matrix[0][val_idx] = 0
    else:
        patience_num_matrix[0][val_idx] += 1

if __name__ == '__main__':
    args = parameters_set()
    args.hypergraph_loss_ratio = 0.6  # 监督损失占比（可根据需要调整）
    random_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # ============================ 数据路径（请根据你的实际路径修改） ============================
    DATA_PATH = "../Data"  # 推荐使用相对路径，假设 Data 文件夹与 py 文件同级
    # 如果你的路径不同，请修改为：
    # DATA_PATH = r"D:\研究生阶段学习\学习\HHCL_DTD(version01)-main 测试版本 - 副本\Data"

    drug_smiles_file = os.path.join(DATA_PATH, "drug_smiles_124.csv")
    target_seq_file = os.path.join(DATA_PATH, "target_sequence.txt")
    dtd_file = os.path.join(DATA_PATH, "dtd_interaction_adjusted_modified.txt")

    # ============================ 加载特征 ============================
    # Drug ESPF
    drug_features = []
    with open(drug_smiles_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            smiles = row[1]
            feature = espf_feature_extraction(smiles, is_smiles=True)
            drug_features.append(feature)
    drug_features = torch.tensor(drug_features, dtype=torch.float)
    drug_num = drug_features.shape[0]

    # Target ESPF
    target_cache_file = "../target_features_espsf.pkl"
    if os.path.exists(target_cache_file):
        with open(target_cache_file, 'rb') as f:
            target_features = pickle.load(f)
    else:
        target_features = []
        with open(target_seq_file, 'r', encoding='utf-8') as f:
            for line in f:
                seq = line.strip()
                if seq:
                    feature = espf_feature_extraction(seq, is_smiles=False)
                    target_features.append(feature)
        target_features = torch.tensor(target_features, dtype=torch.float)
        with open(target_cache_file, 'wb') as f:
            pickle.dump(target_features, f)
    target_num = target_features.shape[0]

    # Disease one-hot
    disease_num = 177  # 请根据你的实际数据修改
    disease_features = torch.eye(disease_num)

    drug_features = drug_features.to(device)
    target_features = target_features.to(device)
    disease_features = disease_features.to(device)

    # ============================ 加载三元组并构建超图 ============================
    adj_data = np.loadtxt(dtd_file, dtype=int, delimiter='\t')  # 假设为 csv 格式
    adj_data = series_num(adj_data, drug_num, target_num, disease_num)
    np.random.shuffle(adj_data)

    cv_data = adj_data[int(0.1 * len(adj_data)):, :]
    test_data = adj_data[:int(0.1 * len(adj_data)), :]

    resultFileName = './results_wo_Hetero.txt'

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold_num = 0

    for train_index, val_index in kf.split(cv_data):
        fold_num += 1
        patience_num_matrix.fill(0)
        hits_max_matrix.fill(0)
        ndcg_max_matrix.fill(0)

        train_data_pos = cv_data[train_index]
        val_data_pos = cv_data[val_index]

        # 负采样
        train_data_all, tr_neg_1, tr_neg_2, tr_neg_3, tr_neg_4, te_neg_1, te_neg_2, te_neg_3, te_neg_4 = neg_data_generate(
            adj_data, train_data_pos, val_data_pos, args.seed
        )

        train_data_pos = series_num(train_data_pos, drug_num, target_num, disease_num)
        train_data_neg_1 = series_num(np.array(tr_neg_1), drug_num, target_num, disease_num)

        # 构建超图
        hypergraph_pos = build_hypergraph(train_data_pos)
        hypergraph_neg_1 = build_hypergraph(train_data_neg_1)
        hypergraph_neg_ls = [hypergraph_neg_1]

        train_data_all = series_num(np.array(train_data_all), drug_num, target_num, disease_num)
        val_data_1 = series_num(np.array(te_neg_1), drug_num, target_num, disease_num)
        val_data_2 = series_num(np.array(te_neg_2), drug_num, target_num, disease_num)
        val_data_3 = series_num(np.array(te_neg_3), drug_num, target_num, disease_num)
        val_data_4 = series_num(np.array(te_neg_4), drug_num, target_num, disease_num)

        hypergraph_pos = hypergraph_pos.to(device)
        hypergraph_neg_ls = [hg.to(device) for hg in hypergraph_neg_ls]

        train_data_all = torch.tensor(train_data_all, dtype=torch.long).to(device)
        val_data_list = [torch.tensor(v, dtype=torch.long).to(device) for v in [val_data_1, val_data_2, val_data_3, val_data_4]]

        # ============================ 模型 ============================
        model = H_GCL_wo_Hetero(
            bio_encoder=BioEncoder(128, 128, disease_num, args.bio_out_dim),
            hyper_encoder=HgnnEncoder(args.bio_out_dim, args.hgnn_dim_1),
            decoder=Decoder(args.hgnn_dim_1 * 3),
            embed_dim=args.hgnn_dim_1
        ).to(device)

        my_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for e in range(args.epochs):
            train(drug_features, target_features, disease_features,
                  hypergraph_pos, hypergraph_neg_ls, train_data_all,
                  model, optimizer, my_loss, args, device)

            test(drug_features, target_features, disease_features,
                 hypergraph_pos, hypergraph_neg_ls, val_data_list,
                 model, args, device)

            if patience_num_matrix.min() >= patience:
                break

        print(f'Fold {fold_num} completed')

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

    print("w/o Hetero 消融实验完成，结果保存在:", resultFileName)