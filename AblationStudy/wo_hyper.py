# w_o_Hyper.py
# 消融实验：w/o Hyper（去除超图分支，仅使用异构图 + BioEncoder 进行预测）

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
    write_type_1234, parameters_set, build_hetero_graph,
    hit_ndcg_value, get_metrics
)
from Utils.process_smiles import espf_feature_extraction
from model_hyper_graph import BioEncoder, HeteroEncoder, Decoder
from Utils.right_negative_sample_generate import neg_data_generate

warnings.filterwarnings("ignore")

# ============================ 新增：仅异构图的模型（无超图） ============================
class H_GCL_wo_Hyper(nn.Module):
    def __init__(self, bio_encoder, hetero_encoder, decoder, embed_dim):
        super(H_GCL_wo_Hyper, self).__init__()
        self.bio_encoder = bio_encoder
        self.hetero_encoder = hetero_encoder
        self.decoder = decoder
        self.embed_dim = embed_dim

        # 用于节点特征增强的掩码（与原模型保持一致）
        self.dropout_ratio = 0.15
        self.edge_dropout_ratio = 0.15

        # 对比学习投影头（与原模型一致）
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.temperature = 0.2

    def node_mask_hetero(self, x_dict, mask_ratio):
        for key in x_dict:
            mask = torch.rand(x_dict[key].size(0), device=x_dict[key].device) < mask_ratio
            x_dict[key][mask] = 0
        return x_dict

    def edge_dropout_hetero(self, edge_index_dict, drop_ratio):
        new_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges, device=edge_index.device) > drop_ratio
            new_edge_index_dict[edge_type] = edge_index[:, mask]
        return new_edge_index_dict

    def info_nce_loss(self, z1, z2, neg_z_list):
        z1 = F.normalize(self.projector(z1), dim=-1)
        z2 = F.normalize(self.projector(z2), dim=-1)
        neg_z_list = [F.normalize(self.projector(neg_z), dim=-1) for neg_z in neg_z_list]

        pos_score = (z1 * z2).sum(dim=-1) / self.temperature
        pos_score = torch.exp(pos_score)

        neg_scores = [torch.exp((z1 * neg_z).sum(dim=-1) / self.temperature) for neg_z in neg_z_list]
        neg_score = sum(neg_scores)

        loss = -torch.log(pos_score / (pos_score + neg_score + 1e-15)).mean()
        return loss

    def forward(self, drug_feature, target_fea, disease_fea, hetero_data,
                drug_id, target_id, disease_id, neg_hyper_list, device):
        # 1. 生物特征编码
        x_drug, x_target, x_disease = self.bio_encoder(drug_feature, target_fea, disease_fea)
        x_dict = {'drug': x_drug, 'target': x_target, 'disease': x_disease}

        # 2. 生成两个增强视图（节点掩码 + 边丢弃）
        x_dict_view1 = self.node_mask_hetero({k: v.clone() for k, v in x_dict.items()}, self.dropout_ratio)
        x_dict_view2 = self.node_mask_hetero({k: v.clone() for k, v in x_dict.items()}, self.dropout_ratio)

        edge_index_dict_view1 = self.edge_dropout_hetero(hetero_data.edge_index_dict, self.edge_dropout_ratio)
        edge_index_dict_view2 = self.edge_dropout_hetero(hetero_data.edge_index_dict, self.edge_dropout_ratio)

        # 3. 异构图编码
        hetero_embed_view1 = self.hetero_encoder(x_dict_view1, edge_index_dict_view1)
        hetero_embed_view2 = self.hetero_encoder(x_dict_view2, edge_index_dict_view2)

        # 拼接所有节点嵌入用于对比学习
        embed1 = torch.cat([hetero_embed_view1['drug'],
                            hetero_embed_view1['target'],
                            hetero_embed_view1['disease']], dim=0)
        embed2 = torch.cat([hetero_embed_view2['drug'],
                            hetero_embed_view2['target'],
                            hetero_embed_view2['disease']], dim=0)

        # 这里使用 view1 作为最终图嵌入（也可取平均）
        graph_embed = {
            'drug': hetero_embed_view1['drug'],
            'target': hetero_embed_view1['target'],
            'disease': hetero_embed_view1['disease']
        }

        # 4. 解码器预测三元组得分
        _, res = self.decoder(graph_embed, drug_id, target_id, disease_id)

        # 负样本嵌入（这里仅用一个负超图作为负样本占位，原模型中是超图负样本）
        # 由于无超图，我们随便取一个负样本（这里用 view1 自身作为占位，实际不参与损失）
        graph_embed_neg_list = [embed1]

        return None, res, embed1, embed2, graph_embed_neg_list
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

def train(drug_fea, target_fea, disease_fea, hetero_data, train_data, model, optimizer, my_loss, args, device):
    model.train()
    optimizer.zero_grad()

    _, pred, embed_pos1, embed_pos2, neg_list = model(
        drug_fea, target_fea, disease_fea, hetero_data,
        train_data[:, 0], train_data[:, 1], train_data[:, 2], None, device
    )

    # 监督损失（BCE）
    loss_1 = my_loss(pred.view(-1, 1), train_data[:, 3].view(-1, 1).float().to(device))

    # 对比损失（可选：若完全去除无监督也可注释掉）
    loss_2 = model.info_nce_loss(embed_pos1, embed_pos2, neg_list)

    # 总损失：可调节对比损失权重（原模型中 hypergraph_loss_ratio 控制监督比例）
    loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2

    loss.backward()
    optimizer.step()

    print(f'epoch:{e+1:03d}, loss:{loss.item():.6f}, bce:{loss_1.item():.6f}, contrast:{loss_2.item():.6f}')
    return loss.item()

def test(drug_fea, target_fea, disease_fea, hetero_data, val_data_1, val_data_2, val_data_3, val_data_4,
         model, args, device):
    model.eval()
    with torch.no_grad():
        for val_data in [val_data_1, val_data_2, val_data_3, val_data_4]:
            _, pred_val, _, _, _ = model(
                drug_fea, target_fea, disease_fea, hetero_data,
                val_data[:, 0], val_data[:, 1], val_data[:, 2], None, device
            )
            hits_1, ndcg_1 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_1)
            hits_3, ndcg_3 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_3)
            hits_5, ndcg_5 = hit_ndcg_value(pred_val, val_data.cpu().numpy(), args.top_5)

            real_score = val_data[:, 3].cpu().numpy()
            predict_score = pred_val.cpu().numpy()
            metrics = get_metrics(real_score, predict_score)
            aupr, auc = metrics[0], metrics[1]

            print(f'hits@3:{hits_3:.6f}, ndcg@3:{ndcg_3:.6f}, aupr:{aupr:.6f}, auc:{auc:.6f}')

            # early stop 逻辑（与原代码保持一致）
            early_stop(val_data, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5)

# early stop 相关全局变量（与原代码一致）
hits_max_matrix = np.zeros((4, 3))
ndcg_max_matrix = np.zeros((4, 3))
epoch_max_matrix = np.zeros((1, 4))
patience_num_matrix = np.zeros((1, 4))
patience = 50  # 可调整

def early_stop(val_data, e, hits_1, ndcg_1, hits_3, ndcg_3, hits_5, ndcg_5):
    global hits_max_matrix, ndcg_max_matrix, epoch_max_matrix, patience_num_matrix
    idx = 0 if val_data is val_data_1 else 1 if val_data is val_data_2 else 2 if val_data is val_data_3 else 3
    if hits_1 > hits_max_matrix[idx][0]:
        hits_max_matrix[idx] = [hits_1, hits_3, hits_5]
        ndcg_max_matrix[idx] = [ndcg_1, ndcg_3, ndcg_5]
        epoch_max_matrix[0][idx] = e + 1
        patience_num_matrix[0][idx] = 0
    else:
        patience_num_matrix[0][idx] += 1

if __name__ == '__main__':
    args = parameters_set()
    args.hypergraph_loss_ratio = 0.6  # 控制监督损失占比
    random_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # ============================ 数据路径（请根据实际情况修改） ============================
    data_path = 'D:\\研究生阶段学习\学习\\HHCL_DTD(version01)-main\\Data'
    drug_smiles_file = os.path.join(data_path, "drug_smiles_124.csv")
    target_seq_file = os.path.join(data_path, "target_sequence.txt")
    drug_target_file = os.path.join(data_path, "drug_target.csv")
    drug_disease_file = os.path.join(data_path, "drug_disease.csv")
    dtd_file = os.path.join(data_path, "dtd_interaction_adjusted_modified.txt")  # 三元组文件

    # ============================ 加载特征 ============================
    # Drug: ESPF + Graph (但本消融只用 ESPF 部分，原模型也如此)
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

    # Target: ESPF
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

    # Disease: one-hot
    disease_num = 177  # 根据你的数据集修改
    disease_features = torch.eye(disease_num)

    drug_features = drug_features.to(device)
    target_features = target_features.to(device)
    disease_features = disease_features.to(device)

    hetero_data = build_hetero_graph(drug_target_file, drug_disease_file, drug_num, target_num, disease_num).to(device)

    # ============================ 加载三元组 ============================
    adj_data = np.loadtxt(dtd_file, dtype=int, delimiter='\t')
    adj_data = series_num(adj_data, drug_num, target_num, disease_num)
    np.random.shuffle(adj_data)

    cv_data = adj_data[int(0.1 * len(adj_data)):, :]  # 90% 用于 CV
    test_data = adj_data[:int(0.1 * len(adj_data)), :]

    resultFileName = './results_wo_Hyper.txt'

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold_num = 0

    for train_index, val_index in kf.split(cv_data):
        fold_num += 1
        patience_num_matrix.fill(0)
        hits_max_matrix.fill(0)
        ndcg_max_matrix.fill(0)

        train_data_pos = cv_data[train_index]
        val_data_pos = cv_data[val_index]

        # 负采样（与原模型一致）
        train_data_all, tr_neg_1, tr_neg_2, tr_neg_3, tr_neg_4, te_neg_1, te_neg_2, te_neg_3, te_neg_4 = neg_data_generate(
            adj_data, train_data_pos, val_data_pos, args.seed
        )

        train_data_all = series_num(np.array(train_data_all), drug_num, target_num, disease_num)
        val_data_1 = series_num(np.array(te_neg_1), drug_num, target_num, disease_num)
        val_data_2 = series_num(np.array(te_neg_2), drug_num, target_num, disease_num)
        val_data_3 = series_num(np.array(te_neg_3), drug_num, target_num, disease_num)
        val_data_4 = series_num(np.array(te_neg_4), drug_num, target_num, disease_num)

        train_data_all = torch.tensor(train_data_all, dtype=torch.long).to(device)
        val_data_1 = torch.tensor(val_data_1, dtype=torch.long).to(device)
        val_data_2 = torch.tensor(val_data_2, dtype=torch.long).to(device)
        val_data_3 = torch.tensor(val_data_3, dtype=torch.long).to(device)
        val_data_4 = torch.tensor(val_data_4, dtype=torch.long).to(device)

        # ============================ 模型 ============================
        model = H_GCL_wo_Hyper(
            bio_encoder=BioEncoder(128, 128, disease_num, args.bio_out_dim),
            hetero_encoder=HeteroEncoder(args.bio_out_dim, args.hgnn_dim_1),
            decoder=Decoder(args.hgnn_dim_1 * 3),
            embed_dim=args.hgnn_dim_1
        ).to(device)

        my_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # global val_data_1, val_data_2, val_data_3, val_data_4  # for early_stop

        for e in range(args.epochs):
            train(drug_features, target_features, disease_features, hetero_data,
                  train_data_all, model, optimizer, my_loss, args, device)

            test(drug_features, target_features, disease_features, hetero_data,
                 val_data_1, val_data_2, val_data_3, val_data_4, model, args, device)

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

    print("w/o Hyper 消融实验完成，结果保存在:", resultFileName)