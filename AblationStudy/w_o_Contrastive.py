# w_o_Contrastive.py
# 消融实验：w/o Contrastive（去除对比学习模块，仅保留双视图 + 扩散重建 + 监督损失）

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
    write_type_1234, parameters_set, build_hetero_graph, build_hypergraph,
    hit_ndcg_value, get_metrics
)
from Utils.process_smiles import espf_feature_extraction
from model_hyper_graph import BioEncoder, HeteroEncoder, HgnnEncoder, Decoder, DiffusionModel

from Utils.right_negative_sample_generate import neg_data_generate

warnings.filterwarnings("ignore")

# ============================ 新增：去除 Contrastive 的模型 ============================
class H_GCL_wo_Contrastive(nn.Module):
    def __init__(self, bio_encoder, hetero_encoder, hyper_encoder, decoder, embed_dim):
        super(H_GCL_wo_Contrastive, self).__init__()
        self.bio_encoder = bio_encoder
        self.hetero_encoder = hetero_encoder
        self.hyper_encoder = hyper_encoder
        self.decoder = decoder
        self.diffusion = DiffusionModel(embed_dim)  # 保留扩散模块

        # 增强参数（保留视图增强，但无对比）
        self.node_dropout_ratio = 0.15
        self.edge_dropout_ratio = 0.15
        self.hyperedge_dropout_ratio = 0.15

    def node_mask_hetero(self, x_dict, mask_ratio):
        for key in x_dict:
            mask = torch.rand(x_dict[key].size(0), device=x_dict[key].device) < mask_ratio
            x_dict[key][mask] = 0
        return x_dict

    def node_mask_hyper(self, embed, mask_ratio):
        mask = torch.rand(embed.size(0), device=embed.device) < mask_ratio
        embed_masked = embed.clone()
        embed_masked[mask] = 0
        return embed_masked

    def edge_dropout_hetero(self, edge_index_dict, drop_ratio):
        new_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            num_edges = edge_index.size(1)
            if num_edges == 0:
                new_dict[edge_type] = edge_index
                continue
            mask = torch.rand(num_edges, device=edge_index.device) > drop_ratio
            new_dict[edge_type] = edge_index[:, mask]
        return new_dict

    def hyperedge_dropout(self, edge_index, drop_ratio):
        if edge_index.size(1) == 0:
            return edge_index
        num_hyperedges = edge_index[1].max().item() + 1
        mask = torch.rand(num_hyperedges, device=edge_index.device) > drop_ratio
        kept_edges = mask[edge_index[1]]
        return edge_index[:, kept_edges]

    def diffusion_loss(self, embed_pred, embed_true):
        return F.mse_loss(embed_pred, embed_true)

    def forward(self, drug_feature, target_fea, disease_fea, hetero_data, hyper_pos,
                drug_id, target_id, disease_id, hyper_neg_list, device):
        # 1. 生物特征编码
        x_drug, x_target, x_disease = self.bio_encoder(drug_feature, target_fea, disease_fea)
        x_dict = {'drug': x_drug, 'target': x_target, 'disease': x_disease}
        embed_all = torch.cat([x_drug, x_target, x_disease], dim=0)  # 用于超图

        # 2. 生成增强视图（保留，但仅用于编码多样性）
        x_dict_view1 = self.node_mask_hetero({k: v.clone() for k, v in x_dict.items()}, self.node_dropout_ratio)
        edge_index_dict_view1 = self.edge_dropout_hetero(hetero_data.edge_index_dict, self.edge_dropout_ratio)

        embed_view1 = self.node_mask_hyper(embed_all, self.node_dropout_ratio)
        hyper_pos_view1 = self.hyperedge_dropout(hyper_pos, self.hyperedge_dropout_ratio)

        # 3. 编码
        hetero_embed_view1 = self.hetero_encoder(x_dict_view1, edge_index_dict_view1)
        hetero_embed_concat = torch.cat([hetero_embed_view1['drug'],
                                         hetero_embed_view1['target'],
                                         hetero_embed_view1['disease']], dim=0)

        hyper_embed_pos1 = self.hyper_encoder(embed_view1, hyper_pos_view1.long())

        # 4. 扩散：跨视图加噪与去噪重建（保留）
        t = torch.randint(0, self.diffusion.num_steps, (1,)).item()
        noisy_hetero = self.diffusion.forward_diffusion(hetero_embed_concat, t, device)[0]
        noisy_hyper = self.diffusion.forward_diffusion(hyper_embed_pos1, t, device)[0]

        pred_hetero_to_hyper = self.diffusion.denoise_step(noisy_hetero, t, hyper_embed_pos1, 'hetero_to_hyper', device)
        pred_hyper_to_hetero = self.diffusion.denoise_step(noisy_hyper, t, hetero_embed_concat, 'hyper_to_hetero', device)

        # 5. 最终图嵌入：使用去噪嵌入相加融合
        total_drug = x_drug.size(0)
        total_target = x_target.size(0)

        graph_embed = {
            'drug': hetero_embed_view1['drug'] + pred_hyper_to_hetero[:total_drug],
            'target': hetero_embed_view1['target'] + pred_hyper_to_hetero[total_drug:total_drug + total_target],
            'disease': hetero_embed_view1['disease'] + pred_hyper_to_hetero[total_drug + total_target:]
        }

        # 6. 解码器预测
        _, res = self.decoder(graph_embed, drug_id, target_id, disease_id)

        # 返回：仅用于计算扩散损失的预测嵌入（无对比输出）
        return None, res, pred_hetero_to_hyper, pred_hyper_to_hetero, hyper_embed_pos1, hetero_embed_concat
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

def train(drug_fea, target_fea, disease_fea, hetero_data, hyper_pos, hyper_neg_ls, train_data,
          model, optimizer, my_loss, args, device):
    model.train()
    optimizer.zero_grad()

    _, pred, pred_hetero_to_hyper, pred_hyper_to_hetero, hyper_true, hetero_true = model(
        drug_fea, target_fea, disease_fea, hetero_data, hyper_pos,
        train_data[:, 0], train_data[:, 1], train_data[:, 2], hyper_neg_ls, device
    )

    # 监督损失
    loss_1 = my_loss(pred.view(-1, 1), train_data[:, 3].view(-1, 1).float().to(device))

    # 扩散重建损失（保留）
    loss_3 = model.diffusion_loss(pred_hetero_to_hyper, hyper_true) + \
             model.diffusion_loss(pred_hyper_to_hetero, hetero_true)

    # 总损失（去除对比损失）
    loss = args.hypergraph_loss_ratio * loss_1 + args.diffusion_loss_ratio * loss_3

    loss.backward()
    optimizer.step()

    print(f'epoch:{e+1:03d}, loss:{loss.item():.6f}, bce:{loss_1.item():.6f}, diffusion:{loss_3.item():.6f}')
    return loss.item()

def test(drug_fea, target_fea, disease_fea, hetero_data, hyper_pos, hyper_neg_ls, val_data_list,
         model, args, device):
    model.eval()
    with torch.no_grad():
        for val_idx, val_data in enumerate(val_data_list):
            _, pred_val, _, _, _, _ = model(
                drug_fea, target_fea, disease_fea, hetero_data, hyper_pos,
                val_data[:, 0], val_data[:, 1], val_data[:, 2], hyper_neg_ls, device
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
    args.hypergraph_loss_ratio = 0.8  # 与原模型默认一致（监督占比高）
    args.diffusion_loss_ratio = 0.1   # 与原模型一致
    random_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # ============================ 数据路径（使用相对路径） ============================
    # DATA_PATH = "./Data"
    DATA_PATH = 'D:\\研究生阶段学习\学习\\HHCL_DTD(version01)-main\\Data'
    drug_smiles_file = os.path.join(DATA_PATH, "drug_smiles_124.csv")
    target_seq_file = os.path.join(DATA_PATH, "target_sequence.txt")
    drug_target_file = os.path.join(DATA_PATH, "drug_target.csv")
    drug_disease_file = os.path.join(DATA_PATH, "drug_disease.csv")
    dtd_file = os.path.join(DATA_PATH, "dtd_interaction_adjusted_modified.txt")

    # ============================ 加载特征 ============================
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

    target_cache_file = "target_features_espsf.pkl"
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

    disease_num = 177  # 请根据实际数据修改
    disease_features = torch.eye(disease_num)

    drug_features = drug_features.to(device)
    target_features = target_features.to(device)
    disease_features = disease_features.to(device)

    hetero_data = build_hetero_graph(drug_target_file, drug_disease_file, drug_num, target_num, disease_num).to(device)

    # ============================ 加载三元组 ============================
    adj_data = np.loadtxt(dtd_file, dtype=int)  # 自动按空白分隔（包括\t）
    adj_data = series_num(adj_data, drug_num, target_num, disease_num)
    np.random.shuffle(adj_data)

    cv_data = adj_data[int(0.1 * len(adj_data)):, :]
    resultFileName = './results_wo_Contrastive.txt'

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold_num = 0

    for train_index, val_index in kf.split(cv_data):
        fold_num += 1
        patience_num_matrix.fill(0)
        hits_max_matrix.fill(0)
        ndcg_max_matrix.fill(0)

        train_data_pos = cv_data[train_index]
        val_data_pos = cv_data[val_index]

        train_data_all, tr_neg_1, tr_neg_2, tr_neg_3, tr_neg_4, te_neg_1, te_neg_2, te_neg_3, te_neg_4 = neg_data_generate(
            adj_data, train_data_pos, val_data_pos, args.seed
        )

        train_data_pos = series_num(train_data_pos, drug_num, target_num, disease_num)
        train_data_neg_1 = series_num(np.array(tr_neg_1), drug_num, target_num, disease_num)

        hypergraph_pos = build_hypergraph(train_data_pos).to(device)
        hypergraph_neg_1 = build_hypergraph(train_data_neg_1).to(device)
        hypergraph_neg_ls = [hypergraph_neg_1]  # 保留，但模型中不使用

        train_data_all = series_num(np.array(train_data_all), drug_num, target_num, disease_num)
        val_data_1 = series_num(np.array(te_neg_1), drug_num, target_num, disease_num)
        val_data_2 = series_num(np.array(te_neg_2), drug_num, target_num, disease_num)
        val_data_3 = series_num(np.array(te_neg_3), drug_num, target_num, disease_num)
        val_data_4 = series_num(np.array(te_neg_4), drug_num, target_num, disease_num)

        train_data_all = torch.tensor(train_data_all, dtype=torch.long).to(device)
        val_data_list = [torch.tensor(v, dtype=torch.long).to(device) for v in [val_data_1, val_data_2, val_data_3, val_data_4]]

        # ============================ 模型 ============================
        model = H_GCL_wo_Contrastive(
            bio_encoder=BioEncoder(128, 128, disease_num, args.bio_out_dim),
            hetero_encoder=HeteroEncoder(args.bio_out_dim, args.hgnn_dim_1),
            hyper_encoder=HgnnEncoder(args.bio_out_dim, args.hgnn_dim_1),
            decoder=Decoder(args.hgnn_dim_1 * 3),
            embed_dim=args.hgnn_dim_1
        ).to(device)

        my_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for e in range(args.epochs):
            train(drug_features, target_features, disease_features, hetero_data, hypergraph_pos, hypergraph_neg_ls,
                  train_data_all, model, optimizer, my_loss, args, device)

            test(drug_features, target_features, disease_features, hetero_data, hypergraph_pos, hypergraph_neg_ls,
                 val_data_list, model, args, device)

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

    print("w/o Contrastive 消融实验完成，结果保存在:", resultFileName)