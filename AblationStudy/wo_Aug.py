# w_o_Aug.py
# 消融实验：w/o Aug（去除视图增强模块，直接使用原始视图进行编码）

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

# ============================ 新增：去除 Aug 的模型 ============================
class H_GCL_wo_Aug(nn.Module):
    def __init__(self, bio_encoder, hetero_encoder, hyper_encoder, decoder, embed_dim):
        super(H_GCL_wo_Aug, self).__init__()
        self.bio_encoder = bio_encoder
        self.hetero_encoder = hetero_encoder
        self.hyper_encoder = hyper_encoder
        self.decoder = decoder
        self.diffusion = DiffusionModel(embed_dim)

        # 对比学习投影头（保留）
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.temperature = 0.2

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

    def diffusion_loss(self, embed_pred, embed_true):
        return F.mse_loss(embed_pred, embed_true)

    def forward(self, drug_feature, target_fea, disease_fea, hetero_data, hyper_pos,
                drug_id, target_id, disease_id, hyper_neg_list, device):
        # 1. 生物特征编码
        x_drug, x_target, x_disease = self.bio_encoder(drug_feature, target_fea, disease_fea)
        x_dict = {'drug': x_drug, 'target': x_target, 'disease': x_disease}
        embed_all = torch.cat([x_drug, x_target, x_disease], dim=0)  # 用于超图

        # 2. 无增强：直接使用原始视图
        hetero_embed = self.hetero_encoder(x_dict, hetero_data.edge_index_dict)
        hetero_embed_concat = torch.cat([hetero_embed['drug'],
                                         hetero_embed['target'],
                                         hetero_embed['disease']], dim=0)

        hyper_embed_pos = self.hyper_encoder(embed_all, hyper_pos.long())
        hyper_embed_neg_list = [self.hyper_encoder(embed_all, neg.long()) for neg in hyper_neg_list]

        # 3. 扩散：跨视图加噪与去噪重建（保留）
        t = torch.randint(0, self.diffusion.num_steps, (1,)).item()
        noisy_hetero, _ = self.diffusion.forward_diffusion(hetero_embed_concat, t, device)
        noisy_hyper, _ = self.diffusion.forward_diffusion(hyper_embed_pos, t, device)

        pred_hetero_to_hyper = self.diffusion.denoise_step(noisy_hetero, t, hyper_embed_pos, 'hetero_to_hyper', device)
        pred_hyper_to_hetero = self.diffusion.denoise_step(noisy_hyper, t, hetero_embed_concat, 'hyper_to_hetero', device)

        # 4. 最终图嵌入：使用去噪嵌入相加融合
        total_drug = x_drug.size(0)
        total_target = x_target.size(0)

        graph_embed = {
            'drug': hetero_embed['drug'] + pred_hyper_to_hetero[:total_drug],
            'target': hetero_embed['target'] + pred_hyper_to_hetero[total_drug:total_drug + total_target],
            'disease': hetero_embed['disease'] + pred_hyper_to_hetero[total_drug + total_target:]
        }

        # 5. 解码器预测
        _, res = self.decoder(graph_embed, drug_id, target_id, disease_id)

        # 返回用于对比和扩散的嵌入
        return None, res, hetero_embed_concat, hyper_embed_pos, hyper_embed_neg_list, pred_hetero_to_hyper, pred_hyper_to_hetero, hyper_embed_pos, hetero_embed_concat
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

    _, pred, hetero_embed, hyper_embed_pos, hyper_neg_list, pred_hetero_to_hyper, pred_hyper_to_hetero, hyper_true, hetero_true = model(
        drug_fea, target_fea, disease_fea, hetero_data, hyper_pos,
        train_data[:, 0], train_data[:, 1], train_data[:, 2], hyper_neg_ls, device
    )

    # 监督损失
    loss_1 = my_loss(pred.view(-1, 1), train_data[:, 3].view(-1, 1).float().to(device))

    # 对比损失（保留）
    loss_2 = model.info_nce_loss(hetero_embed, hyper_embed_pos, hyper_neg_list)

    # 扩散重建损失（保留）
    loss_3 = model.diffusion_loss(pred_hetero_to_hyper, hyper_true) + \
             model.diffusion_loss(pred_hyper_to_hetero, hetero_true)

    # 总损失
    loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2 + args.diffusion_loss_ratio * loss_3

    loss.backward()
    optimizer.step()

    print(f'epoch:{e+1:03d}, loss:{loss.item():.6f}, bce:{loss_1.item():.6f}, contrast:{loss_2.item():.6f}, diffusion:{loss_3.item():.6f}')
    return loss.item()

def test(drug_fea, target_fea, disease_fea, hetero_data, hyper_pos, hyper_neg_ls, val_data_list,
         model, args, device):
    model.eval()
    with torch.no_grad():
        for val_idx, val_data in enumerate(val_data_list):
            _, pred_val, _, _, _, _, _, _, _ = model(
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
    args.hypergraph_loss_ratio = 0.8  # 与原模型默认一致
    args.diffusion_loss_ratio = 0.1   # 与原模型一致
    random_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # ============================ 数据路径（使用相对路径） ============================
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
    resultFileName = './results_wo_Aug.txt'

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
        hypergraph_neg_ls = [hypergraph_neg_1]

        train_data_all = series_num(np.array(train_data_all), drug_num, target_num, disease_num)
        val_data_1 = series_num(np.array(te_neg_1), drug_num, target_num, disease_num)
        val_data_2 = series_num(np.array(te_neg_2), drug_num, target_num, disease_num)
        val_data_3 = series_num(np.array(te_neg_3), drug_num, target_num, disease_num)
        val_data_4 = series_num(np.array(te_neg_4), drug_num, target_num, disease_num)

        train_data_all = torch.tensor(train_data_all, dtype=torch.long).to(device)
        val_data_list = [torch.tensor(v, dtype=torch.long).to(device) for v in [val_data_1, val_data_2, val_data_3, val_data_4]]

        # ============================ 模型 ============================
        model = H_GCL_wo_Aug(
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

    print("w/o Aug 消融实验完成，结果保存在:", resultFileName)