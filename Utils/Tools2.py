import argparse
import csv
import numpy as np
import torch
from torch_geometric.data import HeteroData, Data
from Utils.process_smiles import smile_to_graph, GraphDataset_v, collate


def drug_fea_process(smiles_file, drug_num):
    reader = csv.reader(open(smiles_file, encoding='utf-8'))
    smile_graph = []
    for item in reader:
        smile = item[1]
        g = smile_to_graph(smile)
        smile_graph.append(g)
    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False, collate_fn=collate)
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_data


def build_hetero_graph(drug_target_file, drug_disease_file, drug_num, target_num, disease_num):
    # 读取药物-靶标交互
    drug_target = np.loadtxt(drug_target_file, delimiter=',')
    drug_target_edges = torch.tensor(drug_target[:, :2].T, dtype=torch.long)

    # 读取药物-疾病交互
    drug_disease = np.loadtxt(drug_disease_file, delimiter=',')
    drug_disease_edges = torch.tensor(drug_disease[:, :2].T, dtype=torch.long)

    # 验证边索引范围
    assert drug_target_edges[0].max() < drug_num, f"Drug ID exceeds {drug_num - 1}"
    assert drug_target_edges[1].max() < target_num, f"Target ID exceeds {target_num - 1}"
    assert drug_disease_edges[0].max() < drug_num, f"Drug ID exceeds {drug_num - 1}"
    assert drug_disease_edges[1].max() < disease_num, f"Disease ID exceeds {disease_num - 1}"

    # 创建异构图
    hetero_data = HeteroData()

    # 节点数量
    hetero_data['drug'].num_nodes = drug_num
    hetero_data['target'].num_nodes = target_num
    hetero_data['disease'].num_nodes = disease_num

    # 边（添加反向边）
    """创建反向边的作用：
    （1）、 实现双向信息流动
    原始边（如 drug→target）只能让信息从药物节点流向靶标节点
    添加反向边（target→drug）后，信息也可以从靶标节点流回药物节点
    这对于GNN的消息传递机制至关重要，使信息能在图中双向传播
    （2）、 适应GNN的聚合机制
    大多数GNN架构（如GCN、GAT）在聚合邻居信息时：
    默认只考虑"入边"（指向当前节点的边）
    没有反向边时，靶标节点无法感知哪些药物指向它
    反向边确保每个关系都能被双向处理
    （3）、 保持关系对称性
    对于某些对称关系（如"interacts"）：
    药物与靶标的交互本质上是双向的
    只保留单向边会人为地丢失这种对称性
    反向边明确表示了这种双向性质
    （4）、支持异构图卷积操作
    PyG的异构图卷积层（如HeteroConv）通常：
    独立处理每种边类型
    没有反向边时，某些方向的卷积操作无法进行
    例如：没有target→drug边，就无法从靶标角度聚合药物信息
    """
    hetero_data['drug', 'interacts', 'target'].edge_index = drug_target_edges
    hetero_data['target', 'interacts', 'drug'].edge_index = drug_target_edges.flip(0)
    hetero_data['drug', 'treats', 'disease'].edge_index = drug_disease_edges
    hetero_data['disease', 'treats', 'drug'].edge_index = drug_disease_edges.flip(0)

    return hetero_data


def build_hypergraph(data):
    """
    Build a hypergraph edge index tensor for HypergraphConv.
    :param data: Array of shape (N, 4) with columns [drug_idx, target_idx, disease_idx, label]
    :return: edge_index tensor of shape (2, N*3) where first row is node indices, second row is hyperedge indices
    """
    N = len(data)
    # print(N)  N表示N个三元组，代表N条超边
    node_indices = data[:, 0:3].flatten()  # Shape: (N*3,) - [drug_idx, target_idx, disease_idx, ...]
    hyperedge_indices = np.repeat(np.arange(N), 3)  # Shape: (N*3,) - [0, 0, 0, 1, 1, 1, ...]
    edge_index = np.vstack((node_indices, hyperedge_indices))  # Shape: (2, N*3)
    edge_index = torch.from_numpy(edge_index).type(torch.long)
    return edge_index


def parameters_set():
    parser = argparse.ArgumentParser(description='Heterogeneous and Hypergraph Neural Network')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hypergraph_loss_ratio', type=float, default=0.8)# 原默认超图损失率：default=0.8    // 实验改为0.6
    #--------------------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--epochs', type=int, default=800) #原始的默认epochs为800，现在做的是300的实验

    #--------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--top_1', type=int, default=1)
    parser.add_argument('--top_3', type=int, default=3)
    parser.add_argument('--top_5', type=int, default=5)
    parser.add_argument('--bio_out_dim', type=int, default=32)
    parser.add_argument('--hgnn_dim_1', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.005) # 原学习率0.005 #

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--BATCH_SIZE', type=int, default=30)
    args = parser.parse_args()
    return args


class Metrics:
    def __init__(self, step, test_data, predict_1, batch_size, top):
        self.pair = []
        self.step = step
        self.test_data = test_data
        self.predict_1 = predict_1
        self.top = top
        self.dcgsum = 0
        self.idcgsum = 0
        self.hit = 0
        self.ndcg = 0
        self.batch_size = batch_size
        self.val_top = []

    def hits_ndcg(self):
        for i in range(self.step * self.batch_size, (self.step + 1) * self.batch_size):
            g = [self.test_data[i, 3], self.predict_1[i].item()]
            self.pair.append(g)

        np.random.seed(1)
        np.random.shuffle(self.pair)
        pre_val = sorted(self.pair, key=lambda item: item[1], reverse=True)
        self.val_top = pre_val[0:self.top]
        for i in range(len(self.val_top)):
            if self.val_top[i][0] == 1:
                self.hit += 1
                self.dcgsum = (2 ** self.val_top[i][0] - 1) / np.log2(i + 2)
                break
        ideal_list = sorted(self.val_top, key=lambda item: item[0], reverse=True)
        for i in range(len(ideal_list)):
            if ideal_list[i][0] == 1:
                self.idcgsum = (2 ** ideal_list[i][0] - 1) / np.log2(i + 2)
                break


#-----------------------------------------------------------------下方注释部分是之前能运行的，下边是论文github原代码中的代码

        # self.ndcg = self.dcgsum / self.idcgsum if self.idcgsum != 0 else 0
        # return self.hit, self.ndcg
        if self.idcgsum == 0:
            self.ndcg = 0
        else:
            self.ndcg = self.dcgsum / self.idcgsum
        return self.hit, self.ndcg


# 修改后的get_metrics函数
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = (predict_score_matrix * real_score).sum(axis=1)  # Shape: (999,)   #----------------此行修改过

    FP = predict_score_matrix.sum(axis=1) - TP  # Shape: (999,)
    FN = real_score.sum() - TP  # Shape: (999,)
    TN = len(real_score) - TP - FP - FN  # Shape: (999,)

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.asmatrix(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.asmatrix(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)

    #--------------------下面五行有改动------------------------------
    f1_score = f1_score_list[max_index]  # Use single index
    accuracy = accuracy_list[max_index]  # Use single index
    specificity = specificity_list[max_index]  # Use single index
    recall = recall_list[max_index]  # Use single index
    precision = precision_list[max_index]  # Use single index
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def hit_ndcg_value(pred_val, val_data, top):
    loader_val = torch.utils.data.DataLoader(dataset=pred_val, batch_size=30, shuffle=False)
    hits = 0
    ndcg_val = 0
    for step, batch_val in enumerate(loader_val):
        metrix = Metrics(step, val_data, pred_val, batch_size=30, top=top)
        hit, ndcg = metrix.hits_ndcg()
        hits += hit
        ndcg_val += ndcg
    hits = hits / int((len(val_data)) / 30)
    ndcg = ndcg_val / int((len(val_data)) / 30)
    return hits, ndcg


def write_type_1234(file_name, fold_th, hits_1_max, ndcg_1_max, hits_2_max, ndcg_2_max, hits_3_max, ndcg_3_max,
                    hits_4_max, ndcg_4_max, epoch_max_1=None, epoch_max_2=None, epoch_max_3=None,
                    epoch_max_4=None, top='top1'):
    # with open(file_name, 'a', encoding='utf-8') as f:
    #     f.write(f"{fold_th}\t{top}\t{hits_1_max}\t{ndcg_1_max}\t{epoch_max_1}\t{hits_2_max}\t{ndcg_2_max}\t{epoch_max_2}\t{hits_3_max}\t{ndcg_3_max}\t{epoch_max_3}\t{hits_4_max}\t{ndcg_4_max}\t{epoch_max_4}\n")
    with open(file_name, 'a') as f:
        f.write(str(fold_th) + '\t' + str(top) + '\t' + str(hits_1_max) + '\t' + str(ndcg_1_max) + '\t' + str(
            epoch_max_1) + '\t' + str(hits_2_max) + '\t' + str(ndcg_2_max) + '\t' + str(
            epoch_max_2) + '\t' + str(hits_3_max) + '\t' + str(ndcg_3_max) + '\t' + str(
            epoch_max_3) + '\t' + str(hits_4_max) + '\t' + str(ndcg_4_max) + '\t' + str(
            epoch_max_4) + '\n')