import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, SAGEConv, HeteroConv

EPS = torch.tensor(1e-15)

class DiffusionModel(nn.Module):
    def __init__(self, embed_dim, num_steps=100, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        # Use embed_dim for both input and output to match encoder dimensions
        self.denoise_net_hetero_to_hyper = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.denoise_net_hyper_to_hetero = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    # forward_diffusion：往 embedding 里加噪声，模拟扩散过程。
    # denoise_step：训练一个去噪器，把 noisy embedding 还原成另一视图的 embedding

    def forward_diffusion(self, embed, t, device):
        """Add noise to the input embedding."""
        embed = embed.to(device)
        alpha_t = self.alpha_bar[t].sqrt().to(device)
        beta_t = (1 - self.alpha_bar[t]).sqrt().to(device)
        noise = torch.randn_like(embed).to(device)
        noisy_embed = alpha_t * embed + beta_t * noise
        return noisy_embed, noise

    def denoise_step(self, noisy_embed, t, target_embed, view_type, device):

        noisy_embed = noisy_embed.to(device)
        target_embed = target_embed.to(device)
        # Select the appropriate denoising network based on view_type
        if view_type == 'hetero_to_hyper':
            embed_pred = self.denoise_net_hetero_to_hyper(noisy_embed)
        elif view_type == 'hyper_to_hetero':
            embed_pred = self.denoise_net_hyper_to_hetero(noisy_embed)
        else:
            raise ValueError(f"Invalid view_type: {view_type}")
        return embed_pred
# 这里做的不是单纯的去噪，而是 跨视图去噪：
# 如果输入的是 异构图 noisy embedding，那么目标是恢复 超图的 embedding。
# 如果输入的是 超图 noisy embedding，那么目标是恢复 异构图的 embedding。

# ------------------------负责三元关系的聚合---------------------------------
class HgnnEncoder(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, dim_1)
        self.conv2 = HypergraphConv(dim_1, dim_1)

    def forward(self, x, edge):
        num_nodes = x.size(0)
        # 先获得节点数 num_nodes = N（x 形状 [N, D_in]）
        assert edge[0].min() >= 0 and edge[0].max() < num_nodes, f"Node indices out of bounds: min={edge[0].min()}, max={edge[0].max()}, num_nodes={num_nodes}"
        # 检查 edge 中的节点索引是否越界（edge[0] 存放 node indices，edge[1] 存放 hyperedge indices）。这是一个保护性断言，避免传入错误的 edge_index 导致下游出错
        x = torch.relu(self.conv1(x, edge))
        x = torch.relu(self.conv2(x, edge))
        return x
# -------------------------负责二元关系的聚合--------------------------------
class HeteroEncoder(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(HeteroEncoder, self).__init__()
        self.conv = HeteroConv({
            ('drug', 'interacts', 'target'): SAGEConv((-1, -1), dim_1),
            ('drug', 'treats', 'disease'): SAGEConv((-1, -1), dim_1),
            ('target', 'interacts', 'drug'): SAGEConv((-1, -1), dim_1),
            ('disease', 'treats', 'drug'): SAGEConv((-1, -1), dim_1)
        }, aggr='mean')
    # 核心方法：SAGEConv
    # GraphSAGE 卷积：每个节点更新时，会聚合邻居节点的特征（例如药物聚合靶标、疾病的特征）。
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)   #输入，来自 BioEncoder 的三个 embedding： x_dict = {'drug': x_drug, 'target': x_target, 'disease': x_disease}
        return {k: torch.relu(v) for k, v in x_dict.items()}
#     每类节点更新后的特征都经过 ReLU，增强非线性表达能力-------------二元学习的输出：

class BioEncoder(nn.Module):
    # 作用：把三类原始生物学特征映射到统一的低维表示（仅做逐节点的特征映射，不做图聚合）
    def __init__(self, dim_drug, dim_target, dim_disease, output):
        super(BioEncoder, self).__init__()
        self.drug_layer = nn.Linear(dim_drug, output)
        self.target_layer = nn.Linear(dim_target, output)
        self.disease_layer = nn.Linear(dim_disease, output)
        self.batch_drug = nn.BatchNorm1d(output)
        self.batch_target = nn.BatchNorm1d(output)
        self.batch_disease = nn.BatchNorm1d(output)

    def forward(self, drug_feature, target_feature, disease_feature):
        x_drug = self.drug_layer(drug_feature)
        x_drug = self.batch_drug(F.relu(x_drug))
        x_target = self.target_layer(target_feature)
        x_target = self.batch_target(F.relu(x_target))
        x_disease = self.disease_layer(disease_feature)
        x_disease = self.batch_disease(F.relu(x_disease))
        return x_drug, x_target, x_disease
# 返回“三类节点的初始 embedding”（统一维度 output），供后续两个视角的编码器使用。
# 直观理解：这是“特征投影层”，把异构原始属性标准化到统一 embedding 空间，便于在异构 GNN（hetero）与超图（hyper）之间传递和比较。
class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        # 4层的 MLP 解码器，逐层压缩维度
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, in_channels // 8)
        self.batch3 = nn.BatchNorm1d(in_channels // 8)
        self.fc4 = nn.Linear(in_channels // 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, drug_id, target_id, disease_id):
        # 依次通过三层tanh+BatchNorm，逐步压缩特征维度，第四层fc4输出一个标量，代表该三元组的打分。
        h_0 = torch.cat(
            (graph_embed['drug'][drug_id], graph_embed['target'][target_id], graph_embed['disease'][disease_id]), 1)
        # 把三元组的三份节点 embedding 拼成一个总体的三元向量作为解码器输入
        h_1 = torch.tanh(self.fc1(h_0))
        h_1 = self.batch1(h_1)
        h_2 = torch.tanh(self.fc2(h_1))
        h_2 = self.batch2(h_2)
        h_3 = torch.tanh(self.fc3(h_2))
        h_3 = self.batch3(h_3)
        # 逐步压缩特征维度
        h_4 = self.fc4(h_3)
        return h_3, torch.sigmoid(h_4.squeeze(dim=1))#torch.sigmoid 把 score 转为概率（0-1），用于二分类（存在/不存在）

class H_GCL(nn.Module):
    def __init__(self, bio_encoder, hetero_encoder, hyper_encoder, decoder, out_channels, bio_out_dim):
        super(H_GCL, self).__init__()
        self.bio_encoder = bio_encoder
        self.hetero_encoder = hetero_encoder
        self.hyper_encoder = hyper_encoder
        self.decoder = decoder
        # Use out_channels (hgnn_dim_1) for diffusion model to match encoder output dimensions
        self.diffusion = DiffusionModel(out_channels, num_steps=100)
        self.projector = nn.Linear(out_channels, out_channels)
        self.temperature = 0.5

    def node_mask_hetero(self, x_dict, mask_ratio=0.15):
        """Apply node masking to heterogeneous graph node features."""
        x_dict_aug = {}
        for node_type, x in x_dict.items():
            num_nodes = x.size(0)
            mask = torch.rand(num_nodes, device=x.device) > mask_ratio
            mask = mask.unsqueeze(1)
            x_dict_aug[node_type] = x * mask
        return x_dict_aug

    def node_mask(self, x, mask_ratio=0.15):
        """Apply node masking to hypergraph node features."""
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes, device=x.device) > mask_ratio
        mask = mask.unsqueeze(1)
        return x * mask

    def edge_dropout_hetero(self, edge_index_dict, drop_ratio=0.15):
        """Apply edge dropout to heterogeneous graph edges."""
        edge_index_dict_aug = {}
        for edge_type, edge_index in edge_index_dict.items():
            num_edges = edge_index.size(1)
            if num_edges == 0:
                edge_index_dict_aug[edge_type] = edge_index
                continue
            mask = torch.rand(num_edges, device=edge_index.device) > drop_ratio
            edge_index_dict_aug[edge_type] = edge_index[:, mask]
            if edge_index_dict_aug[edge_type].size(1) == 0:
                edge_index_dict_aug[edge_type] = edge_index[:, :1]
        return edge_index_dict_aug

    def hyperedge_dropout(self, edge_index, drop_ratio=0.15):
        """Apply edge dropout to hypergraph edges."""
        num_hyperedges = edge_index.size(1)
        if num_hyperedges == 0:
            return edge_index
        mask = torch.rand(num_hyperedges, device=edge_index.device) > drop_ratio
        edge_index_aug = edge_index[:, mask]
        if edge_index_aug.size(1) == 0:
            edge_index_aug = edge_index[:, :1]
        return edge_index_aug

    def info_nce_loss(self, z1, z2, neg_z_list):
        z1 = F.normalize(self.projector(z1), dim=-1)
        z2 = F.normalize(self.projector(z2), dim=-1)
        neg_z_list = [F.normalize(self.projector(neg_z), dim=-1) for neg_z in neg_z_list]
        pos_score = (z1 * z2).sum(dim=-1) / self.temperature
        pos_score = torch.exp(pos_score)
        neg_scores = [torch.exp((z1 * neg_z).sum(dim=-1) / self.temperature) for neg_z in neg_z_list]
        neg_score = sum(neg_scores)
        loss = -torch.log(pos_score / (pos_score + neg_score + EPS)).mean()
        return loss

    def diffusion_loss(self, embed_pred, embed_true):
        return F.mse_loss(embed_pred, embed_true)

    def forward(self, drug_feature, target_fea, disease_fea, hetero_data, edge_pos, drug_id, target_id,
                disease_id, edge_neg_list, device):
        # Biological feature encoding
        x_drug, x_target, x_disease = self.bio_encoder(drug_feature, target_fea, disease_fea)
        # 调用 BioEncoder 得到三类节点在原子属性空间映射后的表示，这是所有后续编码器的输入特征
        x_dict = {'drug': x_drug, 'target': x_target, 'disease': x_disease}
        embed = torch.cat((x_drug, x_target, x_disease), 0)

        # Generate augmented views
        x_dict_view1 = self.node_mask_hetero(x_dict, mask_ratio=0.15)
        x_dict_view2 = self.node_mask_hetero(x_dict, mask_ratio=0.15)
        # 分别对 hetero 节点特征进行 随机节点遮挡（15% 被置为 0）
        edge_index_dict_view1 = self.edge_dropout_hetero(hetero_data.edge_index_dict, drop_ratio=0.15)
        edge_index_dict_view2 = self.edge_dropout_hetero(hetero_data.edge_index_dict, drop_ratio=0.15)
        # 对异质图的边做随机删除（15% 边被丢弃）；避免某一视图过于接近原图，提高对比训练的鲁棒性
        embed_view1 = self.node_mask(embed, mask_ratio=0.15)
        embed_view2 = self.node_mask(embed, mask_ratio=0.15)
        # 对 embed（用于超图）也做节点掩蔽，生成两个 hyper-view 的节点特征
        edge_pos_view1 = self.hyperedge_dropout(edge_pos, drop_ratio=0.15)
        edge_pos_view2 = self.hyperedge_dropout(edge_pos, drop_ratio=0.15)
        # 对超图的 hyperedges 做随机删除（超边 dropout），生成两个超图视图

        # Heterogeneous graph encoding（异构图编码：二元聚合）
        hetero_embed_view1 = self.hetero_encoder(x_dict_view1, edge_index_dict_view1)
        hetero_embed_view2 = self.hetero_encoder(x_dict_view2, edge_index_dict_view2)

        hetero_embed_concat1 = torch.cat([hetero_embed_view1['drug'], hetero_embed_view1['target'], hetero_embed_view1['disease']], 0)
        hetero_embed_concat2 = torch.cat([hetero_embed_view2['drug'], hetero_embed_view2['target'], hetero_embed_view2['disease']], 0)

        # Diffusion model: Add noise to both views 对异构图加噪
        t = torch.randint(0, self.diffusion.num_steps, (1,)).item()
        noisy_hetero_embed1, noise1 = self.diffusion.forward_diffusion(hetero_embed_concat1, t, device)
        noisy_hetero_embed2, noise2 = self.diffusion.forward_diffusion(hetero_embed_concat2, t, device)

        # Hypergraph encoding for clean embeddings
        hyper_embed_pos1 = self.hyper_encoder(embed_view1, edge_pos_view1.long())
        hyper_embed_pos2 = self.hyper_encoder(embed_view2, edge_pos_view2.long())
        hyper_embed_neg_list = [self.hyper_encoder(embed, edge_neg.long()) for edge_neg in edge_neg_list[:1]]

        # Diffusion model: Cross-view reconstruction
        # Hetero noisy embedding -> Hyper clean embedding，用加噪的异构图恢复出超图。
        embed_pred_hetero_to_hyper1 = self.diffusion.denoise_step(noisy_hetero_embed1, t, hyper_embed_pos1, 'hetero_to_hyper', device)
        embed_pred_hetero_to_hyper2 = self.diffusion.denoise_step(noisy_hetero_embed2, t, hyper_embed_pos2, 'hetero_to_hyper', device)
        # Hyper noisy embedding -> Hetero clean embedding，对超图进行加噪，以及用加噪后的超图恢复异构图
        noisy_hyper_embed1, _ = self.diffusion.forward_diffusion(hyper_embed_pos1, t, device)
        noisy_hyper_embed2, _ = self.diffusion.forward_diffusion(hyper_embed_pos2, t, device)
        embed_pred_hyper_to_hetero1 = self.diffusion.denoise_step(noisy_hyper_embed1, t, hetero_embed_concat1, 'hyper_to_hetero', device)
        embed_pred_hyper_to_hetero2 = self.diffusion.denoise_step(noisy_hyper_embed2, t, hetero_embed_concat2, 'hyper_to_hetero', device)

        # Combine views: Use denoised embeddings
        graph_embed = {
            'drug': hetero_embed_view1['drug'] + embed_pred_hyper_to_hetero1[:len(x_drug)],
            'target': hetero_embed_view1['target'] + embed_pred_hyper_to_hetero1[len(x_drug):len(x_drug) + len(x_target)],
            'disease': hetero_embed_view1['disease'] + embed_pred_hyper_to_hetero1[len(x_drug) + len(x_target):]
        }
        graph_embed_pos1 = torch.cat([graph_embed['drug'], graph_embed['target'], graph_embed['disease']], 0)
        graph_embed_pos2 = torch.cat([
            hetero_embed_view2['drug'] + embed_pred_hyper_to_hetero2[:len(x_drug)],
            hetero_embed_view2['target'] + embed_pred_hyper_to_hetero2[len(x_drug):len(x_drug) + len(x_target)],
            hetero_embed_view2['disease'] + embed_pred_hyper_to_hetero2[len(x_drug) + len(x_target):]
        ], 0)
        graph_embed_neg_list = [
            torch.cat([
                hetero_embed_view1['drug'] + neg[:len(x_drug)],
                hetero_embed_view1['target'] + neg[len(x_drug):len(x_drug) + len(x_target)],
                hetero_embed_view1['disease'] + neg[len(x_drug) + len(x_target):]
            ], 0) for neg in hyper_embed_neg_list
        ]


        # Verify view difference
        diff = (graph_embed_pos1 - graph_embed_pos2).abs().mean().item()

        # Decoder
        emb, res = self.decoder(graph_embed, drug_id, target_id, disease_id)

        # Return cross-view predictions for loss computation
        return emb, res, graph_embed_pos1, graph_embed_pos2, graph_embed_neg_list, embed_pred_hetero_to_hyper1, embed_pred_hyper_to_hetero1, hyper_embed_pos1, hetero_embed_concat1

# 它们分别从不同角度学习节点/关系的表示，但语义不同，导致两边学到的 embedding 不在同一个空间，互补但难以直接融合。
# 于是引入 扩散模型，目的是：
# 在两个视图之间建立跨模态的“翻译”能力，让 异构图的 embedding 可以重构超图的 embedding，反之亦然。




# ---------------------------------消融实验------------------------------------
class H_GCL_wo_Hyper(nn.Module):
    """
    Ablation: w/o Hypergraph
    Only Heterogeneous Graph Encoder is used
    """
    def __init__(self, bio_encoder, hetero_encoder, decoder):
        super().__init__()
        self.bio_encoder = bio_encoder
        self.hetero_encoder = hetero_encoder
        self.decoder = decoder

    def forward(
        self,
        drug_fea,
        target_fea,
        disease_fea,
        hetero_data,
        drug_idx,
        target_idx,
        disease_idx,
        device
    ):
        # 1. Feature initialization
        drug_emb, target_emb, disease_emb = self.bio_encoder(
            drug_fea, target_fea, disease_fea
        )

        # 2. Heterogeneous graph encoding
        hetero_emb = self.hetero_encoder(
            drug_emb, target_emb, disease_emb, hetero_data
        )

        # 3. Gather triple embeddings
        h_d = hetero_emb['drug'][drug_idx]
        h_t = hetero_emb['target'][target_idx]
        h_di = hetero_emb['disease'][disease_idx]

        triple_emb = torch.cat([h_d, h_t, h_di], dim=1)

        # 4. Prediction
        pred = self.decoder(triple_emb)

        return pred, triple_emb
