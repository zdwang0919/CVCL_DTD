import csv
import os
from collections import Counter

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch, Dataset

# 数据路径
DATA_PATH = r"D:\研究生阶段学习\学习\HHCL_DTD(version01)-main\Data"
# DATA_PATH = r"E:\MCCHNN-main\Data"
SMILES_FILE = os.path.join(DATA_PATH, "drug_smiles_124.csv")
TARGET_FILE = os.path.join(DATA_PATH, "target_sequence.txt")

# 全局词汇表（假设预训练保存）
VOCAB_SMILES = None
VOCAB_TARGET = None


def train_bpe(sequences, max_vocab_size=128, theta=5):
    """
    训练 BPE 模型，生成子序列词汇表
    :param sequences: 序列列表（SMILES 或氨基酸序列）
    :param max_vocab_size: 词汇表最大大小
    :param theta: 频率阈值
    :return: 词汇表（子序列到索引的映射）
    """
    # 初始化：将序列分解为字符列表
    tokenized = [[c for c in seq] for seq in sequences]

    # 初始化词汇表：包含所有单字符
    vocab = set()
    for seq in tokenized:
        vocab.update(seq)
    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}

    # 统计初始频率
    pair_counts = Counter()
    for seq in tokenized:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1

    # 迭代合并
    while len(vocab) < max_vocab_size:
        if not pair_counts:
            break
        # 找到最频繁的 pair
        (char1, char2), count = pair_counts.most_common(1)[0]
        if count < theta:
            break
        new_token = char1 + char2

        # 更新序列
        new_tokenized = []
        for seq in tokenized:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == char1 and seq[i + 1] == char2:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_tokenized.append(new_seq)
        tokenized = new_tokenized

        # 更新词汇表
        vocab[new_token] = len(vocab)

        # 更新 pair 统计
        pair_counts = Counter()
        for seq in tokenized:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1

    return vocab
def decompose_sequence(sequence, vocab):
    """
    将序列分解为词汇表中的子序列
    :param sequence: 输入序列
    :param vocab: 词汇表
    :return: 子序列列表
    """
    tokens = [c for c in sequence]
    result = []
    i = 0
    while i < len(tokens):
        matched = False
        # 尝试匹配最长的子序列
        for j in range(len(tokens), i, -1):
            subseq = "".join(tokens[i:j])
            if subseq in vocab:
                result.append(subseq)
                i = j
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1
    return result
def espf_feature_extraction(sequence, is_smiles=True):
    """
    提取 ESPF 特征
    :param sequence: SMILES 字符串或氨基酸序列
    :param is_smiles: 是否为 SMILES（True）或氨基酸序列（False）
    :return: 特征向量（位向量）
    """
    global VOCAB_SMILES, VOCAB_TARGET

    # 初始化词汇表（仅在第一次调用时训练）
    if is_smiles and VOCAB_SMILES is None:
        # 读取 SMILES 数据
        smiles_list = []
        with open(SMILES_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题
            for row in reader:
                smiles_list.append(row[1])
        VOCAB_SMILES = train_bpe(smiles_list, max_vocab_size=128, theta=5)
    elif not is_smiles and VOCAB_TARGET is None:
        # 读取氨基酸序列数据
        target_list = []
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                target_list.append(line.strip())
        VOCAB_TARGET = train_bpe(target_list, max_vocab_size=128, theta=5)

    # 选择词汇表
    vocab = VOCAB_SMILES if is_smiles else VOCAB_TARGET

    # 分解序列
    subseqs = decompose_sequence(sequence, vocab)

    # 生成位向量
    feature = np.zeros(len(vocab), dtype=np.float32)
    for subseq in subseqs:
        if subseq in vocab:
            feature[vocab[subseq]] = 1.0

    return feature


# 以下为原文件其他部分（保持不变）
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_s = []
    edge_d = []
    for e1, e2 in g.edges:
        edge_s.append(e1)
        edge_d.append(e2)
    edge_index = [edge_s, edge_d]
    return c_size, features, edge_index


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class GraphDataset_v(Dataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, dtype=None,
                 xd=None, xc=None, xm=None, y=None, did=None, cid=None, mid=None):
        super(GraphDataset_v, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dtype
        self.process(xc, cid)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xc, cid):
        data_list = []
        data_len = len(xc)
        for i in range(data_len):
            c_size, features, edge_index = xc[i]
            GCNData = Data(x=torch.FloatTensor(features), edge_index=torch.LongTensor(edge_index))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.__setitem__('cid', torch.LongTensor([cid[i]]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA