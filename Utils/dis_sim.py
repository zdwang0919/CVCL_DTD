# import numpy as np
#
# # 文件路径
# input_file = r"E:\药物-靶点-疾病三重关联\PDTDAHN-main\data\net_disease_sim.txt"
# output_file = r"E:\药物-靶点-疾病三重关联\PDTDAHN-main\data\disease_similarity_matrix.txt"
#
# # 读取数据并确定疾病数量
# diseases = set()
# similarity_dict = {}
# with open(input_file, 'r') as f:
#     for line in f:
#         # 分割每行数据
#         d1, d2, sim = map(float, line.strip().split())
#         d1, d2 = int(d1), int(d2)  # 转换为整数
#         # 添加疾病到集合
#         diseases.add(d1)
#         diseases.add(d2)
#         # 存储相似性（双向存储，因为矩阵是对称的）
#         similarity_dict[(d1, d2)] = sim
#         similarity_dict[(d2, d1)] = sim
#
# # 获取疾病数量和编号范围
# n_diseases = len(diseases)
# min_disease = min(diseases)
# max_disease = max(diseases)
# matrix_size = max_disease - min_disease + 1  # 矩阵大小基于编号范围
#
# # 初始化矩阵，填充为 0
# matrix = np.zeros((matrix_size, matrix_size), dtype=float)
#
# # 填充对角线为 1（自相似性）
# for i in range(matrix_size):
#     matrix[i, i] = 1.0
#
# # 填充相似性数据
# for (d1, d2), sim in similarity_dict.items():
#     # 调整索引以适应矩阵（假设疾病编号从最小值开始）
#     i = d1 - min_disease
#     j = d2 - min_disease
#     matrix[i, j] = sim
#
# # 保存矩阵到 TXT 文件，使用科学计数法
# with open(output_file, 'w') as f:
#     for i in range(matrix_size):
#         # 将每行转换为科学计数法字符串
#         row = [f"{x:.18e}" for x in matrix[i]]
#         f.write("\t".join(row) + "\n")
#
# print(f"矩阵已保存到 {output_file}，大小为 {matrix_size}x{matrix_size}")

#处理smiles序列
import pandas as pd
from rdkit import Chem
import sys

# 已有的原子类型列表（从 atom_features 函数中提取）
existing_atom_types = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
    'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
    'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
    'Pt', 'Hg', 'Pb', 'Unknown'
]

# 文件路径
file_path = r"E:\MCCHNN-main\Data\drug_smiles_124.csv"

try:
    # 读取 CSV 文件
    df = pd.read_csv(file_path, header=None, names=['DB_ID', 'SMILES'])
    print(f"成功加载文件，包含 {len(df)} 条记录")
except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到")
    sys.exit(1)
except Exception as e:
    print(f"读取文件时发生错误：{e}")
    sys.exit(1)

# 存储所有出现的原子类型
unique_atoms = set()

# 遍历 SMILES 并提取原子类型
for index, row in df.iterrows():
    smile = row['SMILES']
    db_id = row['DB_ID']
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"警告：第 {index+1} 行 ({db_id}) 的 SMILES 无效: {smile}")
            continue
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            unique_atoms.add(symbol)
    except Exception as e:
        print(f"处理第 {index+1} 行 ({db_id}) 时出错: {e}")
        continue

# 输出所有发现的原子类型
print("\n文件中发现的原子类型：")
print(sorted(unique_atoms))

# 检查是否有未在 existing_atom_types 中的原子类型
missing_atoms = [atom for atom in unique_atoms if atom not in existing_atom_types]

# 输出结果
if missing_atoms:
    print("\n以下原子类型不在已有列表中：")
    for atom in missing_atoms:
        print(f" - {atom}")
else:
    print("\n所有原子类型均在已有列表中")

# 输出统计信息
print(f"\n总计发现 {len(unique_atoms)} 种唯一原子类型")