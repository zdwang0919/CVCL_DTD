#-------------------------------------------版本1-----------------------------------------------------
# from Bio import SeqIO
# from Bio.Align import PairwiseAligner
# import numpy as np
# from multiprocessing import Pool
# import os
#
# # 1. 读取蛋白质序列
# input_file = r"E:\MCHNN-main\Data\target_sequence.fasta"
# if not os.path.exists(input_file):
#     raise FileNotFoundError(f"文件 {input_file} 不存在")
# sequences = list(SeqIO.parse(input_file, "fasta"))
# n = len(sequences)
# matrix = np.zeros((n, n))
#
# # 2. 设置比对器为局部比对
# aligner = PairwiseAligner()
# aligner.mode = 'local'  # 使用局部比对
# aligner.match_score = 1
# aligner.mismatch_score = 0
# aligner.open_gap_score = -0.5  # 减少空位罚分，避免过大影响
# aligner.extend_gap_score = -0.1
#
#
# # 3. 定义比对函数
# def align_pair(args):
#     i, j = args
#     if i == j:
#         return (i, j, 1.0)  # 自比对为 1.0
#     alignments = aligner.align(sequences[i].seq, sequences[j].seq)
#     best_alignment = alignments[0]
#     # 使用最短序列长度归一化，确保局部相似性有意义
#     seq_len = min(len(sequences[i].seq), len(sequences[j].seq))
#     # 局部比对得分总是非负，归一化为 [0, 1]
#     identity = best_alignment.score / seq_len
#     # 限制范围在 [0, 1]，避免因参数设置导致超过 1 的情况
#     identity = min(1.0, max(0.0, identity))
#     return (i, j, identity)
#
#
# # 4. 并行计算
# if __name__ == '__main__':
#     pairs = [(i, j) for i in range(n) for j in range(i, n)]
#     with Pool(processes=os.cpu_count()) as pool:
#         results = pool.map(align_pair, pairs)
#
#     for i, j, score in results:
#         matrix[i][j] = score
#         matrix[j][i] = score  # 对称矩阵
#
#     # 5. 保存结果到文件，使用科学计数法格式
#     output_path = r"E:\MCHNN-main\Data\target_sim.txt"
#     header = "Protein Similarity Matrix (Normalized Score)\n" + "\t".join(seq.id for seq in sequences)
#     # 使用 %e 格式化输出科学计数法
#     np.savetxt(output_path, matrix, fmt="%.18e", header=header, delimiter="\t")
#
#     print(f"相似性矩阵已保存到 {output_path}")
#     print(f"矩阵维度: {matrix.shape}")

#----------------------------------------fat2---------------------------------------------------------------------
# from Bio import SeqIO
# from Bio.Align import PairwiseAligner
# import numpy as np
# from multiprocessing import Pool
# import os
#
# # 1. 读取蛋白质序列
# input_file = r"E:\MCHNN-main\Data\target_sequence.fasta"
# if not os.path.exists(input_file):
#     raise FileNotFoundError(f"文件 {input_file} 不存在")
# sequences = list(SeqIO.parse(input_file, "fasta"))
# n = len(sequences)
# matrix = np.zeros((n, n))
#
# # 2. 设置比对器为局部比对
# aligner = PairwiseAligner()
# aligner.mode = 'local'  # 使用局部比对
# aligner.match_score = 1
# aligner.mismatch_score = 0
# aligner.open_gap_score = -0.5
# aligner.extend_gap_score = -0.1
#
#
# # 3. 定义两两比对函数并应用阈值 0.3
# def align_pair(args):
#     i, j = args
#     if i == j:
#         return (i, j, 1.0)  # 自相似性为 1
#     alignments = aligner.align(sequences[i].seq, sequences[j].seq)
#     best_alignment = alignments[0]
#     # 使用较短序列长度归一化
#     seq_len = min(len(sequences[i].seq), len(sequences[j].seq))
#     # 计算相似性得分
#     similarity = best_alignment.score / seq_len
#     # 应用阈值：小于 0.3 设为 0，大于等于 0.3 设为 1
#     if similarity < 0.3:
#         similarity = 0.0
#     else:
#         similarity = 1.0
#     return (i, j, similarity)
#
#
# # 4. 并行计算两两相似性
# if __name__ == '__main__':
#     # 生成所有两两组合
#     pairs = [(i, j) for i in range(n) for j in range(i, n)]
#     with Pool(processes=os.cpu_count()) as pool:
#         results = pool.map(align_pair, pairs)
#
#     # 填充相似性矩阵
#     for i, j, score in results:
#         matrix[i][j] = score
#         matrix[j][i] = score  # 对称矩阵
#
#     # 5. 保存结果到文件，使用科学计数法
#     output_path = r"E:\MCHNN-main\Data\target_sim.txt"
#     header = "Protein Similarity Matrix (Thresholded: <0.3=0, >=0.3=1)\n" + "\t".join(seq.id for seq in sequences)
#     # 使用 %.18e 格式化输出科学计数法
#     np.savetxt(output_path, matrix, fmt="%.18e", header=header, delimiter="\t")
#
#     print(f"相似性矩阵已保存到 {output_path}")
#     print(f"矩阵维度: {matrix.shape}")


# import numpy as np
#
# # 文件路径
# input_file = r"E:\药物-靶点-疾病三重关联\PDTDAHN-main\data\net_target_sim.txt"
# output_file = r"E:\药物-靶点-疾病三重关联\PDTDAHN-main\data\target_similarity_matrix.txt"
#
# # 读取数据并确定靶标数量
# targets = set()
# similarity_dict = {}
# with open(input_file, 'r') as f:
#     for line in f:
#         # 分割每行数据
#         t1, t2, sim = map(float, line.strip().split())
#         t1, t2 = int(t1), int(t2)  # 转换为整数
#         # 添加靶标到集合
#         targets.add(t1)
#         targets.add(t2)
#         # 存储相似性（双向存储，因为矩阵是对称的）
#         similarity_dict[(t1, t2)] = sim
#         similarity_dict[(t2, t1)] = sim
#
# # 获取靶标数量和编号范围
# n_targets = len(targets)
# min_target = min(targets)
# max_target = max(targets)
# matrix_size = max_target - min_target + 1  # 矩阵大小基于编号范围
#
# # 初始化矩阵，填充为 0
# matrix = np.zeros((matrix_size, matrix_size), dtype=float)
#
# # 填充对角线为 1（自相似性）
# for i in range(matrix_size):
#     matrix[i, i] = 1.0
#
# # 填充相似性数据
# for (t1, t2), sim in similarity_dict.items():
#     # 调整索引以适应矩阵（假设靶标编号从最小值开始）
#     i = t1 - min_target
#     j = t2 - min_target
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
# ------------------------------------------------------------------靶点相似性矩阵-----------------------------------------------------------------
# import numpy as np
#
# # Define input and output file paths
# input_file = r"E:\药物-靶点-疾病三重关联\PDTDAHN-main\data\net_target_sim.txt"
# output_file = r"E:\药物-靶点-疾病三重关联\PDTDAHN-main\data\target_similarity_matrix_01.txt"
#
# # Read data and determine the number of targets
# targets = set()
# similarity_dict = {}
# with open(input_file, 'r') as f:
#     for line in f:
#         # Split each line into columns
#         t1, t2, sim = map(float, line.strip().split())
#         t1, t2 = int(t1), int(t2)  # Convert to integers
#         # Add targets to the set
#         targets.add(t1)
#         targets.add(t2)
#         # Store similarity (bidirectional, as the matrix is symmetric)
#         similarity_dict[(t1, t2)] = sim
#         similarity_dict[(t2, t1)] = sim
#
# # Get the number of targets and the range of target IDs
# n_targets = len(targets)
# min_target = min(targets)
# max_target = max(targets)
# matrix_size = max_target - min_target + 1  # Matrix size based on ID range
#
# # Initialize the matrix with zeros
# matrix = np.zeros((matrix_size, matrix_size), dtype=float)
#
# # Fill the diagonal with 1 (self-similarity)
# for i in range(matrix_size):
#     matrix[i, i] = 1.0
#
# # Fill the similarity data and apply the threshold
# for (t1, t2), sim in similarity_dict.items():
#     # Adjust indices to fit the matrix (assuming target IDs start from min_target)
#     i = t1 - min_target
#     j = t2 - min_target
#     # Apply threshold: < 0.3 becomes 0, >= 0.3 becomes 1
#     matrix[i, j] = 0.0 if sim < 0.3 else 1.0
#
# # Save the matrix to a TXT file using scientific notation
# with open(output_file, 'w') as f:
#     for i in range(matrix_size):
#         # Convert each row to scientific notation strings
#         row = [f"{x:.18e}" for x in matrix[i]]
#         f.write("\t".join(row) + "\n")
#
# print(f"矩阵已保存到 {output_file}，大小为 {matrix_size}x{matrix_size}")

#----------------------------------------------疾病相似性矩阵--------------------------------------------------------------------
import numpy as np

# Define input and output file paths
input_file = r"E:\MCCHNN-main\Data\disease_similarity_matrix.txt"
output_file = r"/Data/disease_similarity_matrix_modified_0.2.txt"

# Read the similarity matrix from the input file
matrix = []
with open(input_file, 'r') as f:
    for line in f:
        # Split each line by tabs and convert to float
        row = [float(x) for x in line.strip().split('\t')]
        matrix.append(row)

# Convert the list of lists to a NumPy array for easier manipulation
matrix = np.array(matrix)

# Apply the threshold: set values < 0.03 to 0, keep values >= 0.03 unchanged
matrix = np.where(matrix < 0.2, 0.0, matrix)

# Save the modified matrix to the output file in scientific notation
with open(output_file, 'w') as f:
    for i in range(matrix.shape[0]):
        # Convert each row to scientific notation strings with 18 decimal places
        row = [f"{x:.18e}" for x in matrix[i]]
        f.write("\t".join(row) + "\n")

print(f"Modified matrix has been saved to {output_file}, size: {matrix.shape[0]}x{matrix.shape[1]}")