# # 读取文件路径
# input_file = r"E:\MCHNN-main\Data\dtd_interaction.txt"
# output_file = r"E:\MCHNN-main\Data\dtd_interaction_adjusted.txt"
#
# # 读取数据并提取第二列和第三列
# data = []
# with open(input_file, 'r') as f:
#     for line in f:
#         # 分割每行数据，假设以空格或制表符分隔
#         cols = line.strip().split()
#         # 转换为整数
#         row = [int(cols[0]), int(cols[1]), int(cols[2]), int(cols[3])]
#         data.append(row)
#
# # 将数据转换为 numpy 数组以便操作
# import numpy as np
# data_array = np.array(data)
#
# # 获取第二列和第三列的最小值
# min_col2 = np.min(data_array[:, 1])  # 第二列最小值
# min_col3 = np.min(data_array[:, 2])  # 第三列最小值
#
# # 调整第二列和第三列：减去最小值后加 1
# data_array[:, 1] = data_array[:, 1] - min_col2 + 1
# data_array[:, 2] = data_array[:, 2] - min_col3 + 1
#
# # 将调整后的数据保存到新文件
# with open(output_file, 'w') as f:
#     for row in data_array:
#         # 将每一行格式化为字符串，以制表符分隔
#         line = "\t".join(map(str, row)) + "\n"
#         f.write(line)
#
# print(f"原始第二列最小值: {min_col2}, 原始第三列最小值: {min_col3}")
# print(f"调整后的数据已保存到: {output_file}")

# Define input and output file paths
# Define input and output file paths
input_file = r"E:\MCHNN-main\Data\dtd_interaction_adjusted.txt"
output_file = r"E:\MCHNN-main\Data\dtd_interaction_adjusted_modified.txt"

# Open the input file for reading and output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Process each line in the input file
    for line in infile:
        # Split the line into columns (assuming tab-separated)
        columns = line.strip().split('\t')

        # Ensure there are at least 4 columns
        if len(columns) >= 4:
            try:
                # Subtract 1 from the first three columns and convert to integers
                col1 = int(columns[0]) - 1
                col2 = int(columns[1]) - 1
                col3 = int(columns[2]) - 1
                col4 = columns[3]  # Keep the fourth column unchanged

                # Write the modified line to the output file
                outfile.write(f"{col1}\t{col2}\t{col3}\t{col4}\n")
            except ValueError as e:
                print(f"Error processing line '{line.strip()}': {e}")
        else:
            print(f"Skipping malformed line: '{line.strip()}'")

print(f"Modified data has been written to {output_file}")