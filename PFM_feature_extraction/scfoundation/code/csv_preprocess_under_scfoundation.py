import os
import sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="VAE Pretraining")

def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')

parser.add_argument('--system_path', type=str, default='./scFoundation-main/preprocessing/',
                    help='Path to the preprocessing module.')
# 如果未提供文件名，则默认处理指定目录下的所有CSV文件；若提供，则只处理该文件（例如传入 "coad_bulk_pre.csv"）
parser.add_argument('--file_name', type=str, default='',
                    help='Name of the file to process (e.g., "coad_bulk_pre.csv"). If empty, all CSV files in data_dir will be processed.')
# 数据目录作为参数传入
parser.add_argument('--data_dir', type=str, default='./PFM_feature_extraction/raw_csvdata/',
                    help='Absolute path to the data directory containing CSV files.')
# 输出目录作为参数传入
parser.add_argument('--output_dir', type=str, default='./PFM_feature_extraction/h5ad_output',
                    help='Absolute path to the output directory.')
parser.add_argument('--sparse_matrix', type=strict_str2bool, default=True,
                    help='Specify whether to use a sparse matrix (True) or not (False).')

args = parser.parse_args()

target_path = args.system_path
data_dir = args.data_dir
output_dir = args.output_dir
sparse_matrix = args.sparse_matrix
print("sparse_matrix =", sparse_matrix)

sys.path.append(target_path)
os.chdir(target_path)

from scRNA_workflow import *

# 根据参数判断处理单个文件还是目录下所有CSV文件
if args.file_name:
    file_list = [args.file_name]
else:
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# 如果输出目录不存在，则创建（绝对路径）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历所有文件进行处理
for file_name in file_list:
    # 拼接完整文件路径
    path = os.path.join(data_dir, file_name)
    
    # 读入CSV文件
    df = pd.read_csv(path, index_col=0)
    adata = sc.AnnData(df)
    adata.obs_names = df.index
    adata.var_names = df.columns

    # 读取基因列表文件
    gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    # 基因筛选：设原数据矩阵为 \(X \in \mathbb{R}^{m \times n}\)
    # 经过基因筛选后得到新的矩阵 \(X' \in \mathbb{R}^{m \times k}\)，其中 \(k \leq n\)
    X_df, to_fill_columns, var = main_gene_selection(df, gene_list)
    adata_uni = sc.AnnData(X_df)
    adata_uni.obs = adata.obs
    adata_uni.uns = adata.uns

    # 细胞和基因过滤，数学上可视为对矩阵的行和列进行约束筛选：
    # 设 \(X'\) 为预处理后的数据矩阵，保留满足每个细胞的表达基因数 \( \ge 200 \) 的细胞，
    # 即集合 \( C = \{ i \mid \sum_{j=1}^{k} \mathbf{1}(X'_{ij}>0) \ge 200 \} \)
    adata_uni = BasicFilter(adata_uni, qc_min_genes=0, qc_min_cells=0)
    adata_uni = QC_Metrics_info(adata_uni)

    # 保存预处理后的数据，输出文件名保留原始文件名，仅在前面添加前缀 "preprocessed_all_"
    file_name_h5ad = file_name.replace('.csv', '.h5ad')
    output_file = os.path.join(output_dir, f"preprocessed_all_{file_name_h5ad}")
    save_adata_h5ad(adata_uni, output_file)

    # 保存过滤后的细胞信息
    cell_ids = adata_uni.obs.index
    print("Processing file:", file_name)
    print("Cell IDs:", cell_ids)
    try:
        batch_data = adata_uni.obs['Batch']
        info_df = pd.DataFrame({
            'Cell_ID': cell_ids,
            'Batch': batch_data
        })
    except KeyError:
        info_df = pd.DataFrame({
            'Cell_ID': cell_ids
        })
        print("The 'Batch' column does not exist in adata_uni.obs.")

    # 将文件名中的 .csv 替换为 _info.xlsx，用于保存细胞信息
    file_name_info = file_name.replace('.csv', '_info.xlsx')
    info_df.to_excel(os.path.join(output_dir, f'preprocessed_all_{file_name_info}'), index=False)
