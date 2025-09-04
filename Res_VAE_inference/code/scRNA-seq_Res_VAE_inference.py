
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
import torch.optim as optim
from optuna.pruners import MedianPruner
import logging
import torch.nn.init as init
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse

#
# open_path = '/home/luozeyu/desktop/VAE_pretraining/data/in'
# save_path = '/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference'
# open_path_conference_data = '/home/luozeyu/desktop/VAE_pretraining/data/conference_data'
# file_prefix = 'scRNA-seq_panglao_0_1_'
# model_parameters_file = 'model_parameters.pth'
# best_parameter_name = 'best_hyperparameters.xlsx'
# execute_model_post_analysis = True

# Create the parser
parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument("--best_parameter_name", type=str, default='best_hyperparameters.xlsx', help="File name for best hyperparameters.")
parser.add_argument("--execute_model_post_analysis", type=bool, default=True, help="Flag to execute model post-analysis.")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
# parser.add_argument('--REC_beta', type=int, default=1000, help='Beta value for the REC loss')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
open_path = args.open_path
save_path = args.save_path
open_path_conference_data = args.open_path_conference_data
file_prefix = args.file_prefix
model_parameters_file = args.model_parameters_file
best_parameter_name = args.best_parameter_name
execute_model_post_analysis = args.execute_model_post_analysis
batch_size = args.batch_size
# REC_beta = args.REC_beta

# Now you can use these variables in your script
print(f"Open Path: {open_path}")
print(f"Save Path: {save_path}")
print(f"Open Path Conference Data: {open_path_conference_data}")
print(f"File Prefix: {file_prefix}")
print(f"Model Parameters File: {model_parameters_file}")
print(f"Best Parameter Name: {best_parameter_name}")
print(f"Execute Model Post Analysis: {execute_model_post_analysis}")



# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

##input_data
filename = os.listdir(open_path)[0]

scFoundation_Embedding = np.load(os.path.join(open_path, filename))


# 检查数据是否含有NaN
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())


# 启用异常检测
torch.autograd.set_detect_anomaly(True)



##vision3
import torch
from torch import nn
from torch.nn import functional as F

class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
            self.bn = nn.BatchNorm1d(out_dim)

            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = F.leaky_relu(self.bn(self.fc(x)))

            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, input_dim,hidden_dim_layer0, Encoder_layer_dims,Decoder_layer_dims,hidden_dim_layer_out_Z, z_dim, loss_type='RMSE', reduction='sum'):
        super().__init__()
        # Encoder

        # self.fc1 = nn.Linear(input_dim, hidden_dim_layer0)
        # init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        # self.bn1 = nn.BatchNorm1d(hidden_dim_layer0)

        # Resblock
        self.Encoder_resblocks = nn.ModuleList()

        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        # Latent space
        self.fc21 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # mu layer
        init.xavier_normal_(self.fc21.weight)  # Xavier Initialization for mu layer
        self.fc22 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)  # Xavier Initialization for logvar layer

        # Decoder

        # self.fc3 = nn.Linear(z_dim, hidden_dim_layer_out_Z)
        # init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        # self.bn3 = nn.BatchNorm1d(hidden_dim_layer_out_Z)

        # Resblock
        self.Decoder_resblocks = nn.ModuleList()

        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1]))

        self.fc4 = nn.Linear(hidden_dim_layer0, input_dim)
        init.xavier_normal_(self.fc4.weight)
        # Add attributes for loss type and reduction type
        self.loss_type = loss_type
        self.reduction = reduction

        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)

    def encode(self, x):
        # h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = x
        # h = self.resblock1(h)
        # h = self.resblock2(h)
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h = F.leaky_relu(self.bn3(self.fc3(z)))
        h = z
        for block in self.Decoder_resblocks:
            h = block(h)
        return self.fc4(h)  # No sigmoid here

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):

        if self.loss_type == 'MSE':
            epsilon = 1e-8
            self.REC = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction) + epsilon
        elif self.loss_type == 'RMSE':
            epsilon = 1e-8
            self.REC = torch.sqrt(F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction) + epsilon)
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.reduction == 'mean':
            self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            self.KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return beta * self.REC + self.KLD


    def get_model_inference_z(self, x, seed=None):
        """
        This function takes input x and returns the corresponding latent vectors z.
        If a seed is provided, it is used to make the random number generator deterministic.
        """
        self.eval()  # switch to evaluation mode
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():  # disable gradient computation
            mu, logvar = self.encode(x.view(-1, x.shape[1]))
            z = self.reparameterize(mu, logvar)
        return z




# 获取输入特征维度
input_dim = scFoundation_Embedding.shape[1]

# 读取标签映射字典
best_params = pd.read_excel(os.path.join(open_path_conference_data,best_parameter_name ))
best_params = dict(zip(best_params.iloc[:,0], best_params.iloc[:,1]))

##capture the best hyperparamter
learning_rate = best_params['learning_rate']
weight_decay = best_params['weight_decay']
num_epochs = int(best_params['num_epochs'])
z_dim = int(best_params['z_dim'])
hidden_dim_layer0 = int(best_params['hidden_dim_layer0'])

hidden_dim_layer_out_Z = int(best_params['hidden_dim_layer_out_Z'])

num_blocks = best_params['num_blocks']
layer_dims = [int(best_params[f'layer_{i+1}_dim']) for i in range(int(num_blocks)-1)]

##layer_encoder_dims = [int(best_params[f'layer_encoder_{i+1}_dim']) for i in range(int(num_blocks)-1)]

##layer_decoder_dims = [int(best_params[f'layer_decoder_{i+1}_dim']) for i in range(int(num_blocks)-1)]

Encoder_layer_dims =[input_dim]+ [hidden_dim_layer0] + layer_dims + [hidden_dim_layer_out_Z]

Decoder_layer_dims = [z_dim] + layer_dims + [hidden_dim_layer0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


##load model

load_model = ContinuousResidualVAE(input_dim=input_dim, hidden_dim_layer0=hidden_dim_layer0,
                                  Encoder_layer_dims=Encoder_layer_dims, Decoder_layer_dims=Decoder_layer_dims,
                                  hidden_dim_layer_out_Z=hidden_dim_layer_out_Z, z_dim=z_dim, loss_type='MSE',
                                  reduction='mean').to(device)

load_model.load_state_dict(torch.load(os.path.join(open_path_conference_data, model_parameters_file)))


inference_dataset = TensorDataset(torch.Tensor(scFoundation_Embedding))
inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=False)


# perform inference directly on the dataset
# latent_vectors = []
# load_model.eval()  # switch to evaluation mode
# with torch.no_grad():  # disable gradient computation
#     for data in inference_dataset:
#         data = data[0].unsqueeze(0).to(device)  # Unsqueeze to add the batch dimension
#         z = load_model.get_model_inference_z(data)
#         latent_vectors.append(z.cpu().detach().numpy())

# latent_vectors = np.concatenate(latent_vectors, axis=0)

latent_vectors = []
load_model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    for data in tqdm(inference_loader, desc="Processing Data"):  # 使用 DataLoader 进行迭代
        data = data[0].to(device)  # 直接使用 DataLoader 进行数据批处理
        z = load_model.get_model_inference_z(data)
        latent_vectors.append(z.cpu().detach().numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)
print(latent_vectors.shape)

# # Convert the latent vectors to DataFrame and reorder it according to the original index
# latent_vectors_df = pd.DataFrame(latent_vectors, columns=[f"VAE_latent_{i}" for i in range(latent_vectors.shape[1])])
#
# latent_vectors_df.to_excel(os.path.join(save_path ,f'{file_prefix}VAE_Embedding.xlsx'))
# Save the NumPy array as an .npy file
np.save(os.path.join(save_path, f'{file_prefix}VAE_Embedding.npy'), latent_vectors)

##judge weather need to execute VAE model post analysis
if execute_model_post_analysis:
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import umap.plot
    from scipy.stats import norm,anderson,entropy

    ## UMAP dimension for VAE_latent space(z) to 2D.

    reducer = umap.UMAP(random_state=0, min_dist=0.5, n_neighbors=15, n_jobs=1)

    mapper = reducer.fit(latent_vectors)

    ## UMAP 2D plot
    plt.figure(figsize=(12, 10))
    embedding = mapper.embedding_
    # plt.scatter(embedding[:, 0], embedding[:, 1], s=0.5, cmap='Spectral')
    umap.plot.points(mapper)

    plt.xlabel('UMAP0')
    plt.ylabel('UMAP1')

    # Define the filename and the path to save the plot
    filename_save = f"{file_prefix}2D_UMAP.pdf"
    file_path = os.path.join(save_path, filename_save)
    # Save the plot
    plt.savefig(file_path)
    plt.close()

    umap_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    # Save the DataFrame to Excel
    umap_df.to_excel(os.path.join(save_path, f'{file_prefix}UMAP_Results.xlsx'))

    ## each Z feature Distribution plot and Normality Testing

    n_cols = 10  # 每行展示5个图表
    n_rows = (z_dim + n_cols - 1) // n_cols  # 计算需要多少行

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))  # 调整图表大小
    axes = axes.flatten()  # 展平axes数组以便索引

    for i in range(z_dim):
        data = latent_vectors[:, i]
        ax = axes[i]
        ax.hist(data, bins=30, density=True, alpha=0.6, color='g')

        mu, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        title = f"mu = {mu:.2f},std = {std:.2f},\nDim{i}"
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(os.path.join(save_path,f"{file_prefix}latent_space_distribution.pdf"),dpi = 1000)  # 保存整个图表

    results = []

    # # 进行Shapiro-Wilk正态性检验
    # for i in range(z_dim):
    #     data = latent_vectors[:, i]
    #     stat, p = shapiro(data)
    #     result = {
    #         "Dimension": i,
    #         "Statistics": stat,
    #         "p-value": p,
    #         "Normality": "Probably Gaussian" if p > 0.05 else "Probably not Gaussian"
    #     }
    #     results.append(result)

    # Perform Anderson-Darling normality test, 后面可以考虑丢弃！！！
    for i in range(z_dim):
        data = latent_vectors[:, i]
        # Perform Anderson-Darling normality test
        result = anderson(data)

        # Calculate mean and standard deviation for the title
        mu = np.mean(data)
        std = np.std(data)
        # 使用较宽松的临界值（例如15%的显著性水平）
        critical_value = result.critical_values[0]  # 选择较高的显著性水平（更宽松）

        # Determine normality based on test statistic and critical values
        normality = "Probably Gaussian" if result.statistic < critical_value else "Probably not Gaussian"

        result_dict = {
            "Dimension": i,
            "Statistic": result.statistic,
            "Critical Value": result.critical_values[2],
            "Normality": normality
        }
        results.append(result_dict)
    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存DataFrame到Excel文件

    results_df.to_excel(os.path.join(save_path,f"{file_prefix}normality_tests.xlsx"))

    results = []

    for i in range(z_dim):
        data = latent_vectors[:, i]
        # 计算数据的直方图作为概率分布
        data_hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 生成与数据直方图相同bins的正态分布概率密度
        norm_pdf = norm.pdf(bin_centers, loc=np.mean(data), scale=np.std(data))

        # 计算KL散度
        kl_divergence = entropy(data_hist, norm_pdf)

        result_dict = {
            "Dimension": i,
            "KL Divergence": kl_divergence
        }
        results.append(result_dict)

    # 计算KL散度的均值和方差
    kl_values = [result['KL Divergence'] for result in results]
    average_kl = np.mean(kl_values)
    variance_kl = np.var(kl_values)

    # 添加平均KL散度和方差到DataFrame
    results_df = pd.DataFrame(results)
    results_df.loc['Average', 'KL Divergence'] = average_kl
    results_df.loc['Variance', 'KL Divergence'] = variance_kl

    # 保存DataFrame到Excel文件
    results_df.to_excel(os.path.join(save_path, f"{file_prefix}kl_divergence_analysis.xlsx"))



    # ## 获取模型每个样本的KL散度。
    #
    # mus = []
    # logvars = []
    #
    # # 切换到模型的评估模式
    # load_model.eval()
    #
    # # 禁用梯度计算，进行模型推断
    # with torch.no_grad():
    #     for data in tqdm(inference_loader, desc="calculating KL value"):
    #         data = data[0].to(device)  # 将数据送到设备
    #
    #         # 获取潜在向量的均值和对数方差
    #         mu, logvar = load_model.encode(data.view(-1, data.shape[1]))
    #         mus.append(mu.cpu().numpy())
    #         logvars.append(logvar.cpu().numpy())
    #
    # mus = np.concatenate(mus, axis=0)
    # logvars = np.concatenate(logvars, axis=0)
    #
    # # 计算每个样本的KL散度
    # kl_divergences = -0.5 * np.sum(1 + logvars - np.square(mus) - np.exp(logvars), axis=1)
    #
    # # 创建DataFrame
    # results_df = pd.DataFrame({
    #     'Sample Index': range(len(kl_divergences)),
    #     'KL Divergence': kl_divergences
    # })
    #
    # # 添加均值和方差
    # results_df.loc['Average', 'KL Divergence'] = np.mean(kl_divergences)
    # results_df.loc['Variance', 'KL Divergence'] = np.var(kl_divergences)
    #
    # # 保存DataFrame到Excel文件
    # save_path = 'path_to_save'  # 修改为实际路径
    # file_prefix = 'your_file_prefix'
    # results_df.to_excel(f"{save_path}/{file_prefix}kl_divergence_analysis.xlsx")
    #
    # print("Excel file has been saved to:", f"{save_path}/{file_prefix}kl_divergence_analysis.xlsx")