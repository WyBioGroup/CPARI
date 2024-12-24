import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import pandas as pd
import os
file_name = r'D:\CPARI\imputation\final_imputed.csv'
x = pd.read_csv(file_name, sep=',', header=None, index_col=None)




# Define the Autoencoder class
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3):
        super(AE, self).__init__()
        # Define the encoder layers
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        # Define the latent space layers
        self.z1_layer = Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)
        # Define the decoder layers
        self.dec_1 = Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        self.cuda()

    def forward(self, x):
        # Forward pass through the encoder
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        # Pass through latent space layers
        z1 = self.BN4(self.z1_layer(enc_h3))
        z2 = self.BN5(self.z2_layer(z1))
        z3 = self.BN6(self.z3_layer(z2))

        # Forward pass through the decoder
        dec_h1 = F.relu(self.BN7(self.dec_1(z3)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = F.relu(self.x_bar_layer(dec_h3))

        return x_bar, z3

# Dataset class for loading data
class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float().cuda(), \
               torch.from_numpy(np.array(idx))

# Pretraining function for the Autoencoder
def train_ae(model, dataset, file_idx):
    train_loader = DataLoader(dataset, batch_size=Para[0], shuffle=True)
    optimizer = Adam(model.parameters(), lr=Para[1])
    best_acc = 0
    best_nmi = 0
    best_ari = 0

    for epoch in range(Para[2]):
        for batch_idx, (x, _) in enumerate(train_loader):
            x_bar, _ = model(x)

            x_bar = x_bar.cpu()
            x = x.cpu()
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            x_bar_np = x_bar.cpu().numpy().T
            if epoch == (Para[2] - 1):
                res_dir = 'D:\\CPARI\\imputation\\'
                np.savetxt(os.path.join(res_dir, f'decoding_{file_idx}.csv'), x_bar_np, delimiter=',', fmt='%.6f')

            # Uncomment the following lines to perform clustering and evaluation
            # kmeans = KMeans(n_clusters=Cluster_para[0], n_init=Cluster_para[1]).fit(x_bar_np)
            # kmeans = KMeans(n_clusters=Cluster_para[0], n_init=Cluster_para[1]).fit(z.data.cpu().numpy())
            # best_acc, best_nmi, best_ari = eva(y, kmeans.labels_, best_acc, best_nmi, best_ari, epoch)

        os.makedirs(os.path.dirname(File[0]), exist_ok=True)
        torch.save(model.state_dict(), File[0])



# Parameters and file paths
File = ['model/pbmc.pkl', 'data/normalized_data_d.csv', 'data/graphclusters_8.csv']
Para = [256, 1e-4, 100]
model_para = [1000, 1000, 4000]
Cluster_para = [8, 20, 1500, 2000, 500, 10]  # The third parameter is the number of rows

# Initialize the model
model = AE(
    n_enc_1=model_para[0], n_enc_2=model_para[1], n_enc_3=model_para[2],
    n_dec_1=model_para[2], n_dec_2=model_para[1], n_dec_3=model_para[0],
    n_input=Cluster_para[2], n_z1=Cluster_para[3], n_z2=Cluster_para[4], n_z3=Cluster_para[5]
).cuda()

# Read the group.txt file
with open('D:\CPARI\group.txt', 'r') as f:
    file_indices = [int(line) for line in f.read().splitlines()]

# Find unique file indices to process
unique_file_indices = sorted(set(file_indices))

# Process each unique file according to group.txt
for file_idx in unique_file_indices:
    file_name = f'D:\\CPARI\\imputation\\divide_{file_idx}.csv'
    x = pd.read_csv(file_name, sep=',', header=None, index_col=None)
    print(f"Processing file: {file_name}, Shape: {x.shape}")
    x = x.T
    dataset = LoadDataset(x.to_numpy())
    train_ae(model, dataset, file_idx)
