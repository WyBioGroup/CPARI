# coding: utf-8
# In[ ]:
import os
import h5py
import sklearn
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
file_path = r'D:\CPARI\test_data\log_gene_cell.csv'
rawdata = pd.read_csv(file_path, sep=',',header=0,index_col=0)
# rawdata=preprocess(rawdata)
print(rawdata.shape)
data_norm = rawdata.values
data_norm1 = data_norm.copy()

#perform PCA
num_components=30 #The number of principal components at least explains 40% of the variance in the data
pca = PCA(n_components=num_components,svd_solver = "randomized")
pca_data = pca.fit_transform(data_norm)
#K-means
clusters=3
#Testing the block, the actual file being read is group.txt.
kmeans = KMeans(n_clusters=clusters, random_state=0,n_init=10).fit(pca_data)
label_pr =  kmeans.labels_#This is a test example, and it should use the labels obtained from the previous cell partitioning step.
def find_cluster_cell_idx(l, label):
    '''

    '''
    return label==l
#The identification of dropout events
st = datetime.datetime.now()
def identify_dropout(cluster_cell_idxs, X):#Temporarily fill missing values with -1
    for idx in cluster_cell_idxs:
        # The dropout rate for each row is the ratio of zeros in the columns of that cell.
        dropout=(X[:,idx]==0).sum(axis=1)/(X[:,idx].shape[1])
        # Determine the maximum and minimum dropout rates based on the upper and lower threshold scores.
        dropout_thr=0.5
        dropout_upper_thr,dropout_lower_thr = np.nanquantile(dropout,q=dropout_thr),np.nanquantile(dropout,q=0)
        gene_index1 = (dropout<=dropout_upper_thr)&(dropout>=dropout_lower_thr)
        print(gene_index1)
        cv = X[:,idx].std(axis=1)/X[:, idx].mean(axis=1)
        cv_thr=0.5
        cv_upper_thr,cv_lower_thr = np.nanquantile(cv,q=cv_thr),np.nanquantile(cv,q=0)
        # print(cv_upper_thr,cv_lower_thr)
        gene_index2 = (cv<=cv_upper_thr)&(cv>=cv_lower_thr)
        print(gene_index2)
        # include_faslezero_gene= list(np.intersect1d(gene_index1,gene_index2))
        include_faslezero_gene = np.logical_and(gene_index1, gene_index2)
        # print(list(include_faslezero_gene).count(True))
        tmp = X[:, idx]
        tmp[include_faslezero_gene] = tmp[include_faslezero_gene]+(tmp[include_faslezero_gene]==0)*-1
        X[:, idx] = tmp
    return X

label_set = np.unique(label_pr)
cluster_cell_idxs = list(map(partial(find_cluster_cell_idx,label=label_pr), label_set))
data_identi=identify_dropout(cluster_cell_idxs, X=data_norm.T).T
print(data_identi)
num_minus_ones = np.count_nonzero(data_identi == -1)



output_file = r'D:\CPARI\imputation\identify.csv'
pd.DataFrame(data_identi).to_csv(output_file, index=False,header=False)
nowdata = pd.read_csv(output_file, sep=',',header=None,index_col=None)
print(nowdata .shape)
ed = datetime.datetime.now()
print('identify_dropout ：', (ed - st).total_seconds())

