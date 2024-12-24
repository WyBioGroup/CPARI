import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from os.path import join
from sklearn.decomposition import PCA
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Note that this data is the raw data without standardization
file_identify = r'D:\CPARI\imputation\identify.csv'
file_encode = r'D:\CPARI\imputation\encoderdata_combined.csv'

# --------------------------------Absolute Imputation-------------------------------------
identifydata = pd.read_csv(file_identify, sep=',', header=None, index_col=None)
print(identifydata.isna().any().any())
encoderdata = pd.read_csv(file_encode, sep=',', header=None, index_col=None)
indices = identifydata == -1
print(encoderdata.isna().any().any())

# Create a new DataFrame by copying identifydata
result_data = identifydata.copy()

# Replace all elements with a value of -1 with the corresponding values from inputdata
print("Dimensions", encoderdata.shape)
print("Dimensions", result_data.shape)
result_data[indices] = encoderdata[indices]
result_data_nan = result_data.isna().any().any()
print(f"Does Identifydata contain NaN values: {result_data_nan}")

# -----------------------------Relative Imputation-----------------------------------------
# In addition to -1 (absolute imputation), add a (relative imputation)
n_components = 30  # Set the number of principal components
pca = PCA(n_components=n_components)
result_data_pca = pca.fit_transform(result_data.T)
result_data_pca = result_data_pca.T
print(result_data_pca.shape)

# Calculate the correlation matrix of result_data_pca
correlation_matrix_pca = pd.DataFrame(result_data_pca).corr()
correlation_matrix_pca = correlation_matrix_pca - np.identity(correlation_matrix_pca.shape[0])
highest_indices = np.argsort(np.abs(correlation_matrix_pca.to_numpy()), axis=1)[:, -4:].astype(int)
non_zero_indices = (result_data == 0) & (encoderdata != 0)
total_false_elements = np.sum(~non_zero_indices.to_numpy())
print(f"Original number of elements with value False: {total_false_elements}")
result_data[non_zero_indices] = -1

# ---------------------------------------------------------
result_data_np = result_data.values
count = 0

# Find the row and column indices of all elements with a value of -1
rows, cols = np.where(result_data == -1)

# Find the top ten correlated columns for each element
relevant_columns = highest_indices[cols]

# Check if the top ten elements in the corresponding columns of result_data are all zero
values_to_check = result_data_np[rows[:, np.newaxis], relevant_columns]

# Check if all values in each row are zero
mask = np.all(values_to_check == 0, axis=1)

# Set the value of elements that meet the condition to 0
result_data_np[rows[mask], cols[mask]] = 0

# Count the number of changes
count = np.sum(mask)
print("Number of changes", count)

result_data_processed = pd.DataFrame(result_data_np, columns=result_data.columns)
indices = result_data_processed == -1
result_data111 = result_data_processed.copy()

# Replace all elements with a value of -1 with the corresponding values from inputdata
print("Dimensions", encoderdata.shape)
print("Dimensions", result_data.shape)
result_data111[indices] = encoderdata[indices]

res_dir = 'D:\\CPARI\\imputation\\'
np.savetxt(join(res_dir, 'final_imputed.csv'), result_data111, delimiter=',', fmt='%.6f')
