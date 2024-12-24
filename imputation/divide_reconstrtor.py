import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rawdata_divide import matrices_by_group
from rawdata_divide import columns_to_select_by_group
import numpy as np
import pandas as pd
from os.path import join

# Read the group.txt file and extract unique file indices
with open('D:\\CPARI\\group.txt', 'r') as f:
    file_indices = [int(line.strip()) for line in f.readlines()]

unique_file_indices = sorted(set(file_indices))

# Path to the raw data file
file_raw = r'D:\\CPARI\\test_data\\log_gene_cell.csv'
rawdata = pd.read_csv(file_raw, sep=',', header=0, index_col=0)

# Create an empty integrated encoderdata
encoderdata = np.zeros((rawdata.shape[0], rawdata.shape[1]))

# Assuming columns_to_select_by_group is a dictionary where the key is the file group and the value is the list of column indices
# Ensure that the columns_to_select_by_group dictionary is valid
for file_idx in unique_file_indices:
    file_name = f'D:\\CPARI\\imputation\\decoding_{file_idx}.csv'

    # Read the encoded data file
    encoderdata_X = pd.read_csv(file_name, sep=',', header=None, index_col=None)

    # Get the list of column indices
    columns_to_select = columns_to_select_by_group[str(file_idx)]

    # Place the data from encoderdata_X into the correct columns in encoderdata
    for i, column_index in enumerate(columns_to_select):
        if column_index < encoderdata.shape[1]:  # Ensure the column index is within the valid range
            encoderdata[:, column_index] = encoderdata_X.iloc[:, i]

# Save the integrated encoderdata
res_dir = 'D:\\CPARI\\imputation\\'
output_file = join(res_dir, 'encoderdata_combined.csv')

print("Original data dimensions", rawdata.shape)
print("encoderdata data dimensions", encoderdata.shape)

encoderdata_df = pd.DataFrame(encoderdata)
encoderdata_df.to_csv(output_file, index=False, header=False, float_format='%.6f')
