import pandas as pd
from os.path import join
import numpy as np

# Read CSV file
file_path = r'D:\CPARI\test_data\log_gene_cell.csv'
data = pd.read_csv(file_path, sep=',', header=0, index_col=0)

# Read classification labels file
with open(r'D:\CPARI\group.txt', 'r') as f:
    groups = f.read().splitlines()
data_by_group = {}  # In the data_by_group dictionary, the keys are labels and the values are the indices of the data columns that match the labels.

# Iterate over data column indices and corresponding labels
for idx, group in enumerate(groups):
    # Check if the label is in the dictionary; if not, create an empty list
    if group not in data_by_group:
        data_by_group[group] = []
    # Add column index to the list corresponding to the label
    data_by_group[group].append(idx + 1)  # Note that indices start from 0, so +1 is needed

# Create datasets divided by label
columns_to_select_by_group = {}
data_split_by_group = {}

for group, columns in data_by_group.items():
    # Select columns of the dataset, including the 1st column (Cell_ID column)
    # When selecting column indices, subtract 1 from each index to adjust for zero-based indexing
    columns_to_select = [0] + [idx - 1 for idx in columns]
    columns_to_select_by_group[group] = columns_to_select[1:]

    # Add the following line to check the content of columns_to_select
    print("Columns to Select:", len(columns_to_select))

    subset_data = data.iloc[:, columns_to_select]
    data_split_by_group[group] = subset_data

# Reassemble matrices for each label
matrices_by_group = {}
for group, subset_data in data_split_by_group.items():
    # Remove the first column (Cell_ID column) and convert the data to a matrix
    matrix = subset_data.iloc[:, 1:].values
    matrices_by_group[group] = matrix

# Save each matrix to a CSV file based on the label
res_dir = r'D:\CPARI\imputation'

for group, matrix in matrices_by_group.items():
    np.savetxt(join(res_dir, f'divide_{group}.csv'), matrix, delimiter=',', fmt='%g')
