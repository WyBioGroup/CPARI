## 1. Introduction

A Novel Approach Combining Cell Partitioning with Absolute and Relative Imputation to Address Dropout in Single-Cell RNA-seq Data

## 2. Run

### 2.1:&nbsp;&nbsp;Step 1: Cell Partitioning (Please run `..\CPARI\cell partitioning\CPARI.mlx`)

The first step is cell partitioning. Please run the MATLAB file CPARI.mlx, which contains detailed execution steps.
Please note that in the Ledein_SNN function, Rscript = 'D:\R\R-4.3.1\bin\Rscript.exe' specifies the path to your R installation. If the operation is successful, there will be a "grout.txt" file in the ..\CPARI\.

<!-- 1.Import raw data
addpath(genpath('./'))
%iniData = readtabe('raw.txt', 'Delimiter', '\t', 'ReadRowNames', true, 'ReadVariableNames',true);
%disp(iniData)
%insertformattxt()%
%iniData = readtable('output_file.txt', 'Delimiter', '\t', 'ReadRowNames', true, 'ReadVariableNames',true);
%iniData = readtable('scRNAseq.txt');
iniData = readtable('gene_cell.csv');
disp(size(iniData))
% iniData(1, :) = [];
matrix_data=table2array(iniData)
%[~, unique_indices, ~] = unique(matrix_data, 'rows', 'stable');
%duplicate_indices = setdiff(1:size(matrix_data, 1), unique_indices);

2.Preprocessing 
tic;
minGenes = 0; minCells = 0; libararyflag = 0; logNormalize = 1;  
proData = preprocessing(iniData, minCells, minGenes, libararyflag,logNormalize);
M=proData.data

3.Select informative genes used for cell partitioning
id = gene_selection(M,iniData);%This section will generate plots to aid reader understanding.

4.Cell partitioning
%Note that in the Ledein_SNN function, Rscript = 'D:\R\R-4.3.1\bin\Rscript.exe'; is the path to your R installation.
M0 = M(id,:);
K = []; % the cluster numbers can be givn by user.
numCores = 1; 
system_used = 'Mac';
accelerate = 1;%0--------0.9982
label = [];
[group,coph] = partitioning(iniData,M0,K,numCores,system_used,accelerate,label); -->
### 2.2:&nbsp;&nbsp;step2: imputation (Please run `..\CPARI\imputation\test.py`)
The second step is data imputation, specifically including our novel absolute imputation and novel relative imputation. For detailed execution processes, please refer to "test.py".
Para = [256, 1e-4, 100]
model_para = [1000, 1000, 4000]
Cluster_para = [8, 20, 1500, 2000, 500, 10] # The third parameter is the number of rows
Readers can adjust these parameters as needed, but in Cluster_para, the third parameter is the number of rows (genes) in the matrix.
<!-- 1.
def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")
if __name__ == "__main__":
    scripts = [
        'identification.py',
        'rawdata_divide.py',
        'Runmodel.py',
        'divide_reconstrtor.py',
        'whole_reconstrtor.py'
    ]
    for script in scripts:
        run_script(script)
-->
## 3. Comparative Models
The following models and methods are referenced:

- **ALRA**: [Nature Communications](https://www.nature.com/articles/s41467-021-27729-z)
- **SAVER**: [Nature Methods](https://www.nature.com/articles/s41592-018-0033-z)
- **scImpute**: [Nature Communications](https://www.nature.com/articles/s41467-018-03405-7)
- **bayNorm**: [Bioinformatics](https://academic.oup.com/bioinformatics/article/36/4/1174/5581401)
- **VIPER**: [Genome Biology](https://link.springer.com/article/10.1186/s13059-018-1575-1)
- **scRecover**: [BioRxiv](https://www.biorxiv.org/content/10.1101/665323v1.abstract)
- **MAGIC**: [Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)30724-4)
- **DeepImpute**: [Genome Biology](https://link.springer.com/article/10.1186/s13059-019-1837-6)
- **GE-Impute**: [Briefings in Bioinformatics](https://academic.oup.com/bib/article/23/5/bbac313/6651303?login=false)
- **DCA**: [Nature Communications](https://www.nature.com/articles/s41467-018-07931-2)
- **CL-Impute**: [Journal of Computational Biology](https://www.sciencedirect.com/science/article/abs/pii/S001048252300728X)
- **TsImpute**: [Bioinformatics](https://academic.oup.com/bioinformatics/article/39/12/btad731/7457483)

## 4. Data availability
### 4.1 Real dataset
The datasets were derived from publicly available sources: 
- The PBMC datasets from [10x Genomics](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k),
- The worm neuron cells from [Cole Trapnell Lab](https://cole-trapnell-lab.github.io/worm-rna/docs/),
- The LPS datasets from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17721),
- The mouse bladder cells from [Figshare](https://figshare.com/s/865e694ad06d5857db4b).
### 4.2 Simulate dataset
The simulated datasets come from: [Bubble](https://academic.oup.com/bib/article/24/1/bbac580/6960616) and [Splatter](https://link.springer.com/article/10.1186/s13059-017-1305-0).  
Since Bubble is constrained by bulk RNA-seq data and introduces other datasets, we will not compare the Bubble method here.
- **Dataset 1**, **Dataset 2**, **Dataset 3**:
  - Each dataset corresponds to dropout rates of 30%, 40%, 50%, 65%, 80%, and 90% dropout dataset.
  - Compute the average of the results with the same dropout rate.
- **Dataset 4**, **Dataset 5**, **Dataset 6**:
  - Each dataset has a dropout rate of 80% dropout dataset.
  - Compute the average of the results with the same dropout rate.





