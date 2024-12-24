args <- commandArgs()
baseName <- "D:\\CPARI\\"
if (!require("Seurat")) {
  install.packages("Seurat", dependencies = TRUE, repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
}
library(Seurat)
library(dplyr)
library(plyr)
library(ggplot2)
library(data.table)
print("666666666666666666666")
#sessionInfo()
data_file <- as.character(file.path(baseName, "selected_genes_expression_matrix.csv"))
# data <- read.table(data_file, sep = '\t', header = TRUE)
#data <- read.table(data_file,sep = '\t')

counts_matrix <- read.table("D:\\CPARI\\selected_genes_expression_matrix.csv", sep = ',')
print("666666666666666666666")
print(dim(counts_matrix))


#counts_matrix <- data[, -1]
#counts_matrix <- data[-1, ]
w10x_new <- CreateSeuratObject(counts = counts_matrix, raw.data = data, min.cells = 3, min.genes = 200, project = "MUT")

# Check the number of retained cells
print(ncol(w10x_new))

print(length(VariableFeatures(w10x_new)))

#w10x_new <- NormalizeData(object = w10x_new, normalization.method = "LogNormalize", scale.factor = 10000)
w10x_new <- FindVariableFeatures(object = w10x_new, mean.function = ExpMean, dispersion.function = LogVMR, x.low.cutoff = 0.01, x.high.cutoff = 5, y.cutoff = 0.01)
#x.low.cutoff = 0.01: 
#x.high.cutoff = 5: 
#y.cutoff = 0.25: 
print("high") 
print(length(VariableFeatures(w10x_new)))
w10x_new <- ScaleData(object = w10x_new)
# Assuming your Seurat object is named w10x_new
w10x_new <- RunPCA(w10x_new, pc.genes = w10x_new@var.genes, pcs.compute = 30, do.print = FALSE) 
w10x_new <- FindNeighbors(w10x_new, dims = 1:30)

# Define the different resolution values to try
res <- c(0.05,0.1,0.15,0.2,0.25)

# Loop through different resolution values
#test run
for (i in 1:length(res)) {
 w10x_new <- FindClusters(w10x_new, reduction.type = "pca",
       print.output = 0,force.recalc = T,
                           algorithm = 1, 
                           n.start = 800,    
                           save.SNN = TRUE, resolution = res[i])  
# Try to test cmeans
#Obtain clustering results
cluster_results <- data.frame(Cell = names(w10x_new@active.ident), Cluster = w10x_new@active.ident)
out <- paste0("D:\\CPARI\\intermediateFiles\\", "identity_clustering_res", res[i], ".txt")
print("666666666666666")
print(out)
write.table(w10x_new@active.ident, file = out, sep = '\t')
  # Save t-SNE plot
  #plot_filename <- paste(baseName, paste("tSNE_plot_res", res[i], ".jpg", sep = ""), sep = "/")
}

  





