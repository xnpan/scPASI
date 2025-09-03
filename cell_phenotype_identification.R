:library(Seurat)
library(Matrix)
library(igraph)
library(leidenAlg)
library(glmnet)
library(SGL)

#dataset
#python UMAP matrix
sc_embedding <- read.csv("umap_embedding_sfvae.csv")
#load scdata
sc_dataset <- read.csv("sc_sfvae.csv",header = TRUE,row.names = 1)
sc_dataset <- t(sc_dataset)
#load bulkdata
bulk_dataset <- read.csv("bulk_sfvae.csv",header = TRUE,row.names=1)
bulk_dataset <- t(bulk_dataset)
#load phenotype
phenotype <- read.csv("phenotype.csv")
name <- phenotype[,1]
phenotype <- data.frame(phenotype[,2])
name <- gsub("-", ".", name)
rownames(phenotype) <- name
phenotype <- unlist(phenotype)
# load leiden cluster
leidengroup <- read.csv("leiden_cluster.csv")
leidengroup <- leidengroup$Leiden_Cluster


# find the intersection between phenotypes and bulkdata
common_samples <- intersect(name, colnames(bulk_dataset))
# Extract the corresponding rows from the bulk_dataset based on the intersection
bulk_dataset <- bulk_dataset[, common_samples]


colnames(sc_embedding) <- c("UMAP_1", "UMAP_2")
sc_embedding$Cluster <- as.factor(leidengroup)


#data construction
shared_genes<-intersect(rownames(bulk_dataset),rownames(sc_dataset))
cor_matrix_embedding <- cor(bulk_dataset[shared_genes,],sc_dataset[shared_genes,])
data_embedding <- list(x=cor_matrix_embedding,y=phenotype)
data_embedding$y<-as.integer(data_embedding$y)

# 1. SGL
fit1 <- SGL(data_embedding, leidengroup, type = "logit", alpha = 0.5)
lam1 <- fit1[['lambdas']]
cvfit1 <- cvSGL(data_embedding, leidengroup, type = 'logit', nfold = 5, alpha = 0.5, lambdas = lam1)
error1 <- cvfit1$lldiff
h1 <- which.min(error1)
b1 <- fit1[["beta"]][, h1]

# 2. LASSO
fit2 <- glmnet(data_embedding$x, data_embedding$y, family = "binomial", alpha = 1)
cvfit2 <- cv.glmnet(data_embedding$x, data_embedding$y, family = "binomial", alpha = 1)
lambda2 <- cvfit2$lambda.min
b2 <- as.numeric(coef(fit2, s = lambda2))[-1]


# Count the number of positive and negative regression coefficients per cell
results_embedding <- data.frame(
  Cell = colnames(sc_dataset),
  b1 = b1,
  b2 = b2
)

results_embedding$positive_count <- rowSums(results_embedding[, c("b1", "b2")] > 0)
results_embedding$negative_count <- rowSums(results_embedding[, c("b1", "b2")] < 0)

# Classify based on the counts of positive and negative values
results_embedding$category <- ifelse(
  results_embedding$positive_count >= 2, "Strongly Pos",
  ifelse(
    (results_embedding$positive_count == 1 & results_embedding$negative_count == 0), "Weakly Pos",
    ifelse(
      results_embedding$negative_count >= 2, "Strongly Neg",
      ifelse((results_embedding$negative_count == 1 & results_embedding$positive_count == 0), "Weakly Neg", "Background")
    )
  )
)

# Extract cell names for different cell subpopulations
embedding_strongly_pos_cells <- results_embedding$Cell[results_embedding$category == "Strongly Pos"]
embedding_weakly_pos_cells <- results_embedding$Cell[results_embedding$category == "Weakly Pos"]
embedding_strongly_neg_cells <- results_embedding$Cell[results_embedding$category == "Strongly Neg"]
embedding_weakly_neg_cells <- results_embedding$Cell[results_embedding$category == "Weakly Neg"]
embedding_background_cells <- results_embedding$Cell[results_embedding$category == "Background"]
