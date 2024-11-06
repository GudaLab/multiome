library(Seurat)
library(harmony)
library(zellkonverter)
library(SingleCellExperiment)
library(ggplot2)
library(dplyr)
library(Matrix)
library("openai")
library("GPTCelltype")

barcode_file <- "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/filtered_feature_bc_matrix/barcodes.tsv.gz"
features_file <- "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/filtered_feature_bc_matrix/features.tsv.gz"
matrix_file <- "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/filtered_feature_bc_matrix/matrix.mtx.gz"

expression_matrix <- Read10X(data.dir = "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/filtered_feature_bc_matrix")

seurat_object <- CreateSeuratObject(counts = expression_matrix, project = "snRNAseq")

DefaultAssay(seurat_object) <- "RNA"

seurat_object <- NormalizeData(seurat_object, normalization.method = "CLR", assay = "RNA")

s.genes <- cc.genes.updated.2019$s.genes

g2m.genes <- cc.genes.updated.2019$g2m.genes

s.genes <- intersect(rownames(seurat_object[["RNA"]]), s.genes)

g2m.genes <- intersect(rownames(seurat_object[["RNA"]]), g2m.genes)

rna_layers <- Layers(seurat_object[["RNA"]])

print(rna_layers)

counts_gene_expression <- GetAssayData(seurat_object, assay = "RNA", layer = "counts.Gene Expression")

data_gene_expression <- GetAssayData(seurat_object, assay = "RNA", layer = "data.Gene Expression")

counts_peaks <- GetAssayData(seurat_object, assay = "RNA", layer = "counts.Peaks")

data_peaks <- GetAssayData(seurat_object, assay = "RNA", layer = "data.Peaks")

rna_data <- GetAssayData(seurat_object[["RNA"]], layer = "counts.Gene Expression")

calculate_module_score <- function(data, features) {
avg_expression <- colMeans(data[features, , drop = FALSE])
return(avg_expression)
}

s_score <- calculate_module_score(rna_data, s.genes)

g2m_score <- calculate_module_score(rna_data, g2m.genes)

seurat_object <- AddMetaData(seurat_object, metadata = s_score, col.name = "S.Score")

seurat_object <- AddMetaData(seurat_object, metadata = g2m_score, col.name = "G2M.Score")

seurat_object$Phase <- ifelse(seurat_object$S.Score > seurat_object$G2M.Score, "S", "G2M")

seurat_object <- FindVariableFeatures(seurat_object, assay = "RNA", layer = "counts.Gene Expression", selection.method = "vst", nfeatures = 2000)

seurat_object <- ScaleData(seurat_object, assay = "RNA", layer = "counts.Gene Expression", vars.to.regress = c("S.Score", "G2M.Score"))

seurat_object <- RunPCA(seurat_object, assay = "RNA")

head(seurat_object@meta.data)

seurat_object <- RunUMAP(seurat_object, assay = "RNA", dims = 1:20)

seurat_object <- FindNeighbors(seurat_object, assay = "RNA", dims = 1:20)

seurat_object <- FindClusters(seurat_object, resolution = 0.2)

Sys.setenv(OPENAI_API_KEY = 'Your-Key-Here')

seurat_object <- JoinLayers(seurat_object)

table(Idents(seurat_object))

markers <- FindAllMarkers(seurat_object)

write.csv(markers, file = "normal_cluster_markers.csv", row.names = FALSE)

res <- gptcelltype(markers, model = 'gpt-3.5-turbo-0125')   #gpt-3.5-turbo-0125, gpt-4o-2024-05-13 gpt-4o

res

seurat_object@meta.data$celltype <- as.factor(res[as.character(Idents(seurat_object))])

DimPlot(seurat_object, group.by = 'celltype')

ggsave("/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/GSE230389_mutant_astrocytoma_dimplot_seurat.jpeg", plot = last_plot(), device = "jpeg", width = 10, height = 8, units = "in")

cell_counts <- table(seurat_object$celltype)

cell_counts

sce <- as.SingleCellExperiment(seurat_object)

writeH5AD(sce, "GSE230389_mutant_astrocytoma.h5ad")

saveRDS(seurat_object, file = "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/GSE230389_mutant_astrocytoma.rds")

save.image('/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/GSE230389_mutant_astrocytoma.RData')

barcodes <- rownames(seurat_object@meta.data)

celltypes <- seurat_object@meta.data$celltype

barcode_celltype_df <- data.frame(Barcode = barcodes, CellType = celltypes)

write.csv(barcode_celltype_df, file = "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/GSE230389_mutant_astrocytoma_barcodes_and_celltypes.csv", row.names = FALSE)

write.csv(markers, file = "/path/to/cancer_multiome/GSE230389_mutant_astrocytoma/GSE230389_mutant_astrocytoma_normal_cluster_markers.csv", row.names = FALSE)
