import snapatac2 as snap
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load ATAC fragment data
print(snap.__version__)
fragment_file = "/path/to/cancer_multiome/astrocytoma_atac_possorted_fragment.bed"
chrom_sizes = snap.genome.hg38
data = snap.pp.import_data(
    fragment_file,
    chrom_sizes=chrom_sizes,
    sorted_by_barcode=False,
)
print(data)

# Step 2: Plot fragment size distribution and save it
snap.pl.frag_size_distr(data, interactive=False)
fig = snap.pl.frag_size_distr(data, show=False)
fig.update_yaxes(type="log")
fig.write_image("astrocytoma_fragment_size_distribution.png")

# Step 3: Calculate TSSE metrics and plot
snap.metrics.tsse(data, chrom_sizes)
snap.pl.tsse(data, interactive=False)
snap.pl.tsse(data, interactive=False, show=False, out_file="astrocytoma_tsse_distribution.png")

# Calculate summary statistics for n_fragment (counts) and tsse
count_stats = {
    "min": np.min(data.obs['n_fragment']),
    "max": np.max(data.obs['n_fragment']),
    "25th_percentile": np.percentile(data.obs['n_fragment'], 25),
    "50th_percentile": np.median(data.obs['n_fragment']),
    "75th_percentile": np.percentile(data.obs['n_fragment'], 75),
    "mean": np.mean(data.obs['n_fragment']),
    "std": np.std(data.obs['n_fragment'])
}

tsse_stats = {
    "min": np.min(data.obs['tsse']),
    "max": np.max(data.obs['tsse']),
    "25th_percentile": np.percentile(data.obs['tsse'], 25),
    "50th_percentile": np.median(data.obs['tsse']),
    "75th_percentile": np.percentile(data.obs['tsse'], 75),
    "mean": np.mean(data.obs['tsse']),
    "std": np.std(data.obs['tsse'])
}

# Print the statistics
print("Count Metrics (n_fragment):", count_stats)
print("TSSE Metrics:", tsse_stats)

# Step 4: Filter cells based on counts and TSSE
snap.pp.filter_cells(data, min_counts=3500, min_tsse=4, max_counts=100000)
print(data)

# Step 5: Add tile matrix and select features
snap.pp.add_tile_matrix(data)
snap.pp.select_features(data, n_features=100000)

# Step 6: Remove doublets with Scrublet
snap.pp.scrublet(data)

# Step 7: Perform spectral clustering and UMAP
snap.tl.spectral(data)
snap.tl.umap(data)

# Step 8: KNN and Leiden clustering
snap.pp.knn(data)
snap.tl.leiden(data)

# Step 9: Plot ATAC UMAP with Leiden clusters
snap.pl.umap(data, color='leiden', interactive=False, height=500)
snap.pl.umap(data, color='leiden', interactive=False, height=500, show=False, out_file="FS4_umap_leiden_clusters.png")
snap.pl.umap(data, color='leiden', show=False, out_file="FS4_umap.pdf", height=500)

# Step 10: Generate gene matrix from ATAC data and preprocess
gene_matrix = snap.pp.make_gene_matrix(data, chrom_sizes)
sc.pp.filter_genes(gene_matrix, min_cells=130)
sc.pp.normalize_total(gene_matrix)
sc.pp.log1p(gene_matrix)

# Step 11: MAGIC imputation and UMAP integration
sc.external.pp.magic(gene_matrix, solver="approximate")
gene_matrix.obsm["X_umap"] = data.obsm["X_umap"]

# Step 12: Load marker genes and plot UMAP with gene expression
marker_genes = ['GFAP', 'UGT8', 'MBP', 'CNP', 'PLP1', 'IDH1', 'IDH2']
#marker_genes = ['GFAP', 'UGT8', 'MBP', 'CNP', 'PLP1', 'BRAF', 'CDKN2A', 'IDH1', 'IDH2', 'ATRX', 'TP53', 'PTEN', 'EGFR']
sc.pl.umap(gene_matrix, use_raw=False, color=["leiden"] + marker_genes)
plt.savefig("FS4_umap_marker_genes.png")
gene_matrix.write("FS4_gene_matrix.h5ad", compression='gzip')

# Step 13: Load RNA data and map CellType labels (This is the key integration step)
rna = sc.read_h5ad("/path/to/cancer_multiome/astrocytoma.h5ad")
print(rna)

# Load the CSV with barcodes and CellType labels
celltype_labels = pd.read_csv("/path/to/cancer_multiome/astrocytoma_CellType_labels_with_barcodes.csv")

# Set the 'barcode' column as the index to match it with rna.obs_names
celltype_labels.set_index('barcode', inplace=True)

# Remove 'S1_' prefix in celltype_labels so it can match the RNA and ATAC data later
celltype_labels.index = celltype_labels.index.str.replace('^S1_', '', regex=True)

# Map the CellType labels to the RNA data based on the barcodes
rna.obs['CellType'] = celltype_labels['CellType'].reindex(rna.obs_names).astype('category')

# Verify the CellType mapping
print(rna.obs['CellType'].unique())
print(rna.obs[['CellType']].head())

# Step 14: Process the RNA data (select highly variable genes and normalize)
sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=3000)
rna = rna[:, rna.var.highly_variable]
sc.pp.normalize_total(rna, target_sum=1e4)
rna = rna[:, rna.var.highly_variable].copy()
rna = rna[rna.X.sum(axis=1) > 0, :]
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

# Step 15: Spectral and UMAP embedding on RNA
snap.tl.spectral(rna, features=None)
snap.tl.umap(rna)
snap.pl.umap(rna, color='CellType', interactive=False, show=False, out_file='FS4_rna_umap.png')

# Step 16: Read ATAC gene matrix and handle sparse matrix conversion for spectral embedding
atac = sc.read_h5ad("/path/to/cancer_multiome/astrocytoma_ATAC_gene_matrix.h5ad")
from scipy.sparse import csr_matrix
atac.X = csr_matrix(atac.X)

# Step 17: Perform spectral clustering and UMAP on ATAC data
snap.tl.spectral(atac, features=None)
snap.tl.umap(atac)

# Step 18: Find common cells and map CellType
common_cells = rna.obs_names.intersection(atac.obs_names)
print(f"Number of common cells after removing prefix: {len(common_cells)}")

# Step 19: Subset RNA and ATAC for common cells, map CellType labels to ATAC data
if len(common_cells) > 0:
    rna_common = rna[common_cells]
    atac_common = atac[common_cells]
    atac_common.obs['CellType'] = rna_common.obs['CellType']
    snap.pl.umap(atac_common, color='CellType', interactive=False, show=False, out_file='astrocytoma_atac_umap.png')
else:
    print("No common cells found after preprocessing.")

# Step 20: Multi-modality spectral analysis and joint UMAP
embedding = snap.tl.multi_spectral([rna_common, atac_common], features=None)[1]
atac_common.obsm['X_joint'] = embedding
snap.tl.umap(atac_common, use_rep='X_joint')
snap.pl.umap(atac_common, color="CellType", interactive=False, show=False, out_file='astrocytoma_RNA_atac_joint_umap.png')

# Final note: Ensure all UMAP plots use the 'CellType' labels

import snapatac2 as snap
import numpy as np
import polars as pl

# Assuming your SnapATAC2 data object is named atac_common
data = atac_common

# Plot UMAP of cell types and save as an image
snap.pl.umap(data, color='CellType', interactive=False, show=False, out_file='astrocytoma_celltype_umap.png')

# Run peak calling using MACS3, grouping by cell type
snap.tl.macs3(data, groupby='CellType')

# Merge peaks and save the first few rows of the peaks as a CSV
peaks = snap.tl.merge_peaks(data.uns['macs3'], snap.genome.hg38)
peaks.head().to_csv('astrocytoma_merged_peaks.csv', index=False)

# Create a peak matrix and display its structure
peak_mat = snap.pp.make_peak_matrix(data, use_rep=peaks['Peaks'])
print(peak_mat)  # Display structure if needed

# Identify marker regions by cell type and save the output as CSV
marker_peaks = snap.tl.marker_regions(peak_mat, groupby='CellType', pvalue=0.01)
marker_peaks.to_csv('astrocytoma_marker_peaks.csv', index=False)

# Plot the marker regions by cell type and save as an image
snap.pl.regions(peak_mat, groupby='CellType', peaks=marker_peaks, interactive=False, show=False, out_file='astrocytoma_marker_regions_by_celltype.png')

# Perform motif enrichment analysis and save the results
motifs = snap.tl.motif_enrichment(
    motifs=snap.datasets.cis_bp(unique=True),
    regions=marker_peaks,
    genome_fasta=snap.genome.hg38,
)

# Plot motif enrichment and save as an image
snap.pl.motif_enrichment(motifs, max_fdr=0.0001, height=1600, interactive=False, show=False, out_file='astrocytoma_motif_enrichment.png')

# Specify groups for differential analysis (replace with relevant cell types in your data)
group1 = "1. Oligodendrocyte"
group2 = "3. Proliferating cell (possibly cancer cell)"
naive_B = data.obs['CellType'] == group1
memory_B = data.obs['CellType'] == group2
peaks_selected = np.logical_or(
    peaks[group1].to_numpy(),
    peaks[group2].to_numpy(),
)

# Perform differential peak analysis and filter for significant peaks
diff_peaks = snap.tl.diff_test(
    peak_mat,
    cell_group1=naive_B,
    cell_group2=memory_B,
    features=peaks_selected,
)
diff_peaks = diff_peaks.filter(pl.col('adjusted p-value') < 0.01)
diff_peaks.head().to_csv('diff_peaks.csv', index=False)

# Plot the differential regions by cell type and save as an image
snap.pl.regions(
    peak_mat,
    groupby='CellType',
    peaks={
        group1: diff_peaks.filter(pl.col("log2(fold_change)") > 0)['feature name'].to_numpy(),
        group2: diff_peaks.filter(pl.col("log2(fold_change)") < 0)['feature name'].to_numpy(),
    },
    interactive=False,
    show=False,
    out_file='diff_regions_by_celltype.png'
)

# Create a background set by sampling other cell types
barcodes = np.array(data.obs_names)
background = []
for i in np.unique(data.obs['CellType']):
    if i != group2:
        cells = np.random.choice(barcodes[data.obs['CellType'] == i], size=30, replace=False)
        background.append(cells)
background = np.concatenate(background)

# Perform differential peak analysis with the background
diff_peaks = snap.tl.diff_test(
    peak_mat,
    cell_group1=memory_B,
    cell_group2=background,
    features=peaks[group2].to_numpy(),
    direction="positive",
)
diff_peaks = diff_peaks.filter(pl.col('adjusted p-value') < 0.01)
diff_peaks.head().to_csv('astrocytoma_diff_peaks_vs_background.csv', index=False)

# Plot regions that differ significantly from background and save as an image
snap.pl.regions(
    peak_mat,
    groupby='CellType',
    peaks={ group2: diff_peaks['feature name'].to_numpy() },
    interactive=False,
    show=False,
    out_file='astrocytoma_diff_regions_vs_background.png'
)
