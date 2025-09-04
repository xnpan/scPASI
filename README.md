# scPASI: Single-cell Phenotype-associated Subpopulation Identification via Transfer Foundation Model and Statistical Ensemble Learning

## Introduction

scPASI is a robust framework that extracts features from transcriptomic data and performs statistical analysis on regression coefficients to identify and stratify cell subpopulations into Strongly Positive (SP), Weakly Positive (WP), Strongly Negative (SN), and Weakly Negative (WN) groups.

scPASI includes four sub-modules: PFM-based feature extraction module, Res-VAE transfer learning module, statistical feature learning module, and cell phenotype identification module.

## PFM-based feature extraction module

##  1. Environment Setup

1. Download code from the [official repository](https://github.com/biomap-research/scFoundation) and set up environment
2. This project includes original code in `./PFM_feature_extraction/scfoundation/original_scfoundation_project` 

##  2. Data Preprocessing

Place raw csv files (cells/samples × genes, Gene Symbols) in `./scFoundation-main/preprocessing/raw_csvdata/`, then run:

```bash
python ./scFoundation/code/csv_preprocess_under_scfoundation.py \
  --system_path ./scFoundation-main/preprocessing \
  --file_name ./raw_csvdata \
  --output_dir ./h5ad_output \
  --sparse_matrix True
```

Output will be saved to `./h5ad_output/`

##  3. Feature Extraction

Place preprocessed `.h5ad` files in `./scFoundation-main/model/examples/single_cell_data/`, then run:

### For Bulk RNA-seq data:

```bash
python get_embedding.py \
  --task_name AllbulkDEll \
  --input_type bulk \
  --output_type cell \
  --pool_type all \
  --data_path ./examples/single_cell_data/preprocessed_all_ALL_Bulk_Dell.h5ad \
  --pre_normalized F \
  --version rde \
  --save_path ./output/single_cell_data \
  --tgthighres f1
```

### For Single Cell RNA-seq data:

```bash
python get_embedding.py \
  --task_name FA34 \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --data_path ./examples/single_cell_data/preprocessed_GSE149214.h5ad \
  --pre_normalized F \
  --version rde \
  --save_path ./output/single_cell_data \
  --tgthighres f1
```

## Res-VAE transfer learning module

## 1. Environment Setup

1. Download code from the [official repository](https://github.com/doriszmr/scATD) and set up environment
2. This project includes original code in `./Res_VAE_pretraining/original_project` 

##  2. Res_VAE Training Procedure

### Step 1: Hyperparameter Optimization

```bash
python ./Res_VAE_pretraining/skf_pretraining/code/VAE_sf_Res-VAE_hyperparam_pretraining.py
```

Note: Modify these parameters directly in the script:

- `open_path`: Path to input features (.npy)
- `save_path_outer`: Output directory
- `file_prefix`: File naming prefix

Output: Optimal hyperparameter configuration file (`VAE_sf_best_hyperparameters.xlsx`)

### Step 2: Pretraining with Optimal Hyperparameters

```bash
python ./Res_VAE_pretraining/pretraining_after_hyperparameter/code/VAE_sf_Res-VAEpretraining.py \
    --open_path ./data/ \
    --save_path_outer ./output \
    --open_path_conference_data ./reference_data \
    --file_prefix scRNA-seq_panglao \
    --epoch_start_for_loss_plot_only 1 \
    --batch_size 128 \
    --REC_beta 1000 \
    --best_parameter_name VAE_sf_best_hyperparameters.xlsx
```

## Statistical feature learning module & Cell phenotype identification module
## 1.  Process Transferred Embeddings

Run the Jupyter notebook `transferred_embeddings_process.ipynb` to:

- Process embeddings extracted from scFoundation and pretrained Res-VAE
- Generate Leiden clustering labels
- Compute UMAP coordinates for single cells

### Input Requirements:

- Transferred embeddings from scFoundation and Res-VAE
- Preprocessed data in compatible format

### Output:

- Leiden clustering labels
- UMAP coordinates for visualization
- Processed embeddings for downstream analysis

## 2. Identify Phenotype-Associated Cell Subpopulations

Run the R script `cell_phenotype_identification.R` to:

- Load processed transferred cell and sample embeddings
- Incorporate phenotype information and Leiden clustering labels
- Calculate Pearson correlation matrices between cells and samples
- Input data into regression models
- Compute regression coefficients and count positive/negative associations
- Identify cell subpopulations with varying association strengths to phenotypes

### Key Parameters:

- Input: Preprocessed transferred embeddings
- Method: Pearson correlation + regression analysis
- Output: Cell subpopulations ranked by phenotype association strength
