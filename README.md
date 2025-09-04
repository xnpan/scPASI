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

##  Directory Structure

```
scFoundation-main/
├── preprocessing/
│   ├── code/ (preprocessing scripts)
│   ├── data/ (input data)
│   └── output/ (preprocessing output)
└── model/
    ├── get_embedding.py
    ├── examples/single_cell_data/ (preprocessed .h5ad files)
    └── output/single_cell_data/ (feature output)
```

## Res-VAE transfer learning module
