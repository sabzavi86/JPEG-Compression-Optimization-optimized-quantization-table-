# JPEG Compression Optimization using Autoencode,GMM and PCA

## Overview
This repository contains a project focused on optimizing JPEG compression by using Gaussian Mixture Models (GMM) and Principal Component Analysis (PCA). The goal is to improve the quality and compression ratio of images by adjusting the JPEG quantization tables based on clustering analysis.

## Contents
Due to GitHub's file size limitations, certain large files are not included in this repository. These files include the dataset, trained models, and certain large result files. However, the scripts and smaller files are provided for you to set up and run the project with your own data.

### Excluded Large Files:
- `best_model.h5`: The trained neural network model.
- `dataset_ORG.npy`: The original dataset of images.
- `X_train_ORG.npy`: Training dataset.
- `X_val_ORG.npy`: Validation dataset.

### Included Files:
- **Optimized Tables:**
  - `optimized_tables_gmm.npy`: Optimized quantization tables for different image clusters.
  - `gmm_cluster_labels.npy`: Cluster labels for the dataset images.
  - `gmm_center_indices.npy`: Indices of the cluster centers in the dataset.

- **Scripts:**
  - `center_optimized_tables_gmm.py`: Script to calculate optimized quantization tables for the cluster centers.
  - `clustering_comparison.py`: Script to compare different clustering algorithms.
  - `clustering_gmm.py`: Script to perform GMM-based clustering.
  - `com_decom.py`: Contains functions for compressing and decompressing images.
  - `dataset.py`: Script for preparing the dataset.
  - `network_training.py`: Script for training the neural network model.
  - `one_image_with_optimized_table.py`: Script to compress one image with the optimized quantization table.
  - `optimized_table_cal.py`: Contains functions to calculate the optimized quantization tables.
  - `jpeg_all_images_gmm.py`: Script to compress all images in the dataset using the optimized tables.
  
- **Result Files:**
  - `all_results_gmm.csv`: CSV file containing the results of JPEG compression with optimized quantization tables.

## Installation

1. **Clone the repository:**
   - `git clone https://github.com/sabzavi86/JPEG-Compression-Optimization-optimized-quantization-table-.git`

2. **Install dependencies:**
   - Ensure you have Python 3.x installed. Then, install the required Python packages by running:
   - `pip install -r requirements.txt`

3. **Dataset and Model Files:**
   Since the dataset and model files are not included due to their large size, you will need to:
   - **Dataset:** Use your own dataset or download the required datasets from a suitable source.
   - **Model Files:** If you want to use the pre-trained models, you can contact the repository owner or use your own models.

4. **Create the necessary files:**
   If you're training the model from scratch, the training process will generate the necessary model files.

## Usage

### 1. Train the Model
You can train the neural network model using the `network_training.py` script:

- `python network_training.py`

This script will train the model on your dataset (assuming it's correctly formatted and placed) and save the model files.

### 2. Perform Clustering and Compression
Once the model is trained, you can perform clustering and compress the images using the optimized quantization tables by running:

- `python clustering_gmm.py`
- `python jpeg_all_images_gmm.py`

### 3. Test on a Single Image
To test the compression on a single image with the optimized quantization table:

- `python one_image_with_optimized_table.py`

## Notes
- Due to GitHub's file size limits, the following required files are not included in this repository: `best_model.h5`, `dataset_ORG.npy`, `X_train_ORG.npy`, and `X_val_ORG.npy`. You'll need to generate or obtain these files as outlined above.
- The included scripts are set up to be as general as possible, but you may need to modify paths or parameters to match your specific setup.

For any further assistance or to obtain the large files, please feel free to contact the repository owner.

---

