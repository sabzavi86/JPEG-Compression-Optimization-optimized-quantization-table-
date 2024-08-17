# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:19:52 2024

Author: ASUS
Description: Compare JPEG compression using standard and optimized quantization tables.
"""

import numpy as np
import cv2
import os
import joblib
import imghdr
import pandas as pd
from Helper_functions import jpeg 
from helperFunc import *

# Paths and constants (use relative paths)
dataset_path = "./Dataset_ORG"
gmm_model_path = './gmm_model.pkl'
gmm_labels_path = './gmm_cluster_labels.npy'
optimized_tables_path = './optimized_tables_gmm.npy'
results_output_path = './compression_comparison_results.csv'
QF = 50  # Quality factor

# Load the GMM model and labels
gmm = joblib.load(gmm_model_path)
gmm_labels = np.load(gmm_labels_path)

# Load the optimized tables
optimized_tables = np.load(optimized_tables_path, allow_pickle=True).item()

# Define the standard JPEG quantization table
standard_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Function to process a single image
def process_image(image_path, image_idx):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {'index': image_idx, 'error': 'Image not loaded correctly'}

    # Get the cluster label for this image
    cluster_label = gmm_labels[image_idx]
    key = list(optimized_tables.keys())[cluster_label]

    # Get the optimized table for the cluster
    optimized_table = optimized_tables.get(key, None)
    if optimized_table is None:
        return {'index': image_idx, 'cluster': cluster_label, 'error': 'Optimized table not found'}

    # Compress the image using both quantization tables
    try:
        psnr1, psnr2, bpp_compressed1, bpp_compressed2, bpp_uncompressed, ssim1, ssim2 = jpeg(
            img, standard_quantization_table, optimized_table, QF
        )
    except Exception as e:
        return {'index': image_idx, 'cluster': cluster_label, 'error': str(e)}

    # Store the results
    return {
        'index': image_idx,
        'cluster': cluster_label,
        'psnr_standard': psnr1,
        'psnr_optimized': psnr2,
        'bpp_standard': bpp_compressed1,
        'bpp_optimized': bpp_compressed2,
        'bpp_uncompressed': bpp_uncompressed,
        'ssim_standard': ssim1,
        'ssim_optimized': ssim2
    }

# Process all images in the dataset
all_results = []

# List all files in the dataset path
image_files = os.listdir(dataset_path)

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(dataset_path, image_file)
    if imghdr.what(image_path):  # Check if the file is a valid image
        print(f'Processing image {image_file}')
        result = process_image(image_path, idx)
        if 'error' not in result:
            all_results.append(result)
        else:
            print(f"Error processing image {image_file}: {result['error']}")
    else:
        print(f"Unsupported image format for {image_file}")

# Save the results to a CSV file for further analysis
results_df = pd.DataFrame(all_results)

# Output path for CSV file
csv_output_path = './all_results_gmm.csv'
results_df.to_csv(csv_output_path, index=False)

print("JPEG compression comparison for all images is complete.")
