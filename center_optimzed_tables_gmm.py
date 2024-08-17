# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:19:52 2024

Author: ASUS
Description: Optimize quantization tables for images using GMM clustering.
"""

import numpy as np
import joblib
import cv2
import os
from Helper_functions import * 
from helperFunc import *
from optimized_table_cal import optimized_table
from com_decom import *

# Paths to the dataset and models (use relative paths)
dataset_path = "./Dataset_ORG"  # Ensure this directory contains your images
gmm_model_path = './gmm_model.pkl'
gmm_labels_path = './gmm_cluster_labels.npy'
gmm_centers_path = './gmm_centers.npy'
gmm_center_indices_path = './gmm_center_indices.npy'

# Load the pre-trained GMM model, labels, and centers
gmm = joblib.load(gmm_model_path)
gmm_labels = np.load(gmm_labels_path)
gmm_centers = np.load(gmm_centers_path)
gmm_center_indices = np.load(gmm_center_indices_path)

# Function to optimize quantization tables for images at the cluster centers
def process_center_images(indices, dataset_path):
    optimized_tables = {}
    
    for index in indices:
        image_filename = f'{index:04d}.jpg'  # Format the index with leading zeros
        image_path = os.path.join(dataset_path, image_filename)
        
        # Load the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error: Image at index {index} ({image_filename}) not loaded correctly.")
            continue
        
        # Apply the optimized_table function to get the best quantization table
        bestQuaTable, best_psnr, bestCompression_ratio, best_compressed_KB, best_SSIM_img, iterations, unfitness_scores = optimized_table(img)
        
        # Save the results in a dictionary
        optimized_tables[index] = bestQuaTable
        
        # Optionally, save the optimized table to a file for each image
        np.save(f'optimized_table_{index:04d}.npy', bestQuaTable)
        
        # Print the results for reference
        print(f"Optimized table for image {index} ({image_filename}):")
        print(bestQuaTable)
        print(f"PSNR: {best_psnr}, Compression Ratio: {bestCompression_ratio}, Compressed KB: {best_compressed_KB}, SSIM: {best_SSIM_img}")

    return optimized_tables

# Process the center images
optimized_tables = process_center_images(gmm_center_indices, dataset_path)

# Save all optimized tables in a single file for further reference
np.save('optimized_tables.npy', optimized_tables)

print("Processing complete. Optimized tables saved.")
