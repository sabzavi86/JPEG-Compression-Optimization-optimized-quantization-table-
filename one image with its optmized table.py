# src/main_script.py

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from Helper_functions import jpeg

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

def main():
    # Load image 0375
    image_index = 375
    image_filename = f'{image_index:04d}.jpg'
    image_path = os.path.join("../datasets/Dataset_ORG/", image_filename)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Error: Image {image_path} not loaded correctly.")

    # Load the GMM cluster labels, optimized tables, and center indices
    gmm_labels = np.load('../models/gmm_cluster_labels.npy')
    optimized_tables = np.load('../models/optimized_tables_gmm.npy', allow_pickle=True).item()
    gmm_center_indices = np.load('../models/gmm_center_indices.npy')

    # Find the cluster label for the image
    cluster_label = gmm_labels[image_index]

    # Find the image number of the center that corresponds to this cluster
    center_image_number = gmm_center_indices[cluster_label]

    # Retrieve the optimized table for the center image
    if center_image_number in optimized_tables:
        optimized_table = optimized_tables[center_image_number]
    else:
        raise ValueError(f"No optimized table found for center image {center_image_number}")

    # Perform JPEG compression using the standard and optimized quantization tables
    psnr_standard, psnr_optimized, bpp_standard, bpp_optimized, bpp_uncompressed, ssim_standard, ssim_optimized, img_standard_compressed, img_optimized_compressed = jpeg(
        img, standard_quantization_table, optimized_table, 50)

    # Print results
    print(f"JPEG Compression Results with Standard Quantization Table:")
    print(f"PSNR: {psnr_standard}")
    print(f"BPP (Compressed): {bpp_standard}")
    print(f"BPP (Uncompressed): {bpp_uncompressed}")
    print(f"SSIM: {ssim_standard}")

    print(f"\nJPEG Compression Results with Optimized Quantization Table:")
    print(f"PSNR: {psnr_optimized}")
    print(f"BPP (Compressed): {bpp_optimized}")
    print(f"SSIM: {ssim_optimized}")

    # Define the region to zoom in on (e.g., top-left corner 150x150 pixels)
    start_row, start_col = 100, 100
    end_row, end_col = start_row + 150, start_col + 150

    # Extract the regions of interest (ROIs)
    roi_original = img[start_row:end_row, start_col:end_col]
    roi_standard = img_standard_compressed[start_row:end_row, start_col:end_col]
    roi_optimized = img_optimized_compressed[start_row:end_row, start_col:end_col]

    # Display the zoomed-in regions
    plt.figure(figsize=(15, 5))

    # Original Image Region
    plt.subplot(1, 3, 1)
    plt.imshow(roi_original, cmap='gray')
    plt.title("Original Image Region")
    plt.axis('off')

    # Compressed Image Region with Standard Table
    plt.subplot(1, 3, 2)
    plt.imshow(roi_standard, cmap='gray')
    plt.title("Compressed with Standard Table")
    plt.axis('off')

    # Compressed Image Region with Optimized Table
    plt.subplot(1, 3, 3)
    plt.imshow(roi_optimized, cmap='gray')
    plt.title("Compressed with Optimized Table")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
