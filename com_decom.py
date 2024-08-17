# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:43:07 2021

Author: ASUS
Description: Compress and decompress an image using a specified quantization table, 
and evaluate the quality of the compressed image.
"""

import cv2
import numpy as np
from Helper_functions import *
from helperFunc import *
from skimage.metrics import structural_similarity as ssim

def com_decom(img, Table):
    """
    Compress and decompress an image using a specified quantization table,
    and evaluate the quality of the compressed image.
    
    Parameters:
    img (numpy.ndarray): The input grayscale image.
    Table (numpy.ndarray): The quantization table used for compression.
    
    Returns:
    tuple: compression_ratio, psnr, decodedImg, SSIM_img, compressed_bitK
    """
    
    img_h, img_w = img.shape    
    img_sizeBit = img.size * 8  # Calculate the total number of bits in the original image
    
    # Divide the image into 8x8 blocks and apply DCT
    blocks = toBlocks(img)
    dctBlocks = dct_idct(blocks, "dct")
    
    # Quantize the DCT coefficients using the provided quantization table
    qDctBlocks = np.round(np.divide(dctBlocks, Table))
    
    # Convert the quantized DCT coefficients to zigzag order for entropy coding
    zigzag = zigZag_Img(qDctBlocks)
    
    # Calculate the number of bits in the compressed image
    compressed_bit = bitCount(qDctBlocks)
    compressed_bitK = compressed_bit / 1024  # Convert to kilobits
    bpp_compressed = compressed_bit / img.size
    bpp_uncompressed = img_sizeBit / img.size
    compression_ratio = bpp_uncompressed / bpp_compressed
    
    # Dequantize the DCT coefficients and apply inverse DCT to reconstruct the image
    deDctBlocks = dct_idct(np.multiply(qDctBlocks, Table), "idct")
    decodedImg = blocks2img(deDctBlocks).astype('uint8')
    
    # Calculate PSNR and SSIM between the original and decoded images
    psnr = PSNR_cal(img, decodedImg)
    SSIM_img = ssim(img, decodedImg)
    
    return compression_ratio, psnr, decodedImg, SSIM_img, compressed_bitK
