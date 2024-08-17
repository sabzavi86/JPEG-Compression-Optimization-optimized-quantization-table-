# -*- coding: utf-8 -*-
"""
Helper Functions for JPEG Compression and DCT Transformations

Created on Tue Sep 13 10:48:13 2022
@author: ASUS

This module contains functions for:
- Converting images to DCT blocks and back
- Performing DCT and inverse DCT
- Calculating PSNR and SSIM
- Handling zigzag scanning and Run Length Coding (RLC)
- JPEG compression simulations with various quantization tables
"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct
from math import log10, sqrt, e, log
import random
from scipy.stats import laplace
import statistics
from skimage.metrics import structural_similarity as compare_ssim

# Block-based Image Processing
def toBlock(img):
    """Convert an image into 8x8 blocks."""
    xLen = img.shape[1] // 8
    yLen = img.shape[0] // 8
    blocks = np.zeros((yLen, xLen, 8, 8))
    for y in range(yLen):
        for x in range(xLen):
            blocks[y][x] = img[y*8:(y+1)*8, x*8:(x+1)*8]
    return blocks

def toImg(blocks):
    """Reconstruct an image from 8x8 blocks."""
    Xlen = blocks.shape[0]
    Ylen = blocks.shape[1]
    Ximg = Xlen * 8
    Yimg = Ylen * 8
    img = np.zeros((Ximg, Yimg))
    for x in range(Xlen):
        for y in range(Ylen):
            img[x*8:x*8+8, y*8:y*8+8] = blocks[x][y]
    return img

# DCT and IDCT
def dct_idct(blocks, type="dct"):
    """Apply DCT or IDCT to 8x8 blocks."""
    Xlen = blocks.shape[0]
    Ylen = blocks.shape[1]
    f = dct if type == "dct" else idct
    dctBlocks = np.zeros((Xlen, Ylen, 8, 8))
    for x in range(Xlen):
        for y in range(Ylen):
            block = blocks[x][y]
            if f == dct:
                dctBlocks[x][y] = cv2.dct(block)
            else:
                d = cv2.idct(block)
                d[d < 0] = 0
                d[d > 255] = 255
                dctBlocks[x][y] = d
    return dctBlocks

# PSNR Calculation
def PSNR_cal(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2) + 1e-30
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse

# Zigzag Scanning and Run Length Coding
def zigZag(block):
    """Perform zigzag scan on an 8x8 block."""
    lines = [[] for _ in range(8 + 8 - 1)]
    for x in range(8):
        for y in range(8):
            i = x + y
            lines[i].insert(0, block[x][y]) if i % 2 == 0 else lines[i].append(block[x][y])
    return np.array([coefficient for line in lines for coefficient in line])

def rlc(block):
    """Run Length Coding (RLC) for a block."""
    block = block.astype(np.int)
    rlc = []
    run = 0    
    ac = np.copy(block[1:63])
    for i in range(62):
        if ac[i] == 0:
            run += 1
        else:
            rlc.append(run)
            rlc.append(ac[i])
            run = 0
    if run != 0:
        rlc.append(run)
        rlc.append(0)
    count_rlc_bit = sum(len(bin(x)) for x in rlc)
    return count_rlc_bit

def bitCount(blocks):
    """Count the bits required for blocks using RLC and zigzag."""
    blocks = blocks.astype(np.int)
    dc_bit = sum(len(bin(block[0][0])) for block in blocks.reshape(-1, 8, 8))
    ac_bit = sum(rlc(zigZag(block)) for block in blocks.reshape(-1, 8, 8))
    return dc_bit + ac_bit

# DC and AC Coefficient Extraction
def DC_cof(blocks):
    """Extract DC coefficients from DCT blocks."""
    return [blocks[x][y][0][0] for x in range(blocks.shape[0]) for y in range(blocks.shape[1])]

def AC_cof(blocks, num1, num2):
    """Extract specific AC coefficients from DCT blocks."""
    return [blocks[x][y][num1][num2] for x in range(blocks.shape[0]) for y in range(blocks.shape[1])]

def AC_cofsel(blocks, num1, num2):
    """Extract specific AC coefficients from a flat list of blocks."""
    return [block[num1][num2] for block in blocks]

# Quantization Table Scaling
def table_scale(Table, Q):
    """Scale the quantization table based on quality factor Q."""
    scale_factor = 5000 / Q if Q <= 50 else 200 - 2 * Q if Q < 100 else 1
    TableS = np.floor((scale_factor * Table + 50) / 100)
    TableS[TableS == 0] = 1
    return TableS

# JPEG Compression Simulation
def jpeg(img_in, Table1, Table2, QF):
    """Simulate JPEG compression with two different quantization tables."""
    M, N = img_in.shape
    img_size = M * N
    img_sizeBit = M * N * 8
    MB = M // 8
    NB = N // 8

    # JPEG encoding/decoding with the first table
    blocks = toBlock(img_in)
    dctBlocks = dct_idct(blocks, "dct")
    TableS1, qDctBlocks1 = dct_scale(dctBlocks, QF, Table1)
    decodedImg1 = toImg(dct_idct(np.multiply(qDctBlocks1, TableS1), "idct")).astype('uint8')
    psnr1 = PSNR_cal(decodedImg1, img_in)
    ssim1 = compare_ssim(decodedImg1, img_in)
    compressed_bit1 = bitCount(qDctBlocks1)
    bpp_compressed1 = compressed_bit1 / img_size
    compression_ratio1 = img_sizeBit / compressed_bit1

    # JPEG encoding/decoding with the second table
    TableS2, qDctBlocks2 = dct_scale(dctBlocks, QF, Table2)
    decodedImg2 = toImg(dct_idct(np.multiply(qDctBlocks2, TableS2), "idct")).astype('uint8')
    psnr2 = PSNR_cal(decodedImg2, img_in)
    ssim2 = compare_ssim(decodedImg2, img_in)
    compressed_bit2 = bitCount(qDctBlocks2)
    bpp_compressed2 = compressed_bit2 / img_size
    compression_ratio2 = img_sizeBit / compressed_bit2

    return psnr1[0], psnr2[0], bpp_compressed1, bpp_compressed2, bpp_compressed1, ssim1, ssim2, decodedImg1, decodedImg2
