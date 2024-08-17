# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:14:15 2023

@author: ASUS
"""

import os
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, LayerNormalization, UpSampling2D, Reshape, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
from tensorflow.image import psnr
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from sklearn.model_selection import train_test_split

import numpy as np
from scipy.fftpack import dctn, idctn

from PIL import Image

# Update the path below with the actual path to your dataset
path = "path/to/your/dataset"  # Replace with your dataset directory

data_path_label = [(os.path.join(dirpath, f), dirpath.split('/')[-1]) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]

def dct_idct(blocks, type="dct"):
    Xlen = blocks.shape[0]
    Ylen = blocks.shape[1]
    f = dct if type == "dct" else idct
    dctBlocks = np.zeros((Xlen, Ylen, 8, 8))
    for x in range(Xlen):
        for y in range(Ylen):
            d = np.zeros((8, 8))
            block = blocks[x][y]
            if f == dct:
                d = cv2.dct((block))
            else:
                d = (((cv2.idct((block)))))
                for i in range(8):
                    for j in range(8):
                        if d[i][j] < 0:
                            d[i][j] = 0
                        elif d[i][j] > 255:
                            d[i][j] = 255
            dctBlocks[x][y] = d
    return dctBlocks

def cof(blocks, num1, num2):
    Xlen = blocks.shape[0]
    Ylen = blocks.shape[1]
    cf = []
    for x in range(Xlen):
        for y in range(Ylen):
            cf.append(blocks[x][y][num1][num2])
    return cf

def toBlock(img):
    Xlen = img.shape[0] // 8
    Ylen = img.shape[1] // 8
    blocks = np.zeros((Xlen, Ylen, 8, 8))
    for x in range(Xlen):
        for y in range(Ylen):
            blocks[x][y] = img[x * 8:(x + 1) * 8, y * 8:(y + 1) * 8]
    return blocks

def to_dct(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (512, 512))
    blocks = toBlock(img)
    dctBlocks = dct_idct(blocks, "dct")

    dct_tensor = np.zeros((64, 64, 64))
    for i in range(64):
        a = int(i // 8)
        b = int(i % 8)
        dct_tensor[:, :, i] = np.array(cof(dctBlocks, a, b)).reshape(64, 64)

    return dct_tensor

dataset_X = []
dataset_y = []
for data, label in data_path_label:
    dataset_X.append(to_dct(data))
    dataset_y.append(label)
    print(data)

np.save('dataset_ORG.npy', dataset_X)

def psnr_loss(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=2003)
