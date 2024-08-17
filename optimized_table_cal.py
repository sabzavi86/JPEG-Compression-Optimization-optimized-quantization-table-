# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 11:41:08 2021

Author: ASUS
Description: Optimize quantization table using genetic algorithm for JPEG compression.
"""

import cv2
import numpy as np
import time
from Helper_functions import *
from com_decom import com_decom
from helperFunc import *

def optimized_table(img):
    # Standard JPEG quantization table
    TableS = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Evaluate the standard table
    Compression_ratioS, psnrS, decodedImgS, SSIM_imgS, compressed_KBS = com_decom(img, TableS)
    
    # Genetic Algorithm parameters
    pop_size = 200
    sel_num = 50
    unfitness_pop = np.zeros((pop_size, 1))
    best_population = np.zeros((350, 8, 8))
    best_unfitness = np.zeros((best_population.shape[0], 1))
    
    previous_choice = 0
    new_population = mutate(pop_size, TableS)
    
    print("Start")
    t1 = time.time()
    
    # Evaluate initial population
    for i in range(pop_size):
        Compression_ratioO, psnrO, decodedImgO, SSIM_imgO, compressed_KBO = com_decom(img, new_population[i])
        unfitness_pop[i] = cal_unfitness(compressed_KBS, compressed_KBO, psnrS, psnrO)
        print(i, "-", psnrO)
    
    # Select the best parents from the initial population
    unfitnessO, ind, parents = selection(new_population, unfitness_pop, sel_num)
    t2 = time.time()
    print("Mutation: " + str(t2 - t1) + " seconds")
    
    best_parents = parents
    best_choice = unfitnessO[0]
    
    it_num = 0
    xxx = []
    yyy = []
    
    # Iterate over generations
    while it_num < 45:
        xxx.append(it_num)
        print("Iteration is: ", it_num)   
        previous_choice = best_choice
        
        best_population[300:350][:][:] = best_parents
        
        for k in range(0, 300, 20):
            a, b = crossover(best_parents)
            best_population[k:k+10][:][:] = mutate(10, a)
            best_population[k+10:k+20][:][:] = mutate(10, b)
        
        # Evaluate new population
        for j in range(best_population.shape[0]):
            bestCompression_ratio, best_psnr, best_decodedImgO, best_SSIM_img, best_compressed_KB = com_decom(img, best_population[j])
            best_unfitness[j] = cal_unfitness(compressed_KBS, best_compressed_KB, psnrS, best_psnr)
        
        # Select the best individuals
        best_unfitnessO, best_ind, best_parents = selection(best_population, best_unfitness, sel_num)
        
        best_choice = best_unfitnessO[0]
        print("Best choice is: ", best_choice)
        print("Previous choice is: ", previous_choice)
        it_num += 1
        yyy.append(best_choice)
    
    # Final output after all generations
    bestCompression_ratio, best_psnr, best_decodedImgO, best_SSIM_img, best_compressed_KB = com_decom(img, best_parents[0][:][:])
    bestQuaTable = best_parents[0][:][:]
    
    t3 = time.time()
    print("Table calculation time is:", t3 - t1)
    print(xxx, yyy)
    
    return bestQuaTable, best_psnr, bestCompression_ratio, best_compressed_KB, best_SSIM_img, xxx, yyy
