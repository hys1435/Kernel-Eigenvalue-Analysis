#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:08:34 2019

@author: Zhaoqi Li
"""

import numpy as np
from kpm import select_D, kpca
from helper_fun import compute_gram_mat, init_sim_data_2_sphere
from kernels import arccos

def generate_coeffs(n):
    return np.random.uniform(size = n)

def main():
    # Experiments
    N = 200
    X = init_sim_data_2_sphere(N)
    params = None
    K = compute_gram_mat(X, X, arccos, params)
    D = 3 # take the first three dimension
    Alpha, Lam = kpca(K, D)

    coeffs = generate_coeffs(D)
    print("true coeffs: ", coeffs)
    Phi = K @ Alpha / np.sqrt(N)
    print("Shape of Phi: ", Phi.shape)
    #epsilon = np.random.normal(scale=1/25, size = N)
    epsilon = 0
    y = Phi @ coeffs + epsilon
    print("Shape of y: ", y.shape)

    D_max = 5
    C = 0
    D_opt, mse = select_D(K, y, D_max, C)
    print("D_opt: ", D_opt)
    Alpha, Lam = kpca(K, D_opt)
    Phi = K @ Alpha / np.sqrt(N)
    coeffs = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ y)
    print("predicted coeffs: ", coeffs)

if __name__ == '__main__':
     main()
