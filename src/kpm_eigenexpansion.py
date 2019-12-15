#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for plotting the prediction error of coefficients of eigenfunctions with respect to noise. 

@author: Zhaoqi Li
"""

import numpy as np
from kpm import select_D, kpca
from helper_fun import init_sim_data_2_sphere
from kernels import arccos, compute_gram_mat
import matplotlib.pyplot as plt

def generate_coeffs(n):
    return np.random.uniform(size = n)

def main():
    # Experiments
    N = 400
    X = init_sim_data_2_sphere(N)
    K = compute_gram_mat(X, X, arccos)
    D = 3 # take the first three dimension
    Alpha, Lam = kpca(K, D)

    sim_num = 10
    #coeffs = generate_coeffs(D)
    coeffs = np.array([0.2, 0.4, 0.9])
    print("true coeffs: ", coeffs)
    Phi = K @ Alpha / np.sqrt(N)
    sigma_lst = np.array([0, 0.005, 0.01, 0.02, 0.04])
    pred_coeffs_lst = np.zeros([sigma_lst.size, sim_num, 3]) # three coefficients
    for i, sigma in enumerate(sigma_lst):
        for k in range(sim_num):
            epsilon = np.random.normal(scale=sigma, size = N)
            y = Phi @ coeffs + epsilon
        
            D_max = 5
            C = 0
            D_opt, mse = select_D(K, y, D_max, C)
            Alpha, Lam = kpca(K, D_opt)
            Phi = K @ Alpha / np.sqrt(N)
            pred_coeffs_lst[i,k] = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ y)
    
    pred_coeffs_err = np.std(pred_coeffs_lst, axis = 1)
    pred_coeffs_lst = np.mean(pred_coeffs_lst, axis = 1)
    
    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'purple']
    markers = ['o', 's', '^']
    for i in range(3):
        ax.errorbar(sigma_lst, pred_coeffs_lst[:,i].transpose(), 
                    yerr = pred_coeffs_err[:,i].transpose(), c=cols[i], 
                    marker=markers[i],label='c{}'.format(i+1),capsize=3)
    plt.legend(loc='best')
    plt.xlabel("sigma")
    plt.ylabel("Predicted coefficients")
    plt.savefig("kpm_eigenexpansion")
    
if __name__ == '__main__':
     main()
