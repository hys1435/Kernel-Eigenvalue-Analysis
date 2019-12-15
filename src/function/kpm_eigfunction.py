#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for plotting the eigenfunction of f_star function given below. 

@author: Zhaoqi Li
"""

import os
import sys

sys.path.append('..')

import numpy as np
from helper.kpm import select_D, kpca
from helper.kernels import gaussianRBF, compute_gram_mat
import matplotlib.pyplot as plt

def f_star(x):
    return np.minimum(x, 1-x)

def init_sim_data(N):
    # Initialize data given N number of samples
    X = np.random.uniform(size = N)
    epsilon = np.random.normal(scale = 1/np.sqrt(5), size = N)
    y = f_star(X) + epsilon
    return X, y

# plot eigenfunctions
N = 2000
X, y = init_sim_data(N)
sigma = 0.1
K = compute_gram_mat(X, X, gaussianRBF, sigma)
D_max = 5
C = 0.01
D_opt, mse = select_D(K, y, D_max, C)
Alpha, Lam = kpca(K, D_opt)
X_seq = np.linspace(0, 1, num = 100)
K_seq = compute_gram_mat(X_seq, X, gaussianRBF, 0.1) # sigma is 0.1
print("Shape of K_seq: ", K_seq.shape)
print("Shape of Alpha: ", Alpha.shape)
Phi_seq = K_seq @ Alpha / np.sqrt(N)
Phi = K @ Alpha / np.sqrt(N)
coeffs = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ y)
print("coeffs: ", coeffs)
#print("Lam: ", Lam)

fig, ax1 = plt.subplots()
cols = ['red', 'blue', 'purple']
markers = ['o', '^', 's']
for i in range(D_opt):
    ax1.plot(X_seq, Phi_seq[:,i], c=cols[i],label='{}th Eigenfunction'.format(i+1))
plt.legend(loc='upper right')
plt.xlabel("X")
plt.ylabel("Value")
plt.savefig("kpm_eigfunction")