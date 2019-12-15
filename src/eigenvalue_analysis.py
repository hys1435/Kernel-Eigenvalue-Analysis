#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to check the polynomial decay of the eigenvalues. 

@author: hys1435
"""

import numpy as np
from helper_fun import eigen_n_sphere
from kernels import arccos, compute_gram_mat
import matplotlib.pyplot as plt

# eigenvalues for dim = 2
    
N = 1000
d = eigen_n_sphere(N, 2, arccos)
w = d['eig_val']

# compare it with N**-2

Nls = np.power(np.linspace(start = 1, stop = N, num = N), -2)
wl = w / Nls
fig, ax1 = plt.subplots()
ax1.plot(w)
plt.xlabel("Index")
plt.ylabel("Value")
ax1.set_yscale('log')
plt.show()

fig, ax2 = plt.subplots()
ax2.plot(wl)
plt.xlabel("i")
plt.ylabel("Eigenvalue/i^(-2)")
plt.show()

#%%

# eigenfunctions for dim = 2

N = 1000
d = eigen_n_sphere(N, 2, arccos)
X = d['data']
y = d['label']
K = d['gram_mat']
num = 20
alpha = np.linalg.solve(K, y)
theta_seq = np.linspace(start = 0, stop = 2 * np.pi - 1e-4, num = num)
X_seq = np.array([np.cos(theta_seq), np.sin(theta_seq)]).transpose()
K_test = compute_gram_mat(X_seq, X, arccos)
y_pred = K_test @ alpha
y_pred_lst = np.zeros((N, num))

for i in range(N):
    y_pred_lst[i] = K_test[:,i]

# Plot results
fig, ax = plt.subplots()
cols = ['red', 'blue', 'purple', 'orange']
print(theta_seq)
print(y_pred)
ax.plot(theta_seq, y_pred, label='actual function')
for i in range(N):
    ax.plot(theta_seq, y_pred_lst[i].astype(float), c=cols[i],
            label='{}th eigenfunction'.format(i+1))
    #ax.set_yscale('log')
plt.legend(loc='upper left')
plt.xlabel("x")
plt.ylabel("Value")
plt.show()