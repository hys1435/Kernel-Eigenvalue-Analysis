#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:33:24 2019

@author: hys1435
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from random import shuffle
import matplotlib.pyplot as plt

#%%

def init_sim_data_sphere(npoints, ndim=2):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    X = vec.transpose()
    y = np.prod(np.sign(vec),axis=0) # some random prediction function
    return X, y

def split_into_m_parts(X, m):
    # split the entire dataset into subsets
    n = int(X.shape[0] / m)
    resShape = [m, n]
    if len(X.shape) > 1:
        for item in X.shape[1:]:
            resShape.append(item)
    res = np.zeros(resShape)
    for i in range(m):
        res[i] = X[i*n:(i+1)*n]
    return res

def gaussianRBF(u, v, sigma):
    return np.exp(-np.einsum('ij,ij->i',u-v,u-v)/(2*sigma**2))

def arccos(u, v, params):
    theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta = np.minimum(theta, 1) # remove error due to computation issue
    return 1 - 1/np.pi * np.arccos(theta)

def compute_gram_mat(X1, X2, params):
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, itemi in enumerate(X1):
        for j, itemj in enumerate(X2):
            gram_mat[i,j] = arccos(itemi, itemj, params)
    return gram_mat

#%% 
# Dimension = 3
    
N = 1000
lam = N**(-2/3)
sigma = -1
X, y = init_sim_data_sphere(N, ndim = 2)
ind = [x for x in range(N)]
shuffle(ind)
np.take(X, ind, axis = 0, out = X)
np.take(y, ind, axis = 0, out = y)

#X_train_split = split_into_m_parts(X, m) # shape of m, n, d
#y_train_split = split_into_m_parts(y, m)

num = 20
y_pred_lst = np.zeros((N, num))
K = compute_gram_mat(X, X, sigma)

# eigenvalue analysis 

w, v = np.linalg.eig(1/N * K)
w = -np.sort(-w)
Nls = np.power(np.linspace(start = 1, stop = N, num = N), -2)

wl = w / Nls
#wl = w / np.exp(-np.linspace(start = 1, stop = N, num = N))
print("eigenvalues: ", w)
#eigbd = N * np.exp(-np.linspace(start = 1, stop = N, num = N))
#np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#print("eigbound: ", eigbd)
#print("diff: ", eigbd - w)
fig, ax1 = plt.subplots()
ax1.plot(w[0:N])
plt.xlabel("Index")
plt.ylabel("Value")
ax1.set_yscale('log')
plt.show()

fig, ax2 = plt.subplots()
ax2.plot(wl)
plt.xlabel("i")
plt.ylabel("Eigenvalue/i^(-2)")
plt.show()

alpha = np.linalg.solve(K, y)
#print(alpha)

#%%

xx = np.linspace(start = 1, stop = 20, num = 50)[::-1]
xxlog = np.log(xx)
fig, ax3 = plt.subplots()
ax3.plot(xxlog)
plt.show()

#%%

# eigenfunctions for dim = 2

theta_seq = np.linspace(start = 0, stop = 2 * np.pi - 1e-4, num = num)
X_seq = np.array([np.cos(theta_seq), np.sin(theta_seq)]).transpose()
K_test = compute_gram_mat(X_seq, X, sigma)
y_pred = K_test @ alpha
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

plt.show()

#%%

# Dimension = 3

