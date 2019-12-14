#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:08:34 2019

@author: hys1435
"""

import numpy as np
from scipy.special import gegenbauer, legendre, comb
from kpm import select_D, kpca
from helper_fun import compute_gram_mat, init_sim_data_2_sphere
from kernels import gaussianRBF, arccos
import matplotlib.pyplot as plt

def generate_coeffs(n):
	return np.random.uniform(size = n)

# explicit kernel using eigenfunction, phi is a function with additional params
def kernel_of_phi(phi):
    def ker(x, y, params):
        return np.dot(phi(x), phi(y))
    return ker

def eigfun_n(x, n):
    legendre_n = legendre(n)
    costheta = x[2]
    return legendre_n(costheta) * np.sqrt((2*n+1)/2) # Remove sinphi for case m=0

def gen_f_star(n, coeffs):
    def f_star(x):
        y = 0
        for i in range(1,n+1):
            y = y + coeffs[i-1] * eigfun_n(x, i)
        return y
    return f_star


# plot eigenfunctions
N = 200
k = 3
coeffs = generate_coeffs(k)
coeffs = coeffs / np.sum(coeffs) # sum to be 1
print("True coefficients: ", coeffs)

# Simulate data on a sphere
phi = np.random.random([int(N/2)]) * 2 * np.pi
phi = np.concatenate((phi, -phi), axis = 0)
costheta = 2 * (np.random.random([int(N/2)]) - 1/2)
costheta = np.concatenate((costheta, -costheta), axis = 0)
theta = np.arccos(costheta)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
X = np.array([x,y,z]).transpose()

#--------------------------------------- PRE-TESTING --------------------------------------
# Check if X is centered
print("column mean of X: ", np.mean(X, axis = 0))

# First check if the normalized legendres are orthonormal
n_points = 500
X_seq = np.linspace(-1, 1, num = n_points)
n = 1
legendre_n = legendre(n)
eigval_seq_1 = legendre_n(X_seq) * np.sqrt((2*n+1)/2)
print("inner product with same function: ", 2/n_points*np.dot(eigval_seq_1, eigval_seq_1))
n = 2
legendre_n = legendre(n)
eigval_seq_2 = legendre_n(X_seq)
print("inner product with diff function: ", 2/n_points*np.dot(eigval_seq_1, eigval_seq_2))

# Test if eigfun_n is working
XX = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
val = eigfun_n(XX, 1)

print("test value: ", np.dot(val, val)) # should be 1

#------------------------------------ TEST COMPLETE ---------------------------------------

#epsilon = np.random.normal(scale=1/5, size = N)
f_star = gen_f_star(k, coeffs)
params = None
epsilon = 0 # No noise
y = np.zeros([N])
for i in range(N):
    #print("Xi: ", X[i])
    y[i] = f_star(X[i]) + epsilon
print("shape of Y: ", y.shape)

ker_star = kernel_of_phi(f_star)

K = compute_gram_mat(X, X, ker_star, params)
print("Gram matrix: ", K)
print("Shape of gram matrix: ", K.shape)
D_max = 5
C = 0.01
D_opt, mse = select_D(K, y, D_max, C)
print("D_opt: ", D_opt)
Alpha, Lam = kpca(K, D_opt)

Phi = K @ Alpha / np.sqrt(N)
print("Shape of Phi: ", Phi.shape)
print("Shape of y: ", y.shape)
coeffs = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ y)
print("predicted coeffs: ", coeffs)


"""
# The nth eigenfunction, which is the nth normalized gegenbauer polynomial 
# described in the short version - also not working since geg uses the inner product
def eigfun_n(x, n):
    d = 3
    alpha = (d-2)/2
    h_n_lambda = (alpha / (n+alpha))**2 * (comb(n+d-1, n) - comb(n+d-3, n-2))
    geg = gegenbauer(n, alpha)
    return geg(x) / h_n_lambda

def f_star(x, n, coeffs): 
    y = 0
    for i in range(1,n+1):
        y = y + coeffs[i-1] * eigfun_n(x, i)
    return y

# Polynomial functions - not working since not orthogonal
def eigfun_n(x, k): 
    return 1/factorial(k) * np.linalg.norm(x)**k

# The optimal function used for Sobolev kernel. 
def f_star(x):
    return np.minimum(x, 1-x)

# The init data function associated with the Gegenbauer polynomials. 
def init_sim_data(N, k, coeffs):
    # Initialize data given N number of samples
    ndim = 2
    X = init_sim_data_2_sphere(N, ndim = ndim)
    mean = np.zeros(ndim+1)
    cov = 1/5 * np.eye(ndim+1)
    epsilon = np.random.multivariate_normal(mean, cov, size = N)
    y = f_star(X, k, coeffs) + epsilon
    return X, y


def eigfun_n(phi, costheta, n):
    legendre_n = legendre(n)
    def eigfun(phi, costheta, n):
        return np.sin(phi) * legendre_n(costheta) * np.sqrt((2*n+1)/2)
    return eigfun
"""

"""
#print("Lam: ", Lam)
#X_seq = np.linspace(0, 1, num = 200)
#K_seq = compute_gram_mat(X_seq, X, k3, rho)
#print("Shape of K_seq: ", K_seq.shape)
#print("Shape of Alpha: ", Alpha.shape)
#Phi_seq = K_seq @ Alpha / np.sqrt(N)

fig, ax1 = plt.subplots()
cols = ['red', 'blue', 'purple', 'green', 'yellow']
markers = ['o', '^', 's', 'd', '+', 'x', '*']
for i in range(D_opt):
    ax1.plot(X_seq, Phi_seq[:,i], c=cols[i],label='{}th Eigenfunction'.format(i+1))
plt.legend(loc='upper right')
plt.xlabel("X")
plt.ylabel("Value")
plt.savefig("kpm_eigfunction")
"""