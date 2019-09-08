
import numpy as np

def gaussianRBF(u, v, sigma):
    return np.exp(-np.einsum('ij,ij->i',u-v,u-v)/(2*sigma**2))

def arccos(u, v, params):
    theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta = np.minimum(theta, 1) # remove error due to computation issue
    return 1 - 1/np.pi * np.arccos(theta)