
import numpy as np

def gaussianRBF(u, v, sigma):
    return np.exp(-np.einsum('ij,ij->i',u-v,u-v)/(2*sigma**2))

def arccos(u, v, params):
    theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta = np.minimum(theta, 1) # remove error due to computation issue
    return 1 - 1/np.pi * np.arccos(theta)

def k1(x, y, lambda_0 = 1):
	xy = np.minimum(np.dot(x,y), 1)
	return (1+lambda_0)/(4*np.pi) - 1/(2*np.pi)*np.log(1+np.sqrt((1-xy)/2))

def k2(x, y, lambda_0 = 1):
	xy = np.minimum(np.dot(x,y), 1)
	return (1+3*lambda_0)/(12*np.pi) - 1/(8*np.pi)*np.sqrt((1-xy)/2)

def k3(x, y, rho):
	xy = np.minimum(np.dot(x,y), 1)
	return 1/np.sqrt(1-2*rho*xy+rho**2)

def k4(x, y, kappa):
	xy = np.minimum(np.dot(x,y), 1)
	return kappa * np.exp(kappa * xy) / (4*np.pi*np.sinh(kappa))