import numpy as np
from random import shuffle

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

def compute_gram_mat(X1, X2, kernel, params = None):
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, itemi in enumerate(X1):
        for j, itemj in enumerate(X2):
            #print(itemi)
            #print(itemj)
            gram_mat[i,j] = kernel(itemi, itemj, params)
    return gram_mat

# Simulation function

def eigen_n_sphere(N, ndim, kernel, params = None): 
    # eigenvalue analysis 
    X, y = init_sim_data_sphere(N, ndim = ndim)
    ind = [x for x in range(N)]
    shuffle(ind)
    np.take(X, ind, axis = 0, out = X)
    np.take(y, ind, axis = 0, out = y)
    K = compute_gram_mat(X, X, kernel, params)
    #print("matrix: ", 1/N * K)
    w, v = np.linalg.eig(1/N * K)
    #print("value: ", w)
    w = w.real # ignore complex parts (CAUTION: only valid if it's small)
    w = -np.sort(-w)
    d = dict()
    d['data'] = X
    d['label'] = y
    d['gram_mat'] = K
    d['eig_val'] = w
    return d