#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:42:28 2019

@author: hys1435
"""
import numpy as np
from kpm import kpca
from helper_fun import compute_gram_mat, init_sim_data_2_sphere
from kernels import arccos

import matplotlib
import matplotlib.pyplot as plt

def main():
    # ------------ Plot the eigenfunctions of the arccos kernel ---------------
    # Experiments
    N = 400
    X = init_sim_data_2_sphere(N)
    K = compute_gram_mat(X, X, arccos)
    D = 3 # take the first three dimension
    Alpha, Lam = kpca(K, D)
    # Create a sphere
    npoints = 50
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:50j, 0:2 * pi:50j]
    
    # convert to 2d matrices
    XX = r * sin(phi) * cos(theta) # 50x50
    YY = r * sin(phi) * sin(theta) # 50x50
    ZZ = r * cos(phi) # 50x50
    
    X_seq = np.zeros([npoints,npoints, 3])
    W = np.zeros([npoints, npoints])
    for i in range(npoints):
        for j in range(npoints):
            X_seq[i,j] = np.array([XX[i,j],YY[i,j],ZZ[i,j]])
    X_seq = X_seq.reshape([npoints*npoints, 3])
    K_seq = compute_gram_mat(X_seq, X, arccos)
    for i in range(D):
        Phi_seq = K_seq @ Alpha[:,i] / np.sqrt(N)
        W = Phi_seq.reshape([npoints, npoints])
        
        # fourth dimention - colormap
        # create colormap according to x-value (can use any 50x50 array)
        color_dimension = W # change to desired fourth dimension
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)
        
        # plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(XX,YY,ZZ, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # Add a color bar which maps values to colors.
        
        fig.colorbar(m, shrink=0.5, aspect=5)
        plt.show()
        plt.savefig("{}th eigfunction of arccos kernel".format(i+1))

if __name__ == '__main__':
     main()
