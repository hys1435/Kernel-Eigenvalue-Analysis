#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for simulation study of KRR, KPM and linear regression algorithms. 

@author: Zhaoqi Li
"""

import numpy as np
from kpm_regressor import KPMRegressor
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

def f_star(x):
    return np.minimum(x, 1-x)

def init_sim_data(N):
    # Initialize data given N number of samples
    X = np.random.uniform(size = N)
    epsilon = np.random.normal(scale = 1/np.sqrt(5), size = N)
    y = f_star(X) + epsilon
    return X, y

def main():
    #np.random.seed(123)
    start_time = time.time()
    N_lst = np.array([50, 100, 150, 200, 250])
    D_max = 4
    sim_num = 10
    mse_kpm_lst = np.zeros((N_lst.size, sim_num))
    mse_lr_lst = np.zeros((N_lst.size, sim_num))
    mse_krr_lst = np.zeros((N_lst.size, sim_num))
    for i, N in enumerate(N_lst):
        for k in range(sim_num): 
            X, y = init_sim_data(N)
            clf3 = KPMRegressor(D_max = D_max, sigma=0.2)
            parameters = {'C':[0.02, 0.04, 0.06], 'sigma':[0.05, 0.1, 0.15, 0.2]}
            cv3 = GridSearchCV(clf3, parameters, cv=5, error_score = 'raise')
            cv3.fit(X.reshape(-1, 1), y)
            y_pred = cv3.predict(X.reshape(-1, 1))
            mse_kpm_lst[i,k] = np.mean((y - y_pred)**2)
    		    
            clf = LinearRegression()
            clf.fit(X.reshape(-1, 1), y)
            y_pred = clf.predict(X.reshape(-1, 1))
            mse_lr_lst[i,k] = np.mean((y - y_pred)**2)
            sigma = cv3.best_params_['sigma']
            clf2 = KernelRidge(kernel = 'rbf', gamma = 1/(2*sigma**2))
            clf2.fit(X.reshape(-1, 1), y)
            y_pred = clf2.predict(X.reshape(-1, 1))
            mse_krr_lst[i,k] = np.mean((y - y_pred)**2)
        print("------ loop finished ------ time: ", (time.time() - start_time))

    mse_kpm_err = np.std(mse_kpm_lst, axis = 1)
    mse_kpm = np.mean(mse_kpm_lst, axis = 1)
    mse_lr_err = np.std(mse_lr_lst, axis = 1)
    mse_lr = np.mean(mse_lr_lst, axis = 1)
    mse_krr_err = np.std(mse_krr_lst, axis = 1)
    mse_krr = np.mean(mse_krr_lst, axis = 1)
    

    print("kpm err lst: ", mse_kpm_lst)
    print("ls err lst: ", mse_lr_lst)
    print("krr err lst: ", mse_krr_lst)
	
	# Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'purple']
    markers = ['o', 's', '^']
    ax.errorbar(N_lst, mse_kpm, yerr = mse_kpm_err, c=cols[0], marker=markers[0],label='Kernel Projection Machine',capsize=5)
    ax.errorbar(N_lst, mse_lr, yerr = mse_lr_err, c=cols[1], marker=markers[1],label='Linear Regression',capsize=5)
    ax.errorbar(N_lst, mse_krr, yerr = mse_krr_err, c=cols[2], marker=markers[2],label='Kernel Ridge Regression',capsize=5)
    plt.legend(loc='upper right')
    plt.xlabel("N")
    plt.ylabel("Mean square error")
    plt.savefig("kpm_lr_krr_N")

if __name__ == '__main__':
     main()
