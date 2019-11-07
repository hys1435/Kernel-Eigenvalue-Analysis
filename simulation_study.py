import numpy as np
from kpm import select_D, kpca
from kernels import gaussianRBF
from helper_fun import compute_gram_mat
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
    start_time = time.time()
    N = 500
    X, y = init_sim_data(N)
    C_lst = np.linspace(0, 0.05, num = 10)
    sigma_lst = np.linspace(0.05, 0.2, num = 5)
    D_max = 5
    sim_num = 5
    mse_kpm_lst = np.zeros((5, 10, sim_num))
    D_opt_lst = np.zeros((5, 10))
    mse_lr_lst = np.zeros((5, 10, sim_num))
    mse_krr_lst = np.zeros((5, 10, sim_num))
    for k in range(sim_num): 
        for i, sigma in enumerate(sigma_lst):
            K = compute_gram_mat(X, X, gaussianRBF, sigma)
            for j, C in enumerate(C_lst):
                D_opt_lst[i,j], mse_kpm_lst[i,j,k] = select_D(K, y, D_max, C)
    		    #print("error is: ", mse_lst)
    		    #print("run time is: ", (time.time() - start_time))
                clf = LinearRegression()
                clf.fit(X.reshape(-1, 1), y)
                y_pred = clf.predict(X.reshape(-1, 1))
                mse_lr_lst[i,j,k] = np.mean((y - y_pred)**2)
    		    #print("linear regression error: ", err_ls)
    		    
                clf2 = KernelRidge(kernel = 'rbf', gamma = 1/(2*sigma**2))
                clf2.fit(X.reshape(-1, 1), y)
                y_pred = clf2.predict(X.reshape(-1, 1))
                mse_krr_lst[i,j,k] = np.mean((y - y_pred)**2)
    		    #print("kernel ridge regression error: ", err_krr)
        print("------ loop finished ------ time: ", (time.time() - start_time))

    mse_kpm_err = np.std(mse_kpm_lst, axis = 2)
    mse_kpm_lst = np.mean(mse_kpm_lst, axis = 2)
    mse_lr_err = np.std(mse_lr_lst, axis = 2)
    mse_lr_lst = np.mean(mse_lr_lst, axis = 2)
    mse_krr_err = np.std(mse_krr_lst, axis = 2)
    mse_krr_lst = np.mean(mse_krr_lst, axis = 2)
    
    mse_kpm = np.amin(mse_kpm_lst, axis = 1)
    mse_lr = np.amin(mse_lr_lst, axis = 1)
    mse_krr = np.amin(mse_krr_lst, axis = 1)
    mse_kpm_err_min = np.amin(mse_kpm_err, axis = 1)
    mse_lr_err_min = np.amin(mse_lr_err, axis = 1)
    mse_krr_err_min = np.amin(mse_krr_err, axis = 1)

    print("kpm err lst: ", mse_kpm_lst)
    print("D_opt lst: ", D_opt_lst)
    print("ls err lst: ", mse_lr_lst)
    print("krr err lst: ", mse_krr_lst)
	
	# Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'purple']
    markers = ['o', 's', '^']
    ax.errorbar(sigma_lst, mse_kpm, yerr = mse_kpm_err_min, c=cols[0], marker=markers[0],label='Kernel Projection Machine',capsize=5)
    ax.errorbar(sigma_lst, mse_lr, yerr = mse_lr_err_min, c=cols[1], marker=markers[1],label='Linear Regression',capsize=5)
    ax.errorbar(sigma_lst, mse_krr, yerr = mse_krr_err_min, c=cols[2], marker=markers[2],label='Kernel Ridge Regression',capsize=5)
    plt.legend(loc='upper right')
    plt.xlabel("sigma")
    plt.ylabel("Mean square error")
    plt.savefig("kpm_lr_krr")

if __name__ == '__main__':
     main()
