import numpy as np

def power(A, tol = 1e-4, max_iter = 500):
    v_old = np.random.rand(A.shape[1])
    ct = 0
    v_new = np.ones(v_old.shape)
    while (ct < max_iter and np.linalg.norm(v_old-v_new) > tol):
        z_k = np.dot(A, v_old)
        v_old = np.copy(v_new)
        v_new = z_k / np.linalg.norm(z_k)
        lambda_k = np.dot(v_new.transpose(), np.dot(A, v_new))
        ct += 1
    return (v_new, lambda_k)

def compute_gram_mat(X1, X2, kernel, params = None):
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, itemi in enumerate(X1):
        for j, itemj in enumerate(X2):
            gram_mat[i,j] = kernel(itemi, itemj, params)
    return gram_mat

def gaussianRBF(u, v, sigma):
    return np.exp(-np.linalg.norm(u-v)**2/(2*sigma**2))

def kpca(K, k):
    m = K.shape[0]
    Alpha = np.zeros((m,k))
    Lam = np.zeros(k)
    v, lam = power(K)
    Alpha[:,0] = v/np.sqrt(lam)
    Lam[0] = lam
    for i in range(1,k):
        K = K - lam * np.outer(v, v) # substract lam_1^2 * v * v^T
        v, lam = power(K)
        Alpha[:,i] = v/np.sqrt(lam) # alpha is the normalized eigenvector
        #print("norm: ", np.linalg.norm(Alpha[:,i]))
        # Alpha[:,i] = v/lam # compensate for 1/sqrt(lam) in equation 18 - probably wrong
        Lam[i] = lam
    return (Alpha, Lam)

def compute_coeff_fixD(K, y, D):
    N = K.shape[0]
    Alpha, Lam = kpca(K, D)
    #print("Lam: ", Lam)
    Phi = K @ Alpha / np.sqrt(N) # This should be N-by-D matrix
    #print("Phi: ", Phi)
    coeffs = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ y)
    return (coeffs, Phi)

def pen_D(N, D, C):
    return C * D * np.log(N) / N

def select_D(K, y, D_max, C):
    N = K.shape[0]
    total = np.zeros(D_max-1)
    for D in range(1, D_max):
        coeffs, Phi = compute_coeff_fixD(K, y, D)
        x_pred = Phi @ coeffs
        mse = np.mean((y - x_pred)**2)
        total[D-1] = mse + pen_D(N, D, C)
    tot_min = np.amin(total)
    print("total: ", total)
    D_opt = np.where(total == tot_min)[0][0]
    mse_opt = tot_min - pen_D(N, D_opt, C)
    #print("D_opt: ", D_opt)
    #coeffs_opt = compute_coeff_fixD(K, D_opt, params, dist_metric)
    return (D_opt, mse_opt)





