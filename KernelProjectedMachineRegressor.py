import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from kpm import gaussianRBF, compute_gram_mat, compute_coeff_fixD, pen_D, kpca


class KPMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, C=0.5, D_max=50, kernel="gaussian", sigma=20, other_params=None):
        # C is the penalization constant to the dimension, D_max is the maximum dimension we want to test, 
        self.C = C
        self.D_max = D_max
        self.kernel = kernel
        self.sigma = sigma
        self.other_params = other_params

    def _get_kernel_fun(self, kernel):
        if (kernel == "gaussian"):
            return gaussianRBF


    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Compute the coefficients
        self.kernel_fun = self._get_kernel_fun(self.kernel)
        K = compute_gram_mat(X, X, self.kernel_fun, self.sigma)
        N = K.shape[0]
        total = np.zeros(self.D_max-1)
        for D in range(1, self.D_max):
            coeffs, Phi = compute_coeff_fixD(K, y, D)
            x_pred = Phi @ coeffs
            mse = np.mean((y - x_pred)**2)
            total[D-1] = mse + pen_D(N, D, self.C)
        tot_min = np.amin(total)
        D_opt_lst = np.where(total == tot_min)
        if (np.isnan(D_opt_lst)):
            D_opt = 0
        else:
            D_opt = np.where(total == tot_min)[0][0]
        mse_opt = tot_min - pen_D(N, D_opt, self.C)
        coeffs_opt, _ = compute_coeff_fixD(K, y, D_opt)
        Alpha, Lam = kpca(K, D_opt)
        self.D_opt = D_opt
        self.mse_opt = mse_opt
        self.coeffs_opt = coeffs_opt
        self.X_fit_ = X
        self.Alpha = Alpha
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["D_opt", "mse_opt", "coeffs_opt", "Alpha"])
        # Input validation
        X = check_array(X)
        train_samples = self.X_fit_.shape[0]
        K_test = compute_gram_mat(X, self.X_fit_, self.kernel_fun, self.sigma)
        Phi_test = K_test @ self.Alpha / np.sqrt(train_samples)
        y_pred = Phi_test @ self.coeffs_opt
        self.y_pred = y_pred
        return self.y_pred
