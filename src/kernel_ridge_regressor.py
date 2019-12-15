#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:48:30 2019

@author: Zhaoqi Li
"""

# Kernel Ridge Regression Algorithm reproducing results from zhang15d paper

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from kernels import gaussianRBF, compute_gram_mat

class KRRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel="gaussian", sigma=20, lam=1, other_params=None):
        # C is the penalization constant to the dimension, D_max is the maximum dimension we want to test, 
        self.kernel = kernel
        self.sigma = sigma
        self.lam = lam
        self.other_params = other_params
    
    def _get_kernel_fun(self, kernel):
        if (kernel == "gaussian"):
            return gaussianRBF

    def fit(self, X, y):
        # Key algorithm to compute the kernel ridge coefficients alpha given the gram matrix K,  
        # formula listed as equation (37) of the original paper
        self.kernel_fun = self._get_kernel_fun(self.kernel)
        K = compute_gram_mat(X, X, self.kernel_fun, self.sigma)
        self.alpha = np.linalg.solve(K + self.lam * y.size * np.eye(K.shape[0]), y)
        self.X_fit_ = X
        return self
        
    def predict(self, X):
        # compute the prediction of y using kernel coefficients alpha
        check_is_fitted(self, ["alpha"])
        # Input validation
        X = check_array(X)
        train_samples = self.X_fit_.shape[0]
        K_test = compute_gram_mat(X, self.X_fit_, self.kernel_fun, self.sigma)
        y_pred = np.dot(K_test, self.alpha)
        return y_pred
