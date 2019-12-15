"""
Code for KPM, KRR, logistic regression on MNIST dataset. 
"""

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

from kpm_regressor import KPMRegressor
from kernel_ridge_regressor import KRRRegressor

# Turn down for faster convergence
t0 = time.time()
train_samples = 400
test_samples=2000

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

y = y.astype(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=test_samples)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / train_samples, solver='saga', tol = 0.01) # default l2 penalty
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
mse_lr = np.mean((y_test - y_pred)**2)

# Self-written Kernel Ridge Regression
sigma = 20
lam = 1/train_samples
clf4 = KRRRegressor(kernel = 'gaussian', sigma = 20, lam = lam)
clf4.fit(X_train, y_train)
# score2 = clf2.score(X_test, y_test)
y_pred = clf4.predict(X_test)
mse_krr_self = np.mean((y_test - y_pred)**2)

# Kernel Ridge Regression
#clf2 = KernelRidge(kernel = 'rbf', gamma = 1/(2*sigma**2))
#clf2.fit(X_train, y_train)
# score2 = clf2.score(X_test, y_test)
#y_pred = clf2.predict(X_test)
#mse_krr = np.mean((y_test - y_pred)**2)
#print("Mean-squared error for kernel ridge regression: %.4f" % mse_krr)


# Kernel Projection Machine Regressor
clf3 = KPMRegressor()
#check_estimator(clf3)
parameters = {'C':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'sigma':[20, 30, 40, 50, 100]}
cv = GridSearchCV(clf3, parameters, cv=5)
cv.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(cv.best_params_)
print()
y_pred = cv.predict(X_test)
print("y_pred: ", y_pred)
mse_kpmreg = np.mean((y_test - y_pred)**2)
score = cv.score(X_test, y_test)

print("Test score logistic regression with L2 penalty: %.4f" % score)
print("Mean-squared error for logistic regression: %.4f" % mse_lr)
print("Mean-squared error for kernel ridge regression self: %.4f" % mse_krr_self)
print("Mean-squared error for kernel projection machine regressor: %.4f" % mse_kpmreg)

kpm_best = KPMRegressor(C=cv.best_params_['C'], sigma = cv.best_params_['sigma'])
kpm_best.fit(X_train, y_train)
y_pred = cv.predict(X_test)
mse_kpm = np.mean((y_test - y_pred)**2)
print("Mean-squared error for kernel projection machine: %.4f" % mse_kpm)

coef = kpm_best.Phi_opt.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(kpm_best.D_opt):
    l1_plot = plt.subplot(10, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for MNIST Logistic Regression')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()

coef = cv.Phi_opt_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(clf3.D_opt):
    l1_plot = plt.subplot(10, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for MNIST KPM')
plt.show()
