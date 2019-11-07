"""
=====================================================
MNIST classfification using multinomial logistic + L1
=====================================================

Here we fit a multinomial logistic regression with L1 penalty on a subset of
the MNIST digits classification task. We use the SAGA algorithm for this
purpose: this a solver that is fast when the number of samples is significantly
larger than the number of features and is able to finely optimize non-smooth
objective functions which is the case with the l1-penalty. Test accuracy
reaches > 0.8, while weight vectors remains *sparse* and therefore more easily
*interpretable*.

Note that this accuracy of this l1-penalized linear model is significantly
below what can be reached by an l2-penalized linear model or a non-linear
multi-layer perceptron model on this dataset.

"""
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from KRR_algorithm_copy import KRRRegressor
from sklearn.kernel_ridge import KernelRidge
from KernelProjectedMachineRegressor import KPMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator


print(__doc__)

# Author: Arthur Mensch <arthur.mensch@m4x.org>
# License: BSD 3 clause

# Turn down for faster convergence
t0 = time.time()
train_samples = 200
test_samples=1000

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

y = y.astype(dtype=int)
#print("data: ", X[0:10])
#print("label: ", y[0:10])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=test_samples)

#print("shape of X: ", X_train.shape)
#print("number of nonzero elements in X: ", np.count_nonzero(X_train))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#print("shape of X: ", X_train.shape)
#print("number of nonzero elements in X: ", np.count_nonzero(X_train))

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / train_samples, solver='saga', tol = 0.01) # default l2 penalty
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
#print("y_pred: ", y_pred)
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
parameters = {'C':[0.2, 0.5], 'sigma':[10, 20]}
cv = GridSearchCV(clf3, parameters, cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print("y_pred: ", y_pred)
mse_kpmreg = np.mean((y_test - y_pred)**2)
score = cv.score(X_test, y_test)

print("Test score logistic regression with L2 penalty: %.4f" % score)
print("Mean-squared error for logistic regression: %.4f" % mse_lr)
print("Mean-squared error for kernel ridge regression self: %.4f" % mse_krr_self)
print("Mean-squared error for kernel projection machine regressor: %.4f" % mse_kpmreg)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()
