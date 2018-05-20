import numpy as np
from sklearn.linear_model import SGDRegressor
import  matplotlib.pyplot as plt

X = 3 * np.random.rand(100, 1)
y = X + np.random.randn(100, 1)
# plt.plot(X, y, "o")
# plt.show()

##Ridge Regression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print ridge_reg.predict([[1.5]])

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
print sgd_reg.predict([[1.5]])

##Lasso Regression
