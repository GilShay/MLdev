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
from  sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print lasso_reg.predict([[1.5]])

##Elastic Net
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X,y)
print elastic_net.predict([[1.5]])

##Early Stopping
# from sklearn.base import clone
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant",eta0=0.0005)
#
# minimum_val_error = float("inf")
# best_epoch = None
# best_model = None
# for  epoch in range(1000):
#     sgd_reg.fit(X_train_poly_scaled, y_train)
