import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print X[0]
print X_poly[0]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print lin_reg.intercept_
print lin_reg.coef_

X_new =  np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new =lin_reg.predict(X_new_poly)


plt.plot(X_new, y_new, '-r')
plt.plot(X ,y, 'o')
plt.show()

