import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# print X
#print y
# plt.plot(X, y, 'o')
# plt.show()

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
# print X_b
# print X_b.T
# print X_b.T[1].sum()
# print X_b.T[0].sum()

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print X_b.T.dot(X_b)
# print np.linalg.inv(X_b.T.dot(X_b))
print theta_best

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print y_predict
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print lin_reg.intercept_,  lin_reg.coef_
print lin_reg.predict(X_new)

