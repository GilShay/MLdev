from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)

