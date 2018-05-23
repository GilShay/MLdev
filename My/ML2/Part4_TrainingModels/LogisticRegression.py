from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

##Decision Boundaries
iris = datasets.load_iris()
print list(iris.keys())
print iris['feature_names'][3]
X = iris['data'][:, 3:]
y = (iris['target']==2).astype(np.int)

log_reg = LogisticRegression()
log_reg.fit(X,y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], 'g-', label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], 'b--', label="Not Iris-Virginica")
plt.show()

print log_reg.predict([[1.7], [1.5]])