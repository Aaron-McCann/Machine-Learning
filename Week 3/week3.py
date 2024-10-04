#i 
#(a)

import pandas as p
import numpy as n
import matplotlib.pyplot as m
from mpl_toolkits.mplot3d import Axes3D


df = p.read_csv('Week 3\week3.csv')
print(df.head())

df.columns = ['1', '2' , 'y']
x1 = df['1'].values
x2 = df['2'].values
y = df['y'].values

fig = m.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y)
m.show()

# looks like it lies on a curve icl

# (ii)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso


features = PolynomialFeatures(degree = 5)
x_features = n.column_stack((x1,x2))
polynomial = features.fit_transform(x_features)
print(polynomial.shape)
print(polynomial[:5, :])

C = [1, 10, 100, 1000]
param = {}
for c in C:
    model = Lasso(alpha = 1/c, max_iter= 100) #why this alpha?
    model.fit(polynomial,y)
    param[c] = model.coef_

print(param)

#(c)

grid = n.linspace(-5, 5, 100)
X_test = []
for i in grid:
    for j in grid:
        X_test.append([i, j])
X_test = n.array(X_test)

#continuing from above 
polynomial_X_test = features.transform(X_test)
pred = {}
for c, params in param.items():
    pred[c] = n.dot(polynomial_X_test, params)

# Plot the predictions and training data
fig = m.figure()
ax = fig.add_subplot(111, projection='3d')

for c, pred in pred.items():
    ax.plot_surface(X_test[:, 0].reshape(-1, 100), X_test[:, 1].reshape(-1, 100), pred.reshape(-1, 100), alpha=0.5, label=f"C={c}")

ax.scatter(x1, x2, y, c='k', label='Training data')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target variable')
ax.legend()

m.show()