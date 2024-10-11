#i 
#(a)

import pandas as p
import numpy as n
import matplotlib.pyplot as m
from mpl_toolkits.mplot3d import Axes3D


df = p.read_csv('Machine-Learning/Week 3/week3.csv')
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
    model = Lasso(alpha = 1/2*c, max_iter= 100) #why this alpha?
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


#(e)
from sklearn.linear_model import Ridge


def part_e (polynomial, y, X_test, features):
    # same as for Lasso 
    C = [1, 10, 100, 1000]
    param = {}
    for c in C:
        model = Ridge(alpha = c, max_iter = 100) # why different alpha
        model.fit(polynomial, y )
        param[c] = model.coef_ 


    pred_ridge = {}
    for c, params in param.items():
        pred_ridge[c] = n.dot(features.transform(X_test), params)


    fig = m.figure()
    ax = fig.add_subplot(111, projection='3d')

    for c, p in pred_ridge.items():
        ax.plot_surface(X_test[:, 0].reshape(-1, 100), X_test[:, 1].reshape(-1, 100), p.reshape(-1, 100), alpha=0.5, label=f"C={c}")

    ax.scatter(x1, x2, y, c='k', label='Training data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target variable')
    ax.legend()

    m.show()

    return param



param_ridge = part_e(polynomial, y, X_test, features)


import matplotlib.pyplot as plt
import numpy as np


def compare_params(param_ridge, param_lasso):
    """
    Compare the param dictionaries for Ridge and Lasso regression.

    Parameters:
    param_ridge (dict): Param dictionary for Ridge regression
    param_lasso (dict): Param dictionary for Lasso regression

    Returns:
    None
    """
    print("Comparing param dictionaries:")
    print("-------------------------------")

    # Check if the dictionaries have the same keys
    if set(param_ridge.keys()) != set(param_lasso.keys()):
        print("Error: dictionaries have different keys")
        return

    # Plot the coefficients
    for key in param_ridge.keys():
        ridge_value = param_ridge[key]
        lasso_value = param_lasso[key]
        plt.plot(ridge_value, label=f"Ridge C={key}")
        plt.plot(lasso_value, label=f"Lasso C={key}")
        plt.legend()
        plt.show()

    # Calculate the sparsity
    for key in param_lasso.keys():
        lasso_value = param_lasso[key]
        sparsity = np.mean(lasso_value == 0)
        print(f"Lasso C={key}: Sparsity={sparsity:.2f}")

    # Calculate the magnitude of coefficients
    for key in param_ridge.keys():
        ridge_value = param_ridge[key]
        lasso_value = param_lasso[key]
        ridge_magnitude = np.mean(np.abs(ridge_value))
        lasso_magnitude = np.mean(np.abs(lasso_value))
        print(f"Ridge C={key}: Magnitude={ridge_magnitude:.2f}")
        print(f"Lasso C={key}: Magnitude={lasso_magnitude:.2f}")

    # Compare the feature importance
    for key in param_ridge.keys():
        ridge_value = param_ridge[key]
        lasso_value = param_lasso[key]
        ridge_importance = np.argsort(np.abs(ridge_value))[::-1]
        lasso_importance = np.argsort(np.abs(lasso_value))[::-1]
        print(f"Ridge C={key}: Feature importance={ridge_importance}")
        print(f"Lasso C={key}: Feature importance={lasso_importance}")

    print("-------------------------------")



compare_params(param_ridge, param)



#(ii)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def ii_a (X, y, model, C, cv = 5):
    """
    Plot the mean and standard deviation of the prediction error vs C.

    Parameters:
    X (array): Feature matrix
    y (array): Target variable
    model (object): Lasso or Ridge regression model
    Cs (array): Array of values for C
    cv (int): Number of folds for cross-validation

    Returns:
    None
    """
    errors = []
    for c in C:
        model.set_params(alpha=C)
        scores = cross_val_score (model, X, y, cv = cv, scoring = 'neg_mean_squares_error') # why this loss function. Explain in report 
        errors.append (-scores.mean())
        plt.errorbar(c, errors, yerr=np.std(errors), fmt='o-')
        plt.xlabel('C')
        plt.ylabel('Mean Squared Error')
        plt.title('Error vs C')
        plt.show()

def ii_b (X, y, model, C, cv=5 ):
    """
    Select the best value of C based on cross-validation.

    Parameters:
    X (array): Feature matrix
    y (array): Target variable
    model (object): Lasso or Ridge regression model
    Cs (array): Array of values for C
    cv (int): Number of folds for cross-validation

    Returns:
    C (float): Best value of C
    """
    errors = []
    for c in C:
        model.set_params (alpha = C)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        errors.append(-scores.mean())
        C_best = Cs[np.argmin(errors)]
        return C_best



    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    # Define models
    lasso_model = Lasso(max_iter=100)
    ridge_model = Ridge(max_iter=100)

    # Define range of values for C
    Cs = [0.01, 0.1, 1, 10, 100]

    # Plot error vs C for Lasso
    ii_a(X_poly, y, lasso_model, Cs)

    # Select best value of C for Lasso
    C_lasso_best = ii_b(X_poly, y, lasso_model, Cs)
    print(f"Best value of C for Lasso: {C_lasso_best}")

    # Plot error vs C for Ridge
    ii_a(X_poly, y, ridge_model, Cs)

    # Select best value of C for Ridge
    C_ridge_best = ii_b(X_poly, y, ridge_model, Cs)
    print(f"Best value of C for Ridge: {C_ridge_best}")
