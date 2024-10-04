import numpy as np
import pandas as pd
#df= pd.read_csv("week2.csv" )
df = pd.read_csv('C:/Users/McCannA/Desktop/Machine Learning/Machine-Learning/Week 2/week2.csv')

print( df.head( ) )

df.columns = ('X1', 'X2', 'y')

#plotting
import matplotlib.pyplot

class1 = df[df["y"] == -1]
class2 = df[df["y"] == 1]

# double check that you used the right colurs and markers
matplotlib.pyplot.scatter(class1['X1'], class1['X2'], marker = 'o', color='r', label='-1' , edgecolor = 'k' )
matplotlib.pyplot.scatter(class2['X1'], class2['X2'], marker = '+', color='b', label='+1'  )
#TODO: Include legend
matplotlib.pyplot.xlabel('x_1')
matplotlib.pyplot.ylabel('x_2')
matplotlib.pyplot.title('A(i): Scatter Plot of Features X1 and X2')
matplotlib.pyplot.legend(loc = 'upper right', fontsize =12)
matplotlib.pyplot.show()

#a(ii)

import sklearn.linear_model
from sklearn.linear_model import LogisticRegression

X = np.column_stack((df['X1'], df['X2']))
Y = df['y']

model = LogisticRegression(penalty= None , solver='lbfgs')
model.fit(X,Y)



print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficients (slopes): {model.coef_}")

influence = pd.Series(model.coef_[0] , index = ['X1', 'X2'])
print("Feature influence (absolute value):")
print(influence.abs().sort_values(ascending=False))

#a(iii)
predicted = model.predict(X)

p_class1 = df[predicted == -1]  # Predicted class -1
p_class2 = df[predicted == 1]   # Predicted class +1

w1 = model.coef_[0][0]
w2 = model.coef_[0][1]

b = model.intercept_[0]
#TODO: change this
# Range was going to far 
range = np.linspace(df['X1'].min(), df['X1'].max(), 100 )

boundry = -((w1*range + b) / w2)
#y_values = -(model.coef_[0][0] * range + model.intercept_[0]) / model.coef_[0][1]
matplotlib.pyplot.plot(range, boundry, 'k-', label='Decision Boundary')

matplotlib.pyplot.scatter(class1['X1'], class1['X2'], marker = 'o', color='r', label='Actual -1'  )
matplotlib.pyplot.scatter(class2['X1'], class2['X2'], marker = '+', color='b', label='Actual +1'  )
matplotlib.pyplot.scatter(p_class1['X1'], p_class1['X2'], marker = 's',color='orange', label='Predicted -1')
matplotlib.pyplot.scatter(p_class2['X1'], p_class2['X2'], marker = 'x',color='g', label='Predicted +1')


#TODO: Include legend
matplotlib.pyplot.xlabel('x_1')
matplotlib.pyplot.ylabel('x_2')
matplotlib.pyplot.title('A(iii): Predictions and Descion Boundary')
matplotlib.pyplot.legend(loc= 'upper right')
matplotlib.pyplot.grid(True, linestyle = '--', alpha= 0.7) # aplpha 0.7 makes line slightly visible 
matplotlib.pyplot.show()


#edit to show it working 

#b
from sklearn.svm import LinearSVC

#SVM = LinearSVC()

cval = [0.001, 1, 100]
svms = []
for c in cval:
    SVM = LinearSVC(C=c)
    SVM.fit(X,Y)
    svms.append(SVM)
    print(f"C={c}:")
    print(f"Coefficients:{SVM.coef_}")
    print(f"Intercept = {SVM.intercept_}")




#class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', *, dual='auto', tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)[source]


def desicion_boundry(model, A, Y):
    # - and + 1 are to make sure the grid goes further than the datapoints
    # x1 is x axis and x2 is y axis
    xmin = X[:, 0].min() -1 
    xmax = X[:, 0].max() + 1
    ymin = X[:, 1].min() -1 
    ymax = X[:, 1].max() + 1


    xrange = np.arange(xmin, xmax, 0.01) 
    yrange = np.arange(ymin, ymax, 0.01)

    xx, yy = np.meshgrid(xrange, yrange)

    #flatten
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    labels = labels.reshape(xx.shape)

    matplotlib.pyplot.contourf(xx, yy, labels, alpha=0.8, cmap=  matplotlib.pyplot.cm.coolwarm)
    matplotlib.pyplot.scatter(X[:, 0], X[:, 1], c=Y, edgecolor='k', s=20, cmap=  matplotlib.pyplot.cm.coolwarm)
    matplotlib.pyplot.xlabel('X1')
    matplotlib.pyplot.ylabel('X2')
    #matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()

    
for i, SVM in enumerate(svms):
    desicion_boundry(SVM, X, Y)



#c
#c

c_X1 = X[:, 0]**2
c_X2 = X[:, 1]**2

X_new = np.column_stack((X[:, 0],X[:, 1],c_X1, c_X2 ))

# Add the squared features to the dataframe
df['CX1_squared'] = df['X1'] ** 2
df['CX2_squared'] = df['X2'] ** 2

#feature matrix
c_X = df[['X1', 'X2', 'CX1_squared', 'CX2_squared']].values
Y = df['y'].values

c_model = LogisticRegression(penalty= None , solver='lbfgs')
c_model.fit(c_X,Y)

print(f"Intercept: {c_model.intercept_[0]}")
print(f"Coefficients (slopes): {c_model.coef_}")

c_predicted = c_model.predict(c_X)
#check if this value is already above
X_orig = df[['X1', 'X2']].values #double brackets because we are passing a list

def c_plot(x, y, p):
    class1 = X[Y==-1]
    class2 = X[Y==1]


    p_class1 = X[predicted == -1]
    p_class2 = X[predicted == 1]
        
    matplotlib.pyplot.scatter(class1[:, 0], class1[:, 1], marker = 'o', color='r'  )
    matplotlib.pyplot.scatter(class2[:, 0], class2[:, 1], marker = '+', color='b' )
    matplotlib.pyplot.scatter(p_class1[:, 0], p_class1[:, 1], color='g', label='+1')
    matplotlib.pyplot.scatter(p_class2[:, 0], p_class2[:, 1], color='y', label='+1')
    matplotlib.pyplot.xlabel('X1')
    matplotlib.pyplot.ylabel('X2')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    

c_plot(X_orig, Y, c_predicted)

#(iii)
from sklearn.metrics import accuracy_score
from collections import Counter


# Compare performance against a baseline predictor
# Calculate the most common class in the training data using Counter
most_common_class = Counter(Y).most_common(1)[0][0]
print(f"Most Common Class: {most_common_class}")

# Create baseline predictions: all instances are predicted as the most common class
baseline_predictions = np.full_like(Y, most_common_class)

# Calculate accuracy of the baseline predictor
baseline_accuracy = accuracy_score(Y, baseline_predictions)
print(f"Baseline Accuracy: {baseline_accuracy}")

# Calculate accuracy of the logistic regression model with extended features
model_accuracy = accuracy_score(Y, c_predicted)
print(f"Logistic Regression with Extended Features Accuracy: {model_accuracy}")

# Print comparison
print(f"Improvement over Baseline: {model_accuracy - baseline_accuracy}")