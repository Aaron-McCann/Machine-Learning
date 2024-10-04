import numpy as np
import pandas as pd
df= pd.read_csv("C:\\Users\\New\\Desktop\\Year 5\\Machine Learning\\Week 2\\week2.csv" )
#df = pd.read_csv('C:/Users/McCannA/Desktop/Machine Learning/Machine-Learning/Week 2/week2.csv')

print( df.head( ) )

df.columns = ('X1', 'X2', 'y')

#plotting
import matplotlib.pyplot

class1 = df[df["y"] == -1]
class2 = df[df["y"] == 1]

# double check that you used the right colurs and markers
matplotlib.pyplot.scatter(class1['X1'], class1['X2'], marker = 'o', color='r', label='+1'  )
matplotlib.pyplot.scatter(class2['X1'], class2['X2'], marker = '+', color='b', label='+1'  )
#TODO: Include legend
matplotlib.pyplot.xlabel('x_1')
matplotlib.pyplot.ylabel('x_2')
matplotlib.pyplot.title('a(i)')
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
range = np.linspace(-10, 10, 100)
boundry = -((w1*range + b) / w2)
#y_values = -(model.coef_[0][0] * range + model.intercept_[0]) / model.coef_[0][1]
matplotlib.pyplot.plot(range, boundry, 'k-', label='Decision Boundary')

matplotlib.pyplot.scatter(class1['X1'], class1['X2'], marker = 'o', color='r', label='+1'  )
matplotlib.pyplot.scatter(class2['X1'], class2['X2'], marker = '+', color='b', label='+1'  )
matplotlib.pyplot.scatter(p_class1['X1'], p_class1['X2'], color='b', label='+1')
matplotlib.pyplot.scatter(p_class2['X1'], p_class2['X2'], color='g', label='+1')


#TODO: Include legend
matplotlib.pyplot.xlabel('x_1')
matplotlib.pyplot.ylabel('x_2')
matplotlib.pyplot.title('a(iii)')
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
