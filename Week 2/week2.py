import numpy as np
import pandas as pd
df= pd.read_csv("week2.csv" )
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

model = LogisticRegression(penalty= 'none', solver='lbfgs')
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


matplotlib.pyplot.scatter(class1['X1'], class1['X2'], marker = 'o', color='r', label='+1'  )
matplotlib.pyplot.scatter(class2['X1'], class2['X2'], marker = '+', color='b', label='+1'  )
matplotlib.pyplot.scatter(p_class1['X1'], p_class1['X2'], color='b', label='+1')
matplotlib.pyplot.scatter(p_class2['X1'], p_class2['X2'], color='g', label='+1')


#TODO: Include legend
matplotlib.pyplot.xlabel('x_1')
matplotlib.pyplot.ylabel('x_2')
matplotlib.pyplot.title('a(iii)')
matplotlib.pyplot.show()