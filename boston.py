from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
print(boston_data.DESCR)
df_boston['MEDV']= boston_data.target
X_R1 = np.array(df_boston.drop(['MEDV'],1))
Y_R1 = np.array(df_boston['MEDV'])

X_train,X_test,y_train,y_test = train_test_split(X_R1,Y_R1)
linreg = LinearRegression().fit(X_train,y_train)
x = linreg.score(X_test,y_test)
predictions = linreg.predict(X_test)

for i in range(len(predictions)):
    print('{:,.2f}'.format(predictions[i]), y_test[i])

print("BEST ACCURACY OF THE THE MODEL IS ",x)
print(y_test[0])
c = linreg.predict([X_test[0]])
print(c)
