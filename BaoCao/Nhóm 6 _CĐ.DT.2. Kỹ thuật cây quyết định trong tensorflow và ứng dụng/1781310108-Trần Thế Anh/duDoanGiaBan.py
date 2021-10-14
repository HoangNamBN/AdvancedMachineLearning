import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df =pd.read_csv('train.csv')
# print(df.isnull().sum())
# print(df.shape)
df=df.fillna(df.mean())
# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X=df.drop(['SalePrice','Id'], axis = 1)
# X=df.drop(['SalePrice'], axis = 1)
y=df['SalePrice']
X_dummy = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, random_state=1)
#2. Giải bài toán hồi quy , sử dụng cây quyết định Decision Tree 
from sklearn.tree import DecisionTreeRegressor
mode=DecisionTreeRegressor(criterion='mse', random_state=1,max_leaf_nodes=100)
mode.fit(X_train,y_train)
y_predicted = mode.predict(X_test)
#3. Hiện thị hệ số tìm được và sai số MSE
# The mean squared error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_predicted))
val_mae = mean_absolute_error(y_predicted, y_test)
print("Validation MAE: {:,.0f}".format(val_mae))
stt=[]
mae=[]
for i in range(10,150):
    model = DecisionTreeRegressor(random_state=1,max_leaf_nodes=i)
    model.fit(X_train, y_train)
    y_predicted = mode.predict(X_test)
    val_mae = mean_absolute_error(y_predicted, y_test)
    stt.append(i)
    mae.append(val_mae)
plt.title('MAE')
plt.xlabel('Max leaf nodes')
plt.ylabel('MAE')
plt.plot(stt,mae)
plt.show()
