import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
dataset = pd.read_csv('Data/CarPrice_Assignment.csv')
#Tiền xử lý dữ liệu
dataset.dropna() #Xóa giá trị N/A
dataset=dataset.fillna(dataset.mean()) #giá trị trung bình
dataset.drop('car_ID',axis=1,inplace=True)
X = dataset.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))
#hiển thị dữ liệu trên plt : cylindernumber - price
# plt.scatter(dataset["cylindernumber"],dataset["price"])
# plt.xlabel("cylindernumber")
# plt.ylabel("price")
# plt.show()
#
# plt.scatter(dataset["enginesize"],dataset["price"])
# plt.xlabel("enginesize")
# plt.ylabel("price")
# plt.show()
# #
# plt.scatter(dataset["horsepower"],dataset["price"])
# plt.xlabel("horsepower")
# plt.ylabel("price")
# plt.show()

#Tìm X, y
X=X.drop(['CarName','price'],axis=1)
y=dataset['price']
print(X)
#Chia tập train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test)
#Decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)
val_mae = mean_squared_error(y_test_pred, y_test)
#sai số MSE
print('----------DecisionTreeRegressor----------');
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_test_pred))
print("Validation MAE : {:,.0f}".format(val_mae))
# plt.title('Decision Tree Regressor')
# plt.scatter(y_test,y_test_pred,color='red')
# plt.plot(y_test_pred,y_test_pred,color = 'green')
# plt.xlabel('Prices')
# plt.ylabel('Prices Predict')
# plt.show()
#Random Forest Regressor :
from sklearn.ensemble import RandomForestRegressor
Rand_frst = RandomForestRegressor(criterion = 'mse',
                              random_state = 20,
                              n_jobs = -1)
Rand_frst.fit(X_train,y_train)
Rand_frst_train_pred = Rand_frst.predict(X_train)
Rand_frst_test_pred = Rand_frst.predict(X_test)
y_test_predict = Rand_frst.predict(X_test)
print('----------RandomForestRegressor----------');
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Rand_frst_test_pred))
# plt.title('Random Forest Regressor')
# plt.scatter(y_test,Rand_frst_test_pred,color='red')
# plt.plot(Rand_frst_test_pred,y_test_predict,color = 'green')
# plt.xlabel('Prices')
# plt.ylabel('Prices Predict')
# plt.show()
#Linear Regression :
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
y_test_predict = model.predict(X_test)
X_test_first_value = X_test.iloc[0].to_frame()
X_10 = X_test.head(10)
y_predict_firstValue = model.predict(X_10)
print('----------LinearRegression----------');
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_predict))
print('Mean squared error 10 values: %.2f'
      % mean_squared_error(y_test.head(10), y_predict_firstValue))
# plt.title('Linear Regressor')
# plt.scatter(y_test,y_predict,color='red')
# plt.plot(y_predict,y_test_predict,color = 'green')
# plt.xlabel('Prices')
# plt.ylabel('Prices Predict')
# plt.show()

# ###### HỒI QUY CẦN TÍNH SAI SỐ, SAI SỐ CÀNG THẤP THÌ THUẬT TOÁN CÀNG TỐT