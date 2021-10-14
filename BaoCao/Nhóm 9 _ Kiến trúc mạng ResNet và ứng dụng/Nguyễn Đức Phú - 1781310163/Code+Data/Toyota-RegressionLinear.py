import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


dataset = pd.read_csv('database/ToyotaCorolla.csv')
print(dataset.head(10))
print("---------------------------------------------------------------------")
print(dataset.count())
print("---------------------------------------------------------------------")
print(dataset.describe())
print("---------------------------------------------------------------------")
print(dataset.isnull().sum())
print("---------------------------------------------------------------------")

# dataset = pd.get_dummies(dataset, columns=['FuelType'])
dataset = pd.get_dummies(dataset)
print(dataset.head(10))

X = dataset.drop(['Price'], axis = 1).values
y = dataset.iloc[:, 0].values.reshape(-1,1)

print("---------------------------------------------------------------------")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test", y_test.shape)
print("---------------------------------------------------------------------")

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor_linear = LinearRegression()
regressor_linear.fit(X_train, y_train)

from sklearn.metrics import r2_score

# Predicting Cross Validation Score the Test set results
#Dự đoán kết quả giao của tập Test
cv_linear = cross_val_score(estimator = regressor_linear, X = X_train, y = y_train, cv = 10)

# Predicting R2 Score the Train set results
y_pred_linear_train = regressor_linear.predict(X_train)
r2_score_linear_train = r2_score(y_train, y_pred_linear_train)

# Predicting R2 Score the Test set results
y_pred_linear_test = regressor_linear.predict(X_test)
r2_score_linear_test = r2_score(y_test, y_pred_linear_test)

# Predicting RMSE the Test set results
rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_test)))
print("w :",regressor_linear.coef_)#Sai so
print("CV: ", cv_linear.mean())
print('R2_score (train): ', r2_score_linear_train)
print('R2_score (test): ', r2_score_linear_test)
print("RMSE: ", rmse_linear)#Trung binh sai so


plt.scatter(y_pred_linear_test,y_test)
plt.xlabel("Giá Xe dự đoán của Y")
plt.ylabel("Giá Xe thực của Y")
plt.title("Giá Xe Toyota Corolla dự đoán và thực")
plt.show()

