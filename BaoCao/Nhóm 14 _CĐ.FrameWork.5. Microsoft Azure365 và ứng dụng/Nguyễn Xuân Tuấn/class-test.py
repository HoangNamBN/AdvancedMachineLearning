import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("GlassdoorGendePayGap1.csv")
print(df)
# print (df.columns)
# print (df.columns.values)
# print (df.isnull().any())
print(df.isnull().sum())
print(df.shape)
# df = df.dropna()
# print (df.shape)
X = df.iloc[:, 1:-1]
print(X.shape)
y = df.iloc[:, -1]
# --- LabelEncoder -----
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
# X["outlook"] = lb_make.fit_transform(X["outlook"])
# X_LabelEncoder = X.apply(lb_make.fit_transform)
from sklearn import preprocessing

df_LabelEncoder = df.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))
X_LabelEncoder = df_LabelEncoder.drop('Bonus', axis=1)
y_LabelEncoder = df_LabelEncoder['Bonus']
print(X_LabelEncoder)
print(y_LabelEncoder)
X_train, X_test, y_train, y_test = train_test_split(X_LabelEncoder, y_LabelEncoder, test_size=0.2)
# from sklearn.tree import DecisionTreeClassifier
mode = DecisionTreeRegressor()
# Huấn luyện mô hình với X_train, y_bina_train
mode = mode.fit(X_train, y_train)
# Dự đoán mô hình  với X_test, y_bina_test
y_predicted = mode.predict(X_test)
# Hiện thị KQ
# The mean squared error
print('Mean squared error DecisionTreeRegressor: %.2f'
      % mean_squared_error(y_test, y_predicted))
# from sklearn.metrics import confusion_matrix,classification_report
# print(classification_report(y_test, y_predicted))
# print(confusion_matrix(y_test, y_predicted))
######################################################
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Bias: \n', regr.intercept_)
# The mean squared error(sai số bình phương trung bình)
print('Mean squared error LinearRegression: %.2f'
      % mean_squared_error(y_test, y_pred))
######################################################
# from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
##y_train = PCA(1).fit_transform(y_train)
# X_test = PCA(1).fit_transform(X_test)
# plt.scatter(X_train, y_train,  color='green')
plt.scatter(y_test, y_predict, color='black')
plt.plot(y_predict, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.title('Linear regression for 2021')
plt.xlabel('X')
plt.ylabel('y')

plt.show()
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mse')
# ‘mse‘ (mean squared error) for regression.

# fit the model
model.fit(X_train, y_train, epochs=250, batch_size=32, verbose=0)
###################################
# evaluate the model(đánh giá mô hình)
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))
print("y_test:", y_test)
print("y-du doan:",y_predict)
print(y_predict)

# print ('abc')
#print(y_test[1])
#print(y_predict[1])

