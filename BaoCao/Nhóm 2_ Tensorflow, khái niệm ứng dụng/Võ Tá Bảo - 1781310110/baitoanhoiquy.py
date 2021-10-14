import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
# Đọc dữ liệu
df = pd.read_csv("AirUCI1.csv")
df = df.dropna()
print (df.head())
#df2 = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
#df2.head()

df2 = df.drop('place', axis=1)
print (df2.head())
X = df2.drop(['CO(GT)'], axis = 1)
y = df2["CO(GT)"]
print (X.shape)
print (y.shape)
# giam chieu

from sklearn.decomposition import PCA
X = PCA(1).fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=35)
print (X.shape)
print (X_train.shape)
print (X_test.shape)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
#print(pd.DataFrame({"Name": X.columns, "Coefficients": np.abs(regr.coef_)}).sort_values(by='Coefficients'))
#hệ số
print('Coefficients: \n', regr.coef_)

print('Bias: \n', regr.intercept_)
# The mean squared error
#sai số toàn phương trung bình
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
#hệ số xác định
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
print('Score = %.2f' % regr.score(X_train, y_train))
print ("X_test[0] = ", X_test[0])
print ("y_pred[0] = ", regr.predict(X_test[0].reshape(1, -1)))
print ("y_test[0] = " ,y_test[0])
print ("Sai số = " , regr.predict(X_test[0].reshape(1, -1)) - y_test[0])

# Plot outputs
plt.scatter(X_train, y_train,  color='green')
plt.scatter(X_test, y_test,  color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.title('Linear regression for AirqualityUci')
plt.xlabel('X')
plt.ylabel('y')

plt.show()

