import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("car_ad.csv", sep=',', encoding='latin-1')

df.dropna()
df['car'] = df['car'].str.split(' ', expand=True)
df.head()

from sklearn import preprocessing
df = df.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col.astype(str)))
print(df)
X = df.drop(['price'], axis=1)
y = df['price']
print(X)
print(y)
#Chia tập train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
#Linear
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print("y_test: ", y_test)
print("y_predict: ", y_predict)

# Plot outputs
plt.scatter(y_test, y_predict,  color='black')
plt.plot(y_predict, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.title('Linear regression for price car')
plt.xlabel('X')
plt.ylabel('y')

from sklearn.metrics import mean_squared_error
print('----------LinearRegression----------');
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_predict))
print("Score: ", model.coef_)
#Decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
print('----------DecisionTreeRegressor----------');
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_predict))

plt.show()

