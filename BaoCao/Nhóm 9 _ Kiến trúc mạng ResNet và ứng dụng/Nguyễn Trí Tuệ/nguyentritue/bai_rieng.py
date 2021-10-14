import pandas as pd
import numpy as np

# Xử lý dữ liệu

df = pd.read_csv('zoo.csv')
df.head()
features = list(df.columns)
print(features)
features.remove('class_type')
features.remove('animal_name')


print(features)
X = df[features].values.astype(np.float32)
Y = df.class_type


print(X.shape)
print(Y.shape)
print("///")


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)

print(Y_train)

print("///")
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# huấn luyện

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print("training accuracy :", model.score(X_train, Y_train))
print("testing accuracy :", model.score(X_test, Y_test))





#from sklearn.linear_model import LogisticRegression

#model = LogisticRegression()
#model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
#print("training accuracy :", model.score(X_train, Y_train))
#print("testing accuracy :", model.score(X_test, Y_test))


# Đánh giá
from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(Y_test, y_pred), "\n")
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, y_pred)
print('Confusion matrix:')
print(cnf_matrix)