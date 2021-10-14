import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

#Đọc file dữ liệu, phân tích dữ liệu, chuẩn hoá, chia tỉ lệ tập train-test
df = pd.read_csv("pokemon.csv")
print (df.head())
print("-------------------------------------------")
#dummy:

df = pd.get_dummies(df)
print(df.head(10))

X = df.drop(['isLegendary'], axis = 1)
y = df['isLegendary']
print("-------------------------------------------")

print (X.shape)
print("-------------------------------------------")
print (y.shape)
print("-------------------------------------------")

#kiem tra du lieu thieu
missing=df.isnull().sum()
print(missing)

#chuẩn hoá
from sklearn.preprocessing import StandardScaler
#StandardScaler() sẽ bình thường hóa các tính năng (mỗi cột của X) để mỗi cột/tính năng/biến sẽ có mean = 0 và standard deviation = 1.
std = StandardScaler()
X = std.fit_transform(X)
print("chuẩn hóa: ")
print(X)
print("-------------------------------------------")

#chia tỉ lệ train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
print ("Shape of X_train: ",X_train.shape)
print("-------------------------------------------")
print ("Shape of y_train: ",y_train.shape)
print("-------------------------------------------")
print ("Shape of X_test: ",X_test.shape)
print("-------------------------------------------")
print ("Shape of y_test: ",y_test.head(10).shape)
print("-------------------------------------------")

#------------------- kỹ thuật học máy Logistic Regression --------------
#Chạy mô hình học máy
from sklearn.linear_model import LogisticRegression
#Huấn luyện (với tập dữ liệu train X_train, y_train). Lưu model/Tải model.
log_model = LogisticRegression(max_iter = 1000).fit(X_train,y_train)
#Bước 2.2. Dự đoán (với tập dữ liệu train X_test, y_test)
y_pred = log_model.predict(X_test) #y_prediction là y dự đoán được

print ("Hệ số w: ", log_model.coef_)
print("-------------------------------------------")
print (log_model.coef_.shape)
print("-------------------------------------------")
#Hệ số bias là b trong công thức y=ãx+b
print ("Hệ số bias: ", log_model.intercept_)
print("-------------------------------------------")
print ("Số lớp: ", log_model.classes_)
print("-------------------------------------------")
print ("Số vòng lặp: ", log_model.n_iter_)
print("-------------------------------------------")




logreg = linear_model.LogisticRegression(C=1e5) # just a big number
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print("-------------------------------------------")

#đánh giá mô hình học máy
from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("-------------------------------------------")

from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cnf_matrix)
print('\nAccuracy:', np.diagonal(cnf_matrix).sum()/cnf_matrix.sum())
print("-------------------------------------------")

# --- SMOTE -----
# xử lý dữ liệu thiếu cân bằng với SMOTE so sánh kết quả

print('Before OverSampling:')

print('no =', sum(y_train == 0))

print('yes =', sum(y_train == 1))

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print("After OverSampling, counts of label 0: {}".format(sum(y_train_res == 0)))

print("After OverSampling, counts of label 1: {}".format(sum(y_train_res == 1)))

model = LogisticRegression(max_iter=10000).fit(X_train_res, y_train_res)

yhat = model.predict(X_test)

accuracy = accuracy_score(y_test, yhat)

print('Accuracy using SMOTE: %.2f' % (accuracy * 100))

print(classification_report(y_test, yhat))

print ("Hệ số w: ", model.coef_)
print("-------------------------------------------")
print (model.coef_.shape)
print("-------------------------------------------")
print ("Hệ số bias: ", model.intercept_)
print("-------------------------------------------")
print ("Số lớp: ", model.classes_)
print("-------------------------------------------")
print ("Số vòng lặp: ", model.n_iter_)
print("-------------------------------------------")
print ("Tap y du doan: ")
print("-------------------------------------------")


from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_test, yhat))

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, yhat)
print('Confusion matrix:')
print(cnf_matrix)
print('\nAccuracy:', np.diagonal(cnf_matrix).sum()/cnf_matrix.sum())



