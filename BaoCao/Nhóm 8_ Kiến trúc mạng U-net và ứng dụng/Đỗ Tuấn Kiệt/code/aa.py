import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#Bước 1: Đọc file dữ liệu, phân tích dữ liệu, chuẩn hoá, chia tỉ lệ tập train-test
df = pd.read_csv("heart1.csv")
data_frame = df.dropna()
print (df.head())
objList = df.select_dtypes(include = "object").columns #lấy thông tin các cột object để xử lí
print(objList)
for cols in objList:
    df[cols] = preprocessing.LabelEncoder().fit_transform(df[cols].astype(str))
X = df.drop(["target"], axis = 1)
y = df["target"]
print (X.shape)
print (y.shape)
#chuẩn hoá
#from sklearn.preprocessing import StandardScaler
#std = StandardScaler()
#X = std.fit_transform(X)
#print(X)

#chia tỉ lệ train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
print("chia ti le train tesst")
print (X_train)
print (y_train)
print (X_test)
print (y_test.head(10))
#kiêm tra du lieu thiêu
print("du lieu thieu",df.isnull().sum())
#------------------- kỹ thuật học máy LogisticRegression --------------
#Bước 2: Chạy mô hình học máy
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter = 1000).fit(X_train,y_train)

y_pred = log_model.predict(X_test) #y_prediction là y dự đoán được
print ("Hệ số w", log_model.coef_)
print (log_model.coef_.shape)
print ("Hệ số bias", log_model.intercept_)
print ("Số lớp", log_model.classes_)
#print ("Số vòng lặp", log_model.n_iter_)
print ("Tap y du doan")
print (y_pred)
#Bước 3: Đánh giá mô hình học dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)
print("đánh giá mô hình học máy")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("aaaa")
print("CLass",classification_report(y_test,y_pred))
print ('confusion_matrix =',confusion_matrix(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

#cân bằng smote
print("cân bằng smote")

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
from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_test, yhat))
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, yhat)
print('Confusion matrix:')
print(cnf_matrix)
print('\nAccuracy:', np.diagonal(cnf_matrix).sum()/cnf_matrix.sum())



