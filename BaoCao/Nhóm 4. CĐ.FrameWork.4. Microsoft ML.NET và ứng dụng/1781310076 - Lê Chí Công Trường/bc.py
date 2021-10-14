import pandas as pd

# doc file du lieu csv
df = pd.read_csv('student-por.csv')

# xu ly du lieu
#1. Loại bỏ/fill Dữ liệu NaN 
df = df.dropna()

# Xử lý dữ liệu Category (LabelEncoder, OneHotEncoder/dummy)
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()

df_encoder = df.apply(lb_make.fit_transform)

# df_encoder.to_csv('data_encoder.csv')
# print(df_encoder)
# chia train test

# sap xep va chia train test
df_encoder = df_encoder.sort_values('higher', ascending = True)

Y_train = df_encoder.iloc[0:59,20:21].append(df_encoder.iloc[79:,20:21]) 
Y_test = df_encoder.iloc[59:79,20:21]  
# print(yTrain)
df_encoder = df_encoder.drop('higher', axis = 1)

X_train = df_encoder.iloc[0:59].append(df_encoder.iloc[79:]) 
X_test = df_encoder.iloc[59:79] 
# print(xTrain)
# print(xTest)


# train mo hinh

########logictis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


model = LogisticRegression(max_iter = 10000).fit(X_train, Y_train)
Y_hat = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_hat)

print("============== LR =====================")

print('Accuracy: %.2f' % (accuracy*100))
print(classification_report(Y_test,Y_hat))


######## MVC
from sklearn.svm import SVC

model_SVC = SVC()
model_SVC.fit(X_train, Y_train) 

# test MVC
predict = model_SVC.predict(X_test)

print("============== MVC =====================")

ac_score = accuracy_score(Y_test, predict)
cl_report = classification_report(Y_test, predict)

print("Score = ", ac_score)
print(cl_report)

# 3. Thống kê số lượng của y, vẽ sơ đồ 
# 4. Xử lỹ dữ liệu thiếu cân bằng với SMOTE, so sánh kết quả 


