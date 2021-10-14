
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE 

df = pd.read_csv('G:/hoc_mnc/beophi/dataset_beophi/beophi.csv')
print(df)
print (df.isnull().sum())
print (df.shape)
X = df.iloc[:, 1:-1]
print(X.shape)
y = df.iloc[:, -1]
#--- LabelEncoder -----
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()

from sklearn import preprocessing
df_LabelEncoder = df.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))
X_LabelEncoder = df_LabelEncoder.drop('NObeyesdad',axis=1)
y_LabelEncoder = df_LabelEncoder['NObeyesdad']
print(X_LabelEncoder)
print(y_LabelEncoder.head(10))
X_train, X_test, y_train, y_test = train_test_split(X_LabelEncoder, y_LabelEncoder, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
mode = DecisionTreeClassifier(criterion='gini', random_state = 100,max_depth=5, min_samples_leaf=5)
# Huấn luyện mô hình với X_train, y_bina_train
mode = mode.fit(X_train, y_train)
# Dự đoán mô hình  với X_test, y_bina_test
y_predicted = mode.predict(X_test)
# Hiện thị KQ
from sklearn.metrics import accuracy_score
print("Accuracy Decision Tree Classifier: ", accuracy_score(y_test, y_predicted))
######################################################
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_predicted))
print(confusion_matrix(y_test, y_predicted))
#################RandomForestClassifier###############
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=5, random_state=5)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('Accuracy Randomy_predicted Forest Classifier:',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#######################LogisticRegresstion#################
#Bước 2: Chạy mô hình học máy
# from sklearn.linear_model import LogisticRegression
# #Bước 2.1. Huấn luyện (với tập dữ liệu train X_train, y_train). Lưu model/Tải model.
# log_model = LogisticRegression(max_iter = 10).fit(X_train,y_train)
# #Bước 2.2. Dự đoán (với tập dữ liệu train X_test, y_test)
# y_pred = log_model.predict(X_test) #y_prediction là y dự đoán được
# print ("Hệ số w", log_model.coef_)
# print (log_model.coef_.shape)
# print ("Hệ số bias", log_model.intercept_)
# print ("Số lớp", log_model.classes_)
# # #print ("Số vòng lặp", log_model.n_iter_)
# # print ("Tap y du doan")
# # print (y_pred)

#Bước 3: Đánh giá mô hình học dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)
# from sklearn.metrics import accuracy_score
# print("Accuracy Score:", accuracy_score(y_test, y_pred))
from imblearn.over_sampling import SMOTE 
print ('Before OverSampling:')

print ('truoc khi can bang 0 =', sum(y_train == 0))
print ('truoc khi can bang 1 =', sum(y_train == 1))
print ('truoc khi can bang 2 =', sum(y_train == 2))
print ('truoc khi can bang 3 =', sum(y_train == 3))
print ('truoc khi can bang 4 =', sum(y_train == 4))
print ('truoc khi can bang 5 =', sum(y_train == 5))
print ('truoc khi can bang 6 =', sum(y_train == 6))
sm = SMOTE(random_state = 2) 

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print("After OverSampling, counts of label 0: {}".format(sum(y_train_res == 0))) 
print("After OverSampling, counts of label 1: {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label 2: {}".format(sum(y_train_res == 2))) 
print("After OverSampling, counts of label 3: {}".format(sum(y_train_res == 3)))
print("After OverSampling, counts of label 4: {}".format(sum(y_train_res == 4))) 
print("After OverSampling, counts of label 5: {}".format(sum(y_train_res == 5)))
print("After OverSampling, counts of label 6: {}".format(sum(y_train_res == 6))) 
################RandomForestClassifier###########

classifier = RandomForestClassifier(max_depth=5, random_state=5)
classifier.fit(X_train_res, y_train_res)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('Accuracy Randomy_predicted Forest Classifier:',accuracy_score(y_test, y_pred))

####################
mode = DecisionTreeClassifier(criterion='gini', random_state = 100,max_depth=5, min_samples_leaf=5)
mode = mode.fit(X_train, y_train)
# Dự đoán mô hình  với X_test, y_bina_test
y_predicted = mode.predict(X_test)
# Hiện thị KQ

print("Accuracy Decision Tree Classifier: ", accuracy_score(y_test, y_predicted))
accuracy = accuracy_score(y_test, y_predicted)
print('Accuracy DecisionTreeClassifier using SMOTE: %.2f' % (accuracy*100))
print(classification_report(y_test, y_predicted))