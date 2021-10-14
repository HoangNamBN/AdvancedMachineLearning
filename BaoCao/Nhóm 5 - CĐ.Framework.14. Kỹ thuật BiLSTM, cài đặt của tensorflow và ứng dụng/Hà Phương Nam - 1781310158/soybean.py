import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score,confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE

# Doc file soybean-large .
df = pd.read_csv('soybean-large.csv')
print("Sau khi xu ly NaN :",df.shape)
# Xu ly du lieu NaN
df = df.dropna()
print("Sau khi xu ly NaN :",df.shape)
# Gan gia tri X.
X = df.drop(["Classes"], axis = 1)
#print (X)
# Gan gia tri Y.
y = df["Classes"]
#print(y)
print("------------")
# Chia tap train test theo ti le 50:50
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=1)
#print("X_train",X_train)
#Can bang du lieu
sm = SMOTE(random_state = 2,k_neighbors=1)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
print("Du lieu truoc khi SMOTE")
print("Class 'diaporthe-stem-canker': {}".format(sum(y_train == 'diaporthe-stem-canker')))
print("Class 'charcoal-rot': {}".format(sum(y_train == 'charcoal-rot')))
print("Class 'rhizoctonia-root-rot': {}".format(sum(y_train == 'rhizoctonia-root-rot')))
print("Du lieu sau khi SMOTE")
print("Class 'diaporthe-stem-canker': {}".format(sum(y_train_res == 'diaporthe-stem-canker')))
print("Class 'charcoal-rot': {}".format(sum(y_train_res == 'charcoal-rot')))
print("Class 'rhizoctonia-root-rot': {}".format(sum(y_train_res == 'rhizoctonia-root-rot')))
# su dung ky thuat hoc may SVM
print("Ky thuat hoc may SVM")
from sklearn.svm import SVC
model2 = SVC().fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
print("Accuracy Score 2:", accuracy_score(y_test, y_pred2), "\n")
print(classification_report(y_test,y_pred2))

# su dung ky thuat hoc may SVM voi SMOTE
print("Ky thuat hoc may SVM voi SMOTE")
from sklearn.svm import SVC
model2 = SVC().fit(X_train_res,y_train_res)
y_pred2 = model2.predict(X_test)
print("Accuracy Score 2:", accuracy_score(y_test, y_pred2), "\n")
print(classification_report(y_test,y_pred2))

# # su dung ky thuat hoc may LogisticRegression
# print ("ky thuat hoc may LogisticRegression")
# from sklearn.linear_model import LogisticRegression
# log_model = LogisticRegression(max_iter = 1000).fit(X_train,y_train)
# y_pred = log_model.predict(X_test)
# #Buoc 3: Danh gia mo hinh hoc may
# print("Accuracy Score:", accuracy_score(y_test, y_pred), "\n")
# print(classification_report(y_test,y_pred))
# # su dung ky thuat hoc may logistic voi SMOTE
# print("Ky thuat hoc may logistic voi SMOTE")
# from sklearn.linear_model import LogisticRegression
# log_model = LogisticRegression(max_iter = 1000).fit(X_train_res,y_train_res)
# y_pred = log_model.predict(X_test)
# #Buoc 3: Danh gia mo hinh hoc may
# print("Accuracy Score:", accuracy_score(y_test, y_pred), "\n")
# print(classification_report(y_test,y_pred))


# print (log_model.coef_)
# print (log_model.coef_.shape)
# print (log_model.intercept_)
# print (log_model.classes_)
# print (log_model.n_iter_)
#print (y_pred)