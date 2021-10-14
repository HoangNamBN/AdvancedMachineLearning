import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('./soccer_international_history.csv')
# print(df)
#print (df.columns)
#print (df.columns.values)
#print (df.isnull().any())
print (df.isnull().sum())
print (df.shape)
# df = df.dropna()
# print (df.shape)
#-------------------------------- LabelEncoder --------------------------------------
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()

from sklearn import preprocessing
df_LabelEncoder = df.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))
X_LabelEncoder = df_LabelEncoder.drop('home_team_result',axis=1)
y_LabelEncoder = df_LabelEncoder['home_team_result']
# print(X_LabelEncoder)
# print(y_LabelEncoder)
X_train, X_test, y_train, y_test = train_test_split(X_LabelEncoder, y_LabelEncoder, test_size=0.2)
############################### Kỹ thuật phân lớp Decision Tree Classifier ##############################
from sklearn.tree import DecisionTreeClassifier
mode = DecisionTreeClassifier(criterion='gini', random_state = 100,max_depth=5, min_samples_leaf=5)
# Huấn luyện mô hình với X_train, y_bina_train
mode = mode.fit(X_train, y_train)
# Dự đoán mô hình  với X_test, y_bina_test
y_predicted = mode.predict(X_test)
# Hiện thị KQ
from sklearn.metrics import accuracy_score
print("Accuracy Decision Tree Classifier: ", accuracy_score(y_test, y_predicted))

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
#----------------------------------- Kỹ thuật phân lớp Random ForestClassifier -------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('Accuracy Random Forest Classifier:',accuracy_score(y_test, y_pred))
########################## Ma trận #################################
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
########################## Smote data ############################
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import classification_report
print ('Before OverSampling:')
print ('Draw =', sum(y_train == 0))
print ('Win =', sum(y_train == 1))
print ('Loss =', sum(y_train == 2))

sm = SMOTE(random_state = 3) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
  
print("After OverSampling, counts of label 'Draw': {}".format(sum(y_train_res == 0))) 
print("After OverSampling, counts of label 'Win': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label 'Loss': {}".format(sum(y_train_res == 2)))

model = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train_res, y_train_res)
yhat = model.predict(X_test)

accuracy = accuracy_score(y_test, yhat)
print('Accuracy Random Forest Classifier using SMOTE: %.2f' % (accuracy*100))
print(classification_report(y_test,yhat))
