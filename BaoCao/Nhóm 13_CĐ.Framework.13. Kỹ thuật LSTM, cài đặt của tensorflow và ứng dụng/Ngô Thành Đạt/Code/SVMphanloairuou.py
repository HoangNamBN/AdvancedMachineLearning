import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import pandas as pd
from KNNphanloairuou import y_pred

#Load dữ liệu
wine = datasets.load_wine()
wine_X = wine.data
wine_y = wine.target
print(wine_X.shape)
print ('Number of classes: %d' %len(np.unique(wine_y)))
print ('Number of data points: %d' %len(wine_y))
#whisky
X0 = wine_X[wine_y == 0,:]
print ('\nSamples from class 0:\n', X0[:5,:])

#ruouvodka
X1 = wine_X[wine_y == 1,:]
print ('\nSamples from class 1:\n', X1[:5,:])

#ruoucocktail
X2 = wine_X[wine_y == 2,:]
print ('\nSamples from class 2:\n', X2[:5,:])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     wine_X, wine_y, test_size=50)

print("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))
# # --- SMOTE -----
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE
# print('Before OverSampling:')
# print('class 0 =', sum(y_train == 0))
# print('class 1 =', sum(y_train == 1))
# print('class 2 =', sum(y_train == 2))
# sm = SMOTE(random_state=2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
#
# print("After OverSampling, counts of label 0: {}".format(sum(y_train_res == 0)))
# print("After OverSampling, counts of label 1: {}".format(sum(y_train_res == 1)))
# print("After OverSampling, counts of label 2: {}".format(sum(y_train_res == 2)))
#
# model = LogisticRegression(max_iter=10000).fit(X_train_res, y_train_res)
# yhat = model.predict(X_test)
#
# accuracy = accuracy_score(y_test, yhat)
# print('Accuracy using SMOTE: %.2f' % (accuracy * 100))
#
#
# def classification_report(y_test, yhat):
#      pass
#
#
# print(classification_report(y_test, yhat))
#kỹ thuật học máy SVM
from sklearn.svm import SVC
model2 = SVC().fit(X_train,y_train)
y_pred2 = model2.predict(X_test)

def accuracy_score(y_test, y_pred2):
     pass
print( "IN ra kết quả cho 20 data test:")
print ("Predicted labels: ", y_pred[20:50])
print ("Ground truth    : ", y_test[20:50])




from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))


