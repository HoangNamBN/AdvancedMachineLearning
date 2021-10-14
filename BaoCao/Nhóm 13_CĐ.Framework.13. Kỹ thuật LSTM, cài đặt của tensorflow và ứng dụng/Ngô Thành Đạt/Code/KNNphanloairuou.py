import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import pandas as pd
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

#kỹ thuật học máy KNN
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print( "IN ra kết quả cho 20 data test:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])

# from sklearn.metrics import accuracy_score
# print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))

# print ('confusion_matrix =')
# print(confusion_matrix(y_test,y_pred))
