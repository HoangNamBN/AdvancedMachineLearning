'''Khai báo thư viện cần thiết cho bài toán'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import decomposition
from sklearn.linear_model import RidgeCV
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

'''load dữ liệu cho bài toán'''
X = np.load("./Dataset/X.npy")
y = np.load("./Dataset/Y.npy")
print("Số mẫu của bô dữ liệu :\n", X.shape)
print("Nhãn dữ liệu: \n", y.shape)

'''Hiển thị một số mẫu từ tập dữ liệu'''
# img_index_list = [250, 900, 1800, 1600, 1400, 2061, 700, 500, 1111, 100]
# for i in range(10):
#     plt.figure(figsize=(8, 5))
#     plt.imshow(X[img_index_list[i]].reshape(64, 64))
#     plt.axis('off')
#     title = "Sign " + str(i)
#     plt.title(title)
# plt.show()

'''Chia tập dữ liệu để huấn luyện'''
'''Chuyển 2062 mẫu chuyển thành 4096 vector'''
X = np.array(X).reshape((2062, 64*64))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''Tạo mô hình mới'''
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state= 1)
'''Cho máy học'''
clf.fit(X_train, y_train)

'''Test với bộ dữ liệu thử nghiệm X_test'''
y_predict = clf.predict(X_test)
'''Hiển thị độ chính xác của mô hình'''
print("Accuracy: " + str(accuracy_score(y_test, y_predict)*100))
'''=> Độ chính xác thấp dẫn đến cần phải giảm số chiều'''

pca = PCA