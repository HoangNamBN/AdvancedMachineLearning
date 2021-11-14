'''Khai báo thư viện cần thiết cho bài toán'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import decomposition
from sklearn.linear_model import RidgeCV
import time
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

'''load dữ liệu cho bài toán'''
X = np.load("./Dataset/DigitsData/X.npy")
Y = np.load("./Dataset/DigitsData/Y.npy")
print("Data shape: ", X.shape)
print(X.shape[0], "sample, ", X.shape[1], "x", X.shape[2], 'size grayscall image.\n')
print("Labels shape: ", Y.shape)

'''Hiển thị một số dữ liệu mẫu'''
img_size = 64
print('Examples:')
n = 10
plt.figure(figsize=(20, 4))
image_index_list = [260, 900, 1800, 1600, 1400, 2061, 700, 500, 1111, 100]
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X[image_index_list[i - 1]].reshape(img_size, img_size))
    plt.gray()
    plt.axis('off')
    title = "Sign " + str(i - 1)
    plt.title(title)
plt.show()

'''Chia tập dữ liệu thành tập dữ liệu huấn luyện và dữ liệu thử nghiệm (train và test)'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,', X_train.shape[1], 'x', X_train.shape[2], 'size grayscale image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,', X_test.shape[1], 'x', X_test.shape[2], 'size grayscale image.\n')

'''Trực quan bằng hình ảnh về số lượng mẫu mỗi lớp'''
# sample_per_class = np.unique(Y, return_counts=True)
# sns.barplot(x = sample_per_class[0], y= sample_per_class[1])
# plt.title('Samples per class')
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.show()


# '''Hiển thị một số mẫu từ tập dữ liệu'''
# img_size = 64
# '''một số từ mỗi chữ số'''
# img_index_list = [260, 900, 1800, 1600, 1400, 2061, 700, 500, 1111, 100]
# for i in range(10):
#     plt.figure(figsize=(8, 5))
#     plt.imshow(X[img_index_list[i]].reshape(img_size, img_size))
#     plt.gray()
#     plt.axis('off')
#     title = "Sign " + str(i)
#     plt.title(title)
# plt.show()
#
# X = X.reshape((len(X), -1))
# train = X
# test = X[img_index_list]
# n_pixels = X.shape[1]
# print(n_pixels)

'''Chia tập train test'''

# X_train = train[:, :(n_pixels + 1) // 2]
# X_test = test[:, :(n_pixels + 1) // 2]
# y_train = train[:, n_pixels // 2:]
# y_test = test[:, n_pixels // 2:]

'''Hiển thị thông tin tập dữ liệu train test'''
# print("X train: ", X_train.shape)
# print("y train: ", y_train.shape)
# print("X test: ", X_test.shape)
# print("y test: ", y_test.shape)

'''Tạo mô hình mới'''
# clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state=1)
# '''Cho máy học'''
# clf.fit(X_train, y_train)

# '''Test với bộ dữ liệu thử nghiệm X_test'''
# y_predict = clf.predict(X_test)
# '''Hiển thị độ chính xác của mô hình'''
# print("Accuracy: " + str(accuracy_score(y_test, y_predict) * 100))
# '''=> Độ chính xác thấp dẫn đến cần phải giảm số chiều'''
