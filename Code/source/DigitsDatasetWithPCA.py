'''Bài toán: Trực quan PCA với bộ dữ liệu Digits'''

'''Khai báo thư viện cần thiết cho bài toán'''
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

'''load dữ liệu cho bài toán'''
X = np.load("../Dataset/DigitsData/X.npy")
Y = np.load("../Dataset/DigitsData/Y.npy")
print("Data shape: ", X.shape)
print(X.shape[0], "sample, ", X.shape[1], "x", X.shape[2], 'size grayscall image.\n')
print("Labels shape: ", Y.shape)

'''Hiển thị một số dữ liệu mẫu'''
print('Hiển thị một số mẫu:')
img_size = 64
n = 10
plt.figure(figsize=(20, 4))
image_index_list = [260, 900, 1800, 1600, 1400, 2061, 700, 500, 1111, 100]
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X[image_index_list[i - 1]].reshape(img_size, img_size))
    # plt.gray()
    plt.axis('off')
    title = "Sign " + str(i - 1)
    plt.title(title)
plt.show()

'''Chia tập dữ liệu thành tập dữ liệu huấn luyện và dữ liệu thử nghiệm (train và test)'''
X_fat = np.array(X).reshape(2062, 64*64)
X_train, X_test, Y_train, Y_test = train_test_split(X_fat, Y, test_size=0.2, random_state=42)
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,', X_train.shape[1], 'size grayscale image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,', X_test.shape[1],'size grayscale image.\n')

'''Sử dụng mô hình MLPClasifier khi chưa sử dụng giảm chiều PCA để huấn luyện'''
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100, 100, 100), random_state=1)

start = time.time()
clf.fit(X_train, Y_train)
end = time.time()
print("Training time is " + str(end - start) + " second.")
'''Dự đoán mô hình khi chưa sử dụng phương pháp giảm chiều PCA'''
y_hat = clf.predict(X_test)
print("{}: {:.2f}%".format("Accuracy", accuracy_score(Y_test, y_hat)*100))

'''Tính ra số chiều sau khi đã giảm chiều'''
pca_dims = PCA()
pca_dims.fit(X_train)
cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.7) + 1

'''Tính toán giảm chiều và hiển thị ra số chiều trước và sau khi giảm chiều PCA'''
pca = PCA(n_components= d)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)
print("reduced shape: " + str(X_reduced.shape))
print("recovered shape: " + str(X_recovered.shape))

'''Hiển thị hình ảnh trước và sau khi giảm chiều PCA'''
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("Trước khi giảm chiều")
plt.imshow(X_train[4].reshape((img_size, img_size)))
f.add_subplot(1, 2, 2)
plt.title("Sau khi giảm chiều")
plt.imshow(X_recovered[4].reshape((img_size, img_size)))
plt.show(block = True)

'''Sử dụng mô hình MLPClasifier khi sử dụng giảm chiều PCA để huấn luyện'''
clf_PCA = MLPClassifier(solver='adam', alpha=1e-5,
            hidden_layer_sizes= (100, 100, 100, 100), random_state=1)
start_PCA = time.time()
clf_PCA.fit(X_reduced, Y_train)
end_PCA = time.time()
print("Training time is " + str(end_PCA - start_PCA) + " second using PCA")

'''Hiển thị kết quả dự đoán cũng như số lớp của bài toán'''
X_test_PCA = pca.transform(X_test)
y_hat_PCA = clf_PCA.predict(X_test_PCA)
print("Kết quả dự đoán:\n", y_hat_PCA)
# print("{}: {:.2f}%".format("Accuracy", accuracy_score(Y_test, y_hat_PCA)*100))
print("Số lớp: ", clf_PCA.classes_)

'''Đánh giá mô hình học dựa trên kết quả dự đoán 
    (với độ đo đơn giản Accuracy, Precision, Recall)'''
y_hat_PCA_classes = np.argmax(y_hat_PCA, axis=1)
y_true = np.argmax(Y_test, axis=1)
confusion_mtx = confusion_matrix(y_true, y_hat_PCA_classes)
print("Ma trận dự doán:\n", confusion_mtx)
print("{}: {:.2f}%".format("Accuracy Score: " ,
            accuracy_score(y_true, y_hat_PCA_classes)*100))
print(classification_report(y_true, y_hat_PCA_classes))

'''Vẽ ma trận dự đoán'''
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",
            linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()