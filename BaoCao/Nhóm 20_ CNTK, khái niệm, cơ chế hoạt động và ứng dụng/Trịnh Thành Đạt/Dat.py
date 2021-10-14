import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import  train_test_split
from sklearn import metrics as sq
from sklearn import linear_model
from sklearn.decomposition import PCA
regr = linear_model.LinearRegression()

#Đọc tệp
data = pd.read_csv("computer_hardware.csv", sep=",")
data.head
#In dữ liệu
print(data)
#Lấy dataframe hiệu năng máy tính làm biên mục tiêu
Y = data['ERP']
print(Y)
#Lấy datafarame không chứa hiẹu năng máy tính làm biến giải thích
X = data.drop("ERP", axis = 1)
print(X)
#Phân loại dữ liệu train và test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
#Tạo model suy đoán
regr.fit(X_train, Y_train)
#In hệ số hồi quy của các biến giải thích xếp theo thứ tự tăng dần
print("\nHỆ SỐ HỒI QUY")
print(pd.DataFrame({"Tên": X_train.columns, "Hệ số": np.abs(regr.coef_)}).sort_values(by='Hệ số'))
# BIAS
print("\nBIAS")
print(regr.intercept_)
#print(clf.score())
#Tiến hành dự đoán với bộ dữ liệu test
Y_pred = regr.predict(X_test)
print("\nGIÁ TRỊ Y DỰ ĐOÁN")
print(Y_pred)
#In giá trị y test thực tế
print("\nGIÁ TRỊ Y THỰC TẾ")
print(Y_test)
#Kiểm tra mức độ lỗi của model (Mean Squared Error)
mse = sq.mean_squared_error(Y_test, Y_pred)
print("MSE", mse)
print("SCORE: ", regr.score(X_train, Y_train))
#Giảm chiều
pca = PCA(n_components=1)
pca1 = pca.fit_transform(X_test)
#Bảng biểu so sánh giá trị y dự đoán và y thực tế
plt.scatter(pca1, Y_test)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sự tương quan giữa X và Y")
plt.show()
