import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

#############################
#1 .TIỀN XỬ LÍ DỮ LIỆU
df = pd.read_csv('adult.csv')
print(df.shape)
print(df.iloc[:, 0:7])

#xoa du lieu dang NaN va NULL
print(df.shape)
df = df.dropna()
print(df.shape)
# print('Sau khi xóa bỏ dữ liệu trống :')
# print(df.head(15))

#encoding du lieu LabelEncoder
df['marital.status'] = LabelEncoder().fit_transform(df['marital.status'])
df['workclass'] = LabelEncoder().fit_transform(df['workclass'])
df['education'] = LabelEncoder().fit_transform(df['education'])
df['occupation'] = LabelEncoder().fit_transform(df['occupation'])
df['relationship'] = LabelEncoder().fit_transform(df['relationship'])
df['race'] = LabelEncoder().fit_transform(df['race'])
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['native.country'] = LabelEncoder().fit_transform(df['native.country'])
df['income'] = LabelEncoder().fit_transform(df['income'])
print(df.shape)
print(df.head(10))
print(df.iloc[0:10, 0:7])

X = np.asarray(df[['age','workclass','education','occupation','relationship','sex']])
y = np.asarray(df['marital.status'])
# Tập dữ liệu train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# from imblearn.over_sampling import SMOTE
# print('Trước khi OverSampling:')
# print('số lượng nhãn bằng = 0:', sum(y_train == 0))
# print('số lượng nhãn bằng = 1:', sum(y_train == 1))
# print('số lượng nhãn bằng = 2:', sum(y_train == 2))
# print('số lượng nhãn bằng = 3:', sum(y_train == 3))
# print('số lượng nhãn bằng = 4:', sum(y_train == 4))
# print('số lượng nhãn bằng = 5:', sum(y_train == 5))
# print('số lượng nhãn bằng = 6:', sum(y_train == 6))
#
# sm = SMOTE(random_state=7)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
# print("Sau khi OverSampling, số lương nhãn '0': {}".format(sum(y_train_res == 0)))
# print("Sau khi OverSampling, số lương nhãn '1': {}".format(sum(y_train_res == 1 )))
# print("Sau khi OverSampling, số lương nhãn '2': {}".format(sum(y_train_res == 2 )))
# print("Sau khi OverSampling, số lương nhãn '3': {}".format(sum(y_train_res == 3 )))
# print("Sau khi OverSampling, số lương nhãn '4': {}".format(sum(y_train_res == 4 )))
# print("Sau khi OverSampling, số lương nhãn '5': {}".format(sum(y_train_res == 5 )))
# print("Sau khi OverSampling, số lương nhãn '6': {}".format(sum(y_train_res == 6 )))


###################################
#2. MÔ HÌNH HỌC MÁY

#2.1 Kỹ thuật phân lớp bằng cây quyết định
from sklearn.tree import DecisionTreeClassifier

#huấn luyện
dtree_model = DecisionTreeClassifier(criterion='gini',random_state = 100, max_depth=2,min_samples_leaf=5).fit(X_train, y_train)
#dự đoán
dtree_predictions = dtree_model.predict(X_test)

#2.2. Kỹ thuật SVM
from sklearn.svm import SVC
#huấn luyện
svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
#dự đoán
svm_predictions = svm_model_linear.predict(X_test)

# độ chính xác  mô hình cho y_test
accuracy = svm_model_linear.score(X_test, y_test)


#3 ĐÁNH GIÁ:
# đánh giá bằng ma trận sai lệch
print("Confusion_matrix: ")
print(confusion_matrix(y_test, dtree_predictions))
# print("Classification report: ")
# print(classification_report(y_test, dtree_predictions))
#độ chính xác
print("Độ chính xác kỹ thuật cây quyết định: ",  accuracy_score(y_test, dtree_predictions)*100 ,"%")

# # đánh giá bằng classification_report
print("Confusion_matrix: ")
print(confusion_matrix(y_test, svm_predictions))
# print("Classification report: ")
# print(classification_report(y_test, svm_predictions))
print("Độ chính xác kỹ thuât SVM : ", accuracy_score(y_test, svm_predictions)*100, "%")

