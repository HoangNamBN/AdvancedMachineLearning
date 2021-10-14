import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('mushroom_dataset.csv')
# print(df.head())


# kiểm tra có dữ liệu null không và xóa đi
df = df.dropna()

# drop cột này vì có quá nhiều dữ liệu thiếu
df.drop(['stalk_root'], axis=1, inplace=True)
# df = df.fillna(df.mean())

df.to_csv('mushroom_update.csv', index=False)

print('Dữ liệu sau khi tiền xử lý')
print(df)
#
# # Show dư lieu
import seaborn as sns
# sns.countplot(x='y', data=df)
# plt.show()

# # in các giá trị xuất hiện trong 1 cột ra
# val_x = df.habitat.value_counts()
# print(val_x)


# In cot nhan ra xem gia tri
# import seaborn as sns
# sns.countplot(x='cap_color', data=df)
# plt.title('Màu nắp')
# plt.show()


df = df.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))
print('-'*80)
print('Dữ liệu sau khi chuyển từ categorical về số')
print(df)

X = df.iloc[:, 1:-1]
# Hiện thị điểm dữ liệu X (dòng số ... của X) và nhãn y tương ứng.
# print(X)
# X[0:20]

y = df.iloc[:, -1]
print('y',y)

# x1 = df.iloc[:,0:1]
# # print('x1',x1)
# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,10))
# ax[0,0].plot(x1,'go', color='purple')
# # ax[0,1].set_xlabel(df.mushroom)
#
# ax[0,0].set_title('square')
# ax[1,0].set_title('Cubes')
# plt.show()


# Khi khong su dung smote
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
print('-'*80)
print('Tập dữ liệu train: ', x_train.shape, y_train.shape)
print('Tập dữ liệu test:', x_test.shape, y_test.shape)

model = LogisticRegression(max_iter=10000).fit(x_train, y_train)

print('-'*60)
print("Before OverSampling, counts of label 0: {}".format(sum(y == 0)))
print("Before OverSampling, counts of label 1: {}".format(sum(y == 1)))
print('-'*60)
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_sample(x_train, y_train)
print("After OverSampling, counts of label 0: {}".format(sum(y_sm == 0)))
print("After OverSampling, counts of label 1: {}".format(sum(y_sm == 1)))




model_sm = LogisticRegression(max_iter=10000).fit(X_sm, y_sm)

# dự đoán nhãn dựa trên mô hình đã được đào tạo
prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)
print('-'*60)
print("Logistic regression before smote")
print(classification_report(y_test, prediction))


pre_sm = model_sm.predict(x_test)
acc_sm = accuracy_score(y_test, pre_sm)
print('-'*60)
print("Logistic regression after smote")
print(classification_report(y_test, pre_sm))


#---------------- SVM -------------
clf = SVC()

# clf.fit(x_train, y_train)
# pred = clf.predict(x_test)
# print('-'*70)
# print("SVM no using smote")
# print(classification_report(y_test, pred))

clf.fit(X_sm, y_sm)
pred_sm = clf.predict(x_test)
# a = accuracy_score(y_test, pred)
print('-'*60)
print("SVM using smote")
print(classification_report(y_test, pred_sm))



# -----------Random forest classification-------------

rd_forest = RandomForestClassifier(n_estimators=250, max_depth=25, class_weight='balanced', n_jobs=-1)
rd_forest.fit(X_sm, y_sm)
pre_rd = rd_forest.predict(x_test)
# b = accuracy_score(y_test, pre_rd)
print('-'*60)
print("Randomforest using smote")
print(classification_report(y_test, pre_rd))



# ------------------ DecisionTree Classifier -----------------------

dt_classifier = DecisionTreeClassifier(criterion='gini', random_state=400, max_depth=12, min_samples_leaf=5)
dt_classifier.fit(X_sm, y_sm)
pre_tree = dt_classifier.predict(x_test)
# c = accuracy_score(y_test, pre_tree)
print('-'*60)
print("DecisionTreeClassifier using smote")
print(classification_report(y_test, pre_tree))
