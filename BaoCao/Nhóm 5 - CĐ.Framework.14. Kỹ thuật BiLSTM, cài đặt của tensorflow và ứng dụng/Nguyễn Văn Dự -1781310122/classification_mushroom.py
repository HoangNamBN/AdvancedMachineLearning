import pandas as pd


df_mushroom = pd.read_csv("agaricus-lepiota.csv")
print("data frame mushroom\n", df_mushroom)

# Loại bỏ các giá Na, NaN
df_mushroom_new = df_mushroom.dropna()
print("data frame mushroom new\n", df_mushroom_new)
#
# Gán X bằng các cột 9am của df_weather.
X = df_mushroom_new.drop(["poisonous-or-edible"], axis=1)
print("X là:\n", X)

# Đưa X về dạng số
X_dummy = pd.get_dummies(X)
print("X sau khi đưa về dạng số \n", X_dummy)

# Gán y bằng cột poisonous-or-edible của df_mushroom_new.
y = df_mushroom_new["poisonous-or-edible"]
print("y là:\n", y)

# Chia tập X, y theo tỉ lệ 50-50: X_train, y_train và X_test, y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.5, random_state=1)

print("Số nhãn edible khi chưa cân bằng: {}".format(sum(y_train == 'e')))

print("Số nhãn poisonous khi chưa cân bằng: {}".format(sum(y_train == 'p')))

print("-----------------------------------DecisonTreeClassifier-----------------------------------\n")
# Tạo mô hình học máy, sử dụng kỹ thuật cây quyết định Decision Tree với độ đo 'gini'
from sklearn.tree import DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=2)

# Huấn luyện mô hình với X_train, y_train
clf_gini.fit(X_train, y_train)

# 9. Dự đoán mô hình  với X_test, y_test
y_pred_gini = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

accuracy_dec = accuracy_score(y_test, y_pred_gini)
print("Accuracy decisontree :%.2f " % (accuracy_dec * 100))
print("gini classification_report \n", classification_report(y_test, y_pred_gini))

print("-----------------------------------LogisticRegression-----------------------------------\n")
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000).fit(X_train, y_train)
y_pred_logis = model.predict(X_test)
accuracy_logi = accuracy_score(y_test, y_pred_logis)
print('Accuracy logistic: %.2f' % (accuracy_logi * 100))
print(classification_report(y_test, y_pred_logis))

print("------------------------------------Cân bằng dữ liệu-------------------------------------\n")
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print("số nhãn edible sau khi cân bằng: {}".format(sum(y_train_res == 'e')))

print("số nhãn poisonous sau khi cân bằng: {}".format(sum(y_train_res == 'p')))

print("-----------------------------------DecisonTreeClassifier-----------------------------------\n")

# 8. Huấn luyện mô hình
clf_gini.fit(X_train_res, y_train_res)

# 9. Dự đoán mô hình  với X_test, y_test
y_pred_gini = clf_gini.predict(X_test)
accuracy_dec = accuracy_score(y_test, y_pred_gini)
print("Accuracy decisontree sau khi cân bằng :%.2f " % (accuracy_dec * 100))
print("gini classification_report \n", classification_report(y_test, y_pred_gini))

print("-----------------------------------LogisticRegression-----------------------------------\n")

model = LogisticRegression(max_iter=10000).fit(X_train_res, y_train_res)
y_pred_logis = model.predict(X_test)
accuracy_logi = accuracy_score(y_test, y_pred_logis)
print('Accuracy logistic sau khi cân bằng: %.2f' % (accuracy_logi * 100))
print(classification_report(y_test, y_pred_logis))
