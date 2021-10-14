
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
df_flag = pd.read_csv("datagoc.csv")
print(df_flag)
df_flag = df_flag.dropna()
print(df_flag)
X = df_flag.drop(["quality"], axis = 1)
print(X)
y = df_flag["quality"]
print(y)
X_dummy = pd.get_dummies(X);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred_logis = model.predict(X_test)
accuracy_logi = accuracy_score(y_test, y_pred_logis)
print('Accuracy logistic: %.2f' % (accuracy_logi * 100))
print(classification_report(y_test, y_pred_logis))



# print(" ket qua sau khi can bang")
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state = 2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
# model = LogisticRegression(max_iter=1000).fit(X_train_res, y_train_res)
# y_pred_logis = model.predict(X_test)
# accuracy_logi = accuracy_score(y_test, y_pred_logis)
# print('Accuracy logistic: %.2f' % (accuracy_logi * 100))
# print(classification_report(y_test, y_pred_logis))


