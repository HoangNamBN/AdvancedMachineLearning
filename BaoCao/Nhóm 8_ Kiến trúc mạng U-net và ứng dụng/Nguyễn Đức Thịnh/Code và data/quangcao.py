
import pandas as pd
from imblearn.over_sampling import SMOTE

#Bước 1: Đọc file và tiền xử lí dữ liệu
from sklearn import preprocessing
df = pd.read_csv("advertising.csv")
# objList = df.select_dtypes(include = "object").columns #lấy thông tin các cột object để xử lí
# print(objList)
# for cols in objList:
#     df[cols] = preprocessing.LabelEncoder().fit_transform(df[cols].astype(str))
print('-------------------Xử lý dữ liệu categorical--------------------------')
X= df.drop(["Clicked on Ad","Timestamp","City","Country","Ad Topic Line"], axis = 1)
y = df["Clicked on Ad"]
print('Dữ liệu sau khi xử lý:')
print(X)
print (X.shape)

print('---------------------Kiểm tra missing value---------------------------')
print(df.isnull().sum()) #Kiểm tra missing value

print('-------------------------Chia dữ liệu train-test----------------------')
from sklearn.model_selection import train_test_split #chia tập dữ liêu train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)
print (X_train.head(5))
print (y_train.head(5))
print('--------------------------------------------------------')
print (X_test.head(5))
print (y_test.head(5))

print('---------------------Cân bằng SMOTE---------------------')
#Cân bằng SMOTE
print('Before OverSampling:')
print('No Clicked On Ads =', sum(y_train == 0))
print('Clicked On Ads =', sum(y_train == 1))
print('Dữ liệu đã được cân bằng')
# sm = SMOTE(random_state=2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
# print("After OverSampling, counts of No Clicked On Ads: {}".format(sum(y_train_res == 0)))
# print("After OverSampling, counts of Clicked On Ads: {}".format(sum(y_train_res == 1)))

print('--------------------Chạy kỹ thuật học máy----------------')
#Bước 2: Chạy Kỹ thuật học máy Logistic Regression
from sklearn.linear_model import LogisticRegression
#Bước 2.1: Huấn luyện (với tập dữ liệu train X_train, y_train)
model = LogisticRegression(max_iter = 10000).fit(X_train,y_train)
#Bước 2.2: Dự đoán (với tập dữ liệu train X_test, y_test)
y_pred = model.predict(X_test) #y_prediction là y dự đoán được
print ("Hệ số w:", model.coef_)
print (model.coef_.shape)
print ("Hệ số bias:", model.intercept_)
print ("Số lớp:", model.classes_)
#print ("Số vòng lặp", log_model.n_iter_)
print ("Tap y du doan:")
print (y_pred)

print('------------------Đánh giá mô hình học máy----------------')
#Bước 3: Đánh giá mô hình học dựa trên kết quả dự đoán
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy * 100))
print(classification_report(y_test,y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cnf_matrix)

print('--------------------------------------------------------')
print('--------------------------------------------------------')





