import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def WineClassfication(file_csv, classification_properties, noise):
    df = pd.read_csv(file_csv)
    print("Hiển thị 5 mẫu dữ liệu của file: \n", df.head())
    X_noise = df.drop([noise], axis=1)
    X = X_noise.drop([classification_properties], axis=1)
    y = df[classification_properties]
    print("Dữ liệu X: \n", X)
    print("Dữ liệu Y: \n", y)

    '''Ý 1: chia tập dữ liệu thành D_train và D_test'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Tổng dữ liệu có được: ", X.shape)
    print("Dữ liệu X_test:\n", X_test.shape)
    print("Dữ liệu X_Train:\n", X_train.shape)

    '''Ý 2: dùng PCA để giảm chiều dữ liệu'''
    std = StandardScaler()
    X = std.fit_transform(X)
    print("X sau khi được chuẩn hoá:\n", X)

    print("Dữ liệu trước khi sử dụng PCA: ", X.shape)
    X = PCA(3).fit_transform(X)
    print("Dữ liệu sau kho giảm chiều: ", X.shape)

    models = SVC(kernel='linear', C=1).fit(X_train, y_train)
    y_predict = models.predict(X_test)
    print("Kết quả dự đoán:\n", y_predict)

    '''Đánh giá mô hình học dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)'''
    print("Accuracy Score: ", accuracy_score(y_test, y_predict))

if __name__ == "__main__":
    WineClassfication("../Data/wine.csv", "Wine","Proline")