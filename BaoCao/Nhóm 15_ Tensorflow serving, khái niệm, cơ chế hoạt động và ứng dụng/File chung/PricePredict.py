import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import tempfile

df=pd.read_csv('laptop.csv')
labels=df['PriceUSD']
features=df.drop('PriceUSD',axis=1)
#Chuyển các trường chữ thành số
features=pd.get_dummies(features,drop_first=True)

#Phân tách dữ liệu thành dữ liệu đào tạo và dữ liệu kiểm tra
X_train,X_test,y_train,y_test=train_test_split(features,labels,train_size=0.8,test_size=0.2,random_state=0)

#Chuẩn hóa dữ liệu bằng MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)

#Xây dựng mạng neuron

num_hidden_layers=int(input('Nhap so luong hidden layer:'))
num_neuron_first_hidden_layer=int(input('Nhap so luong neuron lop hidden thu nhat:'))
model = Sequential([Dense(num_neuron_first_hidden_layer, activation='relu', input_shape=(features.shape[1],))])
i=2
num_neuron_previous_layer=num_neuron_first_hidden_layer
while (i<=num_hidden_layers):
    num_neuron=int(input('Nhap so luong neuron lop hidden thu '+str(i) + ':'))
    model.add(Dense(num_neuron,input_dim = num_neuron_previous_layer,activation="relu"))
    num_neuron_previous_layer=num_neuron
    i=i+1
model.add(Dense(1,activation = "linear"))
num_epoch=int(input('Nhap so luong vong lap training:'))

#model.compile(optimizer='adam',loss='mean_squared_error')
model.compile(optimizer='adam',loss='mean_absolute_error')



hist=model.fit(X_train,y_train,epochs=num_epoch)
#Lưu lại các tham số của mô hình
tmpdir = tempfile.mkdtemp()
save_path = os.path.join(os.getcwd(),'laptop_model/1/')
print('Mo hinh da duoc luu vao thu muc ' + save_path + ' ,ban hay su dung de dua len tensorflow serving')
tf.saved_model.save(model,save_path)

pred=model.predict(X_test)
print('10 gia tri du doan dau tien')
print(pred[:10].tolist())
print('10 gia tri thuc te dau tien')
print(list(y_test[:10]))
print('Gia tri MAE tren tap du lieu test la: ' + str(mean_absolute_error(y_test,pred)))

