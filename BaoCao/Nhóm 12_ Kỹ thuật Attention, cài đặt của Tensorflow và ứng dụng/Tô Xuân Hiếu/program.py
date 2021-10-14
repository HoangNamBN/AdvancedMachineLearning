import numpy as np
import pandas as pd
from subprocess import check_output 
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers.recurrent import LSTM 
from keras.models import Sequential 
from sklearn.model_selection import train_test_split 
import time
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from numpy import newaxis

file_name ='GSPC.csv' 
prices_dataset = pd.read_csv(file_name, header=0) 


#plt.plot(prices_dataset.Open.values, color='red', label='open') 
#plt.plot(prices_dataset.Close.values, color='green', label='close')
#plt.plot(prices_dataset.Low.values, color='blue', label='low')
#plt.plot(prices_dataset.High.values, color='black', label='high') 
#plt.title('stock price')
#plt.xlabel('time [days]') 
#plt.ylabel('price') 
#plt.legend(loc='best')
#plt.show()

prices_dataset_dropout = prices_dataset.drop(['Date','Adj Close','Volume'], 1)
#prices_dataset_tail_50 = prices_dataset.tail(50) 
#plt.plot(prices_dataset_tail_50.Open.values, color='red', label='open') 
#plt.plot(prices_dataset_tail_50.Close.values, color='green', label='close') 
#plt.plot(prices_dataset_tail_50.Low.values, color='blue', label='low') 
#plt.plot(prices_dataset_tail_50.High.values, color='black', label='high') 
#plt.title('stock price') 
#plt.xlabel('time [days]') 
#plt.ylabel('price') 
#plt.legend(loc='best') 
#plt.show()



# Scale dữ liệu
def normalize_data(df):
    min_max_scaler = MinMaxScaler() 
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1)) 
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1)) 
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1)) 
    df['Close'] = min_max_scaler.fit_transform(df.Close.values.reshape(-1,1)) 
    return df 
prices_dataset_norm = normalize_data(prices_dataset_dropout)


def generate_data(stock_ds, seq_len): 
    data_raw = stock_ds.to_numpy() 
    data = [] 
    # tạo tất cả các chuỗi có độ dài seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len]) 
    return data 
# dữ liệu dưới dạng numpy array
def generate_train_test(data_ds,split_percent=0.8): 
    print(len(data_ds)) 
    data = np.asarray(data_ds) 
    data_size = len(data) 
    train_end = int(np.floor(split_percent*data_size)) 
    x_train = data[:train_end,:-1,:] 
    y_train = data[:train_end,-1,:] 
    x_test = data[train_end:,:-1,:] 
    y_test = data[train_end:,-1,:] 
    return [x_train, y_train, x_test, y_test] 

seq_len = 20 # chọn sequence length 
seq_prices_dataset = generate_data(prices_dataset_norm,seq_len)

x_train, y_train, x_test, y_test = generate_train_test(seq_prices_dataset, 0.8)

print('x_train.shape = ',x_train.shape) 
print('y_train.shape = ', y_train.shape) 
print('x_test.shape = ', x_test.shape) 
print('y_test.shape = ',y_test.shape)


#model = Sequential() 
#model.add(LSTM( 
    #input_dim=4, 
    #output_dim=50, 
    #return_sequences=True)) 
#model.add(Dropout(0.2)) 
#model.add(LSTM( 100, 
#return_sequences=False)) 
#model.add(Dropout(0.2)) 
#model.add(Dense( output_dim=4)) 
#model.add(Activation('linear')) 
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 
#checkpoint = ModelCheckpoint(filepath='sp500.h5', verbose=1, save_best_only=True) 
#hist = model.fit(x_train, y_train, epochs=300, batch_size=128, verbose=1, callbacks=[checkpoint], validation_split=0.2)

from keras.models import load_model
model =load_model('sp500.h5') 
y_hat = model.predict(x_test) 
ft = 3 # 0 = open, 1 = highest, 2 =lowest , 3 = close 
plt.plot( y_test[:,ft], color='blue', label='target') 
plt.plot( y_hat[:,ft], color='red', label='prediction') 
plt.title('future stock prices') 
plt.xlabel('time [days]') 
plt.ylabel('normalized price') 
plt.legend(loc='best') 

from sklearn.metrics import mean_squared_error 
# 0 = open, 1 = highest, 2 =lowest , 3 = close 
print("open error: ") 
print(mean_squared_error(y_test[:,0], y_hat[ :,0])) 
print("highest error: ") 
print(mean_squared_error(y_test[:,1], y_hat[ :,1])) 
print("lowest error: ") 
print(mean_squared_error(y_test[:,2], y_hat[ :,2])) 
print("close error: ") 
print(mean_squared_error(y_test[:,3], y_hat[ :,3]))
plt.show() 
