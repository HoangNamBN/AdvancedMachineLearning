import warnings
warnings.filterwarnings('ignore')
import cntk
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cntk.ops import *
from cntk import input_variable
from sklearn.decomposition import PCA
import random
from sklearn import preprocessing

# Đọc dữ liệu
data = pd.read_csv("school_grades_dataset.csv", sep=",")
data.head

# In dữ liệu
print("DỮ LIỆU TRƯỚC KHI XỬ LÝ")
print(data.shape)
print(data)

# Xử lý dữ liệu
lbEncoder = preprocessing.LabelEncoder()
data['school_replace'] = lbEncoder.fit_transform(data['school'])
data['sex_repalce'] = lbEncoder.fit_transform(data['sex'])
data['address_replace'] = lbEncoder.fit_transform(data['address'])
data['famsize_replace'] = lbEncoder.fit_transform(data['famsize'])
data['Pstatus_replace'] = lbEncoder.fit_transform(data['Pstatus'])
data['Mjob_replace'] = lbEncoder.fit_transform(data['Mjob'])
data['Fjob_replace'] = lbEncoder.fit_transform(data['Fjob'])
data['reason_replace'] = lbEncoder.fit_transform(data['reason'])
data['guardian_replace'] = lbEncoder.fit_transform(data['guardian'])
data['schoolsup_replace'] = lbEncoder.fit_transform(data['schoolsup'])
data['famsup_replace'] = lbEncoder.fit_transform(data['famsup'])
data['paid_replace'] = lbEncoder.fit_transform(data['paid'])
data['activities_replace'] = lbEncoder.fit_transform(data['activities'])
data['nursery_replace'] = lbEncoder.fit_transform(data['nursery'])
data['higher_replace'] = lbEncoder.fit_transform(data['higher'])
data['internet_replace'] = lbEncoder.fit_transform(data['internet'])
data['romantic_replace'] = lbEncoder.fit_transform(data['romantic'])

data = data.drop(['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                  'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], axis=1)

# Dữ liệu sau khi xử lý
print("DỮ LIỆU SAU KHI XỬ LÝ")
print(data)
Y = data['G3'].values
Y = Y[:, None]
X = data.drop(['G3'], axis=1).values

# Phân loại dữ liệu train, test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)

# Biểu diễn dữ liệu
# Giảm chiều (PCA)
pca = PCA(n_components=1)
X_scat = pca.fit_transform(X_train)
# Plot
plt.scatter(X_scat, Y_train)
plt.xlabel("Gia tri X")
plt.ylabel("Gia tri Y")
plt.title("Bieu do tuong quan X va Y")
plt.show()

# Lấy dữ liệu ngẫu nhiên
def next_batch(x, y, batch_size, num_sample):
    index = random.sample(range(num_sample), batch_size)
    return x[index,:].astype('float64'), y[index,:].astype('float64')

#
input_dim = 32
output_dim = 1

input = input_variable(input_dim, np.float32)
label = input_variable((output_dim), np.float32)

# Tạo neural netwrok với 1 neural
def neural_network(input_var, output_dim):
    with cntk.default_options():
        output_Layer =cntk.layers.Dense(output_dim, activation=None, name='output_Layer')(input)
    model = cntk.ops.alias(output_Layer)
    return model

# Tạo model
model = neural_network(input_dim, output_dim)
loss = cntk.squared_error(model, label)
max_iter = 300
minibatch_size = 25
learning_rate = 0.001
num_samples_to_train = X_test.shape[0]
num_minibatch_to_train =int(num_samples_to_train /minibatch_size)
learner = cntk.sgd(model.parameters, learning_rate)
trainer =cntk.Trainer(model, (loss), [learner])

# Training model
for iter in range(0, num_samples_to_train):
    # Read a mini batch from training data
    batch_data, batch_label = next_batch(X_train, Y_train, batch_size=minibatch_size, num_sample=num_samples_to_train)

    arguments = {input: batch_data, label: batch_label}
    trainer.train_minibatch(arguments=arguments)
    if(iter % int(max_iter /10)):
        mse = trainer.previous_minibatch_loss_average
        print("Batch %6d: mean squared error = %8.4f" % (iter, mse))

# Dự đoán với dữ liệu test
y_pred = []
for x in X_test:
    Y_pre = model.eval(x)
    y_pred.append(Y_pre)
y_pred = np.array(y_pred)
y_pred = y_pred[:, 0]
print("GIÁ TRỊ Y DỰ ĐOÁN:", y_pred.tolist())
print("GIÁ TRỊ Y THỰC TẾ:", Y_train.tolist())
MSE = 0.0
for i in range(0, y_pred.shape[0]):
    MSE += ((1/y_pred.shape[0])*(Y_test[i] - y_pred[i])**2)
print("Mean Square Error: {0:.5f}".format(float(MSE)))