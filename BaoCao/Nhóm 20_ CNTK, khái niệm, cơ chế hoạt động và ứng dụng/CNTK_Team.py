import warnings
warnings.filterwarnings('ignore')
import cntk
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from cntk.ops import *
from cntk import input_variable
from cntk import learning_rate_schedule, UnitType
from sklearn.decomposition import PCA
from cntk.learners import sgd
from cntk.train.trainer import Trainer
from sklearn import preprocessing

# Đọc dữ liệu
data = pd.read_csv("abalone.csv", sep=",")
data.head()
print("Dữ LIỆU TRƯỚC KHI XỬ LÝ")
print(data)
# Xử lý dữ liệu
lbEncoder = preprocessing.LabelEncoder()
data['sex_replace'] = lbEncoder.fit_transform(data['sex'])
print("DỮ LIỆU SAU KHI CHUẨN HÓA")
print(data)

Y = data['old'].values
Y = Y[:, None]
X = data.drop(["sex", "old"], axis=1).values

# Chia dữ liệu train,test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Biểu diễn dữ liệu
pca = PCA(n_components=1)
X_scat = pca.fit_transform(X_train)
plt.scatter(X_scat, Y_train)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Bieu do tuong quan giua X va Y")
plt.show()

# Define the network using CNTK primitives to define a single node with no activation function
# Số chiều input, số chiểu output
input_dim = 8
num_outputs = 1

input = input_variable(input_dim, np.float32)
label = input_variable((num_outputs), np.float32)

mydict = {"w": None, "b": None}
# Tạo mô hình
def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    weight_param = cntk.parameter(shape=(input_dim, output_dim))
    bias_param = cntk.parameter(shape=(output_dim))
    mydict['w'], mydict['b'] = weight_param, bias_param
    return cntk.times(input_var, weight_param) + bias_param


z = linear_layer(input, num_outputs)
# Setup loss and evaluation functions
loss = cntk.squared_error(z, label)
eval_error = cntk.squared_error(z, label)

learning_rate = 0.002 # Adjust according to the model
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, eval_error), [learner])

# Initialize the parameters for the trainer
iter = 3
minibatch_size = 10
num_samples_to_train = X_train.shape[0]
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

for no_iter in range(0, iter): # Adjust according to the model
    for i in range(0, num_minibatches_to_train):
        train_features = X_train[(i * minibatch_size):(i * minibatch_size + minibatch_size), :]
        train_labels = Y_train[(i * minibatch_size):(i * minibatch_size + minibatch_size), :]
        trainer.train_minibatch({input: train_features, label: train_labels})
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(i, training_loss, eval_error))

print("Hệ số: ", mydict['w'].value)
print("BIAS: ", mydict['b'].value)

Y_pre = np.dot(X_test, mydict['w'].value) + mydict['b'].value

print("Y test   :", Y_test.tolist())
print("Y predict: ", Y_pre.tolist())
test_eval_result = trainer.test_minibatch({input: X_test, label: Y_test})
print("Mean Square Error: {0:.2f}".format(test_eval_result))


