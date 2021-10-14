import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#
import cntk as C
from cntk.ops import *
from cntk import default_options, input_variable

data = pd.read_csv("heart_disease.csv", sep=",") # Chu y dau ,
data.head()
print(data.shape)
print(data)
X_data = data.iloc[:, :13].values
print("FEATURE")
print(X_data)
Y = data["num"].values
Y_data = Y[:, None]
print("LABEL")
print(Y_data)
# Phân loại dữ liệu train, test
input_dim = 13
num_output_classes = 2

# Ensure that we always get the same results

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = Y_data
    # Make sure that the data is separable
    X = X_data
    # Specify the data type to match the input variable used later in the tutorial
    # (default type is double)
    X = X.astype(np.float32)

    # convert class 0 into the vector "1 0 0",
    # class 1 into the vector "0 1 0", ...
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


# Create the input variables denoting the features and the label data. Note: the input
# does not need additional info on the number of observations (Samples) since CNTK creates only
# the network topology first
train_size = 250 # Lấy 250 mẫu trong data để train
features, labels = generate_random_data_sample(train_size, input_dim, num_output_classes)

# Plot the data
colors = ['r' if label == 0 else 'b' for label in labels[:, 0]]
plt.scatter(features[:, 0], features[:, 1], c=colors)
plt.xlabel("Gia tri X")
plt.ylabel("Gia tri Y")
plt.show()

feature = C.input_variable(input_dim, np.float32)
print("Feature")
print(input_dim)
print(feature)

# Define a dictionary to store the model parameters
mydict = {}


def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    weight_param = C.parameter(shape=(input_dim, output_dim))
    bias_param = C.parameter(shape=(output_dim))

    mydict['w'], mydict['b'] = weight_param, bias_param

    return C.times(input_var, weight_param) + bias_param


output_dim = num_output_classes
z = linear_layer(feature, output_dim)

label = C.input_variable(num_output_classes, np.float32)
loss = C.cross_entropy_with_softmax(z, label)

eval_error = C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])


# Define a utility function to compute the moving average.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# Define a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

# Initialize the parameters for the trainer
minibatch_size = 25
num_samples_to_train = 20000
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

from collections import defaultdict

# Run the trainer and perform model training
training_progress_output_freq = 50
plotdata = defaultdict(list)

for i in range(0, num_minibatches_to_train):
    features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)
    # Assign the minibatch data to the input variables and train the model on the minibatch
    trainer.train_minibatch({feature: features, label: labels})
    batchsize, loss, error = print_training_progress(trainer, i,
                                                     training_progress_output_freq, verbose=1)


# Run the trained model on a newly generated dataset
test_minibatch_size = X_data.shape[0] - train_size # Lấy dữ liệu còn lạ để test
features, labels = generate_random_data_sample(test_minibatch_size, input_dim, num_output_classes)

trainer.test_minibatch({feature: features, label: labels})

out = C.softmax(z)
result = out.eval({feature: features})

# Model parameters
print("Hệ số: ",mydict['w'].value)
print("Bias: ", mydict['b'].value)

Y_test = [np.argmax(label) for label in labels]
Y_pre = [np.argmax(x) for x in result]
print("Giá trị Y test   :", Y_test)
print("Giá trị Y dự đoán:", Y_pre)

Y_test = np.array(Y_test)
Y_pre = np.array(Y_pre)
num_Y_true = 0
for i in range(0, Y_pre.shape[0]):
    if(Y_pre[i] == Y_test[i]):
        num_Y_true += 1
accuracy = (num_Y_true/Y_test.shape[0]) * 100
print("ACCURACY: ", accuracy, "%")
matrix = confusion_matrix(y_true=Y_test, y_pred=Y_pre)
print("CONFUSION MATRIX")
print(matrix)
print("CLASSFICATION REPORT")
print(classification_report(y_true=Y_test, y_pred=Y_pre))



