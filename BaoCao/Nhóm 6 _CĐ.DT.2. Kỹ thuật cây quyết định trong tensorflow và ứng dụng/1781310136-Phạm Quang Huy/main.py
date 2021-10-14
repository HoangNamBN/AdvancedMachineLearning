import tensorflow as tf
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
# Loading dataset
mushrooms_dataset = pd.read_csv('mushrooms.csv')
print(mushrooms_dataset.head())
# View amount of samples and features
print(mushrooms_dataset.shape)
# Rename Classes of Mushrooms
mushrooms_dataset["class"] = mushrooms_dataset["class"].map({
   "e": "edible",
   "p": "poisonous"
})

# View information about dataset (amount of null values, types of columns, etc...)
print(mushrooms_dataset.info())
# Plotting counts of mushrooms for one class
ax = sns.countplot(x="class", data=mushrooms_dataset)
ax.set(title="Counts of classes", )
plt.show()

ax = sns.countplot(x="cap-color", data=mushrooms_dataset)
ax.set(title="Values of classes")
plt.show()
# Labels Extraction and Binarization
labels = mushrooms_dataset.pop("class")
labels = LabelBinarizer().fit_transform(labels)
# Spliting dataset for train and test
# You can see that I choose dividing our dataset 80%/20% because it is a popular choice and we needn't a big test dataset
train_data, test_data, train_labels, test_labels = train_test_split(mushrooms_dataset, labels, test_size=0.2)
# Dataset Preprocessing
# Creating instance of OneHotEncoder and fit it

preprocessor = OneHotEncoder()
preprocessor.fit(mushrooms_dataset)
# Make OneHotEncoding with our train and test data
train_data = preprocessor.transform(train_data)
test_data = preprocessor.transform(test_data)

# Building model
# I have tried to make the model easy and in same time model, which is the most generalized.
def build_model():
   model = Sequential([
       Dense(16, activation="relu", input_shape=(train_data.shape[1],)),
       Dense(32, activation="relu", input_shape=(train_data.shape[1],)),
       Dropout(0.2),
       Dense(1, activation="sigmoid")
   ])

   lr = 0.01
   adam = tf.keras.optimizers.Adam(lr=lr)
   model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

   return model


model = build_model()
# Model training
epochs = 10
model_history = model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels))
# Accuracy
sns.lineplot(x=range(1, epochs + 1), y=model_history.history["accuracy"], label="Train Accuracy").set_title(
   "Model's accuracies")
plt.show()
sns.lineplot(x=range(1, epochs + 1), y=model_history.history["val_accuracy"], label="Test Accuracy")
plt.show()
# Loss
sns.lineplot(x=range(1, epochs + 1), y=model_history.history["loss"], label="Train Loss").set_title("Model's losses")
plt.show()
sns.lineplot(x=range(1, epochs + 1), y=model_history.history["val_loss"], label="Test Loss")
plt.show()
# comfortable predicting
def predict(x):
   x = preprocessor.transform([x])
   predicted = model.predict(x)
   return predicted


predicted = predict(mushrooms_dataset.values[0])[0, 0]
label = labels[0, 0]
print(f"{predicted} - {label}")