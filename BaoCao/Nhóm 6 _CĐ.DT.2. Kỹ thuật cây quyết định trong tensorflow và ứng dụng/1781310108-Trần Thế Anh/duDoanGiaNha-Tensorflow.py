from __future__ import print_function

# Ignore all GPUs (current TF GBDT does not support GPU).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Dataset parameters.
num_features = 79 

# Training parameters.
max_steps = 100
batch_size = 256
learning_rate = 1.0
l1_regul = 0.0
l2_regul = 0.1

# GBDT parameters.
num_batches_per_layer = 1
num_trees = 10
max_depth = 4

df =pd.read_csv('train.csv')
df=df.fillna(df.mean())
X=df.drop(['SalePrice','Id'], axis = 1)
y=df['SalePrice']
X_dummy = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, random_state=1)
features=list(X_dummy.columns)
fc = tf.feature_column
def one_hot_cat_column(feature_name, vocab):
  return fc.indicator_column(
      fc.categorical_column_with_vocabulary_list(feature_name,vocab))
feature_columns = []
for feature_name in features:
  feature_columns.append(fc.numeric_column(feature_name,dtype=tf.float32))
NUM_EXAMPLES = len(y_train)
def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    dataset = (dataset
      .repeat(n_epochs)
      .batch(NUM_EXAMPLES))
    return dataset
  return input_fn
train_input_fn = make_input_fn(X_train, y_train)
test_input_fn =make_input_fn(X_test, y_test)
gbdt_regressor = tf.estimator.BoostedTreesRegressor(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns, 
    n_trees=num_trees,
    max_depth=max_depth,
    center_bias=True
)
gbdt_regressor.train(train_input_fn, max_steps=max_steps)  
result=gbdt_regressor.evaluate(test_input_fn,steps=100)
print(result)