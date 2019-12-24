from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.contrib import rnn
from tensorflow.contrib.eager.python import tfe

gpu_option = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_option)

tfe.enable_eager_execution(config=config)

tf.logging.set_verbosity(tf.logging.ERROR)

base_path = os.path.dirname(os.path.abspath(__file__))

learning_rate = 0.005
training_epochs = 500
batch_size = 30
display_step = 2

input_rows = 3 # 8 筆資料作為一個 timestamp
num_input = 2 * input_rows

timestamps = 3

num_hidden = 256
num_classes = 1
drop_prob = 0.3

min = tf.constant(0.0, dtype=tf.float32)
max = tf.constant(5583.0, dtype=tf.float32)


train_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))

trainset_list = train_df.values[:, ::2]
testset_list  = test_df.values[:, ::2]

trainlabel = list()
testlabel = list()


for idx, label in enumerate(train_label_df.values):
    if idx % (input_rows * timestamps) == (input_rows * timestamps) - 1:
        trainlabel.append(label)

for idx, label in enumerate(test_label_df.values):
    if idx % (input_rows * timestamps) == (input_rows * timestamps) - 1:
        testlabel.append(label)

trainset = list()
testset  = list()

temp_row = list()
temp_data = list()

for i in range(len(trainset_list) - (len(trainset_list) % input_rows)):
    temp_row += list(trainset_list[i])
    if i % input_rows == input_rows - 1:
        temp_data.append(temp_row)
        temp_row = list()
    if i % (input_rows * timestamps) == (input_rows * timestamps) - 1:
        trainset.append(temp_data)
        temp_data = list()

temp_row = list()
temp_data = list()

for i in range(len(testset_list) - (len(testset_list) % input_rows)):
    temp_row += list(testset_list[i])
    if i % input_rows == input_rows - 1:
        temp_data.append(temp_row)
        temp_row = list()
    if i % (input_rows * timestamps) == (input_rows * timestamps) - 1:
        testset.append(temp_data)
        temp_data = list()


trainset = tf.data.Dataset.from_tensor_slices((trainset, trainlabel))
testset = tf.data.Dataset.from_tensor_slices((testset, testlabel))

trainset = trainset.shuffle(1000)

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]), dtype=tf.float32)
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]), dtype=tf.float32)
}

def RNN(x, weights, biases):

    x = tf.unstack(x, timestamps, axis=1)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.0, name='basic_lstm_cell')

    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1)

    rnn_outputs, states = rnn.static_rnn(dropout_cell, x, dtype=tf.float32)

    outputs = tf.matmul(rnn_outputs[-1], weights['out']) + biases['out']

    activate = tf.math.tanh(outputs)

    return activate

def loss_function(x, y):

    logits = RNN(x, weights, biases)

    denorm_x = denorm(logits, min, max)

    denorm_y = denorm(y, min, max)

    return tf.losses.mean_squared_error(denorm_x, denorm_y)

    # return tf.losses.mean_squared_error(logits, y)

def denorm(x, min, max):

    return x * (max - min) + min

def rmse(x, y):

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, x))))

                        
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


with tf.device("/gpu:1"):
    
    for step in range(1, training_epochs+1):
        train = trainset.batch(batch_size, drop_remainder=True)
        
        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, timestamps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            
            optimizer.minimize(lambda: loss_function(batch_x, batch_y))

        if step % display_step == 0 or step == 1:
            rmse_list = list()
            mse_list = list()

            for batch_x, batch_y in tfe.Iterator(train):
                batch_x = tf.reshape(batch_x, (batch_size, timestamps, num_input))
                batch_x = tf.dtypes.cast(batch_x, tf.float32)
                batch_y = tf.dtypes.cast(batch_y, tf.float32)
                mse_loss = loss_function(batch_x, batch_y)

                logits = RNN(batch_x, weights, biases)

                denorm_x = denorm(logits, min, max)

                denorm_y = denorm(batch_y, min, max)

                rmse_loss = rmse(denorm_x, denorm_y)

                mse_list.append(mse_loss)
                rmse_list.append(rmse_loss)
            
            result_rmse = np.array(rmse_list).mean()
            result_mse = np.array(mse_list).mean()
            print("Step " + str(step) + ", Minibatch MSE Loss= {:.4f}, RMSE Loss= {:.4f}".format(result_mse, result_rmse))


    print("Optimization Finished!")
    # rmse_list = list()
    test = testset.batch(batch_size, drop_remainder=True)
    for batch_x, batch_y in tfe.Iterator(test):
        batch_x = tf.reshape(batch_x, (batch_size, input_rows, num_input))
        batch_x = tf.dtypes.cast(batch_x, tf.float32)
        batch_y = tf.dtypes.cast(batch_y, tf.float32)
        loss = loss_function(batch_x, batch_y)
        logits = RNN(batch_x, weights, biases)

        denorm_x = denorm(logits, 0.0, 5583.0)

        denorm_y = denorm(batch_y, 0.0, 5583.0)

        rmse_loss = rmse(denorm_x, denorm_y)
        print("Testing Loss:", loss.numpy())
        print("Testing RMSE:", rmse_loss.numpy())
