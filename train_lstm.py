from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.contrib.eager.python import tfe

gpu_option = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_option)

tfe.enable_eager_execution(config=config)

tf.logging.set_verbosity(tf.logging.ERROR)

base_path = os.path.dirname(os.path.abspath(__file__))

learning_rate = 0.001
training_epochs = 200
batch_size = 315
display_step = 5


input_rows = 8
num_input = 3 * input_rows
timestamps = 3

num_hidden = 256
num_classes = 1
keep_prob = 0.7

data_path = os.path.join(base_path, "data/2")

train_df = pd.read_excel(os.path.join(data_path, "hour_ahead/train_in.xlsx"))
train_label_df = pd.read_excel(os.path.join(data_path, "hour_ahead/train_out.xlsx"))
test_df = pd.read_excel(os.path.join(data_path, "hour_ahead/test_in.xlsx"))
test_label_df = pd.read_excel(os.path.join(data_path, "hour_ahead/test_out.xlsx"))
min_max = pd.read_excel(os.path.join(data_path, "hour_ahead/max_min.xls"))

_min = tf.constant(min_max["pmin"][0], dtype=tf.float32)
_max = tf.constant(round(min_max["pmax"][0], 2), dtype=tf.float32)

trainset_list = train_df.values
testset_list  = test_df.values
trainlabel = train_label_df.values[timestamps * input_rows - 1:]
testlabel = test_label_df.values[timestamps * input_rows - 1:]

trainset = list()
testset  = list()

for i in range(len(trainset_list) - (timestamps * input_rows - 1)):
    temp = list()
    for j in range(timestamps * input_rows):
        temp.extend(trainset_list[i + j])

    trainset.append(temp)


for i in range(len(testset_list) - (timestamps * input_rows - 1)):
    temp = list()
    for j in range(timestamps * input_rows):
        temp.extend(testset_list[i + j])

    testset.append(temp)

trainset = tf.data.Dataset.from_tensor_slices((trainset, trainlabel))
testset = tf.data.Dataset.from_tensor_slices((testset, testlabel))

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]), dtype=tf.float32)
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]), dtype=tf.float32)
}

def RNN(x, weights, biases):

    x = tf.unstack(x, timestamps, axis=1)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, name='basic_lstm_cell')

    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    rnn_outputs, states = rnn.static_rnn(dropout_cell, x, dtype=tf.float32)

    outputs = tf.matmul(rnn_outputs[-1], weights['out']) + biases['out']

    activate = tf.nn.relu(outputs)

    return activate

def loss_function(x, y):

    logits = RNN(x, weights, biases)

    denorm_x = denorm(logits, _min, _max)

    denorm_y = denorm(y, _min, _max)

    return tf.losses.mean_squared_error(denorm_x, denorm_y)

    # return tf.losses.mean_squared_error(logits, y)

def denorm(x, min, max):

    return x * (max - min) + min

def rmse(x, y):

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, x))))

                        
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


with tf.device("/gpu:1"):
    
    epochs = list(range(display_step, training_epochs + display_step, display_step))
    losses = list()

    for step in range(1, training_epochs+1):
        train = trainset.batch(batch_size, drop_remainder=True)
        
        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, timestamps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            
            optimizer.minimize(lambda: loss_function(batch_x, batch_y))

        if step % display_step == 0:
            
            batch_x = tf.reshape(batch_x, (batch_size, timestamps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            mse_loss = loss_function(batch_x, batch_y)

            logits = RNN(batch_x, weights, biases)

            denorm_x = denorm(logits, _min, _max)

            denorm_y = denorm(batch_y, _min, _max)

            rmse_loss = rmse(denorm_x, denorm_y)

            losses.append(rmse_loss)
            print("Step " + str(step) + ", Minibatch MSE Loss= {:.4f}, RMSE Loss= {:.4f}".format(mse_loss, rmse_loss))
    
    fig, ax = plt.subplots()
    ax.plot(epochs, losses)
    ax.set(xlabel='epochs', ylabel='loss (RMSE)')
    ax.grid()
    fig.savefig(os.path.join(base_path, "Output/lstm_loss.png"))


    print("Optimization Finished!")
    # rmse_list = list()
    predict = list()
    target  = list()
    test = testset.batch(batch_size, drop_remainder=True)
    for batch_x, batch_y in tfe.Iterator(test):
        batch_x = tf.reshape(batch_x, (batch_size, timestamps, num_input))
        batch_x = tf.dtypes.cast(batch_x, tf.float32)
        batch_y = tf.dtypes.cast(batch_y, tf.float32)
        loss = loss_function(batch_x, batch_y)
        logits = RNN(batch_x, weights, biases)

        denorm_x = denorm(logits, _min, _max)

        denorm_y = denorm(batch_y, _min, _max)

        rmse_loss = rmse(denorm_x, denorm_y)
        print("Testing Loss:", loss.numpy())
        print("Testing RMSE:", rmse_loss.numpy())

        for p in list(denorm_x.numpy()):
            predict += list(p)
        for t in list(denorm_y.numpy()):
            target += list(t)


    data = pd.DataFrame({
        "predict": predict,
        "target": target
    })

    data.plot()
    plt.savefig(os.path.join(base_path, "Output/lstm_evaluation.png"))