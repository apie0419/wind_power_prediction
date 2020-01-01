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

learning_rate = 0.001
training_steps = 1000
input_rows = 24
batch_size = 315
display_step = 2

min = tf.constant(0.0, dtype=tf.float32)
max = tf.constant(5583.0, dtype=tf.float32)

num_input = 3 * input_rows
num_hidden = [5, 5, 5]
num_classes = 1

train_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))

train_data_list = train_df.values
test_data_list  = test_df.values

train_data = list()
test_data = list()
train_target = list()
test_target = list()

temp_row = list()

for i in range(len(train_data_list) - (input_rows - 1)):
    for j in range(input_rows):
        temp_row += list(train_data_list[i + j])
    
    train_data.append(temp_row)
    temp_row = list()

temp_row = list()

for i in range(len(test_data_list) - (len(test_data_list) % input_rows)):
    for j in range(input_rows):
        temp_row += list(test_data_list[i + j])
    test_data.append(temp_row)
    temp_row = list()

train_target = train_label_df.values[input_rows-1:]
test_target = test_label_df.values[input_rows-1:]


trainset = tf.data.Dataset.from_tensor_slices((train_data, train_target))
testset = tf.data.Dataset.from_tensor_slices((test_data, test_target))

trainset = trainset.shuffle(1000)

weights = {
    "w1": tf.Variable(tf.random_normal([num_input, num_hidden[0]]), dtype=tf.float32),
    "w2": tf.Variable(tf.random_normal([num_hidden[0], num_hidden[1]]), dtype=tf.float32),
    "w3": tf.Variable(tf.random_normal([num_hidden[1], num_hidden[2]]), dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([num_hidden[2], num_classes]), dtype=tf.float32)
}
biases = {
    'b1': tf.Variable(tf.random_normal([num_hidden[0]]), dtype=tf.float32),
    'b2': tf.Variable(tf.random_normal([num_hidden[1]]), dtype=tf.float32),
    'b3': tf.Variable(tf.random_normal([num_hidden[2]]), dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([num_classes]), dtype=tf.float32)
}

def ANN(x, weights, biases):
    fc1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    fc2 = tf.add(tf.matmul(fc1, weights['w2']), biases['b2'])
    fc3 = tf.add(tf.matmul(fc2, weights['w3']), biases['b3'])
    out = tf.matmul(fc1, weights['out']) + biases['out']
    activate = tf.math.tanh(out)
    return activate

def loss_function(x, y):

    logits = ANN(x, weights, biases)

    denorm_x = denorm(logits, min, max)

    denorm_y = denorm(y, min, max)

    return tf.losses.mean_squared_error(denorm_x, denorm_y)

    # return tf.losses.mean_squared_error(logits, y)

def denorm(x, min, max):

    dn = x * (max - min) + min

    return dn

def rmse(x, y):

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, x))))


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

with tf.device("/gpu:0"):
    
    for step in range(1, training_steps+1):
        train = trainset.batch(batch_size, drop_remainder=True)
        
        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            optimizer.minimize(lambda: loss_function(batch_x, batch_y))
            
        rmse_list = list()
        mse_list  = list()


        if step % display_step == 0 or step == 1:
            
            mse_loss = loss_function(batch_x, batch_y)

            logits = ANN(batch_x, weights, biases)

            denorm_x = denorm(logits, min, max)

            denorm_y = denorm(batch_y, min, max)

            rmse_loss = rmse(denorm_x, denorm_y)

            mse_list.append(mse_loss)
            rmse_list.append(rmse_loss)
            
            result_rmse = np.array(rmse_list).mean()
            result_mse = np.array(mse_list).mean()

            print("Step " + str(step) + ", Minibatch MSE Loss= {:.4f}, RMSE Loss= {:.4f}".format(result_mse, result_rmse))
            

    print("Optimization Finished!")
    
    test = testset.batch(batch_size, drop_remainder=True)
    for batch_x, batch_y in tfe.Iterator(test):
        batch_x = tf.reshape(batch_x, (batch_size, num_input))
        batch_x = tf.dtypes.cast(batch_x, tf.float32)
        batch_y = tf.dtypes.cast(batch_y, tf.float32)
        loss = loss_function(batch_x, batch_y)
        logits = ANN(batch_x, weights, biases)

        denorm_x = denorm(logits, 0.0, 5583.0)
        denorm_y = denorm(batch_y, 0.0, 5583.0)

        rmse_loss = rmse(denorm_x, denorm_y)
        print("Testing Loss:", loss.numpy())
        print("Testing RMSE:", rmse_loss.numpy())
