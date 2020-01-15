from matplotlib import pyplot as plt
from tensorflow.contrib.eager.python import tfe
from tcn import TCN
import os
import pandas as pd
import numpy as np
import tensorflow as tf


tf.enable_eager_execution()

base_path = os.path.dirname(os.path.abspath(__file__))

batch_size    = 32
hidden_units  = 25
learning_rate = 0.001
dropout       = 0.05
epochs        = 20
ksize         = 7
levels        = 8
n_classes     = 1
timesteps    = 24
num_input     = 3

_min = tf.constant(0.0, dtype=tf.float32)
_max = tf.constant(5583.0, dtype=tf.float32)


train_data_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_target_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_data_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_target_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))


train_data_list = train_data_df.values
test_data_list  = test_data_df.values


train_data = list()
test_data = list()
train_target = list()
test_target = list()

temp_row = list()

for i in range(len(train_data_list) - (timesteps - 1)):
    for j in range(timesteps):
        temp_row.append(list(train_data_list[i + j]))
    
    train_data.append(temp_row)
    temp_row = list()

temp_row = list()

for i in range(len(test_data_list) - (len(test_data_list) % timesteps)):
    for j in range(timesteps):
        temp_row.append(list(test_data_list[i + j]))
    test_data.append(temp_row)
    temp_row = list()

train_target = train_target_df.values[timesteps-1:]
test_target = test_target_df.values[timesteps-1:]

trainset = tf.data.Dataset.from_tensor_slices((train_data, train_target))
testset = tf.data.Dataset.from_tensor_slices((test_data, test_target))

channel_sizes = [hidden_units] * levels
optimizer = tf.train.AdamOptimizer(learning_rate)
model = TCN(n_classes, channel_sizes, kernel_size=ksize, dropout=dropout)

def loss_function(x, y, training):

    logits = model(x, training=training)

    denorm_x = denorm(logits, _min, _max)

    denorm_y = denorm(y, _min, _max)

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, x))))

def denorm(x, min, max):

    return x * (max - min) + min

losses = list()

with tf.device("/gpu:1"):

    for epoch in range(1, epochs + 1):
        train = trainset.batch(batch_size, drop_remainder=True)
        
        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, timesteps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            
            optimizer.minimize(lambda: loss_function(batch_x, batch_y, True))
        
        loss = loss_function(batch_x, batch_y, False)
        losses.append(loss)
        print("Epoch " + str(epoch) + ", Minibatch RMSE Loss= {:.4f}".format(rmse_loss))

fig, ax = plt.subplots()
ax.plot(list(range(1, epochs + 1)), losses)
ax.set(xlabel='epochs', ylabel='loss (RMSE)')
ax.grid()
fig.savefig("tcn_loss.png")

print("Optimization Finished!")
test = testset.batch(batch_size, drop_remainder=True)
for batch_x, batch_y in tfe.Iterator(test):
    batch_x = tf.reshape(batch_x, (batch_size, timesteps, num_input))
    batch_x = tf.dtypes.cast(batch_x, tf.float32)
    batch_y = tf.dtypes.cast(batch_y, tf.float32)
    loss = loss_function(batch_x, batch_y, False)

    print("Testing Loss:", loss.numpy())