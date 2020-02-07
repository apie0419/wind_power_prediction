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
hidden_units  = 24
learning_rate = 0.001
dropout       = 0.1
epochs        = 70
ksize         = 3
levels        = 5
n_classes     = 1
timesteps     = 24
num_input     = 3

data_path = os.path.join(base_path, "data/1")

train_data_df = pd.read_excel(os.path.join(data_path, "hour_ahead/train_in.xlsx"))
train_target_df = pd.read_excel(os.path.join(data_path, "hour_ahead/train_out.xlsx"))
test_data_df = pd.read_excel(os.path.join(data_path, "hour_ahead/test_in.xlsx"))
test_target_df = pd.read_excel(os.path.join(data_path, "hour_ahead/test_out.xlsx"))
min_max = pd.read_excel(os.path.join(data_path, "hour_ahead/max_min.xls"))

_min = tf.constant(min_max["pmin"][0], dtype=tf.float32)
_max = tf.constant(round(min_max["pmax"][0], 2), dtype=tf.float32)

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
test_data, test_target = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_target)

channel_sizes = [hidden_units] * levels
optimizer = tf.train.AdamOptimizer(learning_rate)
model = TCN(n_classes, channel_sizes, kernel_size=ksize, dropout=dropout)

def loss_function(x, y, training):

    logits = model(x, training=training)

    denorm_x = denorm(logits, _min, _max)

    denorm_y = denorm(y, _min, _max)

    lossL2 = tf.add_n(model.losses)

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(denorm_y, denorm_x)))) + lossL2, logits
    

def denorm(x, min, max):

    return x * (max - min) + min

train_losses = list()
test_data = tf.dtypes.cast(test_data, tf.float32)
test_target = tf.dtypes.cast(test_target, tf.float32)
test_losses = list()

with tf.device("/gpu:1"):

    for epoch in range(1, epochs + 1):
        train = trainset.batch(batch_size, drop_remainder=True).gpu()
        
        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, timesteps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32).gpu()
            batch_y = tf.dtypes.cast(batch_y, tf.float32).gpu()
            
            optimizer.minimize(lambda: loss_function(batch_x, batch_y, True))
        
        train_loss, logits = loss_function(batch_x, batch_y, False)
        train_losses.append(train_loss.numpy())

        test_loss, logits = loss_function(test_data, test_target, training=False)
        logits = denorm(logits, _min, _max)
        target = denorm(test_target, _min, _max)

        print("Epoch " + str(epoch) + ", Minibatch RMSE Loss= {:.4f}, Test Loss= {:.4f}".format(train_loss, test_loss))

        test_losses.append(test_loss.numpy())
        
if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "train": train_losses,
    "test": test_losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_loss.png"))

print("Optimization Finished!")

pd.DataFrame({
    "predict": logits[:, 0],
    "target": test_target[:, 0]
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_evaluation.png"))
