from matplotlib import pyplot as plt
from tensorflow.contrib.eager.python import tfe
from tcn import TCN
from utils import Dataset, denorm, rmse, mape
import os
import pandas as pd
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

base_path = os.path.dirname(os.path.abspath(__file__))

GPU           = 0
batch_size    = 64
hidden_units  = 16
dropout       = 0.3
epochs        = 50
ksize         = 3
levels        = 5
output_dim    = 8
timesteps     = 8
num_input     = 8
global_step   = tf.Variable(0, trainable=False)
l2_lambda     = 0
starter_learning_rate = 0.0001

data_path = os.path.join(base_path, "../data/1")

dataset = Dataset(data_path, timesteps)
trainset = tf.data.Dataset.from_tensor_slices((dataset.train_data, dataset.train_target))
_min, _max = dataset._min, dataset._max

channel_sizes = [hidden_units] * levels
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.7, staircase=True)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
model = TCN(output_dim, channel_sizes, kernel_size=ksize, dropout=dropout, l2_lambda=l2_lambda)

train_losses = list()
test_losses = list()

def loss_function(batch_x, batch_y):

    logits = model(batch_x, training=True)

    denorm_x = denorm(logits, _min, _max)

    denorm_y = denorm(batch_y, _min, _max)
    
    lossL2 = tf.add_n(model.losses)

    return rmse(denorm_x, denorm_y) + lossL2

with tf.device(f"/gpu:{GPU}"):
    
    for epoch in range(1, epochs + 1):
        train = trainset.batch(batch_size, drop_remainder=True)
        
        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, timesteps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            optimizer.minimize(lambda: loss_function(batch_x, batch_y), global_step=global_step)

        logits = model(batch_x, training=False)
        denorm_x = denorm(logits, _min, _max)
        denorm_y = denorm(batch_y, _min, _max)
        train_loss = rmse(denorm_x, denorm_y)
        train_losses.append(train_loss.numpy())
        
        print("Epoch " + str(epoch) + ", Minibatch Train Loss= {:.4f} LR: {:.5f}".format(train_loss, optimizer._lr()))
          
    test_data, test_target = tf.convert_to_tensor(dataset.test_data, dtype=tf.float32), tf.convert_to_tensor(dataset.test_target, dtype=tf.float32)
    logits = model(test_data, training=False)
    
    denorm_x = denorm(logits, _min, _max)
    denorm_y = denorm(test_target, _min, _max)
    test_loss = rmse(denorm_x, denorm_y)
    test_mape_loss = mape(denorm_x, denorm_y)
    test_losses.append(test_loss.numpy())
        
      
if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))


print("Optimization Finished!")

print ("RMSE Loss: {:.4f}, MAPE Loss: {:.4f}".format(test_loss, test_mape_loss))

pd.DataFrame({
    "predict": np.array(denorm_x).flatten(),
    "target": np.array(denorm_y).flatten()
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_evaluation.png"))
