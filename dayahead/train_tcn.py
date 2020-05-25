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

GPU           = 1
batch_size    = 16
hidden_units  = 16
dropout       = 0.3
epochs        = 50
ksize         = 3
levels        = 5
output_dim    = 1
timesteps     = 8
num_input     = 6
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
        
        predict, target = list(), list()
        for i in range(0, len(dataset.test_data), 8):
            logits = None
            if i + 10 > len(dataset.test_data):
                break
            for j in range(11):
                x, y = dataset.test_data[i + j], dataset.test_target[i + j]
                if logits != None:
                    x[-1][-1] = float(logits.numpy()[0][0])
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                x = tf.reshape(x, (1, timesteps, num_input))
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                
                logits = model(x, training=False)
                if j > 2:
                    denorm_x = denorm(logits, _min, _max)
                    denorm_y = denorm(y, _min, _max)
                    predict.append(denorm_x.numpy()[0][0])
                    target.append(denorm_y.numpy())

        test_predict = np.array(predict)
        test_target = np.array(target)
        test_loss = rmse(predict, target)
        test_mape_loss = mape(predict, target)
        test_losses.append(test_loss.numpy())

        print("Epoch " + str(epoch) + ", Minibatch Train Loss= {:.4f}, Test Loss= {:.4f}, MAPE= {:.4f}, LR= {:.5f}".format(train_loss, test_loss, mape(predict, target), optimizer._lr()))
        
    predict, target = list(), list()
    for i in range(0, len(dataset.train_data), 8):
        logits = None
        if i + 10 > len(dataset.train_data):
            break
        for j in range(11):
            x, y = dataset.train_data[i + j], dataset.train_target[i + j]
            if logits != None:
                x[-1][-1] = float(logits.numpy()[0][0])
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            x = tf.reshape(x, (1, timesteps, num_input))
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            
            logits = model(x, training=False)
            if j > 2:
                denorm_x = denorm(logits, _min, _max)
                denorm_y = denorm(y, _min, _max)
                predict.append(denorm_x.numpy()[0][0])
                target.append(denorm_y.numpy())

    predict = np.array(predict)
    target = np.array(target)
    train_loss = rmse(predict, target)
    print ("Train Loss: {:.4f}".format(train_loss))
      
if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "train": train_losses,
    "test": test_losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_loss.png"))

print("Optimization Finished!")

pd.DataFrame({
    "predict": test_predict,
    "target": test_target
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_evaluation.png"))
