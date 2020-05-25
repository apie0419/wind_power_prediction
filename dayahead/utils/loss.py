import tensorflow as tf

def rmse(x, y):
    sub = tf.subtract(y, x)
    sq = tf.square(sub)
    rm = tf.reduce_mean(sq)

    return tf.sqrt(rm)

def mape(x, y):
    return tf.keras.losses.mean_absolute_percentage_error(y, x)

if __name__ == "__main__":

    tf.enable_eager_execution()
    t1 = tf.convert_to_tensor([[1., 1., 1.], [1., 1., 1.]], dtype=tf.float32)
    t2 = tf.convert_to_tensor([[1., 1., 1.], [2., 2., 2.]], dtype=tf.float32)
    loss = rmse(t1, t2)
    print (loss)