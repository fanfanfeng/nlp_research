# create by fanfan on 2020/3/20 0020
import tensorflow as tf
def contrastive_loss(y,d):
    batch_size = tf.shape(y)[0]
    tmp = y * tf.square(d)
    tmp2 = (1-y) * tf.square(tf.maximum((1-d),0))
    return tf.reduce_sum(tmp + tmp2)/batch_size/2


