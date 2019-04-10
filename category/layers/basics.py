# create by fanfan on 2019/3/25 0025
import tensorflow as tf

def feed_forward(x,num_hiddens,activation=None,reuse=False):
    with tf.variable_scope('feed_forward',reuse=reuse):
        x_forward = tf.layers.dense(x,num_hiddens,activation=activation,reuse=reuse)
    return x_forward


def dropout(x,is_training,rate=0.2):
    return tf.layers.dropout(x,rate,training=tf.convert_to_tensor(is_training))

