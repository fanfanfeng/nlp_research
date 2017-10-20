# create by fanfan on 2017/10/19 0019

import tensorflow as tf
import numpy as np

a = tf.constant(np.arange(24).reshape(2,3,4),dtype=tf.float32)

with tf.Session() as sess:
    print("origin:")
    print(sess.run(a))
    print("on axis 1")
    print(sess.run(tf.reduce_sum(a, axis=1)))
    vector_attn = tf.reduce_sum(a, axis=2, keep_dims=True)
    attention_weights = tf.nn.softmax(vector_attn, dim=1)
    #weighted_projection = tf.multiply(inputs, attention_weights)
    #outputs = tf.reduce_sum(weighted_projection, axis=1)
    print(sess.run(vector_attn))
    print(sess.run(attention_weights))


