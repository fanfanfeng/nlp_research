__author__ = 'fanfan'

import tensorflow as tf
indices = [3,5,0,6]

a = tf.one_hot(indices, depth=7, on_value=None, off_value=None, axis=None, dtype=None, name=None)
print ("a is : ")
print(a)
with tf.Session() as sess:
    print(type(sess.run(a)))
    tf.nn.bidirectional_dynamic_rnn