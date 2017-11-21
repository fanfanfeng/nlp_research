__author__ = 'fanfan'
#tf.squeeze(input, squeeze_dims = None, name = None)
#解释：这个函数的作用是将input中维度是1的那一维去掉。但是如果你不想把维度是1的全部去掉，那么你可以使用squeeze_dims参数，来指定需要去掉的位置。
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    data = tf.constant([[1,2,1],[3,1,1]])
    print(sess.run(data))

    d_1 = tf.expand_dims(data,0)
    print("expand dim at axis 0:")
    print(sess.run(d_1))

    d_1 = tf.expand_dims(d_1,2)
    print("expand dim at axis 2:")
    print(sess.run(d_1))

    d_1 = tf.expand_dims(d_1,-1)
    print("expand dim at axis last(-1):")
    print(sess.run(d_1))

    d_2 = d_1
    print("d_1 after squeeze:")
    print(sess.run(tf.shape(tf.squeeze(d_1))))

    print('d_2 sequeeze at axis [2,4]:')
    print(sess.run(tf.shape(tf.squeeze(d_2,[2,4]))))




