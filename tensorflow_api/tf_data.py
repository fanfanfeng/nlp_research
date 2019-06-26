# create by fanfan on 2019/3/27 0027
# 参考博客：https://www.cnblogs.com/hellcat/p/8569651.html

import tensorflow as tf
import numpy as np


# example 1
print("example 1")
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,20,3.0,4.0,5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))



# example 2
print("\n\nexample 2")
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5,2)))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")


# example 3
print('\n\nexample 3')
dataset = tf.data.Dataset.from_tensor_slices(
    {
        'a':np.array([1.0,2.0,3.0,4.0,5.0]),
        'b':np.random.uniform(size=(5,2))
    }
)

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")


# example 4
print('\n\n example 4')
dataset = tf.data.Dataset.from_tensor_slices(
    (np.array([1.0,2.0,3.0,4.0,5.0]),np.random.uniform(size=(5,2)))
)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")


# example 5
print('\n\n example 5')
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
dataset = dataset.map(lambda x:x + 1)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

# example 6
print("\n\nexmaple 6")
dataset = tf.data.Dataset.from_tensor_slices(
    {
        'a':np.array([1.0,2.0,3.0,4.0,5.0]),
        'b':np.random.uniform(size=(5,2))
    }
)
dataset = dataset.batch(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")



