__author__ = 'fanfan'
#http://www.sohu.com/a/117638763_473283
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import pickle
import os

cifar_data_path = r'E:\tensorflow_data\cifar10_data'

def unpickle(file):
    with open(file,'rb') as f:
        dict = pickle.load(f,encoding='bytes')
    return dict

currentCifar = 1


# residual net实现
def resUnit(input_layer,i):
    with tf.variable_scope("res_unit" + str(i)):
        part1 = slim.batch_norm(input_layer)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,64,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,64,[3,3])

        # y = f(x) + x
        output = input_layer + part6
        return output
# highway 实现
def highway(input_layer,i):
    with tf.variable_scope('highway_unit'+ str(i)):
        H = slim.conv2d(input_layer,64,[3,3])
        T = slim.conv2d(input_layer,64,[3,3],biases_initializer=tf.constant_initializer(-1.0),activation_fn=tf.nn.sigmoid)

        output = H * T + input_layer * (1.0 - T)
    return output

# dense block 实现
def denseBlock(input_layer,i,j):
    with tf.variable_scope('dense_unit'+ str(i)):
        nodes = []
        a = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm)
        nodes.append(a)
        for z in range(j):
            b = slim.conv2d(tf.concat(nodes,3),64,[3,3],normalizer_fn=slim.batch_norm)
            nodes.append(b)
        return b

use_type = 0 # 0 normal network
             # 1 residual network
             # 2 highway_network
             # 3 dense network
total_layers = 5
units_between_stride = total_layers /5

tf.reset_default_graph()
input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(inputs=input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+ str(0))
for i in range(5):
    for j in range(units_between_stride):
        if use_type == 1:
            layer1 = resUnit(layer1,j+ (i*units_between_stride))
        elif use_type == 2:
            layer1 = highway(layer1,j + (i* units_between_stride))
        else:
            layer1 = slim.conv2d(layer1,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str((j+1) + (i*units_between_stride)))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))

top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')
output = slim.layers.softmax(slim.layers.flatten(top))
loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10,axis=1))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)

init = tf.global_variables_initializer()
batch_size = 64
currentCifar = 1
total_steps = 2000
l = []
a = []
aT = []
with tf.Session() as sess:
    sess.run(init)
    draw = range(10000)
    while i < total_steps:
        if i % (10000 / batch_size) != 0:
            batch_index = np.random.choice(draw,size = batch_size,replace=False)
        else:
            draw = range(10000)
            if currentCifar == 5:
                currentCifar =1
                print("switched cifar set to " + str(currentCifar))
            else:
                currentCifar = currentCifar + 1
                print("swithed cifar set to " + str(currentCifar))

            cifar = unpickle(os.path.join(cifar_data_path,"data_batch_" + str(currentCifar)))
            batch_index = np.random.choice(draw,size = batch_size,replace=False)

        x = cifar['data'][batch_index]
