# create by fanfan on 2017/11/24 0024
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from CNN.cifar import cifar10

"""
权重初始化
初始化为一个接近0的很小的正数
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


#读取训练数据集
#images_train---训练的图像数据
#cls_train---以整型返回类的数目(0-9)
#labels_train---标签数组(如[0,0,0,0,0,0,1,0,0,0])
images_train, cls_train, labels_train = cifar10.load_training_data()

#读取测试数据集
images_test, cls_test, labels_test = cifar10.load_test_data()

#显示图片
def show_pic():
    fig, axes = plt.subplots(nrows=3, ncols=20, sharex=True, sharey=True, figsize=(80, 12))
    imgs = images_train[:60]
    for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60]],axes):
        for img,ax in zip(imgs,row):
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    plt.show()


n_class = 10
#with tf.Session() as sess:













#    y_train = sess.run(tf.one_hot(cls_train, depth=n_class))
#    y_test = sess.run(tf.one_hot(cls_test, depth=n_class))


from sklearn.model_selection import train_test_split
train_ratio = 0.95
x_train_,x_val_,y_train_,y_val_ = images_train,images_test[:500],labels_train,labels_test[:500]#train_test_split(x_train,y_train,train_size=train_ratio,random_state=123)


# train parameter
img_shape = x_train_.shape
keep_prob = 0.6
epochs = 5
batch_size = 100

inputs_x = tf.placeholder(tf.float32,[None,32,32,3],name='inputs_x')
target_y = tf.placeholder(tf.float32,[None,n_class],name='target_y')
keep_prob_op = tf.placeholder(tf.float32,name='keep_prob')

# 第一层卷积池化
# 32 * 32 * 3 to 32 * 32 * 64
kernel_1 = tf.get_variable('weights_1',shape=[5,5,3,64],initializer=tf.truncated_normal_initializer)
conv1 = tf.nn.conv2d(inputs_x,kernel_1,[1,1,1,1],padding='SAME')
conv1 = tf.nn.relu(conv1)

norm1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
#conv1 = tf.layers.conv2d(inputs_x,64,(2,2),padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer)
# 32 * 32 * 64 to 16 * 16 * 128
pool1 = tf.layers.max_pooling2d(norm1,(2,2),(2,2),padding="same")

# 第二层卷积加池化
# 16 * 16 * 64 to 16 * 16 * 128
kernel_2 = tf.get_variable('weights_12',shape=[3,3,64,128],initializer=tf.truncated_normal_initializer)
conv2 = tf.nn.conv2d(pool1,kernel_2,[1,1,1,1],padding='SAME')
conv2 = tf.nn.relu(conv2)
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
#conv2 = tf.layers.conv2d(conv1,128,(4,4),padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer)
# 16 * 16 * 128 to 8 * 8 * 128
pool2 = tf.layers.max_pooling2d(conv2,(2,2),(2,2),padding='same')

# 第三层卷积加池化
# 8 * 8 * 128 to 8 * 8 * 128
kernel_3 = tf.get_variable('weights_3',shape=[3,3,128,128],initializer=tf.truncated_normal_initializer)
conv3 = tf.nn.conv2d(pool2,kernel_3,[1,1,1,1],padding='SAME')
conv3 = tf.nn.relu(conv3)
norm3 = tf.nn.lrn(conv3,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
#conv2 = tf.layers.conv2d(conv1,128,(4,4),padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer)
# 8 * 8 * 128 to 4 * 4 * 128
pool3= tf.layers.max_pooling2d(norm3,(2,2),(2,2),padding='same')


pool3 = tf.reshape(pool3,[-1, 4 * 4 * 128])
from tensorflow.contrib import layers
# 第一层全连接
# 4* 4 * 128 to 1 * 512
fc1 = layers.fully_connected(pool3,512,activation_fn=tf.nn.relu)
fc1 = tf.nn.dropout(fc1,keep_prob)

# 第二层全连接
# 1 * 1024 to 1 * 512
#fc2 = layers.fully_connected(fc1,512,activation_fn=tf.nn.relu)

# logits 层
#logits = layers.fully_connected(fc1,10,activation_fn=None)
#logits = tf.identity(logits,name='logits')
W_fc2 = weight_variable([512, 10])
b_fc2 = bias_variable([10])

logits = tf.nn.softmax(tf.matmul(fc1, W_fc2) + b_fc2)
logits = tf.identity(logits,name='logits')

# cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=target_y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(target_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')


save_model_path = 'model/test_cifar'
count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batch_i in range(img_shape[0]//batch_size - 1):
            feature_batch = x_train_[batch_i * batch_size: (batch_i + 1) * batch_size]
            label_batch = y_train_[batch_i * batch_size: (batch_i + 1) * batch_size]
            train_loss,_,train_accuracy = sess.run([cost,optimizer,accuracy],feed_dict={inputs_x:feature_batch,target_y:label_batch,keep_prob_op:keep_prob})




            if (count % 50 ) == 0:
                val_acc = sess.run(accuracy,feed_dict={inputs_x:x_val_,target_y:y_val_,keep_prob_op:1.0})
                print("Epoch {:>2},Train loss {:.4f},Validation Accuracy {:4f},train Accuracy {:4f}".format(epoch + 1,train_loss,val_acc,train_accuracy))

            count += 1
    saver = tf.train.Saver(max_to_keep=1)
    save_path = saver.save(sess,save_model_path)



