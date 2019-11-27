__author__ = 'fanfan'
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

mnist_data_path = r'E:\tensorflow_data\mnist'
mnist = input_data.read_data_sets(mnist_data_path)

img = mnist.train.images[50]
label = mnist.train.labels[50]
plt.imshow(img.reshape((28,28)),cmap='Greys_r')
plt.show()

def get_inputs(real_size,noise_size):
    """
    真实图像tensor与噪声图像tensor
    """
    real_img = tf.placeholder(tf.float32,[None,real_size],name='real_img')
    noise_img = tf.placeholder(tf.float32,[None,noise_size],name='noise_img')
    return real_img,noise_img

def leakyReLU(x):
    return tf.maximum(0.01 * x,x)

def get_generator(noise_img,n_units,out_dim,reuse=False,alpha=0.01):
    """
    生成器

    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为32*32=784
    alpha: leaky ReLU系数
    """
    with tf.variable_scope('generator',reuse=reuse):
        hidden1 = tf.layers.dense(noise_img,n_units,activation=leakyReLU)
        # leaky ReLU
        #hidden1 = tf.maximum(alpha * hidden1,hidden1)
        hidden1 = tf.layers.dropout(hidden1,rate=0.2)

        logits = tf.layers.dense(hidden1,out_dim)
        out_puts = tf.tanh(logits)
        return  logits,out_puts

def get_discriminator(img,n_units,reuse=False):
    """
    判别器

    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
    """
    with tf.variable_scope('discriminator',reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(img,n_units,activation=leakyReLU)

        logits = tf.layers.dense(hidden1,1)
        outputs = tf.sigmoid(logits)

        return logits,outputs

# 定义参数
# 真实图像的size
img_size = mnist.train.images[0].shape[0]
# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128
# leaky ReLU的参数
alpha = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1

tf.reset_default_graph()
real_img,noise_img = get_inputs(img_size,noise_size)

#generator
g_logits,g_outputs = get_generator(noise_img,g_units,img_size)

#discriminator
d_logits_real,d_outputs_real = get_discriminator(real_img,d_units)
d_logits_fake,d_outputs_fake = get_discriminator(g_outputs,d_units,reuse=True)

# discriminator 的loss
# 识别真实图片
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real)) * (1 - smooth))

# 识别生成的图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real,d_loss_fake)

# generator 的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)) * (1-smooth))

train_vars = tf.trainable_variables()
# generator中的tensor
g_vars = [var for var in train_vars if var.name.startswith('generator')]
# discriminator的tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)

# batch_size
batch_size = 64
# 训练迭代轮数
epochs = 300
# 抽取样本数
n_sample = 25

# 存储测试样例
losses = []
samples = []
saver = tf.train.Saver(var_list=g_vars)
train = False

if train:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,784))

                batch_images = batch_images * 2 - 1

                batch_noise = np.random.uniform(-1,1,size=(batch_size,noise_size))

                _ = sess.run(d_train_opt,feed_dict = {real_img:batch_images,noise_img:batch_noise})
                _ = sess.run(g_train_opt,feed_dict = {noise_img:batch_noise})

            # 每一轮结束计算loss
            fetch_result = [d_loss,d_loss_real,d_loss_fake,g_loss]
            train_loss_d,train_loss_d_real,train_loss_d_fake,train_loss_g = sess.run(fetch_result,feed_dict={
                real_img:batch_images,
                noise_img:batch_noise
            })

            print("Epoch {}/{}...".format(e+1, epochs),
            "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
            "Generator Loss: {:.4f}".format(train_loss_g))
            # 记录各类loss值
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

            # 抽取样本后期进行观察
            sample_noise = np.random.uniform(-1,1,size=(n_sample,noise_size))
            gen_samples = sess.run(get_generator(noise_img,g_units,img_size,reuse=True),feed_dict={noise_img:sample_noise})
            samples.append(gen_samples)
            saver.save(sess,'model/gan.ckpt')

    with open('train_samples.pkl','wb') as f:
        pickle.dump(samples,f)
else:
    def view_samples(epoch,sample):
        fig,axes = plt.subplots(figsize=(7,7),nrows=5,ncols=5,sharey=True,sharex=True)
        for ax,img in zip(axes.flatten(),sample[epoch][1]):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28,28)),cmap='Greys_r')
        return fig,axes

    with tf.Session() as sess:
        saver.restore(sess,'model/gan.ckpt')
        samples_noise = np.random.uniform(-1,1,size=(25,noise_size))
        gen_samples = sess.run(get_generator(noise_img,g_units,img_size,reuse=True),
                               feed_dict={noise_img:samples_noise})
        _ = view_samples(0,[gen_samples])
        plt.show()


    with open('train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    _=view_samples(-1,samples)
    plt.show()


