# create by fanfan on 2018/8/15 0015
import os
from glob import glob
import numpy as np
from  scipy.misc import imread
import tensorflow as tf
import math
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import conv2d_transpose
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
import time


from scipy.misc import imsave
def save_image(name,vec):
    '''
    保存图片
    :param name: 图片存储名称路径
    :param vec: 图片向量
    :return: 
    '''
    return imsave(name,vec)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class config():
    learning_rate = 0.0001

    epoch = 30
    train_size = np.inf

    batch_size = 200
    input_height = 28  # , "The size of image to use (will be center cropped). [108]")
    input_width = 28  # , "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    output_height = 28  # , "The size of the output images to produce [64]")
    output_width = 28  # , "The size of the output images to produce. If None, same value as output_height [None]")
    input_fname_pattern = "*.jpg"  # , "Glob pattern of filename of input images [*]")
    checkpoint_dir = "checkpoint"  # , "Directory name to save the checkpoints [checkpoint]")
    data_dir = "./data"  # , "Root directory of dataset [data]")
    sample_dir = "samples"  # , "Directory name to save the image samples [samples]")
    train = True  # , "True for training, False for testing [False]")
    crop = False  # , "True for training, False for testing [False]")
    visualize = False  # , "True for visualizing, False for nothing [False]")
    generate_test_images = 100  # , "Number of images to generate during test. [100]")


class DCGAN(object):
    def __init__(self, sess, batch_size=200, sample_num=200,z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=1,
                 checkoutpoint_dir='model-dcgan', sample_dir='sample'):
        '''
        :param sess:  tensorflow运行session
        :param batch_size: 一次训练的训练个数
        :param sample_num: 生成样本的数目
        :param z_dim: 随机的变量的维书，默认100
        :param gf_dim: 生成器G第一层卷积的fillters数目
        :param df_dim: 判别器D第一层卷积的fillters数目
        :param gfc_dim: 生成器全连接层的维度
        :param dfc_dim: 判别器全连接层的维度
        :param c_dim: 图片的颜色，默认3，如果是灰度图，则为1，
        :param checkoutpoint_dir: 模型保存位置
        :param sample_dir: 生成的样例图片位置
        '''
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = 28
        self.input_width = 28
        self.output_height = 28
        self.output_width = 28

        self.y_dim = 10
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.checkpoint_dir = checkoutpoint_dir
        self.sampler_dir = sample_dir


        self.data_x, self.data_y = self.load_mnist()
        self.c_dim = self.data_x[0].shape[-1]

        self.grayscale = 1
        self.build_model()

    def build_model(self):

        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        image_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_summary = tf.summary.histogram('z', self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(self.inputs, self.y, reuse=False)
        self.sampler_data = self.generator(self.z, self.y,reuse=True)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_summary = tf.summary.histogram('d', self.D)
        self.d__summary = tf.summary.histogram('d_', self.D_)
        self.G_summary = tf.summary.image('G', self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_summary = tf.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar('d_loss_fake', self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_mnist(self):
        mnist_data_path = r'E:\tensorflow_data\mnist'
        mnist = input_data.read_data_sets(mnist_data_path)

        x = np.concatenate([mnist.train.images, mnist.validation.images, mnist.test.images], axis=0)
        y = np.concatenate([mnist.train.labels, mnist.validation.labels, mnist.test.labels], axis=0)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0
        x = x.reshape((70000, 28, 28, 1)).astype(np.float)
        return x, y_vec

    @property
    def model_dir(self):
        return 'mnist_{}_{}_{}'.format( self.batch_size, self.output_height, self.output_width)

    def save_mode(self, checkpoint_dir, step):
        model_name = 'DCGAN.model'
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def restore_mode(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join("checkpoint", self.model_dir) +"\\"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print("[*] Failed to find a checkpoint")
            return False

    def generator(self, z, y,reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
            s_w4, s_w4 = int(s_w / 2), int(s_w / 4)

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            h0 = fully_connected(z, self.gfc_dim, activation_fn=None, scope='g_h0_linear')
            btn_h0 = batch_norm(h0, scope='g_btn_h0')
            h0 = tf.nn.relu(btn_h0)
            h0 = tf.concat([h0, y], 1)

            h1 = fully_connected(h0, self.gf_dim * 2 * s_h4 * s_w4, activation_fn=None, scope='g_h1_linear')
            btn_h1 = batch_norm(h1, scope='g_btn_h1')
            h1 = tf.nn.relu(btn_h1)
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            x_shapes = h1.get_shape()
            y_shapes = yb.get_shape()
            h1 = tf.concat([h1, yb * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

            h2 = conv2d_transpose(h1, self.gf_dim * 2, kernel_size=5, stride=2, scope='g_h2', activation_fn=None)
            btn_h2 = batch_norm(h2, scope="g_btn_h2")
            h2 = tf.nn.relu(btn_h2)
            x_shapes = h2.get_shape()
            h2 = tf.concat([h2, yb * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

            h3 = conv2d_transpose(h2, self.c_dim, kernel_size=5, stride=2, scope='g_h3', activation_fn=None)
            return tf.nn.sigmoid(h3)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x_shapes = image.get_shape()
            y_shapes = yb.get_shape()
            x = tf.concat([image, yb * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

            h0 = conv2d(x, self.c_dim + self.y_dim, kernel_size=5, stride=2, scope='d_h0_conv', activation_fn=None)
            h0 = tf.nn.leaky_relu(h0)
            x_shapes = h0.get_shape()
            h0 = tf.concat([h0, yb * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

            h1 = conv2d(h0, self.df_dim + self.y_dim, kernel_size=5, stride=2, scope='d_h1_conv',
                        activation_fn=None)
            btn_h1 = batch_norm(h1, scope='d_btn_h1')
            h1 = tf.nn.leaky_relu(btn_h1)
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2 = fully_connected(h1, self.dfc_dim, activation_fn=None, scope='d_h2_linear')
            btn_h2 = batch_norm(h2, scope='d_btn_h2')
            h2 = tf.nn.leaky_relu(btn_h2)
            h2 = tf.concat([h2, y], 1)

            h3 = fully_connected(h2, 1, activation_fn=None, scope='d_h3_linear')

            return tf.nn.sigmoid(h3), h3

    def train(self):
        d_optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        g_optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.g_summary_run = tf.summary.merge(
            [self.z_summary, self.d__summary, self.G_summary, self.d_loss_fake_summary, self.g_loss_summary])
        self.d_summary_run = tf.summary.merge(
            [self.z_summary, self.d_summary, self.d_loss_real_summary, self.d_loss_summary])

        self.summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))


        sample_inputs = self.data_x[0:self.sample_num]
        sample_labels = self.data_y[0:self.sample_num]


        count = 1
        start_time = time.time()
        could_load = self.restore_mode(self.checkpoint_dir, count)
        if could_load:
            print("load success")
        else:
            print("load failed..")

        for epoch in range(config.epoch):

            batch_idxs = min(len(self.data_x), config.train_size) // config.batch_size
            for idx in range(0, int(batch_idxs)):

                batch_images = self.data_x[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]


                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # 跟新 Distribute 判别式网络
                _, summary_str = self.sess.run([d_optimizer, self.d_summary_run], feed_dict={
                    self.inputs: batch_images,
                    self.z: batch_z,
                    self.y: batch_labels,
                })
                self.summary_writer.add_summary(summary_str, count)

                _,summary_str = self.sess.run([g_optimizer,self.g_summary_run],feed_dict={
                    self.z:batch_z,
                    self.y:batch_labels
                })
                self.summary_writer.add_summary(summary_str,count)

                _, summary_str = self.sess.run([g_optimizer, self.g_summary_run], feed_dict={
                    self.z: batch_z,
                    self.y: batch_labels
                })
                self.summary_writer.add_summary(summary_str, count)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                    self.y: batch_labels
                })

                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })


                count += 1
                print("Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f,d_loss:%.8f,g_loss:%.8f" % (
                epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if count % 1 == 0:
                    samples, d_loss, g_loss = self.sess.run([self.sampler_data, self.d_loss, self.g_loss],
                                                            feed_dict={
                                                                self.z: sample_z,
                                                                self.inputs: sample_inputs,
                                                                self.y: sample_labels,
                                                            })
                    for index in range(self.sample_num):
                        file_name = os.path.join(self.sampler_dir,'epoch{}_step{}_true{}.png' .format(epoch,index,sample_labels[index].tolist().index(1)))
                        file_vec = np.reshape(samples[index],(28,28))
                        save_image(file_name,file_vec)






                if count % 50 == 0:
                    self.save_mode(config.checkpoint_dir, count)


if __name__ == '__main__':
    with tf.Session() as sess:
            dcgan_obj = DCGAN(sess)
            dcgan_obj.train()

























