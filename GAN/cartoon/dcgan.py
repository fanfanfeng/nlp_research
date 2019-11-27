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
import sys


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
    input_height = 96  # , "The size of image to use (will be center cropped). [108]")
    input_width = 96  # , "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    output_height = 64  # , "The size of the output images to produce [64]")
    output_width = 64  # , "The size of the output images to produce. If None, same value as output_height [None]")
    input_fname_pattern = "*.jpg"  # , "Glob pattern of filename of input images [*]")
    checkpoint_dir = "checkpoint"  # , "Directory name to save the checkpoints [checkpoint]")
    if 'win' in sys.platform:
        data_dir = r"E:\tensorflow_data\faces\1"  # , "Root directory of dataset [data]")
    else:
        data_dir='/data/python_project/output/faces'
    sample_dir = "samples"  # , "Directory name to save the image samples [samples]")
    train = True  # , "True for training, False for testing [False]")
    crop = False  # , "True for training, False for testing [False]")
    visualize = False  # , "True for visualizing, False for nothing [False]")
    generate_test_images = 100  # , "Number of images to generate during test. [100]")
    data_set_name = 'cartoon'

    z_dim = 100 # 随机数据的维度
    sample_num = 100 #生成的随机图的个数
    gf_dim = 28 #生成器G第一层卷积的fillters数目
    df_dim = 28 #判别器D第一层卷积的fillters数目
    gfc_dim = 1024 #生成器全连接层的维度
    dfc_dim = 1024 #判别器全连接层的维度
    c_dim = 3 #图片的维度


class DCGAN(object):
    def __init__(self, sess,):
        '''
        :param sess:  tensorflow运行session
        '''

        self.sess = sess
        self.sample_num = config.sample_num
        self.input_height = config.input_height
        self.input_width = config.input_width
        self.output_height = config.output_height
        self.output_width = config.output_width
        self.z_dim = config.z_dim
        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim

        self.gfc_dim = config.gfc_dim
        self.dfc_dim = config.dfc_dim

        self.checkpoint_dir = config.checkpoint_dir
        self.sampler_dir = config.sample_dir

        self.data_set_name = 'cartoon'
        self.data_dir = config.data_dir
        self.input_fname_pattern = config.input_fname_pattern
        self.data = self.load_data()
        self.c_dim = config.c_dim


        self.build_model()

    def build_model(self):


        self.inputs = tf.placeholder(tf.float32, [None,self.input_height, self.input_width, self.c_dim], name='real_images')
        self.batch_size = tf.shape(self.inputs)[0]

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_summary = tf.summary.histogram('z', self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.sampler_data = self.generator(self.z,reuse=True,is_train=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

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

        self.global_step  = tf.Variable(0,trainable=False,name='gloabl_step')
        self.saver = tf.train.Saver()


    def load_data(self):
        data_path = os.path.join(self.data_dir,self.input_fname_pattern)
        data = glob(data_path)
        np.random.shuffle(data)
        return data

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

    def generator(self, z,reuse=False,is_train=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z_ = fully_connected(z,self.gf_dim*8*s_h16*s_w16,activation_fn=None,scope='g_h0_linear')
            h0 = tf.reshape(z_,[-1,s_h16,s_w16,self.gf_dim*8])
            btn_h0 = batch_norm(h0,scope='g_btn_h0',is_training=is_train)
            h0 = tf.nn.relu(btn_h0)

            h1 = conv2d_transpose(h0,self.gf_dim*4,kernel_size=5,stride=2,activation_fn=None,scope='g_dconv_1')
            btn_h1 = batch_norm(h1,scope='g_btn_h1',is_training=is_train)
            h1 = tf.nn.relu(btn_h1)

            h2 = conv2d_transpose(h1,self.gf_dim*2,kernel_size=5,stride=2,activation_fn=None,scope='g_dconv_2')
            btn_h2 = batch_norm(h2,scope='g_btn_h2',is_training=is_train)
            h2 = tf.nn.relu(btn_h2)

            h3 = conv2d_transpose(h2,self.gf_dim,kernel_size=5,stride=2,activation_fn=None,scope='g_dconv_3')
            btn_h3 = batch_norm(h3,scope='g_btn_h3',is_training=is_train)
            h3 = tf.nn.relu(btn_h3)

            h4 = conv2d_transpose(h3,self.c_dim,kernel_size=5,stride=2,activation_fn=None,scope='g_dconv_4')
            return tf.nn.tanh(h4)


    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = conv2d(image, self.df_dim, kernel_size=5, stride=2, scope='d_h0_conv', activation_fn=None)
            h0 = tf.nn.leaky_relu(h0)


            h1 = conv2d(h0, self.df_dim*2, kernel_size=5, stride=2, scope='d_h1_conv',activation_fn=None)
            btn_h1 = batch_norm(h1, scope='d_btn_h1')
            h1 = tf.nn.leaky_relu(btn_h1)

            h2 = conv2d(h1, self.df_dim*4, kernel_size=5, stride=2,activation_fn=None, scope='d_h2_conv')
            btn_h2 = batch_norm(h2, scope='d_btn_h2')
            h2 = tf.nn.leaky_relu(btn_h2)

            h3 = conv2d(h2, self.df_dim * 8,kernel_size=5, stride=2, activation_fn=None, scope='d_h3_conv')
            btn_h3 = batch_norm(h3, scope='d_btn_h3')
            h3 = tf.nn.leaky_relu(btn_h3)


            h3 = tf.reshape(h3,[-1,6*6*224])
            h4 = fully_connected(h3, 1, activation_fn=None, scope='d_h3_linear')

            return tf.nn.sigmoid(h4),h4

    def train(self):
        d_optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars,global_step=self.global_step)
        g_optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars,global_step=self.global_step)

        tf.global_variables_initializer().run()

        self.g_summary_run = tf.summary.merge(
            [self.z_summary, self.d__summary, self.G_summary, self.d_loss_fake_summary, self.g_loss_summary])
        self.d_summary_run = tf.summary.merge(
            [self.z_summary, self.d_summary, self.d_loss_real_summary, self.d_loss_summary])

        self.summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))


        sample_inputs = self.data[0:self.sample_num]


        count = 1
        start_time = time.time()
        could_load = self.restore_mode(self.checkpoint_dir, count)
        if could_load:
            print("load success")
        else:
            print("load failed..")

        for epoch in range(config.epoch):

            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
            for idx in range(0, int(batch_idxs)):

                batch_images = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # 跟新 Distribute 判别式网络
                _, summary_str = self.sess.run([d_optimizer, self.d_summary_run], feed_dict={
                    self.inputs: batch_images,
                    self.z: batch_z,
                })
                self.summary_writer.add_summary(summary_str, count)

                _,summary_str = self.sess.run([g_optimizer,self.g_summary_run],feed_dict={
                    self.z:batch_z,
                })
                self.summary_writer.add_summary(summary_str,count)

                _, summary_str = self.sess.run([g_optimizer, self.g_summary_run], feed_dict={
                    self.z: batch_z,
                })
                self.summary_writer.add_summary(summary_str, count)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                })

                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                })

                errG = self.g_loss.eval({
                    self.z: batch_z,
                })


                count += 1
                print("Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f,d_loss:%.8f,g_loss:%.8f" % (
                epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if count % 50 == 0:
                    samples, d_loss, g_loss = self.sess.run([self.sampler_data, self.d_loss, self.g_loss],
                                                            feed_dict={
                                                                self.z: sample_z,
                                                                self.inputs: sample_inputs,
                                                            })
                    if not os.path.exists(self.sampler_dir):
                        os.makedirs(self.sampler_dir)
                    for index in range(self.sample_num):
                        file_name = os.path.join(self.sampler_dir,'epoch{}_step{}.png' .format(epoch,index))
                        file_vec = samples[index]
                        save_image(file_name,file_vec)






                if count % 50 == 0:
                    self.save_mode(config.checkpoint_dir, count)


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,)) as sess:
                dcgan_obj = DCGAN(sess)
                dcgan_obj.train()

























