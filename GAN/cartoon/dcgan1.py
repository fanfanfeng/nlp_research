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
from GAN.cartoon import config
from GAN.cartoon import utils
import time


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self,sess,input_height=108,input_width=108,crop=True,batch_size=200,sample_num=200,output_height=64,output_width=64,
                 y_dim=None,z_dim=100,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024,c_dim=3,dataset_name='default',input_fname_pattern='*.jpg',
                 checkoutpoint_dir=None,sample_dir=None,data_dir='./data'):
        '''
        
        :param sess:  tensorflow运行session
        :param input_height: 输入图片的高度   
        :param input_width: 输入图片的宽度
        :param crop: 是否需要剪裁
        :param batch_size: 一次训练的训练个数
        :param sample_num: 生成样本的数目
        :param output_height: 输出图片高度
        :param output_width: 输出图片宽度
        :param y_dim: 如果是mnist，则是10，否则是None
        :param z_dim: 随机的变量的维书，默认100
        :param gf_dim: 生成器G第一层卷积的fillters数目
        :param df_dim: 判别器D第一层卷积的fillters数目
        :param gfc_dim: 生成器全连接层的维度
        :param dfc_dim: 判别器全连接层的维度
        :param c_dim: 图片的颜色，默认3，如果是灰度图，则为1，
        :param dataset_name: 数据集名称，mnist或者自定义
        :param input_fname_pattern: 输入图片名字的正则匹配
        :param checkoutpoint_dir: 模型保存位置
        :param sample_dir: 生成的样例图片位置
        :param data_dir: 数据集的根目录
        '''
        self.sess = sess
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.d_bn1 = utils.batch_norm(scope='d_bn1')
        self.d_bn2 = utils.batch_norm(scope='d_bn2')

        if not self.y_dim:
            self.d_bn3 = utils.batch_norm(scope='d_bn3')

        self.g_bn0 = utils.batch_norm(scope='g_bn0')
        self.g_bn1 = utils.batch_norm(scope='g_bn1')
        self.g_bn2 = utils.batch_norm(scope='g_bn2')

        if not self.y_dim:
            self.g_bn3 = utils.batch_norm(scope='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkoutpoint_dir
        self.data_dir = data_dir


        if self.dataset_name == 'mnist':
            self.data_x ,self.data_y = self.load_mnist()
            self.c_dim = self.data_x[0].shape[-1]
        else:
            data_path = os.path.join(self.data_dir,self.dataset_name,self.input_fname_pattern)
            self.data = glob(data_path)
            if len(self.data) == 0:
                raise Exception("[!] No data found in '" + data_path + "'")
            np.random.shuffle(self.data)
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3:
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1
        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32,[self.batch_size,self.y_dim],name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height,self.output_width,self.c_dim]
        else:
            image_dims = [self.input_height,self.input_width,self.c_dim]

        self.inputs = tf.placeholder(tf.float32,[self.batch_size] + image_dims,name='real_images')

        self.z = tf.placeholder(tf.float32,[None,self.z_dim],name='z')
        self.z_summary = tf.summary.histogram('z',self.z)

        self.G = self.generator(self.z,self.y)
        self.D,self.D_logits = self.discriminator(self.inputs,self.y,reuse=False)
        self.sampler_data =  self.sampler(self.z,self.y)
        self.D_,self.D_logits_ = self.discriminator(self.G,self.y,reuse=True)

        self.d_summary = tf.summary.histogram('d',self.D)
        self.d__summary = tf.summary.histogram('d_',self.D_)
        self.G_summary = tf.summary.image('G',self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,labels=tf.zeros_like(self.D_)))

        self.g_loss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,labels=tf.ones_like(self.D_)))

        self.d_loss_real_summary = tf.summary.scalar('d_loss_real',self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar('d_loss_fake',self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_summary = tf.summary.scalar('g_loss',self.g_loss)
        self.d_loss_summary = tf.summary.scalar('d_loss',self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def sampler(self,z,y=None):
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
        if not self.y_dim:
            s_h,s_w = self.output_height,self.output_width
            s_h2,s_w2 = conv_out_size_same(s_h,2),conv_out_size_same(s_w,2)
            s_h4,s_w4 = conv_out_size_same(s_h2,2),conv_out_size_same(s_w2,2)
            s_h8,s_w8 = conv_out_size_same(s_h4,2),conv_out_size_same(s_w4,2)
            s_h16,s_w16 = conv_out_size_same(s_h8,2),conv_out_size_same(s_w8,2)

            h0 = fully_connected(z,self.gf_dim*8*s_h16*s_w16,scope='g_h0_linear',activation_fn=None)
            h0 = tf.reshape(h0,[-1,s_h16,s_w16,self.gf_dim *8])
            btn_h0 = batch_norm(h0,is_training=False,scope='g_btn_h0')
            h0 = tf.nn.relu(btn_h0)

            h1 = conv2d_transpose(h0,self.gf_dim*4,kernel_size=5,stride=2,scope='g_h1')
            btn_h1 = batch_norm(h1,is_training=False,scope='g_btn_h1')
            h1 = tf.nn.relu(btn_h1)

            h2 = conv2d_transpose(h1,self.gf_dim*2,kernel_size=5,stride=2,scope='g_h2')
            btn_h2 = batch_norm(h2,is_training=False,scope='g_btn_h2')
            h2 =  tf.nn.relu(btn_h2)

            h3 = conv2d_transpose(h2,self.gf_dim,kernel_size=5,stride=2,scope='g_h3')
            btn_h3 = batch_norm(h3,is_training=False,scope='g_btn_h3')
            h3 = tf.nn.relu(btn_h3)

            h4 = conv2d_transpose(h3,self.c_dim,kernel_size=5,stride=2,scope='g_h4')
            return tf.nn.tanh(h4)
        else:
            s_h,s_w = self.output_height,self.output_width
            s_h2,s_h4 = int(s_h/2),int(s_h/4)
            s_w2,s_w4 = int(s_w/2),int(s_w/4)
            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])
            z = tf.concat([z,y],1)

            h0 = fully_connected(z,self.gfc_dim,activation_fn=None,scope='g_h0_linear')
            btn_h0 = batch_norm(h0,is_training=False,scope='g_btn_h0')
            h0 = tf.nn.relu(btn_h0)
            h0 = tf.concat([h0,y],1)

            h1 = fully_connected(h0,self.gf_dim *2*s_h4*s_w4,activation_fn=None,scope='g_h1_linear')
            btn_h1 = batch_norm(h1,is_training=False,scope='g_btn_h1')
            h1 = tf.nn.relu(btn_h1)
            h1 = tf.reshape(h1,[self.batch_size,s_h4,s_w4,self.gf_dim*2])
            x_shapes = h1.get_shape()
            y_shapes = yb.get_shape()
            h1 = tf.concat([h1,yb* tf.ones([x_shapes[0],x_shapes[1],x_shapes[2],y_shapes[3]])],3)

            h2 = conv2d_transpose(h1,self.gf_dim*2,kernel_size=5,stride=2,scope='g_h2')
            btn_h2 = batch_norm(h2,is_training=False,scope="g_btn_h2")
            h2 = tf.nn.relu(btn_h2)
            x_shapes = h2.get_shape()
            h2 = tf.concat([h2,yb*tf.ones([x_shapes[0],x_shapes[1],x_shapes[2],y_shapes[3]])],3)

            h3 = conv2d_transpose(h2,self.c_dim,kernel_size=5,stride=2,scope='g_h3')
            return tf.nn.sigmoid(h3)

    def load_mnist(self):
        mnist_data_path = r'E:\tensorflow_data\mnist'
        mnist = input_data.read_data_sets(mnist_data_path)


        x = np.concatenate([mnist.train.images,mnist.validation.images,mnist.test.images],axis=0)
        y = np.concatenate([mnist.train.labels,mnist.validation.labels,mnist.test.labels],axis=0)

        y_vec = np.zeros((len(y),self.y_dim),dtype=np.float)
        for i ,label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        x = x.reshape((70000,28,28,1)).astype(np.float)
        return x,y_vec

    @property
    def model_dir(self):
        return '{}_{}_{}_{}'.format(self.dataset_name,self.batch_size,self.output_height,self.output_width)

    def save_mode(self,checkpoint_dir,step):
        model_name = 'DCGAN.model'
        checkpoint_dir = os.path.join(checkpoint_dir,self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir,model_name),global_step=step)


    def restore_mode(self,checkpoint_dir,step):
        checkpoint_dir = os.path.join(checkpoint_dir,self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(checkpoint_dir,ckpt_name))
            return True
        else:
            print("[*] Failed to find a checkpoint")
            return False

    def generator(self,z,y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h,s_w = self.output_height,self.output_width
                s_h2,s_w2 = conv_out_size_same(s_h,2),conv_out_size_same(s_w,2)
                s_h4,s_w4 = conv_out_size_same(s_h2,2),conv_out_size_same(s_w2,2)
                s_h8,s_w8 = conv_out_size_same(s_h4,2),conv_out_size_same(s_w4,2)
                s_h16,s_w16 = conv_out_size_same(s_h8,2),conv_out_size_same(s_w8,2)


                self.z_linear = fully_connected(z,self.gf_dim*8*s_h16*s_w16,scope='g_h0_linear',activation_fn=None)

                h0 = tf.reshape(self.z_linear,[-1,s_h16,s_w16,self.gf_dim * 8])
                btn_h0 = self.g_bn0(h0)
                h0 = tf.nn.relu(btn_h0)

                h1 = conv2d_transpose(h0,self.gf_dim *4,kernel_size=5,stride=2,scope='g_h1')
                btn_h1 = self.g_bn1(h1)
                h1 = tf.nn.relu(btn_h1)

                h2 = conv2d_transpose(h1,self.gf_dim *2,kernel_size=5,stride=2,scope='g_h2')
                btn_h2 = self.g_bn2(h2)
                h2 = tf.nn.relu(btn_h2)

                h3 = conv2d_transpose(h2,self.gf_dim*1,kernel_size=5,stride=2,scope='g_h3')
                btn_h3 = self.g_bn3(h3)
                h3 = tf.nn.relu(btn_h3)

                h4 = conv2d_transpose(h3,self.c_dim,kernel_size=5,stride=2,scope='g_h4')
                return tf.nn.tanh(h4)
            else:
                s_h,s_w = self.output_height,self.output_width
                s_h2,s_h4 = int(s_h/2),int(s_h/4)
                s_w4,s_w4 = int(s_w/2),int(s_w/4)

                yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])
                z = tf.concat([z,y],1)

                h0 = fully_connected(z,self.gfc_dim,activation_fn=None,scope='g_h0_linear')
                btn_h0 = self.g_bn0(h0)
                h0 = tf.nn.relu(btn_h0)
                h0 = tf.concat([h0,y],1)

                h1 = fully_connected(h0,self.gf_dim*2*s_h4*s_w4,activation_fn=None,scope='g_h1_linear')
                btn_h1 = self.g_bn1(h1)
                h1 = tf.nn.relu(btn_h1)
                h1 = tf.reshape(h1,[self.batch_size,s_h4,s_w4,self.gf_dim *2])
                h1 = utils.conv_cond_concat(h1,yb)

                h2 = conv2d_transpose(h1,self.gf_dim*2,kernel_size=5,stride=2,scope='g_h2')
                btn_h2 = self.d_bn2(h2)
                h2 = tf.nn.relu(btn_h2)
                h2 = utils.conv_cond_concat(h2,yb)

                h3 = conv2d_transpose(h2,self.c_dim,kernel_size=5,stride=2,scope='g_h3')
                return  tf.nn.sigmoid(h3)

    def discriminator(self,image,y=None,reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = conv2d(image,self.df_dim,kernel_size=5,stride=2,scope='d_h0_conv')
                h0 = tf.nn.leaky_relu(h0)

                h1 = conv2d(h0,self.df_dim*2,scope='d_h1_conv')
                btn_h1 = batch_norm(h1,scope='d_btn_h1')
                h1 = tf.nn.leaky_relu(btn_h1)

                h2 = conv2d(h1,self.df_dim*4,scope='d_h2_conv')
                btn_h2 = batch_norm(h2,scope='d_btn_h2')
                h2 = tf.nn.leaky_relu(btn_h2)

                h3 = conv2d(h2,self.df_dim*8,scope='d_h3_conv')
                btn_h3 = batch_norm(h3,scope='d_btn_h3')
                h3 = tf.nn.leaky_relu(btn_h3)

                h3 = tf.reshape(h3,[self.batch_size,-1])
                h4 = fully_connected(h3,1,activation_fn=None,scope='d_h4_linear')
                return tf.nn.sigmoid(h4),h4
            else:
                yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])
                x = utils.conv_cond_concat(image,yb)

                h0 = conv2d(x,self.c_dim+self.y_dim,kernel_size=5,stride=2,scope='d_h0_conv')
                h0 = tf.nn.leaky_relu(h0)
                h0 = utils.conv_cond_concat(h0,yb)

                h1 = conv2d(h0,self.df_dim + self.y_dim,kernel_size=5,stride=2,scope='d_h1_conv')
                btn_h1 = self.d_bn1(h1)
                h1 = tf.nn.leaky_relu(btn_h1)
                h1 = tf.reshape(h1,[self.batch_size,-1])
                h1 = tf.concat([h1,y],1)

                h2 = fully_connected(h1,self.dfc_dim,activation_fn=None,scope='d_h2_linear')
                btn_h2 = self.d_bn2(h2)
                h2 = tf.nn.leaky_relu(btn_h2)
                h2 = tf.concat([h2,y],1)

                h3 = fully_connected(h2,1,activation_fn=None,scope='d_h3_linear')

                return tf.nn.sigmoid(h3),h3


    def train(self):
        d_optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
        g_optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss,var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.g_summary_run = tf.summary.merge([self.z_summary,self.d__summary,self.G_summary,self.d_loss_fake_summary,self.g_loss_summary])
        self.d_summary_run = tf.summary.merge([self.z_summary,self.d_summary,self.d_loss_real_summary,self.d_loss_summary])

        self.summary_writer = tf.summary.FileWriter('./logs',self.sess.graph)
        sample_z = np.random.uniform(-1,1,size=(self.sample_num,self.z_dim))

        if config.dataset == 'mnist':
            sample_inputs = self.data_x[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            pass

        count = 1
        start_time  = time.time()
        could_load = self.restore_mode(self.checkpoint_dir,count)
        if could_load:
            print("load success")
        else:
            print("load failed..")

        for epoch in range(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(self.data_x),config.train_size)// config.batch_size
            else:
                batch_idxs = 0

            for idx in range(0,int(batch_idxs)):
                if config.dataset == 'mnist':
                    batch_images = self.data_x[idx * config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = self.data_y[idx * config.batch_size:(idx+1)*config.batch_size]
                else:
                    pass


                batch_z = np.random.uniform(-1,1,[config.batch_size, self.z_dim]).astype(np.float32)
                if config.dataset == 'mnist':
                    # 跟新 Distribute 判别式网络
                    _,summary_str = self.sess.run([d_optimizer,self.d_summary_run],feed_dict={
                        self.inputs:batch_images,
                        self.z:batch_z,
                        self.y:batch_labels,
                    })
                    self.summary_writer.add_summary(summary_str,count)


                   # _,summary_str = self.sess.run([g_optimizer,self.g_summary_run],feed_dict={
                    #    self.z:batch_z,
                    #    self.y:batch_labels
                    #})
                    #self.summary_writer.add_summary(summary_str,count)

                    _, summary_str = self.sess.run([g_optimizer, self.g_summary_run], feed_dict={
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    self.summary_writer.add_summary(summary_str, count)

                    errD_fake =  self.d_loss_fake.eval({
                        self.z:batch_z,
                        self.y:batch_labels
                    })

                    errD_real = self.d_loss_real.eval({
                        self.inputs:batch_images,
                        self.y:batch_labels
                    })

                    errG = self.g_loss.eval({
                        self.z:batch_z,
                        self.y:batch_labels
                    })
                else:
                    pass

                count += 1
                print("Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f,d_loss:%.8f,g_loss:%.8f" % (epoch,config.epoch,idx,batch_idxs,time.time() - start_time,errD_fake + errD_real,errG))

                if count % 10 == 0:
                    if config.dataset == "mnist":
                        samples,d_loss,g_loss = self.sess.run([self.sampler_data,self.d_loss,self.g_loss],feed_dict={
                            self.z:sample_z,
                            self.inputs:sample_inputs,
                            self.y:sample_labels,
                        })

                if count % 50 == 0:
                    self.save_mode(config.checkpoint_dir, count)

























