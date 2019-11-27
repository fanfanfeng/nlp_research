# create by fanfan on 2018/8/16 0016
from GAN.cartoon.dcgan import DCGAN
from GAN.cartoon.dcgan import config
import tensorflow as tf

with tf.Session() as sess:
    if config.dataset == "mnist":
        dcgan_obj = DCGAN(sess,y_dim=10,checkoutpoint_dir=config.checkpoint_dir,sample_dir=config.sample_dir,c_dim=1,dataset_name='mnist',input_height=28,input_width=28,output_height=28,output_width=28,batch_size=config.batch_size,sample_num= config.batch_size,crop=False)
        dcgan_obj.train()