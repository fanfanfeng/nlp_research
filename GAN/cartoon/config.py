# create by fanfan on 2018/8/16 0016
import numpy as np
learning_rate = 0.0001

epoch = 30
train_size = np.inf

batch_size = 64
input_height = 108 #, "The size of image to use (will be center cropped). [108]")
input_width = None #, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
output_height = 64 #, "The size of the output images to produce [64]")
output_width = None #, "The size of the output images to produce. If None, same value as output_height [None]")
dataset = "mnist" #, "The name of dataset [celebA, mnist, lsun]")
input_fname_pattern = "*.jpg" #, "Glob pattern of filename of input images [*]")
checkpoint_dir = "checkpoint" #, "Directory name to save the checkpoints [checkpoint]")
data_dir = "./data" #, "Root directory of dataset [data]")
sample_dir = "samples"#, "Directory name to save the image samples [samples]")
train = False #, "True for training, False for testing [False]")
crop = False #, "True for training, False for testing [False]")
visualize = False #, "True for visualizing, False for nothing [False]")
generate_test_images = 100 #, "Number of images to generate during test. [100]")