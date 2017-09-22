# create by fanfan on 2017/8/25 0025
import tensorflow as tf
import os


projectDir =  os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(projectDir,'data')
train_file = "train.txt"
test_file = "test.txt"

model_dir = os.path.join(projectDir,'nn_models')
graph_dir = os.path.join(projectDir,'graph')

tf.app.flags.DEFINE_string('data_dir',data_dir,"训练预料目录" )
tf.app.flags.DEFINE_string('model_dir',model_dir,"模型保存目录")


#model config
tf.app.flags.DEFINE_float('learning_rate',0.5,'学习率')
tf.app.flags.DEFINE_float('learning_rate_decay_factor',0.9,'学习率衰减比例')
tf.app.flags.DEFINE_float('max_gradient_norm',5.0,"梯度截断最大值")
tf.app.flags.DEFINE_integer('batch_size',128,'批量训练的batch大小')
tf.app.flags.DEFINE_integer('lstm_size',256,'每一个lstm的大小')
tf.app.flags.DEFINE_integer('num_layers',2,'lstm的层数')
tf.app.flags.DEFINE_integer('steps_per_checkpoint',100,"每训多少次，保存一次模型")
tf.app.flags.DEFINE_float('dropout',0.5,"随机抛弃概率")

FLAGS = tf.app.flags.FLAGS

vocabulary_size = 40000 #词汇库大小
vocabulary_fliter_num = 3 #词库里面出现词语的最小频率





BUCKETS = [(5,10),(10,15),(20,25)]


