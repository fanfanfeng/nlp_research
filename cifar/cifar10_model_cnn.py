__author__ = 'fanfan'
import re
import tensorflow as tf

class settings(object):
    batch_size = 128  #训练时的batch_size
    data_dir = r'data/cifar10_data' #数据目录


    image_size = 32
    num_class = 10
    num_examples_per_epoch_for_train = 50000
    num_examples_per_epoch_for_eval = 10000

    #训练设置参数
    moving_average_decay = 0.999
    num_epoch_per_decay = 350
    learning_rate_decap_factor = 0.1
    initial_learning_rate = 0.1

    #使用gpu的时候，标识op name
    tower_name = 'tower'
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    '''
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    '''
    tensor_name = re.sub("%s_[0-9]*/" % settings.tower_name,"",x.op.name)
    tf.summary.histogram(tensor_name + "/activations",x)
    tf.summary.scalar(tensor_name + "/sparsity",tf.nn.zero_fraction(x))

def _variable_on_cpu(name,shape,initializer):
    with tf.device("/cpu:0"):
        dtype = tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    dtype = tf.float32
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losse',weight_decay)
    return var


