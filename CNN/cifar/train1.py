from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = r"E:\tensorflow_data\cifar10_data\cifar-10-batches-py"

# Use Floyd's cifar-10 dataset if present
floyd_cifar10_location = r'E:\tensorflow_data\cifar10_data/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()

from CNN.cifar import helper
import numpy as np

# Explore the dataset
batch_id = 1
sample_id = 2
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    return x/255

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    out = np.zeros((len(x), 10))
    out[np.arange(len(x)), x] = 1
    return out

helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle



# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, [None]+list(image_shape), name = 'x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, [None, n_classes], name = 'y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, name = 'keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function

    bias = tf.Variable(tf.truncated_normal([conv_num_outputs], stddev=0.05))
    weight = tf.Variable(
        tf.truncated_normal([conv_ksize[0], conv_ksize[1], int(x_tensor.shape.as_list()[3]), conv_num_outputs],
                            stddev=0.05), name='weights')

    # Convolution Layer with relu activation
    network = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding="SAME")
    network = tf.nn.bias_add(network, bias)
    network = tf.nn.relu(network)

    # Max Pool Layer

    network = tf.nn.max_pool(network, strides=[1, pool_strides[0], pool_strides[1], 1],
                             ksize=[1, pool_ksize[0], pool_ksize[1], 1], padding="SAME")
    return network
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    return tf.reshape(x_tensor, [-1, x_tensor.shape.as_list()[1]* x_tensor.shape.as_list()[2]* x_tensor.shape.as_list()[3]])


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function

    weight = tf.Variable(tf.truncated_normal([x_tensor.shape.as_list()[1], num_outputs]))
    bias = tf.Variable(tf.truncated_normal([num_outputs]))

    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    weight = tf.Variable(tf.truncated_normal([x_tensor.shape.as_list()[1], num_outputs]))
    bias = tf.Variable(tf.truncated_normal([num_outputs]))

    return tf.add(tf.matmul(x_tensor, weight), bias)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)



    network = conv2d_maxpool(x, conv_num_outputs=50,
                             conv_ksize=(5, 5),
                             conv_strides=(1, 1),
                             pool_ksize=(2, 2),
                             pool_strides=(2, 2))
    network = conv2d_maxpool(x, conv_num_outputs=100,
                             conv_ksize=(5, 5),
                             conv_strides=(1, 1),
                             pool_ksize=(2, 2),
                             pool_strides=(2, 2))
    network = conv2d_maxpool(network, conv_num_outputs=150,
                             conv_ksize=(5, 5),
                             conv_strides=(1, 1),
                             pool_ksize=(2, 2),
                             pool_strides=(2, 2))
    # network = conv2d_maxpool(network, 64, (5,5), (1,1), (2,2), (2,2))


    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    network = flatten(network)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)

    network = fully_conn(network, 1024)
    network = tf.nn.dropout(network, keep_prob)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)

    logits = output(network, 10)

    # TODO: return output
    return logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer,feed_dict={x: feature_batch,y: label_batch, keep_prob: keep_probability})


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss = session.run(cost,
                       feed_dict={x: feature_batch,
                                  y: label_batch,
                                  keep_prob: 1.0})

    valid_acc = session.run(accuracy,
                            feed_dict={x: valid_features,
                                       y: valid_labels,
                                       keep_prob: 1.0})

    print('Loss: {:>15.4f} Validation Accuracy: {:.4f}'.format(loss, valid_acc))\

# TODO: Tune Parameters
epochs = 50
batch_size = 256
keep_probability = 0.8

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from CNN.cifar import cifar10
images_train, cls_train, labels_train = cifar10.load_training_data()
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        #for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
        for batch_i in range(50000//batch_size - 1):
            feature_batch = images_train[batch_i * batch_size: (batch_i + 1) * batch_size]
            label_batch = labels_train[batch_i * batch_size: (batch_i + 1) * batch_size]
            train_neural_network(sess, optimizer, keep_probability, feature_batch, label_batch)
            print_stats(sess, feature_batch, label_batch, cost, accuracy)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')

