import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def _int64_feature(value):
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# #写TFrecord文件
#minist = input_data.read_data_sets("mnist_data/", dtype=tf.uint8, one_hot=True) #mnist_data/目录下若没有mnist文件，则会自动下载
#images = minist.train.images
#lables = minist.train.labels
#pixs = images.shape[1]
#num_examples = minist.train.num_examples
#filename = "minist.tfrecords"
#writer = tf.python_io.TFRecordWriter(filename)
#for i in range(num_examples):
#    break
#    images_raw = images[i].tostring()
#    example = tf.train.Example(features=tf.train.Features(feature={'label': _int64_feature(np.argmax(lables[i])),'image_raw': _bytes_feature(images_raw)}))
#    writer.write(example.SerializeToString())
#writer.close()

# 读TFrecord文件
import tensorflow as tf
import os

def input_fn(filename):
    """
     build tf.data set for input pipeline

    :param classify_config: classify config dict
    :param shuffle_num: type int number , random select the data
    :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
    :return: set() with type of (tf.data , and labels)
    """

    def parse_single_tfrecord(serializer_item):
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'sentence': tf.FixedLenFeature([], tf.string)
        }

        features_var = tf.parse_single_example(serializer_item, features)

        labels = tf.cast(features_var['label'], tf.int64)
        sentence = tf.decode_raw(features_var['sentence'], tf.uint8)
        sentence = tf.cast(sentence, tf.int64)
        return sentence, labels

    tf_record_filename = filename
    if not os.path.exists(tf_record_filename):
        raise FileNotFoundError("tfrecord not found")
    tf_record_reader = tf.data.TFRecordDataset(tf_record_filename)

    dataset = tf_record_reader.map(parse_single_tfrecord).shuffle(50000).batch(
            10).repeat(1)
    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels

filename_name = r"E:\nlp-data\tmp\models\default\classify\train.tfrecord"  # TFrecord文件名

image,labels = input_fn(filename_name)
image_reshape = tf.reshape(image,[-1,50])
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(10000):
    image_var, label = sess.run([image_reshape, labels])
    print(image_var)
    #ss = np.reshape(images, [28, 28, 1])
    #cv2.imshow("mnist", ss)  # 显示出从TFrecord文件中读出来的图片
    #print(label)

