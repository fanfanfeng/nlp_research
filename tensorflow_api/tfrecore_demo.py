import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def _int64_feature(value):
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# #写TFrecord文件
minist = input_data.read_data_sets("mnist_data/", dtype=tf.uint8, one_hot=True) #mnist_data/目录下若没有mnist文件，则会自动下载
images = minist.train.images
lables = minist.train.labels
pixs = images.shape[1]
num_examples = minist.train.num_examples
filename = "minist.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for i in range(num_examples):
    images_raw = images[i].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={'label': _int64_feature(np.argmax(lables[i])),'image_raw': _bytes_feature(images_raw)}))
    writer.write(example.SerializeToString())
writer.close()

# 读TFrecord文件
import tensorflow as tf

filename_name = "minist.tfrecords"  # TFrecord文件名
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer([filename_name])
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={"image_raw": tf.FixedLenFeature([], tf.string),
                                                                 "label": tf.FixedLenFeature([], tf.int64)})
image = tf.decode_raw(features["image_raw"], tf.uint8)
labels = tf.cast(features["label"], tf.int32)
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(10000):
    images, label = sess.run([image, labels])
    ss = np.reshape(images, [28, 28, 1])
    #cv2.imshow("mnist", ss)  # 显示出从TFrecord文件中读出来的图片
    print(label)

