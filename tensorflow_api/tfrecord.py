# create by fanfan on 2019/4/4 0004
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import os

tfrecode_path = r'E:\tensorflow_data\tfrecord'

def get_tfrecords_example(feature,label):
    '''
    吧数据写入 Example
    :param feature: 
    :param label: 
    :return: 
    '''
    tfrecoreds_features = {}
    tfrecoreds_features['img_raw'] = tf.train.Feature(float_list = tf.train.FloatList(value=feature.reshape((28*28,)).tolist()))
    tfrecoreds_features['label'] = tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
    return tf.train.Example(features=tf.train.Features(feature=tfrecoreds_features))

def make_tfrecord(data,out_name='mninst_train',save_path=tfrecode_path):
    '''
    将所有数据都写入tfrecord文件
    :param data: 
    :param out_name: 
    :return: 
    '''
    feats,labels = data
    out_name += '.tfrecord'
    out_name = os.path.join(save_path,out_name)
    tfrecord_writer = tf.python_io.TFRecordWriter(out_name)
    data_length = len(labels)
    for i in range(data_length):
        exmp = get_tfrecords_example(feats[i],labels[i])
        exmp_serial = exmp.SerializeToString()
        tfrecord_writer.write(exmp_serial)
    tfrecord_writer.close()

import random

def generator_tfrecord():
    mnist = read_data_sets(r'E:\tensorflow_data\mnist',one_hot=False)
    n_datas = len(mnist.train.labels)
    index_list = list(range(n_datas))
    random.shuffle(index_list)

    ntrains = int(0.85 * n_datas)

    # make training set
    data = ([mnist.train.images[i] for i in index_list[:ntrains]],\
            [mnist.train.labels[i] for i in index_list[:ntrains]])
    make_tfrecord(data,out_name='mnist_train')

    # make validation set
    data = ([mnist.train.images[i] for i in index_list[ntrains:]],\
            [mnist.train.labels[i] for i in index_list[ntrains:]])

    make_tfrecord(data,out_name='mnist_val')



    # make test set
    data = (mnist.test.images,mnist.test.labels)
    make_tfrecord(data,out_name='mnist_test')

batch_size= 100
def parse_exmp(serial_exmp):
    features = {
        'img_raw':tf.FixedLenFeature([28*28],tf.float32),
        'label':tf.FixedLenFeature([],tf.int64)
    }

    feats = tf.parse_single_example(serial_exmp,features=features)
    #image = tf.decode_raw(feats['img_raw'],tf.uint8)
    img = feats['img_raw'] #tf.reshape(feats['img_raw'],[28,28,1])
    img = tf.cast(img,tf.float32)
    label = tf.cast(feats['label'],tf.int64)
    return img,label


def read_and_decode(file_name,shuffle=True,epoch=1):
    dataset = tf.data.TFRecordDataset(file_name)
    if shuffle:
        dataset = dataset.map(parse_exmp).repeat(epoch).batch(batch_size=batch_size).shuffle(buffer_size=1000)
    else:
        dataset = dataset.map(parse_exmp).repeat(epoch).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    img_input,label = iterator.get_next()
    return img_input,label


from tensorflow.contrib import slim


def cnn_model(input_x, input_y,reuse=False):
    x_image = tf.reshape(input_x, [-1, 28, 28, 1])
    h_conv1 = slim.conv2d(x_image, 32, 5,reuse=reuse)
    h_pool1 = slim.max_pool2d(h_conv1, 2)

    h_conv2 = slim.conv2d(h_pool1, 64, 5,reuse=reuse)
    h_pool2 = slim.max_pool2d(h_conv2, 2)

    h_flat = slim.flatten(h_pool2)

    h_fc1 = slim.fully_connected(h_flat, 1024,reuse=reuse)
    h_fc2 = slim.fully_connected(h_fc1, 10,activation_fn=None,reuse=reuse)

    logit = tf.nn.softmax(h_fc2)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=input_y))

    correct_prediction = tf.equal(tf.argmax(logit, 1), input_y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logit,loss,accuracy


def train():
    train_record,val_record,test_record = [os.path.join(r'E:\tensorflow_data\tfrecord','mnist_%s.tfrecord') % i for i in ['train','val','test']]

    with tf.Session() as sess:
        train_input,train_label = read_and_decode(train_record,shuffle=True,epoch=16)
        logits,loss,accuracy = cnn_model(train_input,train_label)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        global_step = tf.Variable(0,trainable=False)

        sess.run(tf.global_variables_initializer())

        try:
            while True:
                _,current_loss,current_acc,step = sess.run([train_op,loss,accuracy,global_step])
                if step % 100 ==0:
                    print("step:%s current loss:%s current_acc: %s" % (step,current_loss,current_acc))

        except tf.errors.OutOfRangeError:
            print("train over")


def print_tfrecord():
    train_record, val_record, test_record = ['mnist_%s.tfrecord' % i for i in ['train', 'val', 'test']]
    train_input, train_label = read_and_decode(os.path.join(r'E:\tensorflow_data\tfrecord',train_record), shuffle=True, epoch=16)

    with tf.Session() as sess:
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 因为我这里只有 2 张图片，所以下面循环 2 次
        for i in range(2):
            # 获取一张图片和其对应的类型
            label, image = sess.run([train_label, train_input])
            # 这里特别说明下：
            #   因为要想把图片保存成 TFRecord，那就必须先将图片矩阵转换成 string，即：
            #       pic2tfrecords.py 中 image_raw = image.tostring() 这行
            #   所以这里需要执行下面这行将 string 转换回来，否则会无法 reshape 成图片矩阵，请看下面的小例子：
            #       a = np.array([[1, 2], [3, 4]], dtype=np.int64) # 2*2 的矩阵
            #       b = a.tostring()
            #       # 下面这行的输出是 32，即： 2*2 之后还要再乘 8
            #       # 如果 tostring 之后的长度是 2*2=4 的话，那可以将 b 直接 reshape([2, 2])，但现在的长度是 2*2*8 = 32，所以无法直接 reshape
            #       # 同理如果你的图片是 500*500*3 的话，那 tostring() 之后的长度是 500*500*3 后再乘上一个数
            #       print len(b)
            #
            print(label)
            print(image)



if __name__ == '__main__':
    #generator_tfrecord()
    train()
    #print_tfrecord()
