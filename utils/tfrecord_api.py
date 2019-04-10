# create by fanfan on 2019/4/8 0008
import tensorflow as tf


def _int64_feature(value,need_list=True):
    if need_list:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value,need_list=True):
    if need_list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float32_feature(value,need_list=True):
    if need_list:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))