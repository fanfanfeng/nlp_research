# create by fanfan on 2019/5/24 0024

import json
import os
import sys
import tensorflow as tf
from dialog_system.attention_QAbot.data_process import NormalData
from utils.tfrecord_api import _int64_feature
import re

def find_and_validate(line):
    return ''.join(re.findall(r'[\u4e00-\u9fff]+', line))

def pad_sentence(sentence, max_sentence,vocabulary):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')
    PAD_ID = vocabulary.get('_PAD')
    sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
    if len(sentence_batch_ids) > max_sentence:
        sentence_batch_ids = sentence_batch_ids[:max_sentence]
    else:
        sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))
    return sentence_batch_ids

def make_tfrecord_files(args):
    tfrecord_save_path = os.path.join(args.output_path, "train.tfrecord")
    if not os.path.exists(tfrecord_save_path):
        data_processer = NormalData(args.origin_data,output_path=args.output_path)
        if os.path.exists(os.path.join(args.output_path,'vocab.txt')):
            vocab,vocab_list = data_processer.load_vocab_and_intent()
        else:
            vocab,vocab_list = data_processer.create_vocab_dict()

        # tfrecore 文件写入
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)

        def thread_write_to_file(file,tfrecord_writer):
            for source, target in data_processer.load_single_file(file):
                source_ids = pad_sentence(find_and_validate(source), args.max_sentence_len, vocab)
                target_ids = pad_sentence(find_and_validate(target),args.max_sentence_len,vocab)



                train_feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'source': _int64_feature(source_ids, need_list=False),
                    'target': _int64_feature(target_ids, need_list=False)
                }))
                tfrecord_writer.write(train_feature_item.SerializeToString())


        for file  in data_processer.getTotalfiles():
            thread_write_to_file(file,tfrecord_writer)
        tfrecord_writer.close()

def input_fn(input_file, batch_size,max_sentence_length, mode=tf.estimator.ModeKeys.TRAIN):
    """
     build tf.data set for input pipeline

    :param input_file: classify config dict
    :param shuffle_num: type int number , random select the data
    :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
    :return: set() with type of (tf.data , and labels)
    """
    def parse_single_tfrecord(serializer_item):
        features = {
            'source': tf.FixedLenFeature([max_sentence_length],tf.int64),
            'target' : tf.FixedLenFeature([max_sentence_length],tf.int64)
        }

        features_var = tf.parse_single_example(serializer_item,features)

        source = tf.cast(features_var['source'],tf.int64)
        target = tf.cast(features_var['target'],tf.int64)
        return source,target


    if not os.path.exists(input_file):
        raise FileNotFoundError("tfrecord not found")


    tf_record_reader = tf.data.TFRecordDataset(input_file)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_record_reader = tf_record_reader.repeat()
        tf_record_reader = tf_record_reader.shuffle(buffer_size=batch_size*1000)
    dataset = tf_record_reader.apply(tf.data.experimental.map_and_batch(lambda record:parse_single_tfrecord(record),
                                                   batch_size,num_parallel_calls=8))

    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels