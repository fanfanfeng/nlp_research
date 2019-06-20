# create by fanfan on 2019/4/10 0010

import jieba
import json
import os
import sys
import tensorflow as tf
from category.data_process import RasaData,NormalData
import threading
from utils.tfrecord_api import _int64_feature

_START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
if 'win' in sys.platform:
    user_dict_path = r'E:\nlp-data\jieba_dict\dict_modify.txt'
else:
    user_dict_path = r'/data/python_project/rasa_corpus/jieba_dict/dict_modify.txt'

jieba.load_userdict(user_dict_path)




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

    if max(sentence_batch_ids) == 0:
        print(sentence_batch_ids)
    return sentence_batch_ids





def make_tfrecord_files(params):
    tfrecord_train_path = os.path.join(params.output_path, "train.tfrecord")
    tfrecord_test_path = os.path.join(params.output_path, "test.tfrecord")
    if not os.path.exists(tfrecord_train_path):
        if params.data_type == 'default':
            data_processer = NormalData(params.origin_data,output_path=params.output_path)
        else:
            data_processer = RasaData(params.origin_data, output_path=params.output_path)

        if os.path.exists(os.path.join(params.output_path,'vocab.txt')):
            vocab,vocab_list,intent = data_processer.load_vocab_and_intent()
        else:
            vocab,vocab_list,intent = data_processer.create_vocab_dict()

        intent_ids = {key:index for index,key in enumerate(intent)}
        # tfrecore 文件写入
        tfrecord_train_writer = tf.python_io.TFRecordWriter(tfrecord_train_path)
        tfrecord_test_writer = tf.python_io.TFRecordWriter(tfrecord_test_path)


        if params.data_type == 'default':
            for file ,folder_intent in data_processer.getTotalfiles():
                for index,(sentence, intent) in enumerate(data_processer.load_single_file(file)):
                    sentence_ids = pad_sentence(sentence, params.max_sentence_length, vocab)
                    if folder_intent == "":
                        intent_to_writer = intent
                    else:
                        intent_to_writer = folder_intent

                    feature_item = tf.train.Example(features=tf.train.Features(feature={
                        'label': _int64_feature(intent_ids[intent_to_writer]),
                        'sentence': _int64_feature(sentence_ids, need_list=False)
                    }))

                    if index % 10 == 1:
                        tfrecord_test_writer.write(feature_item.SerializeToString())
                    else:
                        tfrecord_train_writer.write(feature_item.SerializeToString())
        else:
            for sentence,intent in data_processer.load_folder_data(data_processer.train_folder):
                sentence_ids = pad_sentence(sentence, params.max_sentence_length, vocab)
                feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(intent_ids[intent]),
                    'sentence': _int64_feature(sentence_ids, need_list=False)
                }))
                tfrecord_train_writer.write(feature_item.SerializeToString())


            for sentence,intent in data_processer.load_folder_data(data_processer.test_folder):
                sentence_ids = pad_sentence(sentence, params.max_sentence_length, vocab)
                feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(intent_ids[intent]),
                    'sentence': _int64_feature(sentence_ids, need_list=False)
                }))
                tfrecord_test_writer.write(feature_item.SerializeToString())

        tfrecord_train_writer.close()
        tfrecord_test_writer.close()

def input_fn(input_file, batch_size,max_sentence_length,shuffle_num, mode=tf.estimator.ModeKeys.TRAIN):
    """
     build tf.data set for input pipeline

    :param input_file: classify config dict
    :param shuffle_num: type int number , random select the data
    :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
    :return: set() with type of (tf.data , and labels)
    """
    def parse_single_tfrecord(serializer_item):
        features = {
            'label': tf.FixedLenFeature([],tf.int64),
            'sentence' : tf.FixedLenFeature([max_sentence_length],tf.int64)
        }

        features_var = tf.parse_single_example(serializer_item,features)

        labels = tf.cast(features_var['label'],tf.int64)
        #sentence = tf.decode_raw(features_var['sentence'],tf.uint8)
        sentence = tf.cast(features_var['sentence'],tf.int64)
        return sentence,labels


    if not os.path.exists(input_file):
        raise FileNotFoundError("tfrecord not found")


    tf_record_reader = tf.data.TFRecordDataset(input_file)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print(mode)
        tf_record_reader = tf_record_reader.repeat()
        tf_record_reader = tf_record_reader.shuffle(buffer_size=shuffle_num)
    dataset = tf_record_reader.apply(tf.data.experimental.map_and_batch(lambda record:parse_single_tfrecord(record),
                                                   batch_size,num_parallel_calls=8))

    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels



