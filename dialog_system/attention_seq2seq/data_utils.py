# create by fanfan on 2019/5/24 0024

import os
import tensorflow as tf
from dialog_system.attention_seq2seq.data_process import NormalData
from utils.tfrecord_api import _int64_feature
import re

_PAD = '_PAD'
_GO = '_GO'
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def find_and_validate(line):
    return ''.join(re.findall(r'[\u4e00-\u9fff]+', line))

def pad_sentence(sentence, max_sentence,vocabulary,decoder=False):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')
    PAD_ID = vocabulary.get('_PAD')
    START_ID = vocabulary.get("_GO")
    END_ID = vocabulary.get("_EOS")
    sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
    if len(sentence_batch_ids) > max_sentence - 1 :
        if decoder:
            sentence_in = [START_ID] + sentence_batch_ids[:max_sentence -1]
            sentence_out =  sentence_batch_ids[:max_sentence -1] + [END_ID]
        else:
            sentence_in = sentence_batch_ids[:max_sentence ]
            sentence_out = ""
    else:
        if decoder:
            sentence_in = [START_ID] + sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids) - 1)
            sentence_out = sentence_batch_ids + [END_ID] + [PAD_ID]* (max_sentence - len(sentence_batch_ids) - 1)
        else:
            sentence_in = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))
            sentence_out = ""

    return sentence_in,sentence_out

def make_tfrecord_files(args):
    tfrecord_train_save_path = os.path.join(args.output_path, "train.tfrecord")
    tfrecord_test_save_path = os.path.join(args.output_path, "test.tfrecord")
    if not os.path.exists(tfrecord_train_save_path):
        data_processer = NormalData(args.origin_data,output_path=args.output_path,max_vocab_size=args.vocab_size,min_freq=args.filter_size)
        if os.path.exists(os.path.join(args.output_path,'vocab.txt')):
            vocab,vocab_list = data_processer.load_vocab_and_intent()
        else:
            vocab,vocab_list = data_processer.create_vocab_dict()

        # tfrecore 文件写入
        train_tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_train_save_path)
        test_tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_test_save_path)




        for file  in data_processer.getTotalfiles():
            for index, (source, target) in enumerate(data_processer.load_single_file(file)):
                #source,target = find_and_validate(source),find_and_validate(target)
                if source == [] or target == []:
                    continue
                source_ids,_ = pad_sentence(source, args.max_seq_length, vocab)
                target_input,target_out = pad_sentence(target,args.max_seq_length,vocab,decoder=True)



                train_feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'source': _int64_feature(source_ids, need_list=False),
                    'target_input': _int64_feature(target_input, need_list=False),
                    'target_output': _int64_feature(target_out, need_list=False),
                }))

                if index % 10 == 1:
                    test_tfrecord_writer.write(train_feature_item.SerializeToString())
                else:
                    train_tfrecord_writer.write(train_feature_item.SerializeToString())
        test_tfrecord_writer.close()
        train_tfrecord_writer.close()

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
            'target_input' : tf.FixedLenFeature([max_sentence_length],tf.int64),
            'target_output': tf.FixedLenFeature([max_sentence_length], tf.int64)
        }

        features_var = tf.parse_single_example(serializer_item,features)

        source = tf.cast(features_var['source'],tf.int64)
        target_input = tf.cast(features_var['target_input'],tf.int64)
        target_output = tf.cast(features_var['target_output'],tf.int64)

        return source,target_input,target_output


    if not os.path.exists(input_file):
        raise FileNotFoundError("tfrecord not found")


    tf_record_reader = tf.data.TFRecordDataset(input_file)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_record_reader = tf_record_reader.repeat()
        tf_record_reader = tf_record_reader.shuffle(buffer_size=batch_size*100)
    dataset = tf_record_reader.apply(tf.data.experimental.map_and_batch(lambda record:parse_single_tfrecord(record),
                                                   batch_size,num_parallel_calls=8))

    iterator = dataset.make_one_shot_iterator()

    input, decoder_inpout,decoder_output = iterator.get_next()
    return input, decoder_inpout,decoder_output



