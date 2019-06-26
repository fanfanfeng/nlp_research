# create by fanfan on 2019/4/10 0010
import tensorflow as tf
import os
from ner import data_process
_START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
from utils.tfrecord_api import _int64_feature



def pad_sentence(sentence, max_sentence,vocabulary,label_dict):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')
    PAD_ID = vocabulary.get('_PAD')

    sentence_ids = []
    label_ids = []
    for token in sentence:
        word_and_type = token.split("\\")
        sentence_ids.append(word_and_type[0])
        if len(word_and_type) == 2:
            label_ids.append(word_and_type[1])
        else:
            label_ids.append('O')

    sentence_ids = [vocabulary.get(w, UNK_ID) for w in sentence_ids]
    label_ids = [label_dict.get(w, 0) for w in label_ids]
    if len(sentence_ids) > max_sentence:
        sentence_ids = sentence_ids[:max_sentence]
        label_ids = label_ids[:max_sentence]
    else:
        sentence_ids = sentence_ids + [PAD_ID] * (max_sentence - len(sentence_ids))
        label_ids = label_ids + [0] * (max_sentence - len(label_ids))
    return sentence_ids,label_ids


def pad_sentence_rasa(sentence,labels, max_sentence,vocabulary,label_dict):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')
    PAD_ID = vocabulary.get('_PAD')


    sentence_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
    label_ids = [label_dict.get(w, 0) for w in labels]
    if len(sentence_ids) > max_sentence:
        sentence_ids = sentence_ids[:max_sentence]
        label_ids = label_ids[:max_sentence]
    else:
        sentence_ids = sentence_ids + [PAD_ID] * (max_sentence - len(sentence_ids))
        label_ids = label_ids + [0] * (max_sentence - len(label_ids))
    return sentence_ids,label_ids



def input_fn(tf_record_filename,batch_size, shuffle_num,max_sentence_length,mode=tf.estimator.ModeKeys.TRAIN):
    """
     build tf.data set for input pipeline

    :param classify_config: classify config dict
    :param shuffle_num: type int number , random select the data
    :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
    :return: set() with type of (tf.data , and labels)
    """
    def parse_single_tfrecord(serializer_item):
        features = {
            'label': tf.FixedLenFeature([max_sentence_length],tf.int64),
            'sentence' : tf.FixedLenFeature([max_sentence_length],tf.int64)
        }

        features_var = tf.parse_single_example(serializer_item,features)

        labels = tf.cast(features_var['label'],tf.int64)
        sentence = tf.cast(features_var['sentence'],tf.int64)
        return sentence,labels



    tf_record_reader = tf.data.TFRecordDataset(tf_record_filename)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print(mode)
        tf_record_reader = tf_record_reader.repeat()
        tf_record_reader = tf_record_reader.shuffle(buffer_size=shuffle_num)
    dataset = tf_record_reader.apply(tf.data.experimental.map_and_batch(lambda record: parse_single_tfrecord(record),
                                                                        batch_size, num_parallel_calls=8))

    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels


def make_tfrecord_files(params):
    if params.data_type == 'default':
        data_processer = data_process.NormalData(params.origin_data,output_path=params.output_path)
    else:
        data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)

    if os.path.exists(os.path.join(params.output_path,'vocab.txt')):
        vocab,vocab_list,labels = data_processer.load_vocab_and_labels()
    else:
        vocab,vocab_list,labels = data_processer.create_vocab_dict()

    labels_ids = {key:index for index,key in enumerate(labels)}
    # tfrecore 文件写入
    tfrecord_save_path = os.path.join(params.output_path,"train.tfrecord")
    tfrecord_train_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)

    # tfrecore 文件写入
    tfrecord_test_path = os.path.join(params.output_path, "test.tfrecord")
    tfrecord_test_writer = tf.python_io.TFRecordWriter(tfrecord_test_path)

    if params.data_type == 'default':
        for file, folder_intent in data_processer.getTotalfiles():
            for index, sentence in enumerate(data_processer.load_single_file(file)):
                sentence_ids, sentence_labels_ids = pad_sentence(sentence, params.max_sentence_length, vocab,
                                                                 labels_ids)
                feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(sentence_labels_ids, need_list=False),
                    'sentence': _int64_feature(sentence_ids, need_list=False)
                }))
                if index % 10 == 1:
                    tfrecord_test_writer.write(feature_item.SerializeToString())
                else:
                    tfrecord_test_writer.write(feature_item.SerializeToString())
    else:
        for sentence, sentence_labels in data_processer.load_folder_data(data_processer.train_folder):
            sentence_ids,sentence_labels_ids = pad_sentence_rasa(sentence, sentence_labels,params.max_sentence_length, vocab,labels_ids)
            feature_item = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(sentence_labels_ids, need_list=False),
                'sentence': _int64_feature(sentence_ids, need_list=False)
            }))
            tfrecord_train_writer.write(feature_item.SerializeToString())

        for sentence, sentence_labels in data_processer.load_folder_data(data_processer.test_folder):
            sentence_ids, sentence_labels_ids = pad_sentence_rasa(sentence, sentence_labels, params.max_sentence_length,
                                                                  vocab, labels_ids)
            feature_item = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(sentence_labels_ids, need_list=False),
                'sentence': _int64_feature(sentence_ids, need_list=False)
            }))
            tfrecord_test_writer.write(feature_item.SerializeToString())

    tfrecord_train_writer.close()
    tfrecord_test_writer.close()




