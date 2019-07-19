# create by fanfan on 2019/4/22 0022
from category.tf_utils import bert_data_process
from utils.tfrecord_api import _int64_feature
import tensorflow as tf
import os

def pad_sentence(sentence, max_sentence_length,vocabulary):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    tokens = []
    for token in sentence:
        word_and_type = token.split("\\")
        tokens.append(word_and_type[0])

            # 序列截断
    if len(tokens) >= max_sentence_length - 1:
        tokens = tokens[0:(max_sentence_length - 2)]


    ntokens = []
    segment_ids = []


    # 加第一个开始字符
    ntokens.append('[CLS]')
    segment_ids.append(0)


    ## 整个句子转换
    for i,token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)


    # 句尾添加[SEP] 标志
    ntokens.append('[SEP]')
    segment_ids.append(0)



    unk_id = vocabulary.get('[UNK]')
    input_ids = [vocabulary.get(token,unk_id) for token in ntokens]
    input_mask = [1] * len(input_ids)


    while len(input_ids) < max_sentence_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

        # we don't concerned about it!
        ntokens.append("**NULL**")

    assert len(input_ids) == max_sentence_length
    assert len(input_mask) == max_sentence_length
    assert len(segment_ids) == max_sentence_length

    return input_ids,segment_ids,input_mask

def make_tfrecord_files(params):
    # tfrecore 文件写入
    tfrecord_save_path = os.path.join(params.output_path, "train.tfrecord")
    # tfrecore 文件写入
    tfrecord_test_path = os.path.join(params.output_path, "test.tfrecord")

    if not os.path.exists(tfrecord_save_path):
        if params.data_type == 'default':
            data_processer = bert_data_process.NormalData(params.origin_data,output_path=params.output_path)
        else:
            data_processer = bert_data_process.RasaData(params.origin_data, output_path=params.output_path)


        if os.path.exists(os.path.join(params.output_path,'vocab.txt')):
            vocab,vocab_list,labels = data_processer.load_vocab_and_intent()
        else:
            vocab,vocab_list,labels = data_processer.create_label_dict(bert_model_path=params.bert_model_path)

        # tfrecore 文件写入
        tfrecord_train_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)
        tfrecord_test_writer = tf.python_io.TFRecordWriter(tfrecord_test_path)

        labels_ids = {label:index for index,label in enumerate(labels)}

        if params.data_type == 'default':
            for file ,folder_intent in data_processer.getTotalfiles():
                for index,(sentence, intent) in enumerate(data_processer.load_single_file(file)):
                    input_ids, segment_ids, input_mask = pad_sentence(sentence, params.max_sentence_length, vocab)
                    # sentence_ids_string = np.array(sentence_ids).tostring()
                    if folder_intent == "":
                        intent_to_writer = intent
                    else:
                        intent_to_writer = folder_intent

                    feature_item = tf.train.Example(features=tf.train.Features(feature={
                        'input_ids': _int64_feature(input_ids, need_list=False),
                        'label_ids': _int64_feature(labels_ids[intent_to_writer]),
                        'segment_ids': _int64_feature(segment_ids, need_list=False),
                        'input_mask': _int64_feature(input_mask, need_list=False)
                    }))

                    if index % 10 == 1:
                        tfrecord_test_writer.write(feature_item.SerializeToString())
                    else:
                        tfrecord_train_writer.write(feature_item.SerializeToString())
        else:
            for sentence,intent in data_processer.load_folder_data(data_processer.train_folder):
                input_ids, segment_ids, input_mask = pad_sentence(sentence, params.max_sentence_length, vocab)
                feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'input_ids': _int64_feature(input_ids, need_list=False),
                    'label_ids': _int64_feature(labels_ids[intent]),
                    'segment_ids': _int64_feature(segment_ids, need_list=False),
                    'input_mask': _int64_feature(input_mask, need_list=False)
                }))
                tfrecord_train_writer.write(feature_item.SerializeToString())


            for sentence,intent in data_processer.load_folder_data(data_processer.test_folder):
                input_ids, segment_ids, input_mask = pad_sentence(sentence, params.max_sentence_length, vocab)
                feature_item = tf.train.Example(features=tf.train.Features(feature={
                    'input_ids': _int64_feature(input_ids, need_list=False),
                    'label_ids': _int64_feature(labels_ids[intent]),
                    'segment_ids': _int64_feature(segment_ids, need_list=False),
                    'input_mask': _int64_feature(input_mask, need_list=False)
                }))
                tfrecord_test_writer.write(feature_item.SerializeToString())

        tfrecord_train_writer.close()
        tfrecord_test_writer.close()




def input_fn(input_file,batch_size,max_sentence_length,shuffle_num,mode=tf.estimator.ModeKeys.TRAIN):
    name_to_features = {
        'input_ids':tf.FixedLenFeature([max_sentence_length],tf.int64),
        'input_mask':tf.FixedLenFeature([max_sentence_length],tf.int64),
        'segment_ids':tf.FixedLenFeature([max_sentence_length],tf.int64),
        'label_ids':tf.FixedLenFeature([],tf.int64)
    }

    def _decode_record(record):
        example = tf.parse_single_example(record,name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    tf_record_reader = tf.data.TFRecordDataset(input_file)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print(mode)
        tf_record_reader = tf_record_reader.repeat()
        tf_record_reader = tf_record_reader.shuffle(buffer_size=shuffle_num)
    dataset = tf_record_reader.apply(tf.data.experimental.map_and_batch(lambda record:_decode_record(record),
                                                   batch_size,num_parallel_calls=8))
    #if mode == tf.estimator.ModeKeys.TRAIN:
    #    print(mode)
    #    d = d.repeat()
    #    d = d.shuffle(buffer_size=batch_size*1000)
    #if mode == tf.estimator.ModeKeys.TRAIN:
    #    dataset = tf_record_reader.map(_decode_record).repeat(30).shuffle(batch_size*10000).batch(batch_size)
    #else:
    #    dataset = tf_record_reader.map(_decode_record).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    example_item = iterator.get_next()
    return example_item