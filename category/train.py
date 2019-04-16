# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import tensorflow as tf
from utils.tfrecord_api import _int64_feature
from category.data_utils import pad_sentence
from category.tf_models.classify_cnn_model import CNNConfig,ClassifyCnnModel
import os
import tqdm
import argparse
from category import  data_process
import threadpool



def argument_parser():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--output_path',type=str,default='output/',help="中间文件生成目录")
    parser.add_argument('--origin_data', type=str, default=None, help="原始数据地址")
    parser.add_argument('--data_type', type=str, default="default", help="原始数据格式，，目前支持默认的，还有rasa格式")

    return parser.parse_args()




def make_tfrecord_files(arguments,classify_config):
    if arguments.data_type == 'default':
        data_processer = data_process.NormalData(arguments.origin_data,output_path=arguments.output_path)
    else:
        data_processer = data_process.RasaData(arguments.origin_data, output_path=arguments.output_path)
    if os.path.exists(os.path.join(arguments.output_path,'vocab.txt')):
        vocab,vocab_list,intent = data_processer.load_vocab_and_intent()
    else:
        vocab,vocab_list,intent = data_processer.create_vocab_dict()

    intent_ids = {key:index for index,key in enumerate(intent)}
    # tfrecore 文件写入
    tfrecord_save_path = os.path.join(classify_config.save_path,"train.tfrecord")
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)

    def thread_write_to_file(file,intent_folder):
        for sentence, intent in data_processer.load_single_file(file):
            sentence_ids = pad_sentence(sentence, classify_config.max_sentence_length, vocab)
            # sentence_ids_string = np.array(sentence_ids).tostring()

            if intent_folder == "":
                intent_to_writer = intent
            else:
                intent_to_writer = intent_folder
            print('writer intent:%s' % intent_to_writer)
            train_feature_item = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(intent_ids[intent_to_writer]),
                'sentence': _int64_feature(sentence_ids, need_list=False)
            }))
            tfrecord_writer.write(train_feature_item.SerializeToString())

    pool = threadpool.ThreadPool(20)
    requests = threadpool.makeRequests(thread_write_to_file,[(file_and_intent,None) for file_and_intent in data_processer.getTotalfiles()])
    [pool.putRequest(req) for req in requests]
    pool.wait()
    tfrecord_writer.close()

def input_fn(classify_config, shuffle_num, mode,epochs):
    """
     build tf.data set for input pipeline

    :param classify_config: classify config dict
    :param shuffle_num: type int number , random select the data
    :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
    :return: set() with type of (tf.data , and labels)
    """
    def parse_single_tfrecord(serializer_item):
        features = {
            'label': tf.FixedLenFeature([],tf.int64),
            'sentence' : tf.FixedLenFeature([classify_config.max_sentence_length],tf.int64)
        }

        features_var = tf.parse_single_example(serializer_item,features)

        labels = tf.cast(features_var['label'],tf.int64)
        #sentence = tf.decode_raw(features_var['sentence'],tf.uint8)
        sentence = tf.cast(features_var['sentence'],tf.int64)
        return sentence,labels



    tf_record_filename = os.path.join(classify_config.save_path,'train.tfrecord')
    if not os.path.exists(tf_record_filename):
        raise FileNotFoundError("tfrecord not found")
    tf_record_reader = tf.data.TFRecordDataset(tf_record_filename)


    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf_record_reader.map(parse_single_tfrecord).shuffle(shuffle_num).batch(classify_config.batch_size).repeat(epochs)
    else:
        dataset = tf_record_reader.map(parse_single_tfrecord).batch(classify_config.batch_size)

    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    data = tf.reshape(data,[-1,classify_config.max_sentence_length])
    return data, labels

def train(arguments):
    if arguments.data_type == 'default':
        data_processer = data_process.NormalData(arguments.origin_data,output_path=arguments.output_path)
    else:
        data_processer = data_process.RasaData(arguments.origin_data, output_path=arguments.output_path)
    classify_config = CNNConfig()
    classify_config.save_path = output_path
    if not os.path.exists(classify_config.save_path):
        os.makedirs(classify_config.save_path)

    vocab,vocab_list,intent = data_processer.load_vocab_and_intent()


    classify_config.vocab_size = len(vocab_list)
    classify_config.label_nums = len(intent)

    os.environ["CUDA_VISIBLE_DEVICES"] = classify_config.CUDA_VISIBLE_DEVICES
    with tf.Graph().as_default():
        training_input_x,training_input_y = input_fn(classify_config,
                                                          shuffle_num=500000,
                                                          mode=tf.estimator.ModeKeys.TRAIN,
                                                          epochs=classify_config.epochs)

        classify_model = ClassifyCnnModel(classify_config)
        classify_model.train(training_input_x,training_input_y)

        classify_model.make_pb_file(classify_config.save_path)

if __name__ == '__main__':

    argument_dict = argument_parser()

    classify_config = CNNConfig()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), argument_dict.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    classify_config.save_path = output_path
    argument_dict.output_path = output_path
    make_tfrecord_files(argument_dict, classify_config)
    train(argument_dict)
