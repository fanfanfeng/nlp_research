# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import tensorflow as tf
from utils.tfrecord_api import _int64_feature
from category.data_utils import pad_sentence
from category.tf_models.classify_cnn_model import ClassifyCnnModel
from category.tf_models.base_classify_model import ClassifyConfig
import os
import tqdm
import argparse
from category import  data_process
from category.data_utils import input_fn,make_tfrecord_files



def argument_parser():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--output_path', type=str, default='output/', help="中间文件生成目录")
    parser.add_argument('--origin_data', type=str, default=None, help="原始数据地址")
    parser.add_argument('--data_type', type=str, default="default", help="原始数据格式，，目前支持默认的，还有rasa格式")
    parser.add_argument('--category_type', type=str, default="bilstm", help="神经网络类型：cnn or bilstm,如果是空字符串，则直接接一个全连接层输出")

    parser.add_argument('--device_map', type=str, default="0", help="gpu 的设备id")
    parser.add_argument('--use_bert', action='store_true', help='是否使用bert')
    parser.add_argument('--bert_model_path', type=str, help='bert模型目录')

    parser.add_argument('--max_sentence_len', type=int, default=50, help='一句话的最大长度')

    return parser.parse_args()






def train(arguments):
    if arguments.data_type == 'default':
        data_processer = data_process.NormalData(arguments.origin_data,output_path=arguments.output_path)
    else:
        data_processer = data_process.RasaData(arguments.origin_data, output_path=arguments.output_path)
    classify_config = ClassifyConfig(vocab_size=None)
    classify_config.output_path = arguments.output_path
    if not os.path.exists(classify_config.output_path):
        os.makedirs(classify_config.output_path)

    vocab,vocab_list,intent = data_processer.load_vocab_and_intent()


    classify_config.vocab_size = len(vocab_list)
    classify_config.num_tags = len(intent)

    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.device_map
    with tf.Graph().as_default():
        training_input_x,training_input_y = input_fn(os.path.join(classify_config.output_path,"train.tfrecord"),
                                                     classify_config.batch_size,
                                                     classify_config.max_sentence_length,
                                                     mode=tf.estimator.ModeKeys.TRAIN)

        classify_model = ClassifyCnnModel(classify_config)
        classify_model.train(training_input_x,training_input_y)

        classify_model.make_pb_file(classify_config.output_path)

if __name__ == '__main__':

    argument_dict = argument_parser()


    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), argument_dict.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    argument_dict.output_path = output_path
    if argument_dict.use_bert:
        #bert_make_tfrecord_files(argument_dict)
        #bert_train(argument_dict)
        pass
    else:
        make_tfrecord_files(argument_dict)
        train(argument_dict)
