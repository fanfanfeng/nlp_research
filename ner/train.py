# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import os

import argparse
from ner import data_process
from ner.tf_models.base_ner_model import NerConfig
from ner.tf_models.bilstm import BiLSTM
from ner.tf_models.idcnn import IdCnn
from ner.tf_models.bert_ner_model import BertNerModel
import tensorflow as tf
from ner.data_utils import input_fn,make_tfrecord_files
from ner.bert_data_utils import input_fn as bert_input_fn
from ner.bert_data_utils import make_tfrecord_files as bert_make_tfrecord_files
from third_models.bert import modeling as bert_modeling



def argument_parser():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--output_path',type=str,default='output/',help="中间文件生成目录")
    parser.add_argument('--origin_data', type=str, default=None, help="原始数据地址")
    parser.add_argument('--data_type', type=str, default="default", help="原始数据格式，，目前支持默认的，还有rasa格式")
    parser.add_argument('--ner_type', type=str, default="bilstm", help="神经网络类型：idcnn or bilstm" )

    parser.add_argument('--device_map', type=str, default="0", help="gpu 的设备id")
    parser.add_argument('--use_bert',action='store_true',help='是否使用bert')
    parser.add_argument('--bert_model_path',type=str,help='bert模型目录')

    parser.add_argument('--max_sentence_len', type=int,default=50, help='一句话的最大长度')
    return parser.parse_args()


def train(arguments):
    if arguments.data_type == 'default':
        data_processer = data_process.NormalData(arguments.origin_data,output_path=arguments.output_path)
    else:
        data_processer = None #data_process.RasaData(arguments.origin_data, output_path=arguments.output_path)

    vocab, vocab_list, labels = data_processer.load_vocab_and_labels()

    ner_config = NerConfig(vocab_size=len(vocab_list),num_tags=len(labels),max_seq_length=50)
    ner_config.output_path = output_path
    if not os.path.exists(ner_config.output_path):
        os.makedirs(ner_config.output_path)


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with tf.Graph().as_default():
        training_input_x,training_input_y = input_fn(os.path.join(arguments.output_path,'train.tfrecord'),
                                                     shuffle_num=500000,
                                                     mode=tf.estimator.ModeKeys.TRAIN,
                                                     epochs= 30,
                                                     batch_size= ner_config.batch_size,
                                                     max_sentence_length=ner_config.max_seq_length,
                                                     )
        if arguments.ner_type == "idcnn":
            ner_config.filter_width = 3
            ner_config.num_filter = ner_config.hidden_size
            ner_config.repeat_times = 4
            classify_model = IdCnn(ner_config)
        else:
            ner_config.cell_type = 'lstm'
            ner_config.bilstm_layer_nums = 2
            classify_model = BiLSTM(ner_config)
        classify_model.train(training_input_x,training_input_y)

        classify_model.make_pb_file(ner_config.output_path)

def bert_train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    if args.data_type == 'default':
        data_processer = data_process.NormalData(args.origin_data,output_path=args.output_path)
    else:
        data_processer = None #data_process.RasaData(arguments.origin_data, output_path=arguments.output_path)

    vocab, vocab_list, labels = data_processer.load_vocab_and_labels()

    bert_config = bert_modeling.BertConfig.from_json_file(os.path.join(args.bert_model_path,"bert_config.json"))
    if args.max_sentence_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_sentence_len, bert_config.max_position_embeddings)
        )

    ner_config = NerConfig(vocab_size=len(vocab_list), num_tags=len(labels), max_seq_length=args.max_sentence_len)
    ner_config.output_path = output_path
    if not os.path.exists(ner_config.output_path):
        os.makedirs(ner_config.output_path)


    with tf.Graph().as_default():
        bert_input = bert_input_fn(os.path.join(args.output_path,'train.tfrecord'),
                                                     mode=tf.estimator.ModeKeys.TRAIN,
                                                     batch_size= ner_config.batch_size,
                                                     max_sentence_length=ner_config.max_seq_length
                                                     )
        if args.ner_type == "idcnn":
            ner_config.filter_width = 3
            ner_config.num_filter = ner_config.hidden_size
            ner_config.repeat_times = 4
        else:
            ner_config.cell_type = 'lstm'
            ner_config.bilstm_layer_nums = 2
        ner_config.ner_type = args.ner_type
        ner_config.embedding_size = bert_config.hidden_size
        ner_config.bert_model_path = args.bert_model_path
        model = BertNerModel(ner_config,bert_config)
        model.train(bert_input['input_ids'],bert_input['input_mask'],bert_input['segment_ids'],bert_input['label_ids'])
        model.make_pb_file(ner_config.output_path)


if __name__ == '__main__':

    argument_dict = argument_parser()


    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), argument_dict.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    argument_dict.output_path = output_path
    if argument_dict.use_bert:
        bert_make_tfrecord_files(argument_dict)
        bert_train(argument_dict)
    else:
        make_tfrecord_files(argument_dict)
        train(argument_dict)
