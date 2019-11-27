# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import os

import argparse
from ner.tf_utils import data_process
from ner.tf_models.params import TestParams
from ner.tf_models.bilstm import BiLSTM
from ner.tf_models.idcnn import IdCnn
from ner.tf_models.bert_ner_model import BertNerModel
import tensorflow as tf
from ner.tf_utils.data_utils import input_fn,make_tfrecord_files
from ner.tf_utils.bert_data_utils import input_fn as bert_input_fn
from ner.tf_utils.bert_data_utils import make_tfrecord_files as bert_make_tfrecord_files
from third_models.bert import modeling as bert_modeling
import tqdm
from sklearn.metrics import f1_score



def argument_parser():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--output_path',type=str,default='output/',help="中间文件生成目录")
    parser.add_argument('--origin_data', type=str, default=None, help="原始数据地址")
    parser.add_argument('--data_type', type=str, default="default", help="原始数据格式，，目前支持默认的，还有rasa格式")
    parser.add_argument('--ner_type', type=str, default="idcnn", help="神经网络类型：idcnn or bilstm" )

    parser.add_argument('--device_map', type=str, default="0", help="gpu 的设备id")
    parser.add_argument('--use_bert',action='store_true',help='是否使用bert')
    parser.add_argument('--bert_model_path',type=str,help='bert模型目录')

    parser.add_argument('--max_sentence_length', type=int,default=20, help='一句话的最大长度')
    return parser.parse_args()


def train(params):
    if params.data_type == 'default':
        data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
    else:
        data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)

    vocab, vocab_list, labels = data_processer.load_vocab_and_labels()

    params.vocab_size = len(vocab_list)
    params.num_tags = len(labels)
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)


    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            training_input_x,training_input_y = input_fn(os.path.join(params.output_path,'train.tfrecord'),
                                                         shuffle_num=params.shuffle_num,
                                                         mode=tf.estimator.ModeKeys.TRAIN,
                                                         batch_size= params.batch_size,
                                                         max_sentence_length=params.max_sentence_length,
                                                         )
            if params.ner_type == "idcnn":
                ner_model = IdCnn(params)
            else:
                ner_model = BiLSTM(params)
            loss, global_step, train_op, merger_op = ner_model.make_train(training_input_x, training_input_y)
            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            ner_model.model_restore(sess, saver)

            best_f1 = 0
            for _ in tqdm.tqdm(range(params.total_train_steps), desc="steps", miniters=10):
                sess_loss, steps, _ = sess.run([loss, global_step, train_op])

                if steps % params.evaluate_every_steps == 1:
                    test_input_x, test_input_y = input_fn(os.path.join(params.output_path, "test.tfrecord"),
                                                          shuffle_num=params.shuffle_num,
                                                          batch_size=params.batch_size,
                                                          max_sentence_length=params.max_sentence_length,
                                                          mode=tf.estimator.ModeKeys.EVAL)
                    loss_test, predict_test,sentence_length = ner_model.make_test(test_input_x, test_input_y)

                    predict_var = []
                    train_y_var = []
                    loss_total = 0
                    num_batch = 0
                    try:
                        while 1:
                            loss_, predict_, test_input_y_,length = sess.run([loss_test, predict_test, test_input_y,sentence_length])
                            loss_total += loss_
                            num_batch += 1
                            for p_,t_,len_ in zip(predict_.tolist(),test_input_y_.tolist(),length.tolist()):

                                predict_var += p_[:len_]
                                train_y_var += t_[:len_]
                    except tf.errors.OutOfRangeError:
                        print("eval over")
                    if num_batch > 0:

                        f1_val = f1_score(train_y_var, predict_var, average='micro')
                        print("current step:%s ,loss:%s , f1 :%s" % (steps, loss_total / num_batch, f1_val))

                        if f1_val >= best_f1:
                            saver.save(sess, params.model_path, steps)
                            print("new best f1: %s ,save to dir:%s" % (f1_val, params.output_path))
                            best_f1 = f1_val

            ner_model.make_pb_file(params.output_path)

def bert_train(params):
    os.environ['CUDA_VISIBLE_DEVICES'] = params.device_map
    if params.data_type == 'default':
        data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
    else:
        data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)

    vocab, vocab_list, labels = data_processer.load_vocab_and_labels()

    bert_config = bert_modeling.BertConfig.from_json_file(os.path.join(params.bert_model_path,"bert_config.json"))
    params.vocab_size = len(vocab_list)
    params.num_tags = len(labels)
    if params.max_sentence_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (params.max_sentence_length, bert_config.max_position_embeddings)
        )

    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)


    with tf.Graph().as_default():
        bert_input = bert_input_fn(os.path.join(params.output_path,'train.tfrecord'),
                                                     mode=tf.estimator.ModeKeys.TRAIN,
                                                     batch_size= params.batch_size,
                                                     max_sentence_length=params.max_sentence_length
                                                     )

        model = BertNerModel(params,bert_config)
        model.train(bert_input['input_ids'],bert_input['input_mask'],bert_input['segment_ids'],bert_input['label_ids'])
        model.make_pb_file(params.output_path)


if __name__ == '__main__':

    argument_dict = argument_parser()

    params = TestParams()
    params.update_object(argument_dict)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), argument_dict.output_path)
    params.output_path = output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if params.use_bert:
        bert_make_tfrecord_files(params)
        bert_train(params)
    else:
        make_tfrecord_files(params)
        train(params)
