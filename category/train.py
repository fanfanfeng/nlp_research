# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import tensorflow as tf
from utils.tfrecord_api import _int64_feature
from category.data_utils import pad_sentence
from category.tf_models.classify_cnn_model import ClassifyCnnModel
from category.tf_models.classify_bilstm_model import  ClassifyBilstmModel
from category.tf_models.classify_rcnn_model import ClassifyRcnnModel
import os
import tqdm
import argparse
from category import  data_process
from category.data_utils import input_fn,make_tfrecord_files
from category.tf_models.params import Params,TestParams
from sklearn.metrics import f1_score



def argument_parser():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--output_path', type=str, default='output/', help="中间文件生成目录")
    parser.add_argument('--origin_data', type=str, default="", help="原始数据地址")
    parser.add_argument('--data_type', type=str, default="default", help="原始数据格式，，目前支持默认的，还有rasa格式")
    parser.add_argument('--category_type', type=str, default="bilstm", help="神经网络类型：cnn or bilstm,如果是空字符串，则直接接一个全连接层输出")

    parser.add_argument('--device_map', type=str, default="0", help="gpu 的设备id")
    parser.add_argument('--use_bert', action='store_true', help='是否使用bert')
    parser.add_argument('--bert_model_path', type=str, help='bert模型目录')

    parser.add_argument('--max_sentence_length', type=int, default=20, help='一句话的最大长度')

    return parser.parse_args()






def train(params):
    if params.data_type == 'default':
        data_processer = data_process.NormalData(params.origin_data,output_path=params.output_path)
    else:
        data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)

    vocab,vocab_list,intent = data_processer.load_vocab_and_intent()

    params.vocab_size = len(vocab_list)
    params.num_tags = len(intent)

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            training_input_x,training_input_y = input_fn(os.path.join(params.output_path,"train.tfrecord"),
                                                         params.batch_size,
                                                         params.max_sentence_length,
                                                         params.shuffle_num,
                                                         mode=tf.estimator.ModeKeys.TRAIN)

            if params.category_type == 'cnn':
                classify_model = ClassifyCnnModel(params)
            elif params.category_type == "bilstm":
                classify_model = ClassifyBilstmModel(params)
            elif params.category_type == "rcnn":
                classify_model = ClassifyRcnnModel(params)

            loss,global_step,train_op,merger_op = classify_model.make_train(training_input_x,training_input_y)


            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            classify_model.model_restore(sess, saver)

            best_f1 = 0
            for _ in tqdm.tqdm(range(params.total_train_steps), desc="steps", miniters=10):
                sess_loss, steps, _ = sess.run([loss, global_step, train_op])

                if steps % params.evaluate_every_steps == 0:
                    test_input_x, test_input_y = input_fn(os.path.join(params.output_path, "test.tfrecord"),
                                                                  params.batch_size,
                                                                  params.max_sentence_length,
                                                                  params.shuffle_num,
                                                                  mode=tf.estimator.ModeKeys.EVAL)
                    loss_test,predict_test = classify_model.make_test(test_input_x,test_input_y)

                    predict_var = []
                    train_y_var = []
                    loss_total = 0
                    num_batch = 0
                    try:
                        while 1:
                            loss_,predict_,test_input_y_ = sess.run([loss_test,predict_test,test_input_y])
                            loss_total += loss_
                            num_batch += 1
                            predict_var += predict_.tolist()
                            train_y_var += test_input_y_.tolist()
                    except tf.errors.OutOfRangeError:
                        print("eval over")
                    if num_batch > 0:

                        f1_val = f1_score(train_y_var, predict_var, average='micro')
                        print("current step:%s ,loss:%s , f1 :%s" % (steps, loss_total/num_batch, f1_val))

                        if f1_val >= best_f1:
                            saver.save(sess, params.model_path, steps)
                            print("new best f1: %s ,save to dir:%s" % (f1_val, params.output_path))
                            best_f1 = f1_val




        classify_model.make_pb_file(params.output_path)

if __name__ == '__main__':

    argument_dict = argument_parser()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), argument_dict.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    argument_dict.output_path = output_path

    params = Params()
    params.update_object(argument_dict)

    if params.use_bert:
        #bert_make_tfrecord_files(argument_dict)
        #bert_train(argument_dict)
        pass
    else:
        make_tfrecord_files(params)
        train(params)
