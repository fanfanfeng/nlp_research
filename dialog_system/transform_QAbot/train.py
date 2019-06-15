# create by fanfan on 2019/5/24 0024
# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import tensorflow as tf
import os
import tqdm
import argparse
from dialog_system.transform_QAbot import  data_process
from dialog_system.transform_QAbot.data_utils import input_fn,make_tfrecord_files
from dialog_system.transform_QAbot.tf_models_new.model import Transformer
from tensorflow.contrib.seq2seq import sequence_loss
from dialog_system.transform_QAbot.params import Params,TestParams






def get_setence_length( data):
    used = tf.sign(tf.abs(data))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def input_test(params):
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Graph().as_default():
        with tf.Session(config=session_conf) as sess:
            training_input_x, training_decoder_input, training_decoder_output = input_fn(
                os.path.join(params.output_path, "train.tfrecord"),
                params.batch_size,
                params.maxlen,
                mode=tf.estimator.ModeKeys.TRAIN)
            print(os.path.join(params.output_path, "train.tfrecord"))
            a,b,c = sess.run([training_input_x, training_decoder_input, training_decoder_output])
            print(a)
            print(b)
            print(c)

def train(params):
    data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)

    vocab, vocab_list = data_processer.load_vocab_and_intent()
    params.vocab_size = len(vocab_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map


    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Graph().as_default():
        with tf.Session(config=session_conf) as sess:
            training_input_x, training_decoder_input, training_decoder_output = input_fn(
                os.path.join(params.output_path, "train.tfrecord"),
                params.batch_size,
                params.max_seq_length,
                mode=tf.estimator.ModeKeys.TRAIN)

            transformer_obj = Transformer(params,train=True)
            loss,train_op,globalStep,train_summaries = transformer_obj.make_train(training_input_x,training_decoder_input,training_decoder_output)


            init = tf.global_variables_initializer()



            sess.run(init)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            sess.run(tf.global_variables_initializer())
            tf_save_path = os.path.join(params.output_path, 'tf')

            best_f1 = 100

            for _ in tqdm.tqdm(range(2000), desc="steps", miniters=10):

                sess_loss, steps, _ = sess.run([loss, globalStep, train_op])


                if steps % params.evaluate_every_steps == 0:
                    f1_val = sess_loss #f1_score(train_y_var, predict_var, average='micro')
                    print("current step:%s ,loss:%s , f1 :%s" % (steps, sess_loss, f1_val))

                    if f1_val < best_f1:
                        saver.save(sess, tf_save_path, steps)
                        print("new best f1: %s ,save to dir:%s" % (f1_val, params.output_path))
                        best_f1 = f1_val

            print("save to dir:%s" % params.output_path)

    #make_pb_file(model_config,config_dict)

def make_pb_file(model_config,config_dict):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf, graph=graph)
        with sess.as_default():
            input = tf.placeholder(dtype=tf.int32, shape=(None, model_config.max_sentence_length),name='input')
            target = tf.placeholder(dtype=tf.int32, shape=(None, model_config.max_sentence_length),name='target')


            batch_size = tf.shape(target)[0]
            target_with_start = tf.strided_slice(target, [0, 0], [model_config.batch_size, -1],
                                                          [1, 1])
            decoder_start_token = tf.ones(
                shape=[batch_size, 1], dtype=tf.int32)
            target_with_start = tf.concat([decoder_start_token, target_with_start], axis=1)
            model = Transformer(config_dict, False)
            logits = model(input, target_with_start)
            predicts = tf.argmax(logits, axis=2, name="predict")

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            checkpoint = tf.train.latest_checkpoint(model_config.output_path)
            if checkpoint:
                saver.restore(sess, checkpoint)
            else:
                raise FileNotFoundError("模型文件未找到")

            output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                    ["predict"])

            with tf.gfile.GFile(os.path.join(model_config.output_path, 'transform.pb'), 'wb') as gf:
                gf.write(output_graph_with_weight.SerializeToString())
    return os.path.join(model_config.output_path, 'transform.pb')




if __name__ == '__main__':
    params = Params()
    make_tfrecord_files(params)
    train(params)