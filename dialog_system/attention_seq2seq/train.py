# create by fanfan on 2019/5/24 0024
# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import tensorflow as tf
import os
import tqdm
from dialog_system.attention_seq2seq import  data_process
from dialog_system.attention_seq2seq.data_utils import input_fn,make_tfrecord_files
from dialog_system.attention_seq2seq.tf_model.attention_sea2seq import AttentionSeq
from dialog_system.attention_seq2seq.params import Params,TestParams




def train(params):
    data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)

    vocab, vocab_list = data_processer.load_vocab_and_intent()
    params.vocab_size = len(vocab_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map


    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Graph().as_default():
        with tf.Session(config=session_conf) as sess:
            training_input_x, training_decoder_input, training_decoder_output = input_fn(
                os.path.join(params.output_path, "train.tfrecord"),
                params.batch_size,
                params.max_seq_length,
                mode=tf.estimator.ModeKeys.TRAIN)

            test_input_x, test_decoder_input, test_decoder_output = input_fn(
                os.path.join(params.output_path, "test.tfrecord"),
                params.batch_size,
                params.max_seq_length,
                mode=tf.estimator.ModeKeys.EVAL)

            attention_seq2seq_obj = AttentionSeq(params)
            loss,globalStep,train_summaries ,train_op= attention_seq2seq_obj.make_train(training_input_x,training_decoder_input,training_decoder_output)

            loss_eval = attention_seq2seq_obj.create_model(test_input_x,test_decoder_input,test_decoder_output)

            init = tf.global_variables_initializer()



            sess.run(init)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            attention_seq2seq_obj.model_restore(sess,saver)

            best_f1 = 100

            for _ in tqdm.tqdm(range(60), desc="steps", miniters=10):

                sess_loss, steps, _ = sess.run([loss, globalStep, train_op])


                if steps % params.valid_freq == 0:
                    f1_val = sess_loss #f1_score(train_y_var, predict_var, average='micro')
                    print("current step:%s ,loss:%s , f1 :%s" % (steps, sess_loss, f1_val))

                    if f1_val < best_f1:
                        saver.save(sess, params.model_path, steps)
                        print("new best f1: %s ,save to dir:%s" % (f1_val, params.model_path))
                        best_f1 = f1_val

            print("save to dir:%s" % params.model_path)

    make_pb_file(params)

def make_pb_file(params):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf, graph=graph)
        with sess.as_default():
            input = tf.placeholder(dtype=tf.int32, shape=(None, params.max_seq_length),name='input')
            params.dropout_rate = 0.
            attent_seq2seq_obj = AttentionSeq(params)
            attent_seq2seq_obj.create_model_predict(input)


            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            attent_seq2seq_obj.model_restore(sess,saver)

            output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                    ["predicts"])

            with tf.gfile.GFile(os.path.join(params.output_path, 'transform.pb'), 'wb') as gf:
                gf.write(output_graph_with_weight.SerializeToString())
    return os.path.join(params.output_path, 'transform.pb')




if __name__ == '__main__':
    params = TestParams()

    if not os.path.exists(params.output_path):
        os.mkdir(params.output_path)





    make_tfrecord_files(params)
    train(params)