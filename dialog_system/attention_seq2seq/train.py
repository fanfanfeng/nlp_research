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

from dialog_system.attention_seq2seq.utils_other.data_utils import batchnize_dataset,load_data


def _add_placeholders(max_sent_len):
    # shape = (batch_size, max_words_len)
    enc_source = tf.placeholder(dtype=tf.int32, shape=[None, max_sent_len], name="encoder_input")
    dec_target_in = tf.placeholder(dtype=tf.int32, shape=[None, max_sent_len], name="decoder_input")
    dec_target_out = tf.placeholder(dtype=tf.int32, shape=[None, max_sent_len], name="decoder_output")
    return enc_source, dec_target_in, dec_target_out

def _get_feed_dict(batch_data, enc_source, dec_target_in, dec_target_out):
    feed_dict = {
        enc_source: batch_data["source_in"],
        dec_target_in: batch_data["target_in"],
        dec_target_out: batch_data["target_out"]
    }
    return feed_dict
import random
import os
def train_another(params):
    current_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(params.output_path+"_other"):
        os.makedirs(params.output_path+"_other")
    dataset = os.path.join(current_path,'data/dataset.json')
    vocabulary = os.path.join(current_path,'data/vocabulary.json')
    dict_data = load_data(vocabulary)
    target_dict = dict_data["target_dict"]
    del dict_data
    train_set, test_set = batchnize_dataset(dataset, params.batch_size, target_dict)
    params.vocab_size = len(target_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Graph().as_default():
        with tf.Session(config=session_conf) as sess:
            training_input_x, training_decoder_input, training_decoder_output = _add_placeholders(params.max_seq_length)

            attention_seq2seq_obj = AttentionSeq(params)
            loss, globalStep, train_summaries, train_op = attention_seq2seq_obj.make_train(training_input_x,
                                                                                           training_decoder_input,
                                                                                           training_decoder_output)

            init = tf.global_variables_initializer()

            sess.run(init)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            attention_seq2seq_obj.model_restore(sess, saver)

            best_f1 = 100
            for _ in range(params.max_epochs):
                random.shuffle(train_set)
                bat = BatchManager(train_set, params.batch_size)
                bat_test = BatchManager(test_set, params.batch_size)

                for i, batch_data in enumerate(bat):
                    batch_data = batch_data[0]
                    feed_dict = _get_feed_dict(batch_data, training_input_x, training_decoder_input,
                                               training_decoder_output)
                    sess_loss, steps, _ = sess.run([loss, globalStep, train_op],feed_dict=feed_dict)

                    if steps % params.display_freq == 0:
                        print("step:%s , current loss: %s" % (steps, sess_loss))

                    if steps % params.valid_freq == 0:
                        total_loss = 0
                        count = 0
                        for batch in bat_test:
                            b_ = batch[0]
                            feed_dict = _get_feed_dict(b_, training_input_x, training_decoder_input,
                                                   training_decoder_output)
                            sess_loss = sess.run(loss, feed_dict=feed_dict)
                            total_loss += sess_loss
                            count += 1

                        sess_loss = total_loss/count #f1_score(train_y_var, predict_var, average='micro')
                        print("current  step:%s ,eval loss:%s " % (steps, sess_loss))

                        if sess_loss < best_f1:
                            saver.save(sess, params.model_path, steps)
                            print("new best f1: %s ,save to dir:%s" % (sess_loss, params.model_path))
                            best_f1 = sess_loss

            print("save to dir:%s" % params.model_path)

    make_pb_file(params)

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



            attention_seq2seq_obj = AttentionSeq(params)
            loss,globalStep,train_summaries ,train_op= attention_seq2seq_obj.make_train(training_input_x,training_decoder_input,training_decoder_output)



            init = tf.global_variables_initializer()



            sess.run(init)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            attention_seq2seq_obj.model_restore(sess,saver)

            best_f1 = 100

            for _ in tqdm.tqdm(range(params.total_steps), desc="steps", miniters=10):

                sess_loss, steps, _ = sess.run([loss, globalStep, train_op])

                if steps % params.display_freq == 0:
                    print("step:%s , current loss: %s" % (steps,sess_loss))

                if steps % params.valid_freq == 0:
                    test_input_x, test_decoder_input, test_decoder_output = input_fn(
                        os.path.join(params.output_path, "test.tfrecord"),
                        params.batch_size,
                        params.max_seq_length,
                        mode=tf.estimator.ModeKeys.EVAL)
                    loss_eval, _ = attention_seq2seq_obj.create_model(test_input_x, test_decoder_input,
                                                                      test_decoder_output)
                    total_loss = 0
                    count = 0
                    try:
                        while 1:
                            loss_eval_batch = sess.run(loss_eval)
                            total_loss += loss_eval_batch
                            count += 1
                    except tf.errors.OutOfRangeError:
                        print("eval over")

                    #vim total_loss = total_loss/count #f1_score(train_y_var, predict_var, average='micro')
                    total_loss = total_loss/count
                    print("current  step:%s ,eval loss:%s " % (steps, total_loss))

                    if total_loss < best_f1:
                        saver.save(sess, params.model_path, steps)
                        print("new best f1: %s ,save to dir:%s" % (total_loss, params.model_path))
                        best_f1 = total_loss

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


class BatchManager():
    def __init__(self, data, batch_size):
        self.data = data
        self.source = []
        self.target_in = []
        self.target_out =  []
        for item in data:
            self.source.extend(item['source_in'])
            self.target_in.extend(item['target_in'])
            self.target_out.extend(item['target_out'])
        self.batch_size = batch_size

    def shuffle(self):
        total = []
        for i,j,k in zip(self.source,self.target_in,self.target_out):
            total.append((i,j,k))
        random.shuffle(total)
        self.source = []
        self.target_in = []
        self.target_out = []
        for item in total:
            self.source.append(item[0])
            self.target_in.append(item[1])
            self.target_out.append(item[2])


    def __iter__(self):
        self.shuffle()
        number = len(self.data)
        for i in range(number):
            yield [{
                "source_in":self.source[i * self.batch_size:(i + 1) * self.batch_size],
                "target_in":self.target_in[i * self.batch_size:(i + 1) * self.batch_size],
                "target_out": self.target_out[i * self.batch_size:(i + 1) * self.batch_size],
                    }]

if __name__ == '__main__':
    params = Params()

    if not os.path.exists(params.output_path):
        os.mkdir(params.output_path)



    params.save_to_file()

    make_tfrecord_files(params)
    train(params)