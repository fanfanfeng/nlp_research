# create by fanfan on 2017/7/26 0026
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

from ner.tv import ner_setting as ner_tv
from ner.tv import data_util
import pickle


def change_gensim_mode2array():
    model_path = ner_tv.word2vec_path
    with open(model_path,'rb') as f:
        w2v = pickle.load(f)
    return np.asarray(w2v)

class Model():
    def __init__(self):
        self.learning_rate = ner_tv.initial_learning_rate
        self.num_hidden = ner_tv.hidden_neural_size  #lstm隐层个数
        self.embedding_size = ner_tv.embedding_dim
        self.num_tags = ner_tv.tags_num
        self.max_grad_norm = ner_tv.max_grad_norm
        self.max_sentence_len = ner_tv.sentence_length
        self.w2v_model_path = ner_tv.word2vec_path
        self.model_save_path = ner_tv.train_model_bi_lstm
        self.train_epoch = ner_tv.num_epochs
        self.dropout_train = ner_tv.dropout

        self.initializer = initializers.xavier_initializer()

        self.inputs = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sentence_len],name="inputs")
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sentence_len],name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')


        with tf.variable_scope("word2vec_embedding"):
            self.embedding_vec = tf.Variable(change_gensim_mode2array(), name='word2vec', dtype=tf.float32)
            inputs_embedding = tf.nn.embedding_lookup(self.embedding_vec,self.inputs)
            lengths = self.get_length(inputs_embedding)
            self.lengths = tf.cast(lengths, tf.int32)
        lstm_outputs = self.biLSTM_layer(inputs_embedding,self.lengths)

        self.logits = self.project_layer(lstm_outputs)

        self.loss = self.loss_layer(self.logits,self.lengths)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=3)



    def biLSTM_layer(self,inputs,lengths):
        with tf.variable_scope('bi_lstm'):
            lstm_cell = {}
            for direction in ['forward','backward']:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.LSTMCell(self.num_hidden,use_peepholes=True,initializer=self.initializer)


            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell['forward'],lstm_cell['backward'],
                                                        inputs,dtype=tf.float32,sequence_length=lengths)
        return tf.concat(outputs,axis=2)

    def get_length(self,data):
        used = tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        length = tf.reduce_sum(used,reduction_indices=1)
        length = tf.cast(length,tf.int32)
        return length

    def project_layer(self,lstm_outputs):

        with tf.variable_scope('logits'):
            output = tf.reshape(lstm_outputs, shape=[-1, self.num_hidden * 2])
            W = tf.get_variable("W",shape=[self.num_hidden*2,self.num_tags],dtype=tf.float32,initializer=self.initializer)
            b = tf.get_variable('b',shape=[self.num_tags],dtype=tf.float32,initializer=tf.zeros_initializer)
            predict = tf.nn.xw_plus_b(output,W,b)

        return  tf.reshape(predict,[-1,self.max_sentence_len,self.num_tags])


    def loss_layer(self,logits,lengths):
        with tf.variable_scope("crf_loss"):
            self.trans = tf.get_variable('transitions',shape=[self.num_tags,self.num_tags],
                                         initializer=self.initializer)
        log_likelihood,self.trans = crf.crf_log_likelihood(logits,self.labels,transition_params=self.trans,
                                                           sequence_lengths=lengths)
        return tf.reduce_mean(-log_likelihood)

    def model_restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(self.model_save_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("restore model from {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print("init new model")
            sess.run(tf.global_variables_initializer())

    def create_feed_dict(self,inputs,labels,is_train):
        feed_dict = {
            self.inputs:inputs,
            self.dropout:1.0
        }
        if is_train:
            feed_dict[self.labels] = labels
            feed_dict[self.dropout] = self.dropout_train

        return feed_dict

    def run_step(self,sess,inputs,labels,is_train):
        feed_dict = self.create_feed_dict(inputs,labels,is_train)
        if is_train:
            fetch_list = [self.global_step,self.loss,self.train_op]
            global_step ,loss,_ = sess.run(fetch_list,feed_dict)
            return global_step,loss
        else:
            print(self.lengths.name)
            print(self.logits.name)
            fetch_list= [self.lengths,self.logits]
            lengths,logits = sess.run(fetch_list,feed_dict)
            return lengths,logits

    def predict(self,sess,inputs):
        crf_trans_matrix = self.trans.eval()
        lengths,scores = self.run_step(sess,inputs,None,False)
        paths = []
        for score,length in zip(scores,lengths):
            score = score[:length]
            path,_ = crf.viterbi_decode(score,crf_trans_matrix)
            paths.append(path[:length])
        return paths



    def test_accuraty(self,sess,inputs,labels):
        crf_trans_matrix = self.trans.eval()
        lengths, scores = self.run_step(sess, inputs, None, False)
        correct_num = 0
        total_labels = 0
        for score_, length_, label_ in zip(scores, lengths, labels):
            if length_ == 0:
                continue
            score = score_[:length_]
            path, _ = crf.viterbi_decode(score, crf_trans_matrix)
            label_path = label_[:length_]
            correct_num += np.sum(np.equal(path, label_path))
            total_labels += length_

        accuracy = 100.0 * correct_num / float(total_labels)
        return accuracy











