# create by fanfan on 2017/7/26 0026
import sys
sys.path.append(r'/data/python_project/nlp_research')
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

from ner.tv.gpu import ner_setting
import pickle
from sklearn.metrics import f1_score


def change_gensim_mode2array():
    model_path = ner_setting.word2vec_path
    with open(model_path,'rb') as f:
        w2v = pickle.load(f)
    return np.asarray(w2v)

class Model():
    def __init__(self):
        self.learning_rate = ner_setting.initial_learning_rate
        self.num_hidden = ner_setting.hidden_neural_size  #lstm隐层个数
        self.embedding_size = ner_setting.embedding_dim
        self.num_tags = ner_setting.tags_num
        self.max_grad_norm = ner_setting.max_grad_norm
        self.max_sentence_len = ner_setting.max_document_length
        self.w2v_model_path = ner_setting.word2vec_path
        self.model_save_path = ner_setting.train_model_bi_lstm
        self.initializer = initializers.xavier_initializer()

        #初始化placeholder
        self.inputs = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sentence_len],name="inputs")
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sentence_len],name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')


    def logits_and_loss(self,input_x,input_y):
        with tf.variable_scope("word2vec_embedding"):
            embedding_vec = tf.Variable(change_gensim_mode2array(), name='word2vec', dtype=tf.float32,trainable=True)
            inputs_embedding = tf.nn.embedding_lookup(embedding_vec,input_x)
            lengths = self.get_length(input_x)
            lengths = tf.cast(lengths, tf.int32)
        lstm_outputs = self.biLSTM_layer(inputs_embedding,lengths)
        logits = self.project_layer(lstm_outputs)
        loss,transition = self.loss_layer(logits,lengths,input_y)
        return loss,logits,lengths,transition



    def biLSTM_layer(self,inputs,lengths):
        with tf.variable_scope('bi_lstm'):
            lstm_cell = {}
            for direction in ['forward','backward']:
                with tf.variable_scope(direction):
                    cell = rnn.LSTMCell(self.num_hidden,use_peepholes=True,initializer=self.initializer)
                    cell = rnn.DropoutWrapper(cell,self.dropout)
                    lstm_cell[direction] = cell
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell['forward'],lstm_cell['backward'],
                                                        inputs,dtype=tf.float32,sequence_length=lengths)
        return tf.concat(outputs,axis=2)

    # 获取输入句子的真是长度
    def get_length(self,data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def project_layer(self,lstm_outputs):
        with tf.variable_scope('logits'):
            output = tf.reshape(lstm_outputs, shape=[-1, self.num_hidden * 2])
            W = tf.get_variable("W",shape=[self.num_hidden*2,self.num_tags],dtype=tf.float32,initializer=self.initializer)
            b = tf.get_variable('b',shape=[self.num_tags],dtype=tf.float32,initializer=tf.zeros_initializer)
            predict = tf.nn.xw_plus_b(output,W,b)
        return  tf.reshape(predict,[-1,self.max_sentence_len,self.num_tags])


    def loss_layer(self,logits,lengths,input_y):
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable('transitions',shape=[self.num_tags,self.num_tags],initializer=self.initializer)
            log_likelihood,trans = crf.crf_log_likelihood(logits,input_y,transition_params=trans,sequence_lengths=lengths)
        return tf.reduce_mean(-log_likelihood),trans

    def model_restore(self,sess,saver):
        ckpt = tf.train.get_checkpoint_state(self.model_save_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("restore model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print("init new model")
            sess.run(tf.global_variables_initializer())





    def test_accuraty(self,lengths,scores,trans_matrix,labels):
        total_labels = []
        predict_labels = []
        for score_, length_, label_ in zip(scores, lengths, labels):
            if length_ == 0:
                continue
            score = score_[:length_]
            path, _ = crf.viterbi_decode(score, trans_matrix)
            label_path = label_[:length_]
            predict_labels.extend(path)
            total_labels.extend(label_path)

        return total_labels,predict_labels

    def average_gradients(self,tower_grads):
        avarage_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g,_ in grad_and_vars:
                expand_g = tf.expand_dims(g,0)
                grads.append(expand_g)
            grad = tf.concat(grads,0)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad,v)
            avarage_grads.append(grad_and_var)
        return avarage_grads















