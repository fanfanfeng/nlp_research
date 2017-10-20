__author__ = 'fanfan'
import bi_lstm_setting
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from gensim.models import Word2Vec
import os

def change_gensim_mode2array():
    model_path = bi_lstm_setting.word2vec_path
    word2vec_model = Word2Vec.load(model_path)
    array_list = []
    for i in word2vec_model.wv.index2word:
        array_list.append(word2vec_model.wv[i])

    w2v = np.array(array_list)
    return w2v
class Model(object):
    def __init__(self):
        #初始化一些基本参数
        self._init_config()

        #初始化placeholder
        self._init_placeholders()

        #初始化embedding向量
        self._init_embeddings()

        #训练网络
        self._build_network()


        self._init_optimizer()
        self.saver = tf.train.Saver(max_to_keep=3)
        self.merge = tf.summary.merge_all()

    def _init_config(self):
        self.initial_learning_rate      = bi_lstm_setting.initial_learning_rate
        self.min_learning_rate          = bi_lstm_setting.min_learning_rate
        self.decay_step                 = bi_lstm_setting.decay_step
        self.decay_rate                 = bi_lstm_setting.decay_rate
        self.sentence_length            = bi_lstm_setting.sentence_length
        self.sentence_classes           = bi_lstm_setting.sentence_classes
        self.hidden_neural_size         = bi_lstm_setting.hidden_neural_size
        self.hidden_layer_num           = bi_lstm_setting.hidden_layer_num
        self.max_grad_norm              = bi_lstm_setting.max_grad_norm
        self.train_drop_out             = bi_lstm_setting.dropout
        self.model_save_path            = bi_lstm_setting.model_save_path


    def _init_placeholders(self):
        with tf.name_scope("placeholder_layer"):
            self.input_x = tf.placeholder(tf.int32,[None,self.sentence_length])
            self.input_y = tf.placeholder(tf.int32,[None,self.sentence_classes])
            self.dropout = tf.placeholder(tf.float32,name='dropout')

    def _init_embeddings(self):
        with tf.name_scope("embedding_layer"):
            self.embedding_vec = tf.Variable(change_gensim_mode2array(),name='embedding')

            inputs_embedding = tf.nn.embedding_lookup(self.embedding_vec,self.input_x)
             # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            self.inputs_embedding = tf.unstack(inputs_embedding,self.sentence_length,1)


    def lstm_cell(self):
        lstm_cell = rnn.LSTMCell(self.hidden_neural_size)
        lstm_cell = rnn.DropoutWrapper(lstm_cell,self.dropout)
        return lstm_cell

    def _build_network(self):
        with tf.name_scope("bilstm_layer"):
            with tf.name_scope("rnn_cell"):
                if self.hidden_layer_num > 1:
                    lstm_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
                    lstm_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.hidden_layer_num)])
                else:
                    lstm_fw = self.lstm_cell()
                    lstm_bw = self.lstm_cell()

                outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lstm_bw,self.inputs_embedding,dtype=tf.float32)

            with tf.name_scope('softmax_layer'):
                softmax_w = tf.get_variable('softmax_w',[2 * self.hidden_neural_size,self.sentence_classes],dtype=tf.float32,initializer=tf.random_normal_initializer)
                softmax_b = tf.get_variable('softmax_b',[self.sentence_classes],dtype=tf.float32,initializer=tf.random_normal_initializer)
                self.logits = tf.add(tf.matmul(outputs[-1],softmax_w),softmax_b,name='logits')

    def _init_optimizer(self):
        with tf.name_scope("output"):
            self.cross_entry = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entry,name='loss')
            tf.summary.scalar('loss',self.loss)
            self.prediction = tf.argmax(self.logits,1,name='prediction')
            correct_predict = tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32),name='accuracy')

        with tf.name_scope("optimizer"):
            self.global_step = tf.Variable(0,name='global_step',trainable= False)
            self.learning_rate = tf.maximum(tf.train.exponential_decay(self.initial_learning_rate,
                                                                       self.global_step,
                                                                       self.decay_step,
                                                                       self.decay_rate,
                                                                       staircase=True),
                                            self.min_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            t_vars = tf.trainable_variables()
            grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,t_vars),self.max_grad_norm)

            self.train_op = optimizer.apply_gradients(zip(grads,t_vars),global_step=self.global_step)

    #根据is_train字段创建feed_dict
    def create_feed_dict(self,inputX,inputY,is_train):
        feed_dict = {
            self.input_x:inputX,
            self.input_y:inputY,
            self.dropout:1.0
        }

        if is_train:
            feed_dict[self.dropout] = self.train_drop_out

        return feed_dict

    #训练跟新模型
    def train_step(self,sess,inputX,inputY,is_train):
        feed_dict = self.create_feed_dict(inputX,inputY,is_train)

        fetches = [ self.train_op,self.loss,self.accuracy,self.merge ]
        _,loss,accuracy,merge = sess.run(fetches,feed_dict)
        return loss,accuracy,merge

    #根据输入预测分类
    def predict(self,sess,inputX):
        feed_dict = {
            self.dropout:1.0,
            self.input_x:inputX
        }

        fetches = [ self.logits,self.prediction]
        logit,predict = sess.run(fetches,feed_dict)
        return predict


    #重新载入模型或者初始化参数
    def restore_model(self,sess):
        check_point = tf.train.get_checkpoint_state(os.path.dirname(self.model_save_path))
        if check_point:
            self.saver.restore(sess,check_point.model_checkpoint_path)
        else:
            print("init model")
            sess.run(tf.global_variables_initializer())
















