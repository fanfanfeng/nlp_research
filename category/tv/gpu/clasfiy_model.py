# create by fanfan on 2018/1/23 0023
from category.tv.gpu import classfy_setting
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os


def load_word2vec():
    model_path = classfy_setting.word2vec_path
    with open(model_path,'rb') as f:
        w2v = pickle.load(f)
    return np.asarray(w2v)

def load_word2_id():
    model_path = classfy_setting.word2id_path
    with open(model_path, 'rb') as f:
        w2id = pickle.load(f)
    return w2id

class Attention_lstm_model():
    def __init__(self):
        self.initial_learning_rate = classfy_setting.initial_learning_rate
        self.min_learning_rate = classfy_setting.min_learning_rate
        self.decay_step = classfy_setting.decay_step
        self.decay_rate = classfy_setting.decay_rate
        self.sentence_length = classfy_setting.sentence_length
        self.sentence_classes = classfy_setting.sentence_classes
        self.hidden_neural_size = classfy_setting.hidden_neural_size
        # self.input_dim_size = flags.input_dim_size
        self.hidden_layer_num = classfy_setting.hidden_layer_num
        # self.w2v =
        self.train_model_save_path = classfy_setting.train_model_bi_lstm
        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length])
        self.input_y = tf.placeholder(tf.int32, [None])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.l2_reg = 0.001
        self.attention_size = classfy_setting.attention_size

        # L2 正则化损失
        self.l2_loss = tf.constant(0.0)

        # with tf.name_scope("embedding_layer"):
        self.W = tf.Variable(load_word2vec(), name="w", trainable=True)

    def lstm_fw(self):
        lstm_fw = rnn.LSTMCell(self.hidden_neural_size)
        lstm_fw = rnn.DropoutWrapper(lstm_fw, self.dropout)

        return lstm_fw

    def lstm_bw(self):
        lstm_bw = rnn.LSTMCell(self.hidden_neural_size)
        lstm_bw = rnn.DropoutWrapper(lstm_bw, self.dropout)
        return lstm_bw

    def bilstm_layer(self,inputs):

        if self.hidden_layer_num >1:
            lstm_fw = rnn.MultiRNNCell([self.lstm_fw() for _ in range(self.hidden_layer_num)])
            lstm_bw = rnn.MultiRNNCell([self.lstm_bw() for _ in range(self.hidden_layer_num)])
        else:
            lstm_fw = self.lstm_fw()
            lstm_bw = self.lstm_bw()

        outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw,cell_bw=lstm_bw,inputs=inputs,dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        return outputs

    def attention_layer(self,inputs,attention_size,l2_reg):
        """
           Attention mechanism layer.
           :param inputs: outputs of RNN/Bi-RNN layer (not final state)
           :param attention_size: linear size of attention weights
           :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
           """
        # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
        if isinstance(inputs,tuple):
            inputs = tf.concat(inputs,2)
        sequence_length = inputs.get_shape()[1].value
        hidden_size = inputs.get_shape()[2].value

        # Attention
        W_omega = tf.get_variable('W_omega',initializer=tf.random_normal([hidden_size,attention_size],stddev=0.1))
        b_omega = tf.get_variable('b_omega',initializer=tf.random_normal([attention_size],stddev=0.1))
        u_omega = tf.get_variable('u_omega',initializer=tf.random_normal([attention_size],stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs,[-1,hidden_size]),W_omega)) + tf.reshape(b_omega,[1,-1])
        vu = tf.matmul(v,tf.reshape(u_omega,[-1,1]))
        exps = tf.reshape(tf.exp(vu),[-1,sequence_length])
        alphas = exps/tf.reshape(tf.reduce_sum(exps,1),[-1,1])

        output = tf.reduce_sum(inputs * tf.reshape(alphas,[-1,sequence_length,1]),1)
        return output


    def tower_loss(self,scope,input_x,input_y):
        with tf.name_scope("embedding_layer"):
            inputs = tf.nn.embedding_lookup(self.W,input_x)
            inputs = tf.nn.dropout(inputs,self.dropout,name="drouout_input")
            # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            #inputs = tf.unstack(inputs,self.sentence_length,1)
            rnn_features = self.bilstm_layer(inputs)

        with tf.name_scope('attention_layer'):
            rnn_features = tf.reshape(rnn_features,[-1,self.sentence_length,self.hidden_neural_size *2])
            rnn_attention_outputs = self.attention_layer(rnn_features,self.attention_size,self.l2_reg)
            rnn_attention_outputs = tf.nn.dropout(rnn_attention_outputs, self.dropout)

        with tf.name_scope('softmax_layer'):
            softmax_w = tf.get_variable('softmax_w',[2 * self.hidden_neural_size,self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.2,stddev=2))
            softmax_b = tf.get_variable('softmax_b',[self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.1,stddev=2))
            logits = tf.add(tf.matmul(rnn_attention_outputs,softmax_w),softmax_b,name='logits')

        with tf.name_scope("output"):
            cross_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
            loss = tf.reduce_mean(cross_entry, name="loss")
        return loss,logits

    def average_gradients(self,tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g,_ in grad_and_vars:
                expand_g = tf.expand_dims(g,0)
                grads.append(expand_g)
            grad = tf.concat(grads,0)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad,v)
            average_grads.append(grad_and_var)
        return average_grads


    def restore_model(self,sess,saver):
        check_point = tf.train.get_checkpoint_state(self.train_model_save_path)
        if not check_point:
            sess.run(tf.global_variables_initializer())
            print("init new model")
        else:
            file_name = os.path.basename(check_point.model_checkpoint_path)
            real_path = os.path.join(self.train_model_save_path, file_name)
            saver.restore(sess,real_path)
            print("restore from model")