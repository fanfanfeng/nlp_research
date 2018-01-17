# create by fanfan on 2017/7/3 0003
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
from tensorflow.contrib import rnn
import os
import pickle
from category.tv import  classfication_setting



def change_gensim_mode2array():
    model_path = classfication_setting.word2vec_path
    with open(model_path,'rb') as f:
        w2v = pickle.load(f)
    return np.asarray(w2v)

def load_word2_id():
    model_path = classfication_setting.word2id_path
    with open(model_path, 'rb') as f:
        w2id = pickle.load(f)
    return w2id


class Bi_lstm():
    def __init__(self,flags = classfication_setting):
        self.initial_learning_rate = flags.initial_learning_rate
        self.min_learning_rate = flags.min_learning_rate
        self.decay_step = flags.decay_step
        self.decay_rate = flags.decay_rate
        self.sentence_length = flags.sentence_length
        self.sentence_classes = flags.sentence_classes
        self.hidden_neural_size = flags.hidden_neural_size
        #self.input_dim_size = flags.input_dim_size
        self.hidden_layer_num = flags.hidden_layer_num
        #self.w2v =
        self.train_model_save_path = flags.train_model_bi_lstm
        self.batch_size = flags.batch_size
        self.input_x = tf.placeholder(tf.int32,[None,self.sentence_length])
        self.input_y = tf.placeholder(tf.int32,[None,self.sentence_classes])

        self.dropout = tf.placeholder(tf.float32,name='dropout')
        self.l2_reg = 0.001
        self.attention_size = flags.attention_size

        # L2 正则化损失
        self.l2_loss = tf.constant(0.0)



        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(change_gensim_mode2array(),name="w",trainable=True)
            #self.W = tf.Variable(tf.truncated_normal([400000,200],mean=1.1,stddev=2.0),name='w')
            inputs = tf.nn.embedding_lookup(self.W,self.input_x)
            inputs = tf.nn.dropout(inputs,self.dropout,name="drouout_input")

            # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            #inputs = tf.unstack(inputs,self.sentence_length,1)

            rnn_features = self.bilstm_layer(inputs)

        with tf.variable_scope('attention_layer'):
            rnn_features = tf.reshape(rnn_features,[-1,self.sentence_length,self.hidden_neural_size *2])
            rnn_attention_outputs = self.attention_layer(rnn_features,self.attention_size,self.l2_reg)
            rnn_attention_outputs = tf.nn.dropout(rnn_attention_outputs, self.dropout)

        with tf.name_scope('softmax_layer'):
            softmax_w = tf.get_variable('softmax_w',[2 * self.hidden_neural_size,self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.2,stddev=2))
            softmax_b = tf.get_variable('softmax_b',[self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.1,stddev=2))

            self.logits = tf.add(tf.matmul(rnn_attention_outputs,softmax_w),softmax_b,name='logits')

        with tf.name_scope("output"):
            self.cross_entry = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entry,name="loss")

            tf.summary.scalar("loss", self.loss)

            self.prediction = tf.argmax(self.logits,1,name="prediction")

            correction_prediction = tf.equal(self.prediction,tf.argmax(self.input_y,1))

            self.accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32),name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            self.initial_learning_rate,self.global_step,self.decay_step,self.decay_rate,staircase=True
        ),self.min_learning_rate)


        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),flags.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars),global_step=self.global_step)
        self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables())



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

        #used = tf.sign(tf.abs(self.input_x))
        #length = tf.reduce_sum(used,reduction_indices=1)
        #length = tf.cast(length,tf.int32)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw,cell_bw=lstm_bw,inputs=inputs,dtype=tf.float32)
        #outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw,lstm_bw,inputs,dtype=tf.float32)#,sequence_length=length)
        #output_new ,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,inputs,dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        #output = outputs[-1]
        return outputs

    def train_step(self,sess,is_train,inputX,inputY):
        feed_dict = self.create_feed_dict(is_train,inputX,inputY)
        fetches = [self.train_op,self.global_step,self.learning_rate,self.loss,self.accuracy,self.summary]
        _,global_step,learning_rate,loss,accuracy,summary_op = sess.run(fetches,feed_dict)

        return global_step,learning_rate,loss,accuracy,summary_op

    def predict(self,sess,inputX):
        feed_dict = {
            self.dropout:1.0,
            self.input_x:inputX
        }

        fetches = [self.logits,self.prediction]


        logit,predict = sess.run(fetches,feed_dict)
        return predict,logit

    def create_feed_dict(self,is_train,innputX,inputY):
        feed_dict = {
            self.input_x:innputX,
            self.input_y:inputY,
            self.dropout:1.0
        }
        if is_train:
            feed_dict[self.dropout] = classfication_setting.dropout

        return  feed_dict

    def restore_model(self,sess):
        sess.run(tf.global_variables_initializer())
        check_point = tf.train.get_checkpoint_state(self.train_model_save_path)
        if not check_point:
            raise FileNotFoundError("not found model")
        file_name = os.path.basename(check_point.model_checkpoint_path)
        real_path = os.path.join(self.train_model_save_path, file_name)

        self.saver.restore(sess,real_path)

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

        output = tf.reduce_sum(inputs*tf.reshape(alphas,[-1,sequence_length,1]),1)
        return output

if __name__ == '__main__':
    w = change_gensim_mode2array()
    print(np.asarray(w).shape)





