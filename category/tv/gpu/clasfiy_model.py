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
        self.embedding_dim = classfy_setting.embedding_dim
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
        self.W_cnn = tf.Variable(load_word2vec(), name="w_cnn", trainable=True)

        self.num_filters = 128
        self.filter_sizes = [1,2,3, 4, 5]
        self.l2_reg_lambda = 0.001

        self.fc_hidden_size = 200

    def lstm_fw(self):
        lstm_fw = rnn.LSTMCell(self.hidden_neural_size)
        lstm_fw = rnn.DropoutWrapper(lstm_fw, self.dropout)

        return lstm_fw

    def lstm_bw(self):
        lstm_bw = rnn.LSTMCell(self.hidden_neural_size)
        lstm_bw = rnn.DropoutWrapper(lstm_bw, self.dropout)
        return lstm_bw

    def bilstm_layer(self,inputs,length):

        if self.hidden_layer_num >1:
            lstm_fw = rnn.MultiRNNCell([self.lstm_fw() for _ in range(self.hidden_layer_num)])
            lstm_bw = rnn.MultiRNNCell([self.lstm_bw() for _ in range(self.hidden_layer_num)])
        else:
            lstm_fw = self.lstm_fw()
            lstm_bw = self.lstm_bw()

        outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw,cell_bw=lstm_bw,inputs=inputs,sequence_length=length,dtype=tf.float32)
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

    def get_length(self,data):
        #used = tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        #length = tf.reduce_sum(used,reduction_indices=1)
        #length = tf.cast(length,tf.int32)
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def tower_loss(self,scope,input_x,input_y):
        with tf.name_scope("embedding_layer"):
            length = self.get_length(input_x)
            inputs = tf.nn.embedding_lookup(self.W,input_x)
            inputs = tf.nn.dropout(inputs,self.dropout,name="drouout_input")
            rnn_features = self.bilstm_layer(inputs,length)

        with tf.name_scope("embedding_layer_cnn"):
            inputs_cnn = tf.nn.embedding_lookup(self.W_cnn, input_x)
            # 因为卷积操作conv2d的input要求4个维度的tensor, 所以需要给embedding结果增加一个维度来适应conv2d的input要求
            # 传入的-1表示在最后位置插入, 得到[None, sequence_length, embedding_size, 1]
            inputs_cnn = tf.expand_dims(inputs_cnn, -1)

        with tf.variable_scope("cnn_layer"):
            cnn_features, num_filters_total = self.cnn_filter_pool_layer(inputs_cnn)


        with tf.name_scope('attention_layer'):
            rnn_features = tf.reshape(rnn_features,[-1,self.sentence_length,self.hidden_neural_size *2])
            rnn_attention_outputs = self.attention_layer(rnn_features,self.attention_size,self.l2_reg)
            rnn_attention_outputs = tf.nn.dropout(rnn_attention_outputs, self.dropout)

        out_put_total = tf.concat([rnn_attention_outputs, cnn_features], axis=1)

        l2_loss = 0
        with tf.name_scope('hidden_layer'):
            hidden_w = tf.get_variable('hidden_w',[self.hidden_neural_size*2 + num_filters_total,self.hidden_neural_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.2,stddev=2))
            hidden_b = tf.get_variable('hidden_b',[self.hidden_neural_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.1,stddev=2))
            hidden_output = tf.add(tf.matmul(out_put_total,hidden_w),hidden_b,name='hidden_output')
            l2_loss += tf.nn.l2_loss(hidden_w)

        with tf.name_scope('softmax_layer'):
            softmax_w = tf.get_variable('softmax_w',[self.hidden_neural_size,self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.2,stddev=2))
            softmax_b = tf.get_variable('softmax_b',[self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.1,stddev=2))
            l2_loss += tf.nn.l2_loss(softmax_w)
            logits = tf.add(tf.matmul(hidden_output,softmax_w),softmax_b,name='logits')

        with tf.name_scope("output"):
            cross_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
            loss = tf.reduce_mean(cross_entry, name="loss") + self.l2_reg_lambda * l2_loss
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


    def cnn_filter_pool_layer(self,inputs):
        pooled_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('conv_maxpool_%s' % filter_size):
                # 卷积层
                filter_shape = [filter_size,self.embedding_dim,1,self.num_filters]
                weight = tf.get_variable('weight',shape=filter_shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
                bias = tf.get_variable('bias',shape=[self.num_filters],dtype=tf.float32,initializer=tf.constant_initializer(0.1))

                # 卷积操作, “VALID”表示使用narrow卷积，得到的结果大小为[batch, sequence_length - filter_size + 1, 1, num_filters]
                conv = tf.nn.conv2d(inputs,weight,strides=[1,1,1,1],padding='VALID',name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv,bias))

                # 用max-pooling处理上层的输出结果,每一个卷积结果
                # pooled的大小为[batch, 1, 1, num_filters]
                pooled = tf.nn.max_pool(h,ksize=[1,self.sentence_length - filter_size + 1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pooled)

        with tf.variable_scope("full_connect_layer"):
            # 全连接输出层
            # 将上面的pooling层输出全连接到输出层
            num_filters_total = self.num_filters * len(self.filter_sizes)

            # 把相同filter_size的所有pooled结果concat起来，再将不同的filter_size之间的结果concat起来
            # tf.concat按某一维度进行合并, h_pool的大小为[batch, 1, 1, num_filters_total]
            self.h_pool = tf.concat(pooled_outputs,3)
            self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout)

        return  self.h_drop,num_filters_total

    def batchnorm(self,Ylogits,offset):
        """batchnormalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        #exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,self.global_step)
        bnepsilon = 1e-5
        mean,variance = tf.nn.moments(Ylogits,[0])
        #update_moving_everages = exp_moving_avg.apply([mean,variance])
        Ybn = tf.nn.batch_normalization(Ylogits,mean,variance,offset,None,bnepsilon)
        return Ybn

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