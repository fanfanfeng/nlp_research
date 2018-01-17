# create by fanfan on 2017/11/20 0020
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


class Bi_lstm_cnn():
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

        # todo ,后期可以配置到setting文件里面去
        self.num_filters = 128
        self.filter_sizes = [3, 4, 5]
        self.l2_reg_lambda = 0.001

        self.fc_hidden_size = 200
        self.embedding_dim = flags.embedding_dim

        with tf.name_scope("embedding_layer_lstm"):
            self.W_lstm = tf.Variable(change_gensim_mode2array(),name="w",trainable=True)
            inputs_lstm = tf.nn.embedding_lookup(self.W_lstm,self.input_x)
            inputs_lstm = tf.nn.dropout(inputs_lstm,self.dropout,name="drouout_input")

        with tf.variable_scope('bi_lstm_layer'):
            rnn_features = self.bilstm_layer(inputs_lstm)


        with tf.name_scope("embedding_layer_cnn"):
            self.W_cnn = tf.Variable(change_gensim_mode2array(), name="w", trainable=True)
            inputs_cnn = tf.nn.embedding_lookup(self.W_cnn, self.input_x)
            # 因为卷积操作conv2d的input要求4个维度的tensor, 所以需要给embedding结果增加一个维度来适应conv2d的input要求
            # 传入的-1表示在最后位置插入, 得到[None, sequence_length, embedding_size, 1]
            inputs_cnn = tf.expand_dims(inputs_cnn, -1)

        with tf.variable_scope("cnn_layer"):
            cnn_features, num_filters_total = self.cnn_filter_pool_layer(inputs_cnn)

        with tf.variable_scope('attention_layer'):
            rnn_features = tf.reshape(rnn_features,[-1,self.sentence_length,self.hidden_neural_size *2])
            rnn_attention_outputs = self.attention_layer(rnn_features,self.attention_size,self.l2_reg)
            rnn_attention_outputs = tf.nn.dropout(rnn_attention_outputs, self.dropout)

        with tf.variable_scope("fc_bn_layer"):
            out_put_total = tf.concat([rnn_attention_outputs,cnn_features],axis=1)
            w_fc = tf.get_variable("Weight_fc",shape=[self.hidden_neural_size*2 + num_filters_total,self.fc_hidden_size],initializer=tf.truncated_normal_initializer)
            beta = tf.get_variable('beta_fc',shape=[self.fc_hidden_size],initializer=tf.truncated_normal_initializer)
            h_fc = tf.matmul(out_put_total,w_fc,name='h_fc')
            fc_bn = self.batchnorm(h_fc,beta)
            self.fc_bn_relu = tf.nn.relu(fc_bn,name='relu')
            fc_bn_drop = tf.nn.dropout(self.fc_bn_relu,self.dropout)




        with tf.name_scope('softmax_layer'):
            softmax_w = tf.get_variable('softmax_w',[self.fc_hidden_size,self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.2,stddev=2))
            softmax_b = tf.get_variable('softmax_b',[self.sentence_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.1,stddev=2))

            self.logits = tf.add(tf.matmul(fc_bn_drop,softmax_w),softmax_b,name='logits')

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





