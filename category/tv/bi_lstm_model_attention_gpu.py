# create by fanfan on 2017/7/3 0003
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
from tensorflow.contrib import rnn
import os
import pickle
from category.tv import  classfication_setting
from category.tv import data_util
import time
from  _datetime import datetime



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



        #with tf.name_scope("embedding_layer"):
        self.W = tf.Variable(change_gensim_mode2array(),name="w",trainable=True)
            #self.W = tf.Variable(tf.truncated_normal([400000,200],mean=1.1,stddev=2.0),name='w')

        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            self.initial_learning_rate,self.global_step,self.decay_step,self.decay_rate,staircase=True
        ),self.min_learning_rate)


        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
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

        output = tf.reduce_sum(inputs * tf.reshape(alphas,[-1,sequence_length,1]),1)
        return output

    def tower_loss(self,scope):
        with tf.name_scope("embedding_layer"):
            inputs = tf.nn.embedding_lookup(self.W,self.input_x)
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
            cross_entry = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            loss = tf.reduce_mean(cross_entry, name="loss")
        return loss

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






num_gpus = 1
TOWER_NAME = 'tower'
log_device_placement = False
def train():
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        model = Bi_lstm()
        tower_grads = []
        with tf.variable_scope("train"):
            for i in range(num_gpus):
                with tf.device('cpu:%d' % i):
                    with tf.name_scope("%s_%d" % (TOWER_NAME,i)) as scope:

                        loss = model.tower_loss(scope)
                        tf.get_variable_scope().reuse_variables()
                        grads = model.optimizer.compute_gradients(loss)
                        tower_grads.append(grads)

        grads = model.average_gradients(tower_grads)
        train_op = model.optimizer.apply_gradients(grads,global_step=model.global_step)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = log_device_placement
        ))
        sess.run(tf.global_variables_initializer())

        tv_data = data_util.BatchManager(classfication_setting.tv_data_path, classfication_setting.batch_size)
        for num_epoch in range(classfication_setting.num_epochs):
            print("epoch  {}".format(num_epoch + 1))
            for train_x, train_y in tv_data.train_iterbatch():
                start_time = time.time()
                feed_dict = model.create_feed_dict(is_train=True, innputX=train_x, inputY=train_y)
                _,loss_value,step = sess.run([train_op,loss,model.global_step],feed_dict)
                duration = time.time() - start_time
                if step % classfication_setting.show_every == 0:
                    num_example_per_step = classfication_setting.batch_size * num_gpus
                    examples_per_sec = num_example_per_step / duration
                    sec_per_batch = duration / num_gpus
                    format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec /batch)"
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                    #graph_writer.add_summary(summary_op, step)
                if step % classfication_setting.valid_every == 0 and 0:
                    avg_loss = 0
                    avg_accuracy = 0
                    for test_x, test_y in tv_data.test_iterbatch():
                        _, _, loss, accuracy, _ = model.train_step(sess, False, test_x, test_y)
                        avg_loss += loss
                        avg_accuracy += accuracy

                    avg_loss = avg_loss / tv_data.test_epoch
                    avg_accuracy = avg_accuracy / tv_data.test_epoch
                    print(
                        "验证模型, 训练步数 {} ,学习率 {:g}, 损失值 {:g}, 精确值 {:g}".format(step, model.learning_rate.eval(), avg_loss,
                                                                             avg_accuracy))

                if step % classfication_setting.checkpoint_every == 0:
                    path = model.saver.save(sess, classfication_setting.train_model_bi_lstm, step)
                    print("模型保存到{}".format(path))



if __name__ == '__main__':
    train()





