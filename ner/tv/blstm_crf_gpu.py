# create by fanfan on 2017/7/26 0026
import sys
sys.path.append(r'/data/python_project/nlp_research')
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

from ner.tv import ner_setting as ner_tv
from ner.tv import data_util
import pickle
from sklearn.metrics import f1_score


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
        self.decay_step = ner_tv.decay_step
        self.decay_rate = ner_tv.decay_rate
        self.min_learning_rate = ner_tv.min_learning_rate
        self.initializer = initializers.xavier_initializer()

        with tf.variable_scope("word2vec_embedding"):
            self.embedding_vec = tf.Variable(change_gensim_mode2array(), name='word2vec', dtype=tf.float32,trainable=False)
            print(self.embedding_vec.name)

        self.inputs = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sentence_len],name="inputs")
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.max_sentence_len],name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.train_learning_rate = tf.maximum(tf.train.exponential_decay(
            self.learning_rate, self.global_step, self.decay_step, self.decay_rate, staircase=True
        ), self.min_learning_rate)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.train_learning_rate)

        #tvars = tf.trainable_variables()
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
        #self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def logits_and_loss(self):
        with tf.variable_scope("word2vec_embedding"):
            #embedding_vec = tf.Variable(change_gensim_mode2array(), name='word2vec', dtype=tf.float32,
                                             #trainable=True)
            inputs_embedding = tf.nn.embedding_lookup(self.embedding_vec,self.inputs)
            lengths = self.get_length(self.inputs)
            lengths = tf.cast(lengths, tf.int32)
        lstm_outputs = self.biLSTM_layer(inputs_embedding,lengths)
        logits = self.project_layer(lstm_outputs)
        loss = self.loss_layer(logits,lengths)
        with tf.control_dependencies(None):
            loss = tf.identity(loss)
        return loss,logits,lengths



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

    def get_length(self,data):
        #used = tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        #length = tf.reduce_sum(used,reduction_indices=1)
        #length = tf.cast(length,tf.int32)
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
            fetch_list= [self.lengths,self.logits]
            lengths,logits = sess.run(fetch_list,feed_dict)
            return lengths,logits

    def predict(self,sess,inputs,inputs_y =[]):
        crf_trans_matrix = self.trans.eval()
        lengths,scores = self.run_step(sess,inputs,None,False)
        paths = []
        for score,length in zip(scores,lengths):
            score = score[:length]
            path,_ = crf.viterbi_decode(score,crf_trans_matrix)
            paths.append(path[:length])

        if len(inputs_y)!= 0:
            paths_y = []
            for y,length in zip(inputs_y,lengths):
                paths_y.append(y[:length])

            return paths,paths_y


        return paths



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


num_gpus = 1
TOWER_NAME = 'tower'
log_device_placement = False
def train():
    g = tf.Graph()
    with g.as_default(),tf.device('/cpu:0'):
        model_obj = Model()
        tower_grads = []
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope("%s_%d" % (TOWER_NAME,i)):
                    loss,_,_ = model_obj.logits_and_loss()
                    grads = model_obj.optimizer.compute_gradients(loss)
                    #tf.get_variable_scope().reuse_variables()

                    tower_grads.append(grads)

        grads = model_obj.average_gradients(tower_grads)
        train_op = model_obj.optimizer.apply_gradients(grads,global_step=model_obj.global_step)

        tf.get_variable_scope().reuse_variables()
        _, test_logits, test_lengths = model_obj.logits_and_loss()
        print("trans_model:",model_obj.trans.name)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = log_device_placement
        ),graph=g)
        model_obj.saver = tf.train.Saver(max_to_keep=3)
        model_obj.model_restore(sess)
        data_manager = data_util.BatchManager(ner_tv.tv_data_path, ner_tv.batch_size)

        best_f1 = 0
        for epoch in range(model_obj.train_epoch):
            print("start epoch {}".format(str(epoch)))
            average_loss = 0
            for train_inputs, train_labels in data_manager.train_iterbatch():
                feed_dict = model_obj.create_feed_dict(inputs=train_inputs,labels=train_labels,is_train=True)
                step, loss_val,_ = sess.run([model_obj.global_step,loss,train_op],feed_dict=feed_dict)
                average_loss += loss_val
                if step % ner_tv.show_every == 0:
                    average_loss = average_loss / ner_tv.show_every
                    print("iteration:{} step:{},NER loss:{:>9.6f}".format(epoch, step, average_loss))
                    average_loss = 0

                if step % ner_tv.valid_every == 0:
                    real_total_labels = []
                    predict_total_labels = []
                    for test_inputs, test_labels in data_manager.test_iterbatch():
                        feed_dict = model_obj.create_feed_dict(inputs=test_inputs, labels=test_labels, is_train=False)
                        logits_test_var,lengths_test_var,trans_matrix = sess.run([test_logits,test_lengths,model_obj.trans],feed_dict=feed_dict)
                        real_labels,predict_labels= model_obj.test_accuraty(lengths_test_var,logits_test_var,trans_matrix,test_labels)
                        real_total_labels.extend(real_labels)
                        predict_total_labels.extend(predict_labels)
                    f1_score_value = f1_score(real_total_labels,predict_total_labels,labels=[0,1,2,3],average='micro')
                    print("iteration:{},NER ,f1 score:{:>9.6f}".format(epoch, f1_score_value))
                    if best_f1 < f1_score_value:
                        print("mew best f1_score,save model ")
                        model_obj.saver.save(sess, model_obj.model_save_path, global_step=step)
                        best_f1 = f1_score_value

import os

def make_pb_file_from_model():
    #g = tf.Graph()
    #with g.as_default(),tf.device('/cpu:0'):
    model_obj = Model()
    _,logits, lengths = model_obj.logits_and_loss()
    print("Model trans_matrix:",model_obj.trans.name)
    sess = tf.Session()
    model_obj.saver = tf.train.Saver()
    model_obj.model_restore(sess)
    print(model_obj.trans)
    output_tensor = []
    output_tensor.append(model_obj.trans.name.replace(":0", ""))
    output_tensor.append(lengths.name.replace(":0", ""))
    output_tensor.append(logits.name.replace(":0", ""))
    output_tensor.append(model_obj.dropout.name.replace(":0", ""))


    output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_tensor)
    with tf.gfile.FastGFile(os.path.join(ner_tv.train_model_bi_lstm, "weight_ner.pb"),
                            'wb') as gf:
        gf.write(output_graph_with_weight.SerializeToString())

if __name__ == '__main__':
    #make_pb_file_from_model()
    train()










