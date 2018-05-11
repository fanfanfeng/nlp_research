# create by fanfan on 2018/4/16 0016
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from DualLSTMEncoderRankModel import config
import os
class Ranker:
    def __init__(self,mode="train"):
        '''
        model:["train", "valid", "eval", "predict", "cache"]
        :param mode: 
        '''
        self.mode = mode
        self.dtype = tf.float32


        self.buildNetwork()

    def _create_rnn_cell(self):
        cell = rnn.LSTMCell(config.hiddenSize,use_peepholes=True,state_is_tuple=True)
        cell = rnn.DropoutWrapper(cell,output_keep_prob=config.dropout)
        return cell

    def buildNetwork(self):
        # 假设 batchsize = 100 ,句子长度20
        with tf.name_scope('placeholder_query'):
            # shape = [100,20]
            self.query_seqs = tf.placeholder(tf.int32,[None,None],name='query')
            # shape = [100]
            self.query_length = tf.placeholder(tf.int32,[None],name='query_length')

        with tf.name_scope('placeholder_labels'):
            # shape = [100]
            self.targets = tf.placeholder(tf.int32,[None,1],name='targets')

        with tf.name_scope('placeholder_response'):
            # shape = [100,20]
            self.response_seqs = tf.placeholder(tf.int32,[None,None],name='response')
            # shape = [100]
            self.response_length = tf.placeholder(tf.int32,[None],name='response_length')

        with tf.name_scope('embedding_layer'):
            # shape=[40000,200]
            self.embedding = tf.get_variable('embedding',[config.max_vocab_size, config.embedding_size])
            # shape=[100,20,200]
            self.embed_query = tf.nn.embedding_lookup(self.embedding,self.query_seqs)
            # shape=[100,20,200]
            self.embed_response = tf.nn.embedding_lookup(self.embedding,self.response_seqs)

        encoder_cell = rnn.MultiRNNCell([self._create_rnn_cell() for _ in range(config.layer_num)])

        # shape = [100,20, 256] ,[h:(100,256),c:(100,256)]
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            encoder_cell,
            tf.concat([self.embed_query, self.embed_response], 0),
            sequence_length=tf.concat([self.query_length, self.response_length], 0),
            dtype=tf.float32)
        query_final_state, response_final_state = tf.split(rnn_states[-1].h, 2, 0)

        with tf.variable_scope('bilinar_regression'):
            # shape= [256,256]
            W = tf.get_variable('bilinear_W',shape=[config.hiddenSize,config.hiddenSize],initializer=tf.truncated_normal_initializer())

            # "Predict" a  response: c * M
            generated_response = tf.matmul(response_final_state, W)
            generated_response = tf.expand_dims(generated_response, 2)
            query_final_state = tf.expand_dims(query_final_state, 2)

            # Dot product between generated response and actual response
            # (c * M) * r
            logits = tf.matmul(generated_response, query_final_state, True)
            logits = tf.squeeze(logits, [2])

            # Apply sigmoid to convert logits to probabilities
            self.probs = tf.sigmoid(logits)


        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(self.targets))

        self.means_loss = tf.reduce_mean(losses,name='mean_loss')
        train_loss_summary = tf.summary.scalar('loss',self.means_loss)

        self.training_summaries = tf.summary.merge([train_loss_summary])

        opt = tf.train.AdamOptimizer(learning_rate= config.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
        self.train_op = opt.minimize(self.means_loss)


    def step(self,batch):
        feedDict = {}
        ops = None


        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            ops = (self.train_op,self.means_loss,self.training_summaries)
            feedDict[self.targets] = np.eye(len(batch.query_seqs))

        return ops,feedDict






if __name__ == '__main__':
    Ranker(mode='eval')
