# create by fanfan on 2019/7/18 0018
import tensorflow as tf
from dialog_system.MultiTurnResponseSelection.params import Params
import pickle

params = Params()
class SequentialMatchingNetwork():
    def __init__(self):
        pass


    def create_gru_cell(self):
        cell = tf.nn.rnn_cell.GRUCell(params.rnn_units,kernel_initializer=tf.orthogonal_initializer())
        return cell


    def build_model(self):
        self.utterance_ph = tf.placeholder(tf.int32,shape=(None,params.max_num_utterance,params.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32,shape=(None,params.max_sentence_len))

        self.y_true = tf.placeholder(tf.int32,shape=(None,))

        self.embedding_ph = tf.placeholder(tf.float32,shape=(params.total_words,params.word_embedding_size))

        self.response_len = tf.placeholder(tf.int32,shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32,shape=(None,params.max_num_utterance))

        word_embeddings = tf.get_variable('word_embeddings',shape=(params.total_words,params.word_embedding_size))

        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings,self.utterance_ph)
        response_embeddings = tf.nn.embedding_lookup(word_embeddings,self.response_ph)



        with tf.variable_scope("sentence_layer"):
            gru_cell = self.create_gru_cell()

            response_GRU_embeddings, _ = tf.nn.dynamic_rnn(gru_cell, response_embeddings,
                                                           sequence_length=self.response_len, dtype=tf.float32,
                                                           scope='sentence_GRU')


        with tf.variable_scope("UtteranceResponseMatching"):
            response_embeddings_reverse = tf.transpose(response_embeddings,perm=[0,2,1])
            response_gru_embeddings_reverse = tf.transpose(response_GRU_embeddings,perm=[0,2,1])

            matching_vectors = []

            all_utterance_embeddings  = tf.unstack(all_utterance_embeddings,num=params.max_num_utterance,axis=1)
            all_utterance_len = tf.unstack(self.all_utterance_len_ph,num=params.max_num_utterance,axis=1)

            A_matrix = tf.get_variable('A_matrix_v',shape=(params.rnn_units,params.rnn_units),initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)

            for utterance_embeddings,utterance_len in zip(all_utterance_embeddings,all_utterance_len):
                # 相关度矩阵 1
                matrix1 = tf.matmul(utterance_embeddings,response_embeddings_reverse)

                utterance_GRU_embeddins,_ = tf.nn.dynamic_rnn(gru_cell,response_embeddings,sequence_length=utterance_len, dtype=tf.float32,
                                                            scope='sentence_GRU')
                maxtrix2 = tf.einsum('aij,jk->aik',utterance_GRU_embeddins,A_matrix)
                maxtrix2 = tf.matmul(maxtrix2,response_gru_embeddings_reverse)

                matrix = tf.stack([matrix1,maxtrix2],axis=3,name='matrix_stack')
                conv_layer = tf.layers.conv2d(matrix,filters=8,kernel_size=(3,3),padding='VALID',kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.nn.relu,reuse=tf.AUTO_REUSE,name='conv')

                pooling_layer = tf.layers.max_pooling2d(conv_layer,(3,3),strides=(3,3),
                                                        padding='VALID',name='max_pooling')
                matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer),50,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   activation=tf.tanh,
                                                   reuse=tf.AUTO_REUSE,
                                                   name='matching_V')
                matching_vectors.append(matching_vector)

            total_matching_vector = tf.stack(matching_vectors,axis=0,name='matching_stack')
            _,last_state = tf.nn.dynamic_rnn(self.create_gru_cell(),total_matching_vector,dtype=tf.float32,time_major=True,scope='final_GRU')

            logits = tf.layers.dense(last_state,2,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='final_V')

            self.y_pred = tf.nn.softmax(logits)
            self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true,logits=logits))

            tf.summary.scalar('loss',self.total_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(self.total_loss)









