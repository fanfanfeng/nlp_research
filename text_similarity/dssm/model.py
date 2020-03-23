
# create by fanfan on 2020/3/18 0018
import tensorflow as tf
from tensorflow.contrib.rnn import  LSTMCell
from text_similarity.config import Config
from tensorflow.contrib.layers import fully_connected

class Dssm(object):
    def __init__(self,params):
        self.params = params

    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def init_embedding(self):
        with tf.variable_scope('embedding_define'):
            initial_value = tf.truncated_normal([self.params.vocab_size, self.params.embedding_size], mean=0.0, stddev=0.1, dtype=tf.float32)
            embedding = tf.Variable(initial_value=initial_value, trainable=True, dtype=tf.float32)
        return embedding

    def layer_bilstm(self,input,input_lengths):
        with tf.variable_scope("layer_bilstm",reuse=tf.AUTO_REUSE):
            cell_fw = LSTMCell(self.params.hidden_num)
            cell_bw = LSTMCell(self.params.hidden_num)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,input,input_lengths,dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs[:,-1,:]

    def layer_full_connect(self,input):
        hiden_size_list = [500,200]
        output = input
        with tf.variable_scope('layer_full_connect',reuse=tf.AUTO_REUSE):
            for index,size in enumerate(hiden_size_list):
                with tf.variable_scope("full_connect_"+str(index)):
                    output = fully_connected(input,size)
        return output


    def vector_input(self,input,dropout,already_embedded=False,real_sentence_length=None):
        with tf.variable_scope("sentence_vec",reuse=tf.AUTO_REUSE) as scope:
            if already_embedded:
                input_embeddings = input
                real_sentence_length = real_sentence_length
            else:
                word_embeddings = self.init_embedding()
                input_embeddings = tf.nn.embedding_lookup(word_embeddings, input)
                real_sentence_length = self.get_setence_length(input)

        lstm_output = self.layer_bilstm(input_embeddings,real_sentence_length)
        fullylayer_output = self.layer_full_connect(lstm_output)

        output = tf.nn.dropout(fullylayer_output,keep_prob=dropout)
        return output








    def create_model(self,input_x,input_y,dropout,already_embedded=False,real_sentence_length=None,real_sentence_length_y=None):
        '''
        :param input_x: shape=[self.batch_size, self.sequence_length]
        :param input_y: shape=[self.batch_size, N, self.sequence_length] ,N = positive+nagavte例如1个正例4个负例，
        :param dropout: 
        :param already_embedded: 
        :param real_sentence_length: 
        :param real_sentence_length_y: 
        :return: 
        '''
        with tf.variable_scope("model_define") as scope:
            input_vec = self.vector_input(input_x,dropout,already_embedded,real_sentence_length)

            input_y_reshape = tf.reshape(input_y,[-1,self.params.max_sentence_len])
            input_other_vec = self.vector_input(input_y_reshape,dropout,already_embedded,real_sentence_length_y)
            input_other_vec = tf.reshape(input_other_vec,[3,7,-1])
            input_vec_expand = tf.expand_dims(input_vec,axis=1)

            query_norm = tf.sqrt(tf.reduce_sum(tf.square(input_vec_expand), axis=2))
            input_other_vec_norm = tf.sqrt(tf.reduce_sum(tf.square(input_other_vec), axis=2))
            norm = tf.multiply(query_norm,input_other_vec_norm)

            dot_vec = tf.reduce_sum(tf.multiply(input_other_vec,input_vec_expand),axis=-1)

            cosine_raw = tf.truediv(dot_vec, norm)

            pos_and_neg_softmax = tf.nn.softmax(cosine_raw,axis=-1)
            pos = tf.slice(pos_and_neg_softmax, [0, 0], [-1, 1])
            loss = -tf.reduce_mean(tf.log(pos))
            return loss




if __name__ == '__main__':
    input_x = tf.placeholder(tf.int32,shape=[3,10])
    input_y = tf.placeholder(tf.int32,shape=[3,7,10])
    dropout = tf.placeholder(tf.float32,shape=())

    config = Config()
    config.max_sentence_len = 10
    obj = Dssm(config)
    obj.create_model(input_x,input_y,dropout)