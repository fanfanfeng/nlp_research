
# create by fanfan on 2020/3/18 0018
import tensorflow as tf
from tensorflow.contrib.rnn import  LSTMCell
from text_similarity.config import Config
from utils.tf_utils.loss import contrastive_loss

class Siamese(object):
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

    def layer_bilstm(self,input,input_lengths,reuse=tf.AUTO_REUSE,scope_name="layer_bilstm"):
        with tf.variable_scope(scope_name,reuse=reuse):
            cell_fw = LSTMCell(self.params.hidden_num)
            cell_bw = LSTMCell(self.params.hidden_num)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,input,input_lengths,dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs[:,-1,:]


    def layer_logit(self,input_x,input_y):
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(input_x, input_y)), 1, keep_dims=True))
        distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(input_x), 1, keep_dims=True)),
                                                     tf.sqrt(tf.reduce_sum(tf.square(input_y), 1, keep_dims=True))))
        distance = tf.reshape(distance, [-1], name="distance")
        return distance

    def layer_loss(self,logit,labels):
        loss = contrastive_loss(labels,logit)
        return loss

    def layer_predict(self,logits,labels):
        temp_sim = tf.subtract(tf.ones_like(logits), tf.rint(logits),
                                    name="temp_sim")  # auto threshold 0.5
        correct_predictions = tf.equal(temp_sim, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return temp_sim,accuracy

    def create_model(self,input_x,input_y,dropout,already_embedded=False,real_sentence_length=None,real_sentence_length_y=None):
        '''
        :param input_x: shape=[self.batch_size, self.sequence_length]
        :param input_y: shape=[self.batch_size, self.sequence_length]
        :param dropout: 
        :param already_embedded: 
        :param real_sentence_length: 
        :param real_sentence_length_y: 
        :return: 
        '''
        with tf.variable_scope("model_define") as scope:
            with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE) as scope:
                if not already_embedded:
                    word_embeddings = self.init_embedding()
                    real_sentence_length = self.get_setence_length(input_x)
                    input_x = tf.nn.embedding_lookup(word_embeddings, input_x)

                    real_sentence_length_y = self.get_setence_length(input_y)
                    input_y = tf.nn.embedding_lookup(word_embeddings, input_y)



            vec_input_x = self.layer_bilstm(input_x,real_sentence_length)
            vec_inout_y = self.layer_bilstm(input_y,real_sentence_length_y)

            logit = self.layer_logit(vec_input_x,vec_inout_y)
            return logit




if __name__ == '__main__':
    input_x = tf.placeholder(tf.int32,shape=[3,10])
    input_y = tf.placeholder(tf.int32,shape=[3,10])
    dropout = tf.placeholder(tf.float32,shape=())

    config = Config()
    config.max_sentence_len = 10
    obj = Siamese(config)
    obj.create_model(input_x,input_y,dropout)