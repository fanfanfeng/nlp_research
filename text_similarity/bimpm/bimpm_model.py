# create by fanfan on 2019/12/13 0013
import tensorflow as tf
from text_similarity.bimpm import utils
import math

num_perspective = 12

class BIMPM():
    def __init__(self,is_training,seq_length,class_num,vocabulary_size,embedding_size,hidden_num,l2_lambda,learning_rate):
        self.create_placeholders(seq_length,class_num,hidden_num)


        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            self.vocab_matrix = tf.get_variable(name='embed', shape=(vocabulary_size, embedding_size),dtype=tf.float32)
            self.text_a_embed = tf.nn.embedding_lookup(self.vocab_matrix,self.text_a)
            self.text_b_embed = tf.nn.embedding_lookup(self.vocab_matrix,self.text_b)


        with tf.name_scope('Input_Encoding'):
            with tf.variable_scope('h'):
                (p_fw,p_bw) = self.biLSTMBlock(self.text_a_embed,hidden_num,'Input_Encoding/biLSTM')#,self.a_length)
            with tf.variable_scope('p'):
                (h_fw,h_bw) = self.biLSTMBlock(self.text_b_embed,hidden_num,'Input_Encoding/biLSTM')#,self.b_length,isReuse=True)
            p_fw = utils.dropout_layer(p_fw,self.dropout_keep_prob)
            p_bw = utils.dropout_layer(p_bw,self.dropout_keep_prob)
            h_fw = utils.dropout_layer(h_fw,self.dropout_keep_prob)
            h_bw = utils.dropout_layer(h_bw,self.dropout_keep_prob)


        # ----- Matching Layer -----
        # 1、Full-Matching
        p_full_fw = utils.full_matching(p_fw,tf.expand_dims(h_fw[:,-1,:],1),self.w1,num_perspective)
        p_full_bw = utils.full_matching(p_bw,tf.expand_dims(h_bw[:,0,:],1),self.w2,num_perspective)

        h_full_fw = utils.full_matching(h_fw,tf.expand_dims(p_fw[:,-1,:],1),self.w1,num_perspective)
        h_full_bw = utils.full_matching(h_bw,tf.expand_dims(p_bw[:,0,:],1),self.w2,num_perspective)

        # 2. Maxpooling-Mathcing
        max_fw = utils.maxpool_full_matching(p_fw,h_fw,self.w3,num_perspective)
        max_bw = utils.maxpool_full_matching(p_bw,h_bw,self.w4,num_perspective)

        # 3. Attentive-Matching
        # 计算权重，用余弦值表示
        fw_cos = utils.cosine(p_fw,h_fw)
        bw_cos = utils.cosine(p_bw,h_bw)

        # 计算attentive-Matching
        p_atten_fw = tf.matmul(fw_cos,p_fw)
        p_atten_bw = tf.matmul(bw_cos,p_bw)
        h_atten_fw = tf.matmul(fw_cos,h_fw)
        h_atten_bw = tf.matmul(bw_cos,h_bw)

        p_mean_fw = p_atten_fw / tf.reduce_sum(fw_cos,axis=2,keepdims=True)
        p_mean_bw = p_atten_bw / tf.reduce_sum(bw_cos,axis=2,keepdims=True)
        h_mean_fw = h_atten_fw / tf.reduce_sum(fw_cos,axis=2,keepdims=True)
        h_mean_bw = h_atten_bw / tf.reduce_sum(bw_cos,axis=2,keepdims=True)

        p_atten_mean_fw = utils.full_matching(p_fw,p_mean_fw,self.w5,num_perspective)
        p_atten_mean_bw = utils.full_matching(p_bw,p_mean_bw,self.w6,num_perspective)
        h_atten_mean_fw = utils.full_matching(h_fw,h_mean_fw,self.w5,num_perspective)
        h_atten_mean_bw = utils.full_matching(h_bw,h_mean_bw,self.w6,num_perspective)

        # 4.Max-Attentive-Matching
        p_max_fw = tf.reduce_max(p_atten_fw,axis=2,keepdims=True)
        p_max_bw = tf.reduce_max(p_atten_bw,axis=2,keepdims=True)
        h_max_fw = tf.reduce_max(h_atten_fw,axis=2,keepdims=True)
        h_max_bw = tf.reduce_max(h_atten_bw,axis=2,keepdims=True)

        p_atten_max_fw = utils.full_matching(p_fw,p_max_fw,self.w7,num_perspective)
        p_atten_max_bw = utils.full_matching(p_bw,p_max_bw,self.w8,num_perspective)
        h_atten_max_fw = utils.full_matching(h_fw,h_max_fw,self.w7,num_perspective)
        h_atten_max_bw = utils.full_matching(h_bw,h_max_bw,self.w8,num_perspective)


        mv_p = tf.concat([p_full_fw,max_fw,p_atten_mean_fw,p_atten_max_fw,
                          p_full_bw,max_bw,p_atten_mean_bw,p_atten_max_bw],axis=2)
        mv_h = tf.concat([h_full_fw,max_fw,h_atten_mean_fw,h_atten_max_fw,
                          h_full_bw,max_bw,h_atten_mean_bw,h_atten_max_bw],axis=2)
        mv_p = utils.dropout_layer(mv_p,self.dropout_keep_prob)
        mv_h = utils.dropout_layer(mv_h,self.dropout_keep_prob)

        mv_p = tf.reshape(mv_p,[-1,mv_p.shape[1],mv_p.shape[2]*mv_p.shape[3]])
        mv_h = tf.reshape(mv_h,[-1,mv_h.shape[1],mv_h.shape[2]*mv_h.shape[3]])


        # Aggregation Layer
        with tf.variable_scope("Aggregation"):
            with tf.variable_scope("bilstm_agg_p", reuse=tf.AUTO_REUSE):
                (p_fw_last, p_bw_last) = self.biLSTMBlock(mv_p, hidden_num, 'biLSTM')#, self.a_length)
            with tf.variable_scope("bilstm_agg_h", reuse=tf.AUTO_REUSE):
                (h_fw_last, h_bw_last) = self.biLSTMBlock(mv_h, hidden_num, 'biLSTM')#, self.b_length)


        with tf.variable_scope("Predict"):
            x = tf.concat((p_fw_last,p_bw_last,h_fw_last,h_bw_last),axis=2)
            x = tf.reshape(x,shape=[-1,x.shape[1] * x.shape[2]])
            x = utils.dropout_layer(x,self.dropout_keep_prob)

            x = tf.layers.dense(x,10000,activation=tf.nn.relu)
            x = utils.dropout_layer(x,self.dropout_keep_prob)
            x = tf.layers.dense(x,2500,activation=tf.nn.relu)
            x = utils.dropout_layer(x,self.dropout_keep_prob)
            x = tf.layers.dense(x,512,activation=tf.nn.relu)
            x = utils.dropout_layer(x,self.dropout_keep_prob)


            self.logits = tf.layers.dense(x, class_num)

            self.score = tf.nn.softmax(self.logits, name='score')
            self.prediction = tf.argmax(self.score, 1, name='prediction')

        with tf.name_scope('cost'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            self.cost = tf.reduce_mean(self.cost)
            #weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            #l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = self.cost#l2_loss + self.cost

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32)
        )


        tvars = tf.trainable_variables()
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        #self.train_op = optimizer.apply_gradients(zip(grads, tvars))









    def create_placeholders(self,seq_length,class_num,hidden_num):
        self.text_a = tf.placeholder(tf.int32, [None, seq_length],name='premise')
        self.text_b = tf.placeholder(tf.int32, [None, seq_length],name='hypothesis')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.a_length = tf.placeholder(tf.int32, [None], name='a_length')
        self.b_length = tf.placeholder(tf.int32, [None], name='b_length')
        self.y = tf.placeholder(tf.int32, [None, class_num], name='y')

        for i in range(1, 9):
            setattr(self, f'w{i}', tf.get_variable(name=f'w{i}', shape=(num_perspective, hidden_num),
                                                   dtype=tf.float32))


    def biLSTMBlock(self,inputs,num_units,scope,seq_len=None,isReuse=False):
        with tf.variable_scope(scope,reuse=isReuse):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            #fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
           # bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)
            (a_outputs, a_output_states) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell_fw,
                cell_bw = cell_bw,
                inputs = inputs,
                sequence_length = seq_len,
                dtype = tf.float32
            )
            return a_outputs


if __name__ == '__main__':
    bimpm = BIMPM(True, 20, 2, 10000, 300, 300, 0.001, 0.0001)