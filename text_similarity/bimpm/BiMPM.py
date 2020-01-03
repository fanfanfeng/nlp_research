import tensorflow as tf

import os

num_perspective=12
class BIMPM:
    def __init__(self,is_training,seq_length,class_num,vocabulary_size,embedding_size,hidden_num,l2_lambda,learning_rate):
        self.hidden_size= hidden_num
        self.class_num=class_num
        self.text_a = tf.placeholder(name='p', shape=(None, seq_length), dtype=tf.int32)
        self.text_b = tf.placeholder(name='h', shape=(None, seq_length), dtype=tf.int32)


        self.dropout_keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)
        self.a_length = tf.placeholder(tf.int32, [None], name='a_length')
        self.b_length = tf.placeholder(tf.int32, [None], name='b_length')
        self.y = tf.placeholder(tf.int32, [None, class_num], name='y')

        self.embed = tf.get_variable(name='embed', shape=(vocabulary_size, embedding_size),
                                     dtype=tf.float32)

        for i in range(1, 9):
            setattr(self, f'w{i}', tf.get_variable(name=f'w{i}', shape=(num_perspective, hidden_num),
                                                   dtype=tf.float32))

        self.forward()
        self.train()

    def BiLSTM(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, activation='tanh')
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, activation='tanh')

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)



    def full_matching(self, metric, vec, w):
        w = tf.expand_dims(tf.expand_dims(w, 0), 2)
        metric = w * tf.stack([metric] * num_perspective, axis=1)
        vec = w * tf.stack([vec] * num_perspective, axis=1)

        m = tf.matmul(metric, tf.transpose(vec, [0, 1, 3, 2]))
        n = tf.norm(metric, axis=3, keep_dims=True) * tf.norm(vec, axis=3, keep_dims=True)
        cosine = tf.transpose(m / n, [0, 2, 3, 1])

        return cosine

    def maxpool_full_matching(self, v1, v2, w):
        cosine = self.full_matching(v1, v2, w)
        max_value = tf.reduce_max(cosine, axis=2, keep_dims=True)
        return max_value

    def cosine(self, v1, v2):
        m = tf.matmul(v1, tf.transpose(v2, [0, 2, 1]))
        n = tf.norm(v1, axis=2, keep_dims=True) * tf.norm(v2, axis=2, keep_dims=True)
        cosine = m / n
        return cosine

    def dropout(self, x):
        return tf.nn.dropout(x, self.dropout_keep_prob)

    def forward(self):
        # ----- Word Representation Layer -----
        # 字嵌入
        p_embedding = tf.nn.embedding_lookup(self.embed, self.text_a)
        h_embedding = tf.nn.embedding_lookup(self.embed, self.text_b)


        p_embedding = self.dropout(p_embedding)
        h_embedding = self.dropout(h_embedding)

        # ----- Context Representation Layer -----
        # 论文中是取context，tf不会输出所有时刻的ctx，这里用输出值代替
        with tf.variable_scope("bilstm_p", reuse=tf.AUTO_REUSE):
            (p_fw, p_bw), _ = self.BiLSTM(p_embedding)
        with tf.variable_scope("bilstm_h", reuse=tf.AUTO_REUSE):
            (h_fw, h_bw), _ = self.BiLSTM(h_embedding)

        p_fw = self.dropout(p_fw)
        p_bw = self.dropout(p_bw)
        h_fw = self.dropout(h_fw)
        h_bw = self.dropout(h_bw)

        # ----- Matching Layer -----
        # 1、Full-Matching
        p_full_fw = self.full_matching(p_fw, tf.expand_dims(h_fw[:, -1, :], 1), self.w1)
        p_full_bw = self.full_matching(p_bw, tf.expand_dims(h_bw[:, 0, :], 1), self.w2)
        h_full_fw = self.full_matching(h_fw, tf.expand_dims(p_fw[:, -1, :], 1), self.w1)
        h_full_bw = self.full_matching(h_bw, tf.expand_dims(p_bw[:, 0, :], 1), self.w2)

        # 2、Maxpooling-Matching
        max_fw = self.maxpool_full_matching(p_fw, h_fw, self.w3)
        max_bw = self.maxpool_full_matching(p_bw, h_bw, self.w4)

        # 3、Attentive-Matching
        # 计算权重即相似度
        fw_cos = self.cosine(p_fw, h_fw)
        bw_cos = self.cosine(p_bw, h_bw)

        # 计算attentive vector
        p_att_fw = tf.matmul(fw_cos, p_fw)
        p_att_bw = tf.matmul(bw_cos, p_bw)
        h_att_fw = tf.matmul(fw_cos, h_fw)
        h_att_bw = tf.matmul(bw_cos, h_bw)

        p_mean_fw = p_att_fw / tf.reduce_sum(fw_cos, axis=2, keep_dims=True)
        p_mean_bw = p_att_bw / tf.reduce_sum(bw_cos, axis=2, keep_dims=True)
        h_mean_fw = h_att_fw / tf.reduce_sum(fw_cos, axis=2, keep_dims=True)
        h_mean_bw = h_att_bw / tf.reduce_sum(fw_cos, axis=2, keep_dims=True)

        p_att_mean_fw = self.full_matching(p_fw, p_mean_fw, self.w5)
        p_att_mean_bw = self.full_matching(p_bw, p_mean_bw, self.w6)
        h_att_mean_fw = self.full_matching(h_fw, h_mean_fw, self.w5)
        h_att_mean_bw = self.full_matching(h_bw, h_mean_bw, self.w6)

        # 4、Max-Attentive-Matching
        p_max_fw = tf.reduce_max(p_att_fw, axis=2, keep_dims=True)
        p_max_bw = tf.reduce_max(p_att_bw, axis=2, keep_dims=True)
        h_max_fw = tf.reduce_max(h_att_fw, axis=2, keep_dims=True)
        h_max_bw = tf.reduce_max(h_att_bw, axis=2, keep_dims=True)

        p_att_max_fw = self.full_matching(p_fw, p_max_fw, self.w7)
        p_att_max_bw = self.full_matching(p_bw, p_max_bw, self.w8)
        h_att_max_fw = self.full_matching(h_fw, h_max_fw, self.w7)
        h_att_max_bw = self.full_matching(h_bw, h_max_bw, self.w8)

        mv_p = tf.concat(
            (p_full_fw, max_fw, p_att_mean_fw, p_att_max_fw,
             p_full_bw, max_bw, p_att_mean_bw, p_att_max_bw),
            axis=2)

        mv_h = tf.concat(
            (h_full_fw, max_fw, h_att_mean_fw, h_att_max_fw,
             h_full_bw, max_bw, h_att_mean_bw, h_att_max_bw),
            axis=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        mv_p = tf.reshape(mv_p, [-1, mv_p.shape[1], mv_p.shape[2] * mv_p.shape[3]])
        mv_h = tf.reshape(mv_h, [-1, mv_h.shape[1], mv_h.shape[2] * mv_h.shape[3]])

        # ----- Aggregation Layer -----
        with tf.variable_scope("bilstm_agg_p", reuse=tf.AUTO_REUSE):
            (p_f_last, p_b_last), _ = self.BiLSTM(mv_p)
        with tf.variable_scope("bilstm_agg_h", reuse=tf.AUTO_REUSE):
            (h_f_last, h_b_last), _ = self.BiLSTM(mv_h)

        x = tf.concat((p_f_last, p_b_last, h_f_last, h_b_last), axis=2)
        x = tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2]])
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = tf.layers.dense(x, 20000, activation='relu')
        x = self.dropout(x)
        x = tf.layers.dense(x, 10000, activation='relu')
        x = self.dropout(x)
        x = tf.layers.dense(x, 512, activation='relu')
        x = self.dropout(x)
        self.logits = tf.layers.dense(x, self.class_num)

    def train(self):

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), tf.cast(tf.argmax(self.y, axis=1), tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

if __name__ == '__main__':
    bimpm = BIMPM(True, 20, 2, 10000, 300, 300, 0.001, 0.0001)

# if __name__ == '__main__':
#     train()
# test()
