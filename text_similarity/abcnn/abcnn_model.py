# create by fanfan on 2019/12/31 0031
import tensorflow as tf

conv_layers = 3 # cnn的层数
model_type = 'ABCNN-2'  #one of BCNN, ABCNN-1, ABCNN-2, ABCNN-3
filter_num = 200
filter_width = 3
use_last_only = True

class ABCNN(object):
    def create_placeholders(self,seq_length,class_num,hidden_num):
        self.text_a = tf.placeholder(tf.int32, [None, seq_length],name='premise')
        self.text_b = tf.placeholder(tf.int32, [None, seq_length],name='hypothesis')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.a_length = tf.placeholder(tf.int32, [None], name='a_length')
        self.b_length = tf.placeholder(tf.int32, [None], name='b_length')
        self.y = tf.placeholder(tf.int32, [None, class_num], name='y')

    def __init__(self,is_training,seq_length,class_num,vocabulary_size,embedding_size,hidden_num,l2_lambda,learning_rate):
        self.create_placeholders(seq_length,class_num,hidden_num)

        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_text_a = tf.nn.embedding_lookup(W, self.text_a)
            self.embedded_text_b = tf.nn.embedding_lookup(W, self.text_b)

            tq1 = tf.transpose(self.embedded_text_a,[0,2,1],name='t_q1')
            tq2 = tf.transpose(self.embedded_text_b,[0,2,1],name='t_q2')

            self.q1 = tf.expand_dims(tq1,-1,name='q1')
            self.q2 = tf.expand_dims(tq2,-1,name='q2')

        layer_input = [self.q1,self.q2]
        sims = []
        for i in range(conv_layers):
            last = (i == conv_layers - 1)
            if model_type == 'BCNN':
                ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool = self._add_BCNN(i,filter_num,layer_input,filter_width)
            elif model_type == 'ABCNN-1':
                ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool = self._add_ABCNN_1(i,filter_num,layer_input,filter_width)
            elif model_type == 'ABCNN-2':
                ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool = self._add_ABCNN_2(i,filter_num,layer_input,filter_width)
            elif model_type == 'ABCNN-3':
                ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool = self._add_ABCNN_3(i,filter_num,layer_input,filter_width)
            else:
                raise ValueError("model_type is error")

            layer_input = [ap1_width_pool,ap2_width_pool]
            if last :
                sims = [ap1_all_pool,ap2_all_pool]




        with tf.variable_scope("fc") as scope:
            fc_in = tf.concat(sims,axis=1)
            fc_in = tf.nn.dropout(fc_in,keep_prob= self.dropout_keep_prob)
            self.logits = tf.layers.dense(fc_in,class_num,kernel_initializer=tf.truncated_normal_initializer)

        with tf.variable_scope('loss') as scope:
            cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.logits))
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = l2_loss + cross_entroy

        self.score = tf.nn.softmax(self.logits,name='score')
        self.prediction = tf.argmax(self.score, 1, name='prediction')

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32)
        )

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def pad_for_wide_conv(self,x,f_width):
        return tf.pad(x, [[0, 0], [0, 0], [f_width - 1, f_width - 1], [0, 0]], "CONSTANT", name="pad_wide_conv")

    def conv_layer(self,filter_width,filter_num,input):
        conv = tf.layers.conv2d(input,filter_num,[input.get_shape()[1],filter_width],activation=tf.nn.relu)
        return conv
    def pool_layer(self,filter_width,input,name=None):
        pool = tf.layers.average_pooling2d(input,[1,filter_width],strides=1,name=name)
        return pool
    def euclidean_score(self,v1,v2):
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1-v2),axis=1))
        return 1 / 1 + euclidean

    def cos_sim(self,v1,v2):
        norm_v1 = tf.norm(v1,axis=1) + 1e-6
        norm_v2 = tf.norm(v2,axis=1) + 1e-6
        dot_products = tf.reduce_sum(v1 *v2,axis=1,name='cos_sim')
        return dot_products/(norm_v1 * norm_v2)

    def make_attention_mat(self,x1, x2):
        #sq_dist = tf.squared_difference(x1, tf.transpose(x2,perm=[0,1,3,2]), name='sq_dist')
        #pair_dist = tf.sqrt(tf.reduce_sum(sq_dist, axis=[1]), name='pair_dist') + 1e-6
        #similarity = tf.reciprocal(tf.add(pair_dist, tf.constant(1.0, dtype=tf.float32, name='one')),
        #                           name='similarity')

        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.transpose(x2,perm=[0,1,3,2])),axis=1) +1e-6)

        attention_matrix = 1 / (euclidean + 1)
        return attention_matrix

    def make_attention_mat_other(self,x1, x2):
        # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
        # x2 => [batch, height, 1, width]
        # [batch, width, wdith] = [batch, s, s]

        # 作者论文中提出计算attention的方法 在实际过程中反向传播计算梯度时 容易出现NaN的情况 这里面加以修改
        # euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
        # return 1 / (1 + euclidean)

        x1 = tf.transpose(tf.squeeze(x1, [-1]), [0, 2, 1])
        attention = tf.einsum("ijk,ikl->ijl", x1, tf.squeeze(x2, [-1]))
        return attention


    def _add_BCNN(self, id, filter_num, layer_input,filter_width):
        scope_name = 'BCNN_' + str(id)
        with tf.variable_scope(scope_name,tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope("conv",reuse=tf.AUTO_REUSE) as scope:
                padded_in1 = self.pad_for_wide_conv(layer_input[0],filter_width)
                padded_in2 = self.pad_for_wide_conv(layer_input[1],filter_width)

                conv1 = self.conv_layer(filter_width,filter_num,padded_in1)
                conv2 = self.conv_layer(filter_width,filter_num,padded_in2)

            with tf.variable_scope("all-ap") as scope:
                ap1_all_pool = tf.reduce_mean(conv1,axis=[1,2])
                ap2_all_pool = tf.reduce_mean(conv2,axis=[1,2])

            with tf.variable_scope("%d-ap" %  filter_width) as scope:
                avg_pool1 = self.pool_layer(filter_width,conv1,name='avg_pool1')
                avg_pool2 = self.pool_layer(filter_width,conv2,name='avg_pool2')

                ap1_width_pool = tf.transpose(avg_pool1,perm=[0,3,2,1],name='ap1')
                ap2_width_pool = tf.transpose(avg_pool2,perm=[0,3,2,1],name='ap2')
            return ap1_width_pool,ap1_all_pool,ap2_width_pool,ap2_all_pool


    def _add_ABCNN_1(self, id, filter_num, layer_input,filter_width):
        scope_name = 'ABCNN1_' + str(id)
        with tf.variable_scope(scope_name,tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope("similary") as scope:
                similarity = self.make_attention_mat(layer_input[0],layer_input[1])

            with tf.variable_scope('attention') as scope:
                W = tf.get_variable("W",[layer_input[0].get_shape()[2],layer_input[1].get_shape()[1]],initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                A1 = tf.matrix_transpose(tf.einsum('ijk,kl->ijl',similarity,W))
                A2 = tf.matrix_transpose(tf.einsum('ijk,kl->ijl',tf.matrix_transpose(similarity),W))




            with tf.variable_scope("conv",reuse=tf.AUTO_REUSE) as scope:
                layer_in1 = tf.concat([layer_input[0],tf.expand_dims(A1,-1)],axis=-1)
                layer_in2 = tf.concat([layer_input[1],tf.expand_dims(A2,-1)],axis=-1)

                padded_in1 = self.pad_for_wide_conv(layer_in1,filter_width)
                padded_in2 = self.pad_for_wide_conv(layer_in2,filter_width)

                conv1 = self.conv_layer(filter_width,filter_num,padded_in1)
                conv2 = self.conv_layer(filter_width,filter_num,padded_in2)

            with tf.variable_scope("all-ap") as scope:
                ap1_all_pool = tf.reduce_mean(conv1, axis=[1, 2])
                ap2_all_pool = tf.reduce_mean(conv2, axis=[1, 2])

            with tf.variable_scope("%d-ap" % filter_width) as scope:
                avg_pool1 = self.pool_layer(filter_width, conv1, name='avg_pool1')
                avg_pool2 = self.pool_layer(filter_width, conv2, name='avg_pool2')

                ap1_width_pool = tf.transpose(avg_pool1, perm=[0, 3, 2, 1], name='ap1')
                ap2_width_pool = tf.transpose(avg_pool2, perm=[0, 3, 2, 1], name='ap2')
            return ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool


    def _add_ABCNN_2(self, id, filter_num, layer_input,filter_width):
        scope_name = 'ABCNN2_' + str(id)
        with tf.variable_scope(scope_name,tf.contrib.layers.xavier_initializer()) as scope:

            with tf.variable_scope("conv",reuse=tf.AUTO_REUSE) as scope:
                padded_in1 = self.pad_for_wide_conv(layer_input[0],filter_width)
                padded_in2 = self.pad_for_wide_conv(layer_input[1],filter_width)

                conv1 = self.conv_layer(filter_width,filter_num,padded_in1)
                conv2 = self.conv_layer(filter_width,filter_num,padded_in2)

            with tf.variable_scope("similary") as scope:
                conv1t = tf.transpose(conv1, [0, 3, 2, 1], name='conv1t')
                conv2t = tf.transpose(conv2, [0, 3, 2, 1], name='conv2t')
                similarity = self.make_attention_mat(conv1t,conv2t)

            with tf.variable_scope('attention') as scope:
                A1 = tf.reduce_sum(similarity,axis=[2],name='attention_map1')
                A2 = tf.reduce_sum(similarity,axis=[1],name='attention_map2')

                A1e = tf.reshape(A1,[-1,1,A1.get_shape()[1],1])
                A2e = tf.reshape(A2,[-1,1,A2.get_shape()[1],1])

                conv1_w = tf.multiply(conv1t,A1e,name='weighted_conv1')
                conv2_2 = tf.multiply(conv2t,A2e,name='weighted_conv2')



            with tf.variable_scope("all-ap") as scope:
                ap1_all_pool = tf.reduce_mean(conv1_w, axis=[1, 2])
                ap2_all_pool = tf.reduce_mean(conv2_2, axis=[1, 2])

            with tf.variable_scope("%d-ap" % filter_width) as scope:
                ap1_width_pool = self.pool_layer(filter_width, conv1_w, name='avg_pool1')
                ap2_width_pool = self.pool_layer(filter_width, conv2_2, name='avg_pool2')

            return ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool

    def _add_ABCNN_3(self, id, filter_num, layer_input,filter_width):
        scope_name = 'ABCNN3_' + str(id)
        with tf.variable_scope(scope_name,tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope('similary_pre') as scope:
                similarity = self.make_attention_mat(layer_input[0],layer_input[1])
            with tf.variable_scope('attention') as scope:
                W = tf.get_variable("W",[layer_input[0].get_shape()[2],layer_input[1].get_shape()[1]],initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                A1 = tf.matrix_transpose(tf.einsum('ijk,kl->ijl',similarity,W))
                A2 = tf.matrix_transpose(tf.einsum('ijk,kl->ijl',tf.matrix_transpose(similarity),W))




            with tf.variable_scope("conv",reuse=tf.AUTO_REUSE) as scope:
                layer_in1 = tf.concat([layer_input[0],tf.expand_dims(A1,-1)],axis=-1)
                layer_in2 = tf.concat([layer_input[1],tf.expand_dims(A2,-1)],axis=-1)
                padded_in1 = self.pad_for_wide_conv(layer_in1,filter_width)
                padded_in2 = self.pad_for_wide_conv(layer_in2,filter_width)

                conv1 = self.conv_layer(filter_width,filter_num,padded_in1)
                conv2 = self.conv_layer(filter_width,filter_num,padded_in2)

            with tf.variable_scope("similary") as scope:
                conv1t = tf.transpose(conv1, [0, 3, 2, 1], name='conv1t')
                conv2t = tf.transpose(conv2, [0, 3, 2, 1], name='conv2t')
                similarity = self.make_attention_mat(conv1t,conv2t)

            with tf.variable_scope('attention') as scope:
                A1 = tf.reduce_sum(similarity,axis=[2],name='attention_map1')
                A2 = tf.reduce_sum(similarity,axis=[1],name='attention_map2')

                A1e = tf.reshape(A1,[-1,1,A1.get_shape()[1],1])
                A2e = tf.reshape(A2,[-1,1,A2.get_shape()[1],1])

                conv1_w = tf.multiply(conv1t,A1e,name='weighted_conv1')
                conv2_2 = tf.multiply(conv2t,A2e,name='weighted_conv2')

            with tf.variable_scope("all-ap") as scope:
                ap1_all_pool = tf.reduce_mean(conv1_w, axis=[1, 2])
                ap2_all_pool = tf.reduce_mean(conv2_2, axis=[1, 2])

            with tf.variable_scope("%d-ap" % filter_width) as scope:
                ap1_width_pool = self.pool_layer(filter_width, conv1_w, name='avg_pool1')
                ap2_width_pool = self.pool_layer(filter_width, conv2_2, name='avg_pool2')

            return ap1_width_pool, ap1_all_pool, ap2_width_pool, ap2_all_pool

if __name__ == '__main__':
    abcnn = ABCNN(True, 20, 2, 10000, 300, 300, 0.001, 0.0001)