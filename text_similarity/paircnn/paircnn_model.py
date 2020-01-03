# create by fanfan on 2019/12/20 0020
import tensorflow as tf

filter_sizes = [2,3,4]
num_filters = 100
k=2

class PairCNN(object):
    def create_placeholders(self,seq_length,class_num,hidden_num):
        self.text_a = tf.placeholder(tf.int32, [None, seq_length],name='premise')
        self.text_b = tf.placeholder(tf.int32, [None, seq_length],name='hypothesis')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.a_length = tf.placeholder(tf.int32, [None], name='a_length')
        self.b_length = tf.placeholder(tf.int32, [None], name='b_length')
        self.y = tf.placeholder(tf.int32, [None, class_num], name='y')


    def cnn_k_pool(self,input_embbed,embedding_size):
        pooled_output = []
        with tf.variable_scope("cnn_filiter",reuse=tf.AUTO_REUSE) :
            for i,filter_size in enumerate(filter_sizes):
                with tf.variable_scope('conv-maxpool-%s' % filter_size):
                    conv = tf.layers.conv2d(input_embbed,num_filters,[filter_size,embedding_size],padding='VALID',name='conv')
                    h = tf.nn.relu(conv,name='relu')

                    # k_max_pooling over the outputs
                    k_max_pooling = tf.nn.top_k(tf.transpose(h,[0,3,2,1]),k=k,sorted=True)[0]
                    k_max_pooling = tf.reshape(k_max_pooling,[-1,k*num_filters])
                    pooled_output.append(k_max_pooling)

        return pooled_output


    def __init__(self,is_training,seq_length,class_num,vocabulary_size,embedding_size,hidden_num,l2_lambda,learning_rate):
        self.create_placeholders(seq_length,class_num,hidden_num)

        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_text_a = tf.expand_dims(tf.nn.embedding_lookup(W, self.text_a), -1)
            self.embedded_text_b = tf.expand_dims(tf.nn.embedding_lookup(W, self.text_b), -1)

        # Create a convolution + maxpool layer for each filter size
        with tf.variable_scope('cnn_pool'):
            pooled_outputs_a = self.cnn_k_pool(self.embedded_text_a,embedding_size)
            pooled_outputs_b = self.cnn_k_pool(self.embedded_text_b,embedding_size)


        with tf.name_scope('similarity'):
            num_filters_total = num_filters * len(filter_sizes) * k
            h_pool_a = tf.reshape(tf.concat(pooled_outputs_a, 1), [-1, num_filters_total],
                                          name='h_pool_left')
            h_pool_b = tf.reshape(tf.concat(pooled_outputs_b, 1), [-1, num_filters_total],
                                           name='h_pool_right')
            W = tf.get_variable('W',shape=[num_filters_total,num_filters_total],initializer=tf.contrib.layers.xavier_initializer())
            transorm_left = tf.matmul(h_pool_a,W)
            similaritys = tf.reduce_sum(tf.multiply(transorm_left,h_pool_b),1,keepdims=True)


        with tf.name_scope('output'):
            all_feature = tf.concat([h_pool_a,similaritys,h_pool_b],axis=1,name='all_feature')
            hidden_output = tf.layers.dense(all_feature,hidden_num,activation=tf.nn.relu,name='hidden_output')
            hidden_output = tf.nn.dropout(hidden_output,self.dropout_keep_prob)
            self.score = tf.layers.dense(hidden_output,class_num)
            self.prediction = tf.argmax(self.score, 1, name="predictions")
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score,labels=self.y)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

if __name__ == '__main__':
    paircnn = PairCNN(True, 20, 2, 10000, 300, 300, 0.001, 0.0001)







