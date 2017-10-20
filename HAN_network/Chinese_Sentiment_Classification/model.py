# create by fanfan on 2017/10/18 0018
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from HAN_network.Chinese_Sentiment_Classification import  settings

class Model(object):
    def __init__(self):
        # 初始化一些基本参数
        self._init_config()

        # 初始化placeholder
        self._init_placeholders()

        # 初始化embedding向量
        self._init_embeddings()

        # 构建网络
        self._build_network()

        # 计算loss
        self._build_loss()


    def _init_config(self):
        self.vocab_size = settings.vocab_size
        self.rnn_size = settings.rnn_size
        self.max_doc_len = settings.max_doc_len
        self.max_sentence_len = settings.max_sentence_len
        self.word_attention_size = settings.word_attention_size
        self.sent_attention_size = settings.sent_attention_size
        self.char_embedding_size = settings.char_embedding_size
        self.keep_prob = settings.keep_prob
        self.l2_reg = 1e-4
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.grad_clip = settings.grad_clip
        self.learning_rate = settings.learning_rate
        self.model_save_path = settings.model_save_dir
        self.batch_size = settings.batch_size

    def _init_placeholders(self):
        with tf.variable_scope("placeholders") :
            self.inputs = tf.placeholder(shape=(self.batch_size, self.max_doc_len, self.max_sentence_len), dtype=tf.int64,name='inputs')
            self.labels = tf.placeholder(shape=(self.batch_size,), dtype=tf.int64,name='labels')
            self.sentence_lengths = tf.placeholder(shape=(self.batch_size,self.max_doc_len), dtype=tf.int64, name='sentence_lengths')
            self.document_lengths = tf.placeholder(shape=(self.batch_size,),dtype=tf.int64,name='document_lengths')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            #self.batch_size = self.inputs.get_shape()[0]

    def _init_embeddings(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding',[self.vocab_size,self.char_embedding_size],initializer=tf.truncated_normal_initializer())

    def _build_network(self):
        inputs_embed = tf.nn.embedding_lookup(self.embedding,self.inputs)
        char_length = tf.reshape(self.sentence_lengths,[-1])
        char_inputs = tf.reshape(inputs_embed,[self.batch_size*self.max_doc_len,self.max_sentence_len,self.char_embedding_size])

        with tf.variable_scope("char_encoder") as scope:
            char_outputs = self.bi_gru_encoder(char_inputs,char_length,scope)

            with tf.variable_scope('attention') as scope:
                char_attn_outputs = self.attention_layer(char_outputs,self.word_attention_size,scope)
                char_attn_outputs = tf.reshape(char_attn_outputs,[self.batch_size,self.max_doc_len,-1])
            with tf.variable_scope('drouput'):
                char_attn_outputs = layers.dropout(char_attn_outputs,keep_prob=self.keep_prob,is_training=self.is_training)

        with tf.variable_scope("sentence_encoder") as scope:
            sent_outputs = self.bi_gru_encoder(char_attn_outputs,self.document_lengths,scope)
            with tf.variable_scope('attention') as scope:
                sent_attn_outputs = self.attention_layer(sent_outputs,self.sent_attention_size,scope)
            with tf.variable_scope('dropout'):
                self.sent_attn_outputs = layers.dropout(sent_attn_outputs,
                                                   keep_prob=self.keep_prob,
                                                   is_training=self.is_training)

    def _build_loss(self):
        with tf.variable_scope('losses'):
            logits = layers.fully_connected(inputs=self.sent_attn_outputs,
                                            num_outputs=2,
                                            activation_fn=None,
                                            weights_regularizer=layers.l2_regularizer(self.l2_reg))
            pred = tf.argmax(logits,1)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                 labels=self.labels))
            tf.summary.scalar("loss",self.loss)

            correct_pred = tf.equal(self.labels,pred)
            correct_pred = tf.cast(correct_pred,tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)
            tf.summary.scalar("accuracy", self.accuracy)

            tvars = tf.trainable_variables()
            grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),self.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.training_op = optimizer.apply_gradients(zip(grads,tvars),global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()





    def bi_gru_encoder(self,inputs,sentence_length,scope = None):
        batch_size = inputs.get_shape()[0]
        with tf.variable_scope(scope or 'bi_gru_encode'):
            fw_cell = rnn.GRUCell(self.rnn_size)
            bw_cell = rnn.GRUCell(self.rnn_size)

            fw_cell_state = fw_cell.zero_state(batch_size=batch_size,dtype=tf.float32)
            bw_cell_state = bw_cell.zero_state(batch_size=batch_size,dtype=tf.float32)

            enc_out,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                        cell_bw=bw_cell,
                                                        inputs=inputs,
                                                        sequence_length=sentence_length,
                                                        initial_state_fw=fw_cell_state,
                                                        initial_state_bw=bw_cell_state)
            enc_outputs = tf.concat(enc_out,2)
        return enc_outputs

    def attention_layer(self,inputs,size,scope):
        with tf.variable_scope(scope or 'attention_layer') as scope:
            attention_context_vector = tf.get_variable(name='attention_contenxt_vector',
                                                       shape=[size],
                                                       regularizer=layers.l2_regularizer(self.l2_reg),
                                                       dtype=tf.float32)
            input_projection = layers.fully_connected(inputs,size,
                                                      activation_fn=tf.tanh,
                                                      weights_regularizer=layers.l2_regularizer(self.l2_reg))
            vector_attn = tf.reduce_sum(tf.multiply(input_projection,attention_context_vector),axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn,dim=1)
            weightd_project = tf.multiply(inputs,attention_weights)
            outputs = tf.reduce_sum(weightd_project,axis=1)
        return outputs

    def restore_model(self,sess):
        checkpoint = tf.train.get_checkpoint_state(self.model_save_path)
        if checkpoint:
            print("restore from folder:%s" % self.model_save_path)
            self.saver.restore(sess,checkpoint.model_checkpoint_path)
        else:
            print("init model")
            sess.run(tf.global_variables_initializer())





