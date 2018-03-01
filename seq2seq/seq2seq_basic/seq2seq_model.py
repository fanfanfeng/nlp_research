# create by fanfan on 2018/2/23 0023
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq

from seq2seq.seq2seq_basic import config
from seq2seq.seq2seq_basic import data_utils
import math
import os

class Seq2seqModel():
    def __init__(self):
        self.learning_rate = config.learning_rate
        self.encoding_embedding_size = config.embedding_size
        self.decoding_embedding_size = config.embedding_size
        self.num_encoder_symbols =  config.num_encoder_symbols
        self.num_decoder_symbols = config.num_decoder_symbols
        self.hidden_units = config.hidden_units
        self.depth = config.depth
        self.model_save_path = os.path.join(config.model_dir,config.model_name)
        self.max_gradient_norm = config.max_gradient_norm

        # encoder sentenct placeholder
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        self.batch_size = tf.shape(self.inputs)[0]

        # decoder sentence placeholder
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.GO_ID
        decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.EOS_ID

        self.decoder_train = tf.concat([decoder_start_token,self.targets],axis=1)
        self.target_sequence_length_train = self.target_sequence_length + 1
        self.targets_train= tf.concat([self.targets,decoder_end_token],axis=1)

        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length_train, name='max_target_len')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

    def build_network(self):
        encoder_output, encoder_state = self.encoder_layer()
        training_decoder_output, predicting_decoder_output = self.decoding_layer(encoder_state)
        self.decoder_predict_ids = tf.expand_dims(predicting_decoder_output.sample_id, -1)
        self.loss = self.loss_layer(training_decoder_output)
        #return loss,decoder_predict_ids


    # 在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
    def encoder_layer(self):
        '''
       构造Encoder层

       参数说明：
       - input_data: 输入tensor
       - rnn_size: rnn隐层结点数量
       - num_layers: 堆叠的rnn cell数量
       - source_sequence_length: 源数据的序列长度
       - source_vocab_size: 源数据的词典大小
       - encoding_embedding_size: embedding的大小
       '''
        # Encoder embedding
        with tf.variable_scope('encoder_layer'):
            #这里不是翻译，所以encode跟decode公用一个embedding
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3,sqrt3,dtype=tf.float32)
            encode_embeddings = tf.get_variable(name='embedding_matrix',shape=[self.num_encoder_symbols,self.encoding_embedding_size],dtype=tf.float32,initializer=initializer)
            encoder_embed_input = tf.nn.embedding_lookup(encode_embeddings,self.inputs)

        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer)
            return lstm_cell

        cell = rnn.MultiRNNCell([get_lstm_cell(self.hidden_units) for _ in range(self.depth)])
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,sequence_length=self.source_sequence_length, dtype=tf.float32)
        return encoder_output, encoder_state

    def decoding_layer(self,encoder_state):
        '''
        构造Decoder层

        参数：
        - num_decoder_symbols: target数据的映射表
        - decoding_embedding_size: embed向量大小
        - num_layers: 堆叠的RNN单元数量
        - rnn_size: RNN单元的隐层结点数量
        - target_sequence_length: target数据序列长度
        - max_target_sequence_length: target数据序列最大长度
        - encoder_state: encoder端编码的状态向量
        - decoder_input: decoder端输入
        '''
        with tf.variable_scope('decoder_layer'):
            # 1. Embedding
            # Encoder embedding
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            decoder_embeddings = tf.get_variable(name='embedding_matrix',
                                                shape=[self.num_decoder_symbols, self.decoding_embedding_size],
                                                dtype=tf.float32, initializer=initializer)
            decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, self.decoder_train)

            # 2 构造Decoder中的RNN单元
            def get_decoder_cell(rnn_size):
                decoder_cell = rnn.LSTMCell(rnn_size, initializer=tf.random_normal_initializer)
                return decoder_cell

            cell = rnn.MultiRNNCell([get_decoder_cell(self.hidden_units) for _ in range(self.depth)])

            # 3. Output全连接层
            output_layer = Dense(self.num_decoder_symbols, kernel_initializer=tf.truncated_normal_initializer)

            # 4 Trainning decoder
            # 得到help对象
            training_helper = seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                     sequence_length=self.target_sequence_length_train,
                                                     time_major=False)
            # 构造decoder
            training_decoder = seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
            (training_decoder_output, _, _) = seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                     maximum_iterations=self.max_target_sequence_length)

        # 5. Predicting decoder
        # 和training 共享参数
        with tf.variable_scope('decoder_layer', reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([data_utils.GO_ID], dtype=tf.int32), [self.batch_size])
            predicting_helper = seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,data_utils.EOS_ID)
            predicting_decoder = seq2seq.BasicDecoder(cell, predicting_helper, encoder_state, output_layer)
            (predicting_decoder_output, _, _) = seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,maximum_iterations=self.max_target_sequence_length)
        return training_decoder_output, predicting_decoder_output

    def loss_layer(self,training_decoder_output):
        with tf.variable_scope("loss_layer"):
            # More efficient to do the projection on the batch-time-concatenated tensor
            # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
            # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
            decoder_logits_train = tf.identity(training_decoder_output.rnn_output,name='logits')


            # Use argmax to extract decoder symbols to emit
            decoder_pred_train = tf.argmax(decoder_logits_train, axis=-1,name='decoder_pre_train')

            # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
            masks = tf.sequence_mask(lengths=self.target_sequence_length_train,
                                     maxlen=self.max_target_sequence_length,
                                     dtype=tf.float32,
                                     name='masks')

            loss = seq2seq.sequence_loss(logits=decoder_logits_train,
                                              targets=self.targets_train,
                                              weights=masks,
                                              average_across_timesteps=True,
                                              average_across_batch=True)
        return loss

        # 模型恢复或者初始化
    def model_restore(self, sess,tf_saver):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.model_save_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("restor model")
            tf_saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model")
            sess.run(tf.global_variables_initializer())

    def _init_optimizer(self):
        with tf.name_scope("optimizer"):
            train_params = tf.trainable_variables()
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.loss,train_params)

            # Clip_gradients by a given maximum_gradient_norm
            clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)

            # Update the model
            self.updates = self.opt.apply_gradients(zip(clip_gradients,train_params),global_step=self.global_step)


    #创建 feed_dict
    def make_feeds_dict(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("encode inputs 和它的长度不一致,%d != %d" % (input_batch_size,encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("encode inputs 和 decode inputs 的长度不一致,%d != %d" % (input_batch_size,target_batch_size))

            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets 和 它的长度不一致%d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))


        feed_dict = {}
        feed_dict[self.inputs.name] = encoder_inputs
        feed_dict[self.source_sequence_length.name] = encoder_inputs_length

        if not decode:
            feed_dict[self.targets.name] = decoder_inputs
            feed_dict[self.target_sequence_length.name] = decoder_inputs_length

        return feed_dict

    #训练函数
    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """

        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          decoder_inputs,decoder_inputs_length,False)
        #input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,self.loss]
        _,loss = sess.run(output_feed,input_feed)
        return loss

    # 训练时，验证模型效果
    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          decoder_inputs,decoder_inputs_length,False)
        #input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.loss]
        loss = sess.run(output_feed,input_feed)
        return  loss


    #根据输入句子进行预测
    def predict(self,sess,encoder_inputs,encoder_inputs_length,vocab_list):
        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          None,None,True)
        #input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = self.decoder_predict_ids
        #for key in input_feed.keys():
        #    print(key)
        print(self.decoder_predict_ids.name)
        predicts = sess.run(output_feed,input_feed)

        outputs = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        print(predicts.shape)
        for token in predicts[0]:
            selected_token_id = int(token)
            if selected_token_id == data_utils.EOS_ID or selected_token_id == data_utils.PAD_ID:
                break
            else:
                outputs.append(selected_token_id)

        # Forming output sentence on natural language
        output_sentence = " ".join([vocab_list[output] for output in outputs])
        return output_sentence




if __name__ == '__main__':
    Seq2seqModel()
