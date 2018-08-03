# create by fanfan on 2017/8/25 0025
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense,array_ops
from tensorflow.python.util import nest
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from seq2seq.seq2seq_dynamic import config


from seq2seq.seq2seq_dynamic import data_utils
import math
import os
from seq2seq.seq2seq_dynamic.encoder import  build_encode
from seq2seq.seq2seq_dynamic.decoder import build_decoder_cell


class Seq2SeqModel(object):
    def __init__(self,config,model):
        self.mode =model
        #初始化一些基本参数
        self._init_config(config)

        #初始化placeholder
        self._init_placeholders()

        #初始化embedding向量
        self._init_embeddings()

        #构建网络
        result = self._build_network()

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.loss = result[1]
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.loss = result[1]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits, _, self.final_context_state, self.sample_id = result
        tf.summary.scalar('loss', result[1])





        self._init_optimizer()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=3)


    def _init_config(self,config):
        self.config = config
        self.cell_type = config.cell_type
        self.hidden_units = config.hidden_units
        self.depth = config.depth
        self.attention_type = config.attention_type
        self.embeddint_size = config.embedding_size

        self.num_encoder_symbols = config.num_encoder_symbols
        self.num_decoder_symbols = config.num_decoder_symbols
        self.use_residual = config.use_residual
        self.attn_input_feeding = config.attn_input_feeding
        self.use_dropout = config.use_dropout
        self.keep_prob = 1.0 - config.dropout_rate

        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate


        self.max_gradient_norm = config.max_gradient_norm
        self.global_step = tf.Variable(0,trainable=False,name='global_step')



        self.dtype =  tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype,shape=[],name='keep_prob')
        self.mode_save_path = os.path.join(config.model_dir,config.model_name)

        self.beam_with = config.beam_with
        self.use_beamsearch_decode = True


    def _init_placeholders(self):
        # encode_inputs:[batch_size,max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None,None),name='encoder_inputs')
        #encode_inputs_lengths [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,shape=(None,),name='encoder_inputs_length')
        # 获取batch_size的大小
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        # decoder_inputs:[batch_size,max_time_steps]
        self.decoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None,None),name='decoder_inputs')
        # decoder_inputs_length: [batch_size]
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32,shape=(None,),name='decoder_inputs_length')

        decoder_start_token = tf.ones(shape=[self.batch_size,1],dtype=tf.int32) * data_utils.GO_ID
        decoder_end_token = tf.ones(shape=[self.batch_size,1],dtype=tf.int32) * data_utils.EOS_ID

        # 在每一句解码的句子前面加一个GO标识，作为训练的时候解码的输入
        self.decoder_inputs_train = tf.concat([decoder_start_token,self.decoder_inputs], axis=1)
        # 句子长度加1
        self.decoder_inputs_length_train = self.decoder_inputs_length + 1
        # 在每一句解码的句子后面加一个END表示，作为训练时解码段的输出
        self.decoder_targets_train = tf.concat([self.decoder_inputs,decoder_end_token], axis=1)



    def _init_embeddings(self):
        with tf.name_scope('word_embedding'):
            #这里不是翻译，所以encode跟decode公用一个embedding
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3,sqrt3,dtype=tf.float32)
            self.embeddings = tf.get_variable(name='embedding_matrix',shape=[self.num_encoder_symbols,self.embeddint_size],dtype=tf.float32,initializer=initializer)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            # 经过一层浓密型网络
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # 输入的句子也经过一层浓密型网络

            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs_train)
            self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)




    def _build_network(self):
        # 编码层网络定义
        self.encoder_outputs, self.encoder_last_state = build_encode(self.encoder_inputs_embedded,self.encoder_inputs_length,self.depth,self.hidden_units,self.keep_prob_placeholder,self.use_residual)

        # 解码层网络定义
        ## Decoder
        logits, sample_id, final_context_state = self.build_decode()

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            loss = self._compute_loss(logits)
        else:
            loss = None
        return  logits,loss,sample_id,final_context_state

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.decoder_targets_train
        max_time = tf.shape(target_output)[1]
        target_weights = tf.sequence_mask(
            self.decoder_inputs_length_train, max_time, dtype=logits.dtype)
        #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    labels=target_output, logits=logits)
        #loss = tf.reduce_sum(
        #    crossent * target_weights) / tf.to_float(self.batch_size)

        loss = seq2seq.sequence_loss(logits=logits,
                                     targets=target_output,
                                     weights=target_weights,
                                     average_across_timesteps=True,
                                     average_across_batch=True)
        return loss




    def build_decode(self):
        # build decoder and attention.
        with tf.variable_scope('decoder'):
            self.decoder_cell,self.decoder_initial_state = build_decoder_cell(self.encoder_outputs,self.encoder_last_state,self.encoder_inputs_length,self.depth,self.hidden_units,self.keep_prob_placeholder,self.use_residual,self.mode,self.beam_with,self.batch_size)
            # Output projection layer to convert cell_outpus to logits
            output_layer = Dense(self.num_decoder_symbols,name='output_project')

            # Maximum decoder time_steps in current batch
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(self.decoder_inputs_length_train)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))

            # train或者eval的时候处理
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_length_train,
                                                         time_major=False,
                                                         name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                        helper=training_helper,
                                                        initial_state=self.decoder_initial_state,
                                                        output_layer=output_layer)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                decoder_outputs_train,final_context_state,_ = seq2seq.dynamic_decode(decoder=training_decoder,output_time_major=False,impute_finished=True,swap_memory=True)
                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                sample_id = decoder_outputs_train.sample_id
                logits = tf.identity(decoder_outputs_train.rnn_output)
            else:
                # Start_tokens: [batch_size,] `int32` vector
                start_token = tf.ones([self.batch_size,],tf.int32) * data_utils.GO_ID
                end_token = data_utils.EOS_ID

                # 由于解码的时候，需要经过一层dense网络，所以定义一个函数转化
                def embed_and_input_proj(inputs):
                    return self.input_layer_out(tf.nn.embedding_lookup(self.embeddings,inputs))

                if not self.use_beamsearch_decode:
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_token,end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    inference_decoder = seq2seq.BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_token,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_with,
                        output_layer=output_layer
                    )

                    # For GreedyDecoder, return
                    # decoder_outputs_decode: BasicDecoderOutput instance
                    #                         namedtuple(rnn_outputs, sample_id)
                    # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                    #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                    #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True

                    # For BeamSearchDecoder, return
                    # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                    #                         namedtuple(predicted_ids, beam_search_decoder_output)
                    # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                    #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                    # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                    #                                                    namedtuple(scores, predicted_ids, parent_ids)

                decoder_outputs_decode,final_context_state,_ = seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    maximum_iterations=maximum_iterations)

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    #self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)
                    logits = decoder_outputs_decode.rnn_output
                    sample_id = decoder_outputs_decode.sample_id
                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    sample_id = decoder_outputs_decode.predicted_ids
                    logits = tf.no_op()
            return  logits,sample_id,final_context_state




    def _init_optimizer(self):
        with tf.name_scope("optimizer"):
            train_params = tf.trainable_variables()
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
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
        feed_dict[self.encoder_inputs.name] = encoder_inputs
        feed_dict[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            feed_dict[self.decoder_inputs.name] = decoder_inputs
            feed_dict[self.decoder_inputs_length.name] = decoder_inputs_length

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
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          decoder_inputs,decoder_inputs_length,False)
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,self.loss,self.summary_op]
        _,loss,summary = sess.run(output_feed,input_feed)
        return loss,summary

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
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.loss,self.summary_op]
        loss,summary_op = sess.run(output_feed,input_feed)
        return  loss,summary_op


    #根据输入句子进行预测
    def predict(self,sess,encoder_inputs,encoder_inputs_length,vocab_list):
        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          None,None,True)
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = self.sample_id
        for key in input_feed.keys():
            print(key)
        print(self.sample_id.name)
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

    # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
    # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
    def predict_beam_search(self, sess, encoder_inputs, encoder_inputs_length, vocab_list):
        input_feed = self.make_feeds_dict(encoder_inputs, encoder_inputs_length,
                                          None, None, True)
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = self.sample_id
        predicts = sess.run(output_feed, input_feed)

        outputs = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        print(predicts.shape)
        result = []
        print(predicts[0])
        for k in range(self.beam_with):
            result.append(self.seq2words(predicts[0][:,k],vocab_list))
        return result


    def seq2words(self,seq, inverse_target_dictionary):
        words = []
        for w in seq:
            if w == data_utils.EOS_ID or w == data_utils.PAD_ID:
                break
            words.append(inverse_target_dictionary[w])
        return ' '.join(words)


    #模型恢复或者初始化
    def model_restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.mode_save_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("restor model")
            self.saver.restore(sess,ckpt.model_checkpoint_path)

        else:
            print("init model")
            sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    from seq2seq.seq2seq_dynamic import config
    Seq2SeqModel(config,tf.contrib.learn.ModeKeys.TRAIN)

