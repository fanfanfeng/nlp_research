# create by fanfan on 2017/11/6 0006
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper,ResidualWrapper
from tensorflow.python.layers.core import  Dense,array_ops
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper,beam_search_decoder

from seq2seq.seq2seq_beamsearch import data_utils,config
import math,os

class Seq2SeqModel(object):
    def __init__(self,config,mode):
        assert mode.lower() in ['train','decode']
        self.mode = mode.lower()
        self.config = config

        # 初始化一些基本参数
        self._init_config()

        # 初始化placeholder
        self._init_placeholders()

        # 构建网络
        self._build_network()

        if self.mode.lower() == "train":
            self._init_optimizer()


        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=3)

    def _init_config(self):
        self.cell_type = self.config.cell_type
        self.hidden_units = self.config.hidden_units
        self.depth = self.config.depth
        self.attention_type = self.config.attention_type
        self.embedding_size = self.config.embedding_size

        self.num_encoder_symbols = self.config.num_encoder_symbols
        self.num_decoder_symbols = self.config.num_decoder_symbols

        self.use_residual = self.config.use_residual
        self.attn_input_feeding = self.config.attn_input_feeding
        self.use_dropout = self.config.use_dropout
        self.keep_prob = 1.0 - self.config.dropout_rate

        self.optimizer = self.config.optimizer
        self.learning_rate = self.config.learning_rate
        self.max_gradient_norm = self.config.max_gradient_norm

        self.global_step = tf.Variable(0,trainable=False,name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.dtype = tf.float16 if self.config.use_fp16 else tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype,shape=[],name='keep_prob')

        self.mode_save_path = os.path.join(config.model_dir, config.model_name)
        self.beam_with = self.config.beam_with
        if  self.config.beam_with >1 and self.mode == "decode":
            self.use_beamsearch_decode = True
        else:
            self.use_beamsearch_decode = False


    def _init_placeholders(self):
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None,None),name='encoder_inputs')

        # encoder_inputs_length:[batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,shape=(None,),name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        # decoder_inputs: [batch_size, max_time_steps]
        self.decoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None,None),name='decoder_inputs')

        # decoder_inputs_length: [batch_size]
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32,shape=(None,),name='decoder_inputs_length')

        decoder_start_token = tf.ones(shape=[self.batch_size,1],dtype=tf.int32) * data_utils.GO_ID
        decoder_end_token = tf.ones(shape=[self.batch_size,1],dtype=tf.int32) * data_utils.PAD_ID

        # decoder_inputs_train: [batch_size , max_time_steps + 1]
        # insert _GO symbol in front of each decoder input
        self.decoder_inputs_train = tf.concat([decoder_start_token,self.decoder_inputs],axis=1)

        # decoder_inputs_length_train: [batch_size]
        self.decoder_inputs_length_train = self.decoder_inputs_length + 1

        # decoder_targets_train: [batch_size, max_time_steps + 1]
        # insert EOS symbol at the end of each decoder input
        self.decoder_target_train = tf.concat([self.decoder_inputs,decoder_end_token],axis=1)

    def _build_network(self):
        self.build_encoder()
        self.build_decoder()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()

    def build_encoder(self):
        print("building encoder layer...")
        with tf.variable_scope("encoder_layer"):
            # Building encoder_cell
            self.encoder_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3,sqrt3,dtype=self.dtype)

            self.encoder_embeddings = tf.get_variable(name='encoder_embedding',shape=[self.num_encoder_symbols,self.embedding_size],initializer=initializer,dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings,ids=self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units,dtype=self.dtype,name='input_project')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            self.encoder_outputs,self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=self.dtype
            )

    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_input_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode :
            encoder_outputs = seq2seq.tile_batch(self.encoder_outputs,multiplier=self.beam_with)
            encoder_last_state = seq2seq.tile_batch(self.encoder_last_state,multiplier=self.beam_with)#nest.map_structure(lambda s:seq2seq.tile_batch(s,multiplier=self.beam_with),self.encoder_last_state)
            encoder_input_length = seq2seq.tile_batch(self.encoder_inputs_length,multiplier=self.beam_with)


        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.hidden_units,
            memory=encoder_outputs,
            memory_sequence_length=encoder_input_length
        )

        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.hidden_units,
                memory = encoder_outputs,
                memory_sequence_length=encoder_input_length
            )

        # Building decoder_cell
        self.decoder_cell_list = [self.build_single_cell() for i in range(self.depth)]

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            cell_input_fn=attn_decoder_input_fn,
            name='Attention_Wrapper'
        )

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode else self.batch_size * self.beam_with
        initial_state = [state for state in encoder_last_state]
        initial_state[-1] = self.decoder_cell_list[-1].zero_state(batch_size,dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list),decoder_initial_state

    #building decoder layer and attention layer
    def build_decoder(self):
        with tf.variable_scope('decoder_layer'):
            self.decoder_cell,self.decoder_initial_state = self.build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.decoder_embeddings = tf.get_variable(name='decoder_embedding',
                                                      shape=[self.num_decoder_symbols, self.embedding_size],
                                                      initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units,dtype=self.dtype,name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols,name='output_projection')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings,ids=self.decoder_inputs_train)

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                trainning_helper = seq2seq.TrainingHelper(
                    inputs=self.decoder_inputs_embedded,
                    sequence_length=self.decoder_inputs_length_train,
                    time_major=False,
                    name="training_helper"
                )

                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=trainning_helper,
                    initial_state=self.decoder_initial_state,
                    output_layer=output_layer
                )
                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train,self.decoder_last_state_train,self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length
                ))

                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)

                self.decoder_pred_train = tf.argmax(self.decoder_logits_train,axis=1,name='decoder_pre_train')
                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')
                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                  targets=self.decoder_target_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
            elif self.mode == 'decode':
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * data_utils.GO_ID
                end_token = data_utils.EOS_ID

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings,inputs))

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    inference_decoder = seq2seq.BasicDecoder(
                        cell= self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=output_layer
                    )
                else:
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
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
                    #
                #
                (self.decoder_outputs_decode,self.decoder_last_state_decode,self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False
                ))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id,-1)
                else:
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def build_single_cell(self):
        cell_type = LSTMCell
        if self.cell_type.lower() == 'gru':
            cell_type = GRUCell

        cell = cell_type(self.hidden_units)
        if self.use_dropout:
            cell = DropoutWrapper(cell,dtype=self.dtype,output_keep_prob=self.keep_prob_placeholder)

        if self.use_residual:
            cell = ResidualWrapper(cell)
        return cell

    # Building encoder cell
    def build_encoder_cell(self):
        return MultiRNNCell([self.build_single_cell() for i in range(self.depth)])


    def _init_optimizer(self):
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss,trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)

        self.updates = self.opt.apply_gradients(
            zip(clip_gradients,trainable_params),global_step=self.global_step
        )

    def save(self,sess,path,var_list=None,global_step = None):
        saver = tf.train.Saver(var_list=var_list)

        save_path = saver.save(sess,save_path=path,global_step=global_step)
        print("model save at %s " % save_path)

    def restore(self,sess,path,var_list):
        saver = tf.train.Saver(var_list)
        saver.restore(sess,save_path=path)
        print("model restored from %s" % path)

    def train(self,sess,encoder_inputs,encoder_inputs_length,decoder_inputs,decoder_inputs_length):
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
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train model")
        input_feed = self.check_feeds(encoder_inputs,encoder_inputs_length,
                                      decoder_inputs,decoder_inputs_length,False)
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob
        output_feed = [self.updates,self.loss,self.summary_op]

        outputs = sess.run(output_feed,input_feed)
        return outputs[1],outputs[2]

    def eval(self,sess,encoder_inputs,encoder_inputs_length,decoder_inputs,decoder_inputs_length):
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
        input_feed = self.check_feeds(encoder_inputs,encoder_inputs_length,decoder_inputs,decoder_inputs_length,False)

        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.loss,self.summary_op]
        outputs = sess.run(output_feed,input_feed)
        return output_feed[0],outputs[1]

    def predict(self,sess,encoder_inputs,encoder_inputs_length):
        input_feed = self.check_feeds(encoder_inputs,encoder_inputs_length,decoder_inputs=None,decoder_inputs_length=None,decode=True)
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed,input_feed)

        # GreedyDecoder: [batch_size, max_time_step]
        # BeamSearchDecoder: [batch_size, max_time_step, beam_width]
        return outputs[0]


    def check_feeds(self,encoder_inputs,encoder_inputs_length,decoder_inputs,decoder_inputs_length,decode):
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
            raise ValueError("Encoder inputs and their lengths must be equal! batch_size,%d != %d" % (input_batch_size,encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal!batch_size,%d != %d" % (input_batch_size,target_batch_size))

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed

    # 模型恢复或者初始化
    def model_restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.mode_save_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("restor model")
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        else:
            print("init model")
            sess.run(tf.global_variables_initializer())






if __name__ == '__main__':
    a = Seq2SeqModel(config=config,mode="train")
