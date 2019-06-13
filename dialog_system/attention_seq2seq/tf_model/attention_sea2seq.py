# create by fanfan on 2019/6/11 0011
import math
import os
import tensorflow as tf
from dialog_system.attention_seq2seq.tf_model.encoder import build_bilstm_encode
from dialog_system.attention_seq2seq.tf_model.decoder import build_decoder_cell
from dialog_system.attention_seq2seq import data_utils
from tensorflow.contrib import  seq2seq
from tensorflow.python.layers.core import Dense

class AttentionSeq():
    def __init__(self,params):
        self.params = params


    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def _create_embedding(self):
        with tf.variable_scope('word_embedding'):
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3,sqrt3,dtype=tf.float32)
            embeddings_matrix = tf.get_variable(
                name='embedding_matrix',
                shape=[self.params.vocab_size,self.params.embedding_size],
                dtype=tf.float32,
                initializer=initializer)
        return embeddings_matrix

    def _create_encoder(self,embeddings_matrix,input,keep_prob):
        with tf.variable_scope('encoder'):
            encoder_inputs_length = self.get_setence_length(input)
            encoder_inputs_embeded = tf.nn.embedding_lookup(embeddings_matrix,input)
            # Input projection layer to feed embedded inputs to the cell
            encoder_inputs_embeded = tf.layers.dense(encoder_inputs_embeded,self.params.hidden_units)
            encoder_outputs,encoder_last_states = build_bilstm_encode(encoder_inputs_embeded,
                                                                      encoder_inputs_length,
                                                                      self.params.depth,
                                                                      self.params.hidden_units,
                                                                      keep_prob,
                                                                      self.params.use_residual)
        return encoder_outputs, encoder_last_states,encoder_inputs_length


    def create_model(self,input,target_input,target_output,mode='train'):
        with tf.variable_scope("attetnion_seq2seq",reuse=tf.AUTO_REUSE):
            embeddings_matrix = self._create_embedding()

            keep_prob = 1 - self.params.dropout_rate
            batch_size = tf.shape(input)[0]
            # encoder
            encoder_outputs,encoder_last_states,encoder_inputs_length = self._create_encoder(embeddings_matrix,input,keep_prob)

            # decoder
            with tf.variable_scope('decoder'):
                # Output projection layer to convert cell_outpus to logits
                output_layer = Dense(self.params.vocab_size, name='output_project')
                input_layer = Dense(self.params.hidden_units * 2, dtype=tf.float32, name='input_projection')
                decoder_cell,decoder_initial_state = build_decoder_cell(encoder_outputs,
                                                                        encoder_last_states,
                                                                        encoder_inputs_length,
                                                                        self.params.depth,
                                                                        self.params.attention_size,
                                                                        self.params.hidden_units,
                                                                        keep_prob,
                                                                        self.params.use_residual,
                                                                        mode,
                                                                        self.params.beam_with,
                                                                        batch_size)

                decoder_inputs_length_train = self.get_setence_length(target_input)
                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(decoder_inputs_length_train)

                target_output = tf.slice(target_output, [0, 0], [batch_size, max_decoder_length])
                target_input = tf.slice(target_input, [0, 0], [batch_size, max_decoder_length])
                decoder_input_embedded = tf.nn.embedding_lookup(embeddings_matrix,target_input)
                decoder_input_embedded = input_layer(decoder_input_embedded)
                inference_decoder = seq2seq.TrainingHelper(inputs=decoder_input_embedded,
                                                         sequence_length=decoder_inputs_length_train,
                                                         #sequence_length= self.params.max_seq_length,
                                                         time_major=False,
                                                         name='training_helper')
                training_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                        helper=inference_decoder,
                                                        initial_state=decoder_initial_state,
                                                        output_layer=output_layer)
                decoder_output,_,_ = seq2seq.dynamic_decode(decoder=training_decoder,
                                                            output_time_major=False,
                                                            impute_finished=False,
                                                            maximum_iterations=max_decoder_length)

                logits = tf.identity(decoder_output.rnn_output)
                predicts = tf.argmax(logits,axis=-1)

                masks = tf.sequence_mask(lengths=decoder_inputs_length_train,
                                         maxlen=max_decoder_length,
                                         dtype=tf.float32,
                                         name='maks')
                loss = seq2seq.sequence_loss(logits=logits,
                                             targets= target_output,
                                             weights=masks,
                                             average_across_batch=True,
                                             average_across_timesteps=True)
                tf.summary.scalar("loss",loss)
        return loss,predicts

    def create_model_predict(self, input,mode='decode'):
        with tf.variable_scope("attetnion_seq2seq", reuse=tf.AUTO_REUSE):
            embeddings_matrix = self._create_embedding()

            keep_prob = 1 - self.params.dropout_rate
            batch_size = tf.shape(input)[0]
            # encoder
            encoder_outputs, encoder_last_states, encoder_inputs_length = self._create_encoder(embeddings_matrix, input,
                                                                                               keep_prob)

            # decoder
            with tf.variable_scope('decoder'):
                # # Output projection layer to convert cell_outpus to logits
                output_layer = Dense(self.params.vocab_size, name='output_project')
                input_layer = Dense(self.params.hidden_units * 2, dtype=tf.float32, name='input_projection')
                decoder_cell,decoder_initial_state = build_decoder_cell(encoder_outputs,
                                                                        encoder_last_states,
                                                                        encoder_inputs_length,
                                                                        self.params.depth,
                                                                        self.params.attention_size,
                                                                        self.params.hidden_units,
                                                                        keep_prob,
                                                                        self.params.use_residual,
                                                                        mode,
                                                                        self.params.beam_with,
                                                                        batch_size)

                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([batch_size,],tf.int32) * data_utils.GO_ID
                end_token = data_utils.EOS_ID

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(embeddings_matrix,inputs))

                if self.params.beam_with <= 1:
                    decode_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                  end_token=end_token,
                                                                  embedding=embed_and_input_proj)
                    inference_decoder = seq2seq.BasicDecoder(cell= decoder_cell,
                                                             helper=decode_helper,
                                                             initial_state=decoder_initial_state,
                                                             output_layer=output_layer)
                    decoder_output, _, _ = seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                  output_time_major=False,
                                                                  impute_finished=True,
                                                                  maximum_iterations=self.params.max_seq_length)
                else:
                    inference_decoder = seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=self.params.beam_with,
                        output_layer=output_layer
                    )

                decoder_output,_,_ = seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    maximum_iterations=self.params.max_seq_length
                )

                if self.params.beam_with <= 1:
                    decoder_predict = tf.expand_dims(decoder_output.sample_id,-1)
                else:
                    decoder_predict = decoder_output.predicted_ids

        decoder_predict = tf.identity(decoder_predict,'predicts')
        return decoder_predict

    #模型恢复或者初始化
    def model_restore(self,sess,tf_saver):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.params.model_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("restor model")
            tf_saver.restore(sess,ckpt.model_checkpoint_path)

        else:
            print("init model")
            sess.run(tf.global_variables_initializer())


    def _init_optimizer(self,loss):
        with tf.name_scope("optimizer"):
            train_params = tf.trainable_variables()
            global_step = tf.train.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(loss,train_params)

            # Clip_gradients by a given maximum_gradient_norm
            clip_gradients,_ = tf.clip_by_global_norm(gradients,self.params.max_gradient_norm)

            # Update the model
            train_op = opt.apply_gradients(zip(clip_gradients,train_params),global_step=global_step)
        return train_op,global_step

    def make_train(self,input,target_input,target_output):
        loss,_ = self.create_model(input,target_input,target_output)
        train_op,global_step = self._init_optimizer(loss)
        summary_op = tf.summary.merge_all()
        return loss,global_step,summary_op,train_op







