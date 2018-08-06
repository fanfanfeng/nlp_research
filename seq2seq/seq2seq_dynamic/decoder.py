# create by fanfan on 2018/7/20 0020
import tensorflow as tf
from tensorflow.contrib import rnn
from seq2seq.seq2seq_dynamic.encoder import build_cell_list
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense,array_ops



def build_decoder_cell(encoder_outputs,encoder_last_state,encoder_inputs_length,layer_num,hidden_units,keep_prob,use_residual,mode,beam_width,batch_size):
    def attn_decoder_input_fn(inputs, attention):

        # Essential when use_residual=True
        _input_layer = Dense(hidden_units * 2, dtype=tf.float32, name='attn_input_feeding')
        return _input_layer(array_ops.concat([inputs, attention], -1))

    memory = encoder_outputs
    memory_sequence_length = encoder_inputs_length
    with tf.variable_scope('decoder'):
        if mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
            memory = seq2seq.tile_batch(memory,multiplier=beam_width)
            memory_sequence_length = seq2seq.tile_batch(memory_sequence_length,multiplier=beam_width)
            encoder_last_state = seq2seq.tile_batch(encoder_last_state,multiplier=beam_width)
            batch_size_real = batch_size * beam_width
        else:
            batch_size_real = batch_size

        attention_mechanism = seq2seq.BahdanauAttention(num_units=hidden_units,
                                                        memory=memory,
                                                        memory_sequence_length=memory_sequence_length)

        decoder_layer = int(layer_num/2)
        decoder_layer_residual = decoder_layer- 1
        rnn_decoder_cell = build_cell_list(hidden_units*2,decoder_layer,decoder_layer_residual,keep_prob)

        decoder_attention_cell = seq2seq.AttentionWrapper(
            cell=rnn_decoder_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=hidden_units,
            #cell_input_fn=attn_decoder_input_fn,
            name='attention'
        )
        decoder_initial_state = decoder_attention_cell.zero_state(batch_size_real,tf.float32).clone(cell_state=encoder_last_state)

    return decoder_attention_cell,decoder_initial_state