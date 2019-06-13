# create by fanfan on 2018/7/20 0020
import tensorflow as tf
from dialog_system.attention_seq2seq.tf_model.encoder import build_cell_list
import tensorflow.contrib.seq2seq as seq2seq



def build_decoder_cell(encoder_outputs,encoder_last_state,encoder_inputs_length,layer_num,attention_size,hidden_units,keep_prob,use_residual,mode,beam_width,batch_size):
    memory = encoder_outputs
    memory_sequence_length = encoder_inputs_length
    with tf.variable_scope('decoder'):
        if mode == 'decode' and beam_width > 1:
            memory = seq2seq.tile_batch(memory,multiplier=beam_width)
            memory_sequence_length = seq2seq.tile_batch(memory_sequence_length,multiplier=beam_width)
            encoder_last_state = seq2seq.tile_batch(encoder_last_state,multiplier=beam_width)
            batch_size_real = batch_size * beam_width
        else:
            batch_size_real = batch_size

        attention_mechanism = seq2seq.BahdanauAttention(num_units=attention_size,
                                                        memory=memory,
                                                        memory_sequence_length=memory_sequence_length)

        decoder_layer = layer_num
        decoder_layer_residual = decoder_layer - 1
        rnn_decoder_cell = build_cell_list(hidden_units*2,decoder_layer,decoder_layer_residual,keep_prob,use_residual)

        decoder_attention_cell = seq2seq.AttentionWrapper(
            cell=rnn_decoder_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_size,
            name='attention',
            initial_cell_state=encoder_last_state
        )
        decoder_initial_state = decoder_attention_cell.zero_state(batch_size_real,tf.float32).clone(cell_state=encoder_last_state)

    return decoder_attention_cell,decoder_initial_state
