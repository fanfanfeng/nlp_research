# create by fanfan on 2019/11/14 0014
from dialog_system.attention_seq2seq.tf_model.encoder import create_cell_list
import tensorflow as tf
from tensorflow.contrib.seq2seq import tile_batch,BahdanauAttention,AttentionWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.util import nest

def create_decoder_cell(enc_outputs,
                        enc_states,
                        enc_seq_len,
                        num_layers,
                        num_units,
                        use_residual,
                        keep_prob,
                        batch_size,
                        top_attention,
                        use_beam_search,
                        beam_size,
                        cell_type="lstm"):
    if use_beam_search:
        enc_outputs = tile_batch(enc_outputs, multiplier=beam_size)
        enc_states = nest.map_structure(lambda s: tile_batch(s,beam_size), enc_states)
        enc_seq_len = tile_batch(enc_seq_len, multiplier=beam_size)

    batch_size_real = batch_size * beam_size if use_beam_search else batch_size
    with tf.variable_scope("attention"):
        attention_mechanism = BahdanauAttention(num_units=num_units, memory=enc_outputs,
                                                memory_sequence_length=enc_seq_len)

    def cell_input_fn(inputs, attention):
        output = tf.concat([inputs, attention], axis=-1)
        if use_residual:
            # define cell input function to keep input/output dimension same
            input_project = tf.layers.Dense(num_units, dtype=tf.float32, name='attn_input_feeding')
            output = input_project(output)
        return output

    if top_attention:  # apply attention mechanism only on the top decoder layer
        cells = create_cell_list(num_layers,num_units,keep_prob,use_residual,cell_type,return_list=True)
        cells[-1] = AttentionWrapper(cells[-1], attention_mechanism=attention_mechanism, name="Attention_Wrapper",
                                     attention_layer_size=num_units, initial_cell_state=enc_states[-1],
                                     cell_input_fn=cell_input_fn)
        initial_state = [state for state in enc_states]
        initial_state[-1] = cells[-1].zero_state(batch_size=batch_size_real, dtype=tf.float32)
        dec_init_states = tuple(initial_state)
        cells = MultiRNNCell(cells)
    else:
        cells = create_cell_list(num_layers,num_units,keep_prob,use_residual,cell_type)
        cells = AttentionWrapper(cells, attention_mechanism=attention_mechanism, name="Attention_Wrapper",
                                 attention_layer_size=num_units, initial_cell_state=enc_states,
                                 cell_input_fn=cell_input_fn)
        dec_init_states = cells.zero_state(batch_size=batch_size_real, dtype=tf.float32).clone(cell_state=enc_states)
    return cells, dec_init_states