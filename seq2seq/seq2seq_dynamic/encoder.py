# create by fanfan on 2018/7/17 0017
import tensorflow as tf
from tensorflow.contrib import rnn



def build_single_cell(hidden_units,keep_prob,use_residual):
    cell = rnn.LSTMCell(hidden_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell,input_keep_prob=keep_prob)
    if use_residual:
        cell = rnn.ResidualWrapper(cell)
    return cell

def build_cell_list(hidden_units,num_layers,num_residual_layers,keep_prob):
    cell_list = []
    for i in range(num_layers):
        single_cell = build_single_cell(hidden_units,keep_prob,(i >= num_layers - num_residual_layers))
        cell_list.append(single_cell)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def build_encode(encoder_inputs_embedded,encoder_inputs_length,layer_num,hidden_units,keep_prob,use_residual):
    with tf.variable_scope('encoder'):
        # 由于用的bilstm，所以layer除以2,输入的last_state跟后面的decode cell能够对应起来
        encoder_layer = int(layer_num/2)
        encoder_layer_residual = int(layer_num/2 -1)

        # building encoder_cell
        encoder_cell_fw = build_cell_list(hidden_units,encoder_layer,encoder_layer_residual,keep_prob)
        encoder_cell_bw = build_cell_list(hidden_units,encoder_layer,encoder_layer_residual,keep_prob)
        #将输入的句子转化为上下文向量
        # encoder_outputs: [batch_size, max_time_step, cell_output_size]
        # encoder_state: [batch_size, cell_output_size]
        encoder_outputs,encoder_last_state = tf.nn.bidirectional_dynamic_rnn(
            encoder_cell_fw,
            encoder_cell_bw,
            inputs=encoder_inputs_embedded,
            sequence_length=encoder_inputs_length,
            dtype=tf.float32,
        )
        # 由于是双向lstm，合并输入
        encoder_outputs = tf.concat(encoder_outputs, 2)

        # 合并输入的states
        encoder_states = []
        for i in range(encoder_layer):
            if isinstance(encoder_last_state[0][i], tf.contrib.rnn.LSTMStateTuple):
                encoder_state_c = tf.concat(values=(encoder_last_state[0][i].c, encoder_last_state[1][i].c), axis=1,
                                            name="encoder_fw_state_c")
                encoder_state_h = tf.concat(values=(encoder_last_state[0][i].h, encoder_last_state[1][i].h), axis=1,
                                            name="encoder_fw_state_h")
                encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            elif isinstance(encoder_last_state[0][i], tf.Tensor):
                encoder_state = tf.concat(values=(encoder_last_state[0][i], encoder_last_state[1][i]), axis=1,
                                          name='bidirectional_concat')
            encoder_states.append(encoder_state)
        encoder_last_state = tuple(encoder_states)

        #encoder_state = []
        #for layer_id in range(encoder_layer):
        #    encoder_state.append(encoder_last_state[0][layer_id])  # forward
        #    encoder_state.append(encoder_last_state[1][layer_id])  # backward
        #encoder_state = tuple(encoder_state)

    return encoder_outputs,encoder_last_state