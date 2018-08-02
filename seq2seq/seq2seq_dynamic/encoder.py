# create by fanfan on 2018/7/17 0017
import tensorflow as tf
from tensorflow.contrib import rnn



def build_single_cell(hidden_units,keep_prob,use_residual):
    cell = rnn.LSTMCell(hidden_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell,output_keep_prob=keep_prob)
    if use_residual:
        cell = rnn.ResidualWrapper(cell)
    return cell

def build_encode(encoder_inputs_embedded,encoder_inputs_length,layer_num,hidden_units,keep_prob,use_residual):
    with tf.variable_scope('encoder'):
        # building encoder_cell
        encoder_cell_fw = rnn.MultiRNNCell(
            [build_single_cell(hidden_units,keep_prob,use_residual) for _ in range(layer_num)])
        encoder_cell_bw = rnn.MultiRNNCell(
            [build_single_cell(hidden_units, keep_prob, use_residual) for _ in range(layer_num)])
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
        for i in range(layer_num):
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

    return encoder_outputs,encoder_last_state