# create by fanfan on 2019/11/14 0014
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,LSTMCell,DropoutWrapper,ResidualWrapper,MultiRNNCell
def create_single_cell(num_units,keep_prob,use_residual,cell_type='lstm'):
    if cell_type == 'lstm':
        cell = LSTMCell(num_units)
    else:
        cell = GRUCell(num_units)

    cell = DropoutWrapper(cell, output_keep_prob=keep_prob)

    if use_residual:
        cell = ResidualWrapper(cell)
    return cell


def create_cell_list(num_layers,num_units,keep_prob,use_residual,cell_type,return_list=False):
    cell_list = [create_single_cell(num_units,keep_prob,use_residual,cell_type) for _ in range(num_layers)]

    if num_layers == 1:
        return cell_list[0]
    else:
        if return_list:
            return cell_list
        else:
            return MultiRNNCell(cell_list)



def create_encoder(source_emb,enc_seq_len,num_units,use_residual,keep_prob,num_layers,cell_type='lstm'):
    '''
    :param source_emb: 经过embedding处理的source向量
    :param enc_seq_len: source的长度
    :param num_units: lstm的size
    :param use_residual: 是否使用残差
    :param keep_prob: 保留概率，1 - dropout
    :param num_layers: rnn的层数
    :param cell_type: rnn的类型
    :return: output,output_states
    '''


    enc_cells_fw = create_cell_list(num_layers,num_units,keep_prob,use_residual,cell_type)
    enc_cells_bw = create_cell_list(num_layers,num_units,keep_prob,use_residual,cell_type)
    enc_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(enc_cells_fw, enc_cells_bw, source_emb,
                                                              sequence_length=enc_seq_len,
                                                              dtype=tf.float32)
    enc_outputs = tf.concat(enc_outputs, 2)
    # 合并输入的states
    encoder_states = []
    for i in range(num_layers):
        if isinstance(enc_states[0][i], tf.contrib.rnn.LSTMStateTuple):
            encoder_state_c = tf.concat(values=(enc_states[0][i].c, enc_states[1][i].c), axis=1,
                                        name="encoder_fw_state_c")
            encoder_state_h = tf.concat(values=(enc_states[0][i].h, enc_states[1][i].h), axis=1,
                                        name="encoder_fw_state_h")
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
        elif isinstance(enc_states[0][i], tf.Tensor):
            encoder_state = tf.concat(values=(enc_states[0][i], enc_states[1][i]), axis=1,
                                      name='bidirectional_concat')
        else:
            raise TypeError("cell type error in encoder cell")
        encoder_states.append(encoder_state)
    enc_states = tuple(encoder_states)
    return enc_outputs,enc_states