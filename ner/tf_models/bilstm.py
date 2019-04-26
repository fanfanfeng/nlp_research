# create by fanfan on 2019/4/17 0017
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers
from ner.tf_models.base_ner_model import BasicNerModel

class BiLSTM(BasicNerModel):
    def __init__(self,config):

        super().__init__(ner_config=config)
        self.cell_type = config.cell_type
        self.bilstm_layer_nums = config.bilstm_layer_nums


    def _witch_cell(self,dropout):
        """
                RNN 类型
                :return:
                """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_size)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_size)

        cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=dropout)
        return cell_tmp

    def model_layer(self,model_inputs,dropout,sequence_length=None):
        with tf.variable_scope('rnn_layer'):
            if self.bilstm_layer_nums > 1:
                cell_fw = rnn.MultiRNNCell([self._witch_cell(dropout) for _ in range(self.bilstm_layer_nums)])
                cell_bw = rnn.MultiRNNCell([self._witch_cell(dropout) for _ in range(self.bilstm_layer_nums)])
            else:
                cell_bw = self._witch_cell(dropout)
                cell_fw = self._witch_cell(dropout)

            outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,model_inputs,dtype=tf.float32,)
            outputs = tf.concat(outputs,axis=2)
            outputs = tf.reshape(outputs,[-1,self.hidden_size * 2])
        return outputs





