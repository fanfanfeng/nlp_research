# create by fanfan on 2019/7/11 0011
# create by fanfan on 2019/3/26 0026
from category.tf_models.base_classify_model import BaseClassifyModel
from category.tf_models import constant
import tensorflow as tf
from tensorflow.contrib import layers
import os
from tensorflow.contrib import rnn




class ClassifyBilstmModel(BaseClassifyModel):
    def __init__(self,params):
        BaseClassifyModel.__init__(self,params)


    def lstm_cell(self,dropout):
        lstm_fw = rnn.LSTMCell(self.params.hidden_size)
        lstm_fw = rnn.DropoutWrapper(lstm_fw, dropout)

        return lstm_fw

    def classify_layer(self, input_embedding,dropout,real_sentence_length):
        if self.params.layer_num >1:
            lstm_fw = rnn.MultiRNNCell([self.lstm_cell(dropout) for _ in range(self.params.layer_num)])
            lstm_bw = rnn.MultiRNNCell([self.lstm_cell(dropout) for _ in range(self.params.layer_num)])
        else:
            lstm_fw = self.lstm_cell(dropout)
            lstm_bw = self.lstm_cell(dropout)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,input_embedding,real_sentence_length,dtype=tf.float32)
        outputs = tf.concat(outputs,axis=2)


        if self.params.use_attention == True:
            with tf.variable_scope('attention_layer'):
                rnn_attention_outputs = self.attention_layer(outputs, self.params.attention_size)
                last_output = tf.nn.dropout(rnn_attention_outputs, dropout)
        else:
            last_output = outputs[:, -1, :]

        with tf.name_scope("output"):
            logits = tf.layers.dense(last_output,self.params.num_tags,)

        return logits

    def attention_layer(self,inputs,attention_size):
        """
           Attention mechanism layer.
           :param inputs: outputs of RNN/Bi-RNN layer (not final state)
           :param attention_size: linear size of attention weights
           :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
           """
        # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
        if isinstance(inputs,tuple):
            inputs = tf.concat(inputs,2)
        sequence_length = inputs.get_shape()[1].value
        hidden_size = inputs.get_shape()[2].value

        # Attention
        W_omega = tf.get_variable('W_omega',initializer=tf.random_normal([hidden_size,attention_size],stddev=0.1))
        b_omega = tf.get_variable('b_omega',initializer=tf.random_normal([attention_size],stddev=0.1))
        u_omega = tf.get_variable('u_omega',initializer=tf.random_normal([attention_size],stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs,[-1,hidden_size]),W_omega)) + tf.reshape(b_omega,[1,-1])
        vu = tf.matmul(v,tf.reshape(u_omega,[-1,1]))
        exps = tf.reshape(tf.exp(vu),[-1,sequence_length])
        alphas = exps/tf.reshape(tf.reduce_sum(exps,1),[-1,1])

        output = tf.reduce_sum(inputs*tf.reshape(alphas,[-1,sequence_length,1]),1)
        return output

    def make_pb_file(self,model_dir):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                input_x = tf.placeholder(dtype=tf.int32,shape=(None,self.params.max_sentence_length),name=constant.INPUT_NODE_NAME)
                dropout = tf.placeholder_with_default(1.0,shape=(), name='dropout')
                logits = self.create_model(input_x, dropout)
                logits_output = tf.nn.softmax(logits,name=constant.OUTPUT_NODE_LOGIT)
                predict = tf.argmax(logits, axis=1, output_type=tf.int32,
                                    name=constant.OUTPUT_NODE_NAME)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                checkpoint = tf.train.latest_checkpoint(model_dir)
                if checkpoint:
                    saver.restore(sess,checkpoint)
                else:
                    raise FileNotFoundError("模型文件未找到")

                output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,[constant.OUTPUT_NODE_NAME,constant.OUTPUT_NODE_LOGIT])

                with tf.gfile.GFile(os.path.join(model_dir,'classify.pb'),'wb') as gf:
                    gf.write(output_graph_with_weight.SerializeToString())
        return os.path.join(model_dir,'classify.pb')

