# create by fanfan on 2019/4/17 0017
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import layers
from ner.tf_models.base_ner_model import BasicNerModel

class IdCnn(BasicNerModel):
    def __init__(self,config):
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]

        super().__init__(ner_config=config)
        self.filter_width = config.filter_width
        self.num_filter = config.num_filter
        self.repeat_times = config.repeat_times


    def model_layer(self,model_inputs,dropout,sequence_length=None):
        model_inputs = tf.expand_dims(model_inputs, 1)
        with tf.variable_scope('idcnn_layer'):
            filter_weights = tf.get_variable('idcnn_filter',
                                             shape=[1,self.filter_width,self.embedding_size,self.num_filter],
                                             initializer=initializers.xavier_initializer())
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1,1,1,1],
                                      padding='SAME',
                                      name='init_layer')

            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope('atrous_conv_layer_%d' % i,reuse=True if j>0 else False):
                        w = tf.get_variable('filter_W',
                                            shape=[1,self.filter_width,self.num_filter,self.num_filter],
                                            initializer=initializers.xavier_initializer())
                        b = tf.get_variable('filter_b',shape=[self.num_filter])

                        conv = tf.nn.atrous_conv2d(layerInput,w,rate=dilation,padding='SAME')

                        conv = tf.nn.bias_add(conv,b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter

            finalOut = tf.concat(axis=3,values=finalOutFromLayers)
            finalOut = tf.nn.dropout(finalOut,dropout)

            finalOut = tf.squeeze(finalOut,[1])
            finalOut = tf.reshape(finalOut,[-1,totalWidthForLastDim])

        return finalOut


    def project_model_layer(self,model_outputs,name=None):
        """
                :param idcnn_outputs: [batch_size, totalWidthForLastDim]  
                :return: [batch_size, num_steps, num_tags]
                """
        with  tf.variable_scope('project' if not name else name):
            # project to score of tags
            with tf.variable_scope('logits'):
                logit = layers.fully_connected(model_outputs,self.num_tags,activation_fn=None)

            return tf.reshape(logit,[-1,self.max_seq_length,self.num_tags])