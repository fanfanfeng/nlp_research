# create by fanfan on 2019/3/26 0026
from category.models.base_classify_model import BaseClassifyModel,ClassifyConfig
import tensorflow as tf
from tensorflow.contrib import layers
import os


class CNNConfig(ClassifyConfig):
    """Configuration for `BertModel`."""

    def __init__(self,
               vocab_size=1000,
               hidden_size=256,
               embedding_size=256,
               num_hidden_layers=2,
               dropout_prob=0.2,
               initializer_range=0.02,
                learning_rate=0.01,
                 max_sentence_length=50
                 ):

        super().__init__(self)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.learning_rate = learning_rate
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size


        self.num_filters = 128
        self.filter_sizes = [2,3,4,5]
        self.label_nums = 4




class ClassifyCnnModel(BaseClassifyModel):
    def __init__(self,classify_config):
        BaseClassifyModel.__init__(self,classify_config)

    def classify_layer(self, input_embedding,dropout):
        input_embedding_expand = tf.expand_dims(input_embedding, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        # 创建卷积和池化层
        for i, filterSize in enumerate(self.classify_config.filter_sizes):
            # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
            # 初始化权重矩阵和偏置
            conv = layers.conv2d(inputs=input_embedding_expand,
                                 num_outputs=self.classify_config.num_filters,
                                 kernel_size=[filterSize,self.classify_config.embedding_size],
                                 stride=1,
                                 padding='VALID',
                                 scope='conv_'+str(i),
                                 )
            # 池化层，最大池化，池化是对卷积后的序列取一个最大值
            pooled = layers.max_pool2d(
                conv,
                kernel_size=[self.classify_config.max_sentence_length - filterSize + 1,1],
                stride=1,
                padding='VALID',
                scope='pool_'+ str(i)
            )

            pooledOutputs.append(pooled)

        # 得到CNN网络的输出长度
        numFilterTotal = self.classify_config.num_filters * len(self.classify_config.filter_sizes)
        # 池化后的维度不变，按照最后的维度channel来concat
        self.h_pool = tf.concat(pooledOutputs, 3)
        # 摊平成二维的数据输入到全连接层
        self.h_poolflat = tf.reshape(self.h_pool, [-1, numFilterTotal])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_poolflat,dropout)

        h_dense = tf.layers.dense(self.h_drop, numFilterTotal, activation=tf.nn.tanh, use_bias=True)

        with tf.name_scope("output"):
            logits = tf.layers.dense(h_dense,self.classify_config.label_nums)

        return logits

    def make_pb_file(self,model_dir):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                input_x = tf.placeholder(dtype=tf.int32,shape=(None,self.classify_config.max_sentence_length),name=self.classify_config.input_node_name)
                dropout = tf.placeholder_with_default(1.0,shape=(), name='dropout')
                logits = self.create_model(input_x, dropout)
                logits_output = tf.identity(logits,name=self.classify_config.output_node_logit)
                predict = tf.argmax(logits, axis=1, output_type=tf.int32,
                                    name=self.classify_config.output_node_name)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                checkpoint = tf.train.latest_checkpoint(model_dir)
                if checkpoint:
                    saver.restore(sess,checkpoint)
                else:
                    raise FileNotFoundError("模型文件未找到")

                output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,[self.classify_config.output_node_name,self.classify_config.output_node_logit])

                with tf.gfile.GFile(os.path.join(model_dir,'classify.pb'),'wb') as gf:
                    gf.write(output_graph_with_weight.SerializeToString())
        return os.path.join(model_dir,'classify.pb')



if __name__ == '__main__':
    config = CNNConfig()
    config.vocab_size = 33901
    config.label_nums = 4
    model = ClassifyCnnModel(config)
    #logit = model.train()
    model.make_pb_file(r'E:\git-project\rasa_nlu\rasa\nlu\tmp')