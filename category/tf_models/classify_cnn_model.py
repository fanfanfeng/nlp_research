# create by fanfan on 2019/3/26 0026
from category.tf_models.base_classify_model import BaseClassifyModel
from category.tf_models import constant
import tensorflow as tf
from tensorflow.contrib import layers
import os





class ClassifyCnnModel(BaseClassifyModel):
    def __init__(self,classify_config):
        BaseClassifyModel.__init__(self,classify_config)
        self.num_filters = 128
        self.filter_sizes = [2, 3, 4, 5]

    def classify_layer(self, input_embedding,dropout):
        input_embedding_expand = tf.expand_dims(input_embedding, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        # 创建卷积和池化层
        for i, filterSize in enumerate(self.filter_sizes):
            # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
            # 初始化权重矩阵和偏置
            conv = layers.conv2d(inputs=input_embedding_expand,
                                 num_outputs=self.num_filters,
                                 kernel_size=[filterSize,self.embedding_size],
                                 stride=1,
                                 padding='VALID',
                                 scope='conv_'+str(i),
                                 )
            # 池化层，最大池化，池化是对卷积后的序列取一个最大值
            pooled = layers.max_pool2d(
                conv,
                kernel_size=[self.max_sentence_length - filterSize + 1,1],
                stride=1,
                padding='VALID',
                scope='pool_'+ str(i)
            )

            pooledOutputs.append(pooled)

        # 得到CNN网络的输出长度
        numFilterTotal = self.num_filters * len(self.filter_sizes)
        # 池化后的维度不变，按照最后的维度channel来concat
        self.h_pool = tf.concat(pooledOutputs, 3)
        # 摊平成二维的数据输入到全连接层
        self.h_poolflat = tf.reshape(self.h_pool, [-1, numFilterTotal])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_poolflat,dropout)

        h_dense = tf.layers.dense(self.h_drop, numFilterTotal, activation=tf.nn.tanh, use_bias=True)

        with tf.name_scope("output"):
            logits = tf.layers.dense(h_dense,self.num_tags)

        return logits

    def make_pb_file(self,model_dir):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                input_x = tf.placeholder(dtype=tf.int32,shape=(None,self.max_sentence_length),name=constant.INPUT_NODE_NAME)
                dropout = tf.placeholder_with_default(1.0,shape=(), name='dropout')
                logits = self.create_model(input_x, dropout)
                logits_output = tf.identity(logits,name=constant.OUTPUT_NODE_LOGIT)
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



if __name__ == '__main__':
    config = CNNConfig()
    config.vocab_size = 33901
    config.label_nums = 4
    model = ClassifyCnnModel(config)
    #logit = model.train()
    model.make_pb_file(r'E:\git-project\rasa_nlu\rasa\nlu\tmp')