from category.tf_models.base_classify_model import BaseClassifyModel
from category.tf_models.classify_cnn_model import ClassifyCnnModel
from category.tf_models.classify_bilstm_model import ClassifyBilstmModel
from category.tf_models.classify_rcnn_model import ClassifyRcnnModel
from category.tf_models import constant
import tensorflow as tf
import os





class ClassifyEnsembleModel(ClassifyRcnnModel,ClassifyCnnModel,ClassifyBilstmModel):
    def __init__(self,params):
        ClassifyRcnnModel.__init__(self,params)
        ClassifyCnnModel.__init__(self,params)
        ClassifyBilstmModel.__init__(self,params)

    def classify_layer(self, input_embedding, dropout, real_sentence_length=None):
        with tf.variable_scope("cnn_layer"):
            cnn_output = ClassifyCnnModel.classify_layer(self,input_embedding,dropout,real_sentence_length)

        with tf.variable_scope('bilstm_layer'):
            bilstm_output = ClassifyBilstmModel.classify_layer(self,input_embedding,dropout,real_sentence_length)

        with tf.variable_scope("rcnn_layer"):
            rcnn_output = ClassifyRcnnModel.classify_layer(self,input_embedding,dropout,real_sentence_length)
        total_output = tf.concat([cnn_output,bilstm_output,rcnn_output],axis=1)
        return total_output

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

