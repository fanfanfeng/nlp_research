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



