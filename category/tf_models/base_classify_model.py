# create by fanfan on 2019/3/26 0026
import tensorflow as tf
import tqdm
import os
from sklearn.metrics import f1_score
from category.tf_models import constant
from category.tf_models.params import Params




class BaseClassifyModel(object):
    def __init__(self,params):
        self.params = params


    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def create_model(self,input_x,dropout,already_embedded=False):
        real_sentence_length = self.get_setence_length(input_x)
        with tf.variable_scope("model_define",reuse=tf.AUTO_REUSE) as scope:
            if already_embedded:
                input_embeddings = input_x
            else:
                with tf.variable_scope('embeddings_layer'):
                    word_embeddings = tf.get_variable('word_embeddings', [self.params.vocab_size, self.params.embedding_size])
                    input_embeddings = tf.nn.embedding_lookup(word_embeddings, input_x)

            with tf.variable_scope('classify_layer'):
                output_layer = self.classify_layer(input_embeddings,dropout,real_sentence_length)
            logits = output_layer
        return logits

    def make_train(self,inputX,inputY):
        dropout = tf.placeholder_with_default(self.params.dropout_prob,(), name='dropout')

        logits = self.create_model(inputX,dropout)
        logits = tf.identity(logits,name=constant.OUTPUT_NODE_LOGIT)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=inputY))
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars,globalStep)

        predict = tf.argmax(logits,axis=1,output_type=tf.int64,name=constant.OUTPUT_NODE_NAME)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,inputY),dtype=tf.float32))


        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()

        return loss,globalStep,train_op,summary_op


    def make_test(self,inputX,inputY):
        dropout = tf.placeholder_with_default(1.0,(), name='dropout')

        logits = self.create_model(inputX,dropout)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=inputY))
        predict = tf.argmax(logits,axis=1,output_type=tf.int64,name=constant.OUTPUT_NODE_NAME)
        return loss,predict


    def model_restore(self, sess, tf_saver):
        '''
        模型恢复或者初始化
        :param sess: 
        :param tf_saver: 
        :return: 
        '''
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.params.model_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("restor model")
            tf_saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model")
            sess.run(tf.global_variables_initializer())


    def classify_layer(self, input_embedding,dropout,real_sentence_length=None):
        """Implementation of specific classify layer"""
        raise NotImplementedError()


    @staticmethod
    def load_model_from_pb(model_path):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        ))

        with tf.gfile.GFile(model_path,'rb') as fr:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fr.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name="")

        input_node = sess.graph.get_operation_by_name(constant.INPUT_NODE_NAME).outputs[0]
        logit_node = sess.graph.get_operation_by_name(constant.OUTPUT_NODE_LOGIT).outputs[0]
        return sess,input_node,logit_node




