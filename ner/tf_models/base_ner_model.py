# create by fanfan on 2019/4/18 0018
import os
import tqdm
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf
from tensorflow.contrib import layers
from sklearn.metrics import f1_score

from ner.tf_models import constant
from ner.tf_models.params import Params
class BasicNerModel():
    def __init__(self,params):
        self.params = params




    def crf_layer_loss(self,logits,labels,seq_lens,transition_params):
        """
                        calculate crf loss
                        :param project_logits: [1, num_steps, num_tags]
                        :return: scalar loss
                        """
        with tf.variable_scope('crf_layer'):
            log_likelihood,_ = crf.crf_log_likelihood(logits,tag_indices=labels,sequence_lengths=seq_lens,transition_params=transition_params)

        return tf.reduce_mean(-log_likelihood)

    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def create_model(self,input_x,dropout,already_embedded=False,real_sentence_length=None):
        with tf.variable_scope("model_define", reuse=tf.AUTO_REUSE) as scope:
            if not already_embedded:
                with tf.variable_scope('embeddings_layer'):
                    word_embeddings = tf.get_variable('word_embeddings', [self.params.vocab_size, self.params.embedding_size])
                    input_embeddings = tf.nn.embedding_lookup(word_embeddings, input_x)
                real_sentence_length = self.get_setence_length(input_x)
            else:
                input_embeddings = input_x
                real_sentence_length = real_sentence_length

            model_output = self.model_layer(input_embeddings,dropout,real_sentence_length)
            logits = self.project_layer(model_output)

            trans = tf.get_variable('transitions',
                                         shape=[self.params.num_tags,self.params.num_tags],)
        return logits,real_sentence_length,trans

    def model_layer(self, model_inputs, dropout,sequence_length=None):
        raise NotImplementedError("")

    def project_layer(self,model_outputs,name=None):
        """
                hidden layer between lstm layer and logits
                :param model_outputs: [batch_size, num_steps, emb_size]  
                :return: [batch_size, num_steps, num_tags]
                """
        with  tf.variable_scope('project' if not name else name):
            with tf.variable_scope('hidden'):
                hidden_output = layers.fully_connected(model_outputs,self.params.hidden_size,activation_fn=tf.tanh)

            # project to score of tags
            with tf.variable_scope('logits'):
                pred = layers.fully_connected(hidden_output,self.params.num_tags,activation_fn=None)

            return tf.reshape(pred,[-1,self.params.max_sentence_length,self.params.num_tags])



    def make_train(self,input_x,input_y):
        dropout = tf.placeholder_with_default(0.8,shape=[], name='dropout')
        logits,real_sentence_length,trans = self.create_model(input_x, dropout)
        global_step = tf.Variable(0, name="globalStep", trainable=False)
        with tf.variable_scope('loss'):
            loss = self.crf_layer_loss(logits,input_y,real_sentence_length,trans)
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=real_sentence_length)
        pred_ids = tf.identity(pred_ids,name=constant.OUTPUT_NODE_NAME)
        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", loss)
            summary_op = tf.summary.merge_all()

        return loss,global_step,train_op,summary_op

    def make_test(self,input_x,input_y):
        dropout = tf.placeholder_with_default(1.0,(), name='dropout')

        logits, real_sentence_length, trans = self.create_model(input_x, dropout)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=input_y))
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=real_sentence_length)
        pred_ids = tf.identity(pred_ids, name=constant.OUTPUT_NODE_NAME)
        return loss,pred_ids,real_sentence_length

    def make_pb_file(self,model_dir):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf,graph=graph)
            with sess.as_default():
                input_x = tf.placeholder(dtype=tf.int32,shape=(None,self.params.max_sentence_length),name=constant.INPUT_NODE_NAME)
                dropout = tf.placeholder_with_default(1.0,shape=(), name='dropout')
                logits,real_sentence_length,trans = self.create_model(input_x, dropout)
                pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans,
                                             sequence_length=real_sentence_length)
                pred_ids = tf.identity(pred_ids, name=constant.OUTPUT_NODE_NAME)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                checkpoint = tf.train.latest_checkpoint(model_dir)
                if checkpoint:
                    saver.restore(sess,checkpoint)
                else:
                    raise FileNotFoundError("模型文件未找到")

                output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,[constant.OUTPUT_NODE_NAME])

                with tf.gfile.GFile(os.path.join(model_dir,'ner.pb'),'wb') as gf:
                    gf.write(output_graph_with_weight.SerializeToString())
        return os.path.join(model_dir,'ner.pb')

    @staticmethod
    def load_model_from_pb(model_path):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        ))

        with tf.gfile.GFile(model_path, 'rb') as fr:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fr.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name="")

        input_node = sess.graph.get_operation_by_name(constant.INPUT_NODE_NAME).outputs[0]
        predict_node = sess.graph.get_operation_by_name(constant.OUTPUT_NODE_NAME).outputs[0]
        return sess, input_node, predict_node


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