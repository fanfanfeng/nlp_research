# create by fanfan on 2019/4/24 0024
import tensorflow as tf
from third_models.bert import modeling as bert_modeling
from category.tf_models.classify_cnn_model import ClassifyCnnModel
from category.tf_models.classify_bilstm_model import ClassifyBilstmModel
from category.tf_models.classify_rcnn_model import ClassifyRcnnModel
from category.tf_models import constant
from sklearn.metrics import f1_score
import tqdm
import os


class BertClassifyModel(object):
    def __init__(self,params,bert_config):
        self.params = params
        self.bert_config = bert_config


    def create_model(self,input_ids, input_mask, segment_ids,is_training,dropout,use_one_hot_embeddings=False):
        with tf.variable_scope("model_define", reuse=tf.AUTO_REUSE) as scope:
            bert_model_layer = bert_modeling.BertModel(
                config=self.bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings
            )



            classify_model = None
            if self.params.category_type == 'cnn':
                classify_model = ClassifyCnnModel(self.params)
            elif self.params.category_type == "bilstm":
                classify_model = ClassifyBilstmModel(self.params)
            elif self.params.category_type == "rcnn":
                classify_model = ClassifyRcnnModel(self.params)


            if classify_model != None:
                real_sentece_length = classify_model.get_setence_length(input_ids)
                embedding_input_x = bert_model_layer.get_sequence_output()
                output_layer = classify_model.create_model(embedding_input_x, dropout, already_embedded=True,
                                                               real_sentence_length=real_sentece_length,need_logit=False)
            else:
                # 获取第一个位置 cls 输入数据[batch_size, embedding_size]
                output_layer = bert_model_layer.get_pooled_output()


            with tf.variable_scope("loss"):
                if is_training:
                    # I.e., 0.1 dropout
                    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            logits = tf.layers.dense(output_layer,self.params.num_tags)

        return logits



    def make_train(self, input_ids, input_mask, segment_ids,label_ids):
        dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        logits = self.create_model(input_ids, input_mask,segment_ids,is_training=True,dropout=dropout)
        logits = tf.identity(logits, name=constant.OUTPUT_NODE_LOGIT)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ids))
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, globalStep)

        predict = tf.argmax(logits, axis=1, output_type=tf.int32, name=constant.OUTPUT_NODE_NAME)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, label_ids), dtype=tf.float32))

        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()

        return loss, globalStep, train_op, summary_op



    def make_test(self,input_ids, input_mask, segment_ids,label_ids):
        dropout = tf.placeholder_with_default(1.0,(), name='dropout')

        logits = self.create_model(input_ids, input_mask, segment_ids, is_training=True, dropout=dropout)
        logits = tf.identity(logits, name=constant.OUTPUT_NODE_LOGIT)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_ids))
        predict = tf.argmax(logits,axis=1,output_type=tf.int64,name=constant.OUTPUT_NODE_NAME)
        return loss,predict

    def make_pb_file(self,model_dir):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                input_ids = tf.placeholder(dtype=tf.int32,shape=(None,self.params.max_sentence_length),name=constant.INPUT_NODE_NAME)
                input_mask = tf.placeholder(dtype=tf.int32,shape=(None,self.params.max_sentence_length),name=constant.INPUT_MASK_NAME)
                dropout = tf.placeholder_with_default(1.0,shape=(), name='dropout')
                logits = self.create_model(input_ids, input_mask, segment_ids=None, is_training=False, dropout=dropout)
                pred_ids = tf.argmax(tf.nn.softmax(logits),axis=1,name=constant.OUTPUT_NODE_NAME)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                self.model_restore(sess,saver)

                output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,[constant.OUTPUT_NODE_NAME])

                with tf.gfile.GFile(os.path.join(model_dir,'classify.pb'),'wb') as gf:
                    gf.write(output_graph_with_weight.SerializeToString())
        return os.path.join(model_dir,'classify.pb')

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
        input_mask_node = sess.graph.get_operation_by_name(constant.INPUT_MASK_NAME).outputs[0]
        logit_node = sess.graph.get_operation_by_name(constant.OUTPUT_NODE_NAME).outputs[0]
        return sess, input_node,input_mask_node, logit_node


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
            sess.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            # 加载BERT模型
            bert_init_checkpoint = os.path.join(self.params.bert_model_path, 'bert_model.ckpt')
            if os.path.exists(self.params.bert_model_path):
                (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                                bert_init_checkpoint)
                tf.train.init_from_checkpoint(bert_init_checkpoint, assignment_map)