# create by fanfan on 2019/3/26 0026
import tensorflow as tf
import tqdm
import os
from sklearn.metrics import classification_report,f1_score

input_node_name = 'input_x'
output_node_name = 'predict_index'
output_node_logit = 'logit'




class ClassifyConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
               vocab_size=1000,
               hidden_size=256,
               embedding_size=256,
               num_hidden_layers=2,
               dropout_prob=0.2,
               initializer_range=0.02,
                learning_rate=0.01,
                 max_sentence_length=50,
                 min_freq= 3
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.learning_rate = learning_rate
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.min_freq = min_freq

        self.batch_size = 300
        self.epochs = 30

        self.evaluate_every_num_epochs =  50  # small values may hurt performance
        self.save_every_num_epochs =  200


        self.save_path = "tmp/"
        # 指定gpu deviceid
        self.CUDA_VISIBLE_DEVICES = "0"


        self.input_node_name = 'input_x'
        self.output_node_name = 'predict_index'
        self.output_node_logit = 'logit'



class BaseClassifyModel(object):
    def __init__(self,classify_config):
        self.classify_config = classify_config


    def create_model(self,input_x,dropout,sentences_lengths=None):
        with tf.variable_scope('embeddings_layer'):
            word_embeddings = tf.get_variable('word_embeddings', [self.classify_config.vocab_size, self.classify_config.embedding_size])
            input_embeddings = tf.nn.embedding_lookup(word_embeddings, input_x)

        with tf.variable_scope('classify_layer'):
            output_layer = self.classify_layer(input_embeddings,dropout)
        logits = tf.nn.softmax(output_layer,name=output_node_logit)
        return logits

    def train(self,inputX,inputY):
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dropout = tf.placeholder(dtype=tf.float32, name='dropout')

            logits = self.create_model(inputX,dropout)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            with tf.variable_scope('loss'):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=inputY))
                optimizer = tf.train.AdamOptimizer(self.classify_config.learning_rate)

                grads_and_vars = optimizer.compute_gradients(loss)
                trainOp = optimizer.apply_gradients(grads_and_vars,globalStep)

            predict = tf.argmax(logits,axis=1,output_type=tf.int64,name=self.classify_config.output_node_name)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,inputY),dtype=tf.float32))


            with tf.variable_scope('summary'):
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("accuracy", accuracy)
                summary_op = tf.summary.merge_all()

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            sess.run(tf.global_variables_initializer())
            steps = 0

            tf_save_path = os.path.join(self.classify_config.save_path,'tf')
            try:
                for _ in tqdm.tqdm(range(40000), desc="steps",miniters=10):
                    sess_loss,predict_var,steps,_,train_y_var = sess.run(
                        [loss,  predict,globalStep,trainOp,inputY],
                        feed_dict={dropout:0.8}
                    )

                    if steps % self.classify_config.evaluate_every_num_epochs == 0:
                        f1 = f1_score(train_y_var,predict_var,average='micro')
                        print("current step:%s ,loss:%s , f1 :%s" % (steps,sess_loss,f1))


                    if (steps+1) % self.classify_config.save_every_num_epochs == 0:
                        saver.save(sess,tf_save_path,steps)
                        print("save to dir:%s" % self.classify_config.save_path)
            except tf.errors.OutOfRangeError:
                print("training end")

            saver.save(sess, tf_save_path, steps)
            print("save to dir:%s" % self.classify_config.save_path)


    def classify_layer(self, input_embedding,dropout):
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

        input_node = sess.graph.get_operation_by_name(input_node_name).outputs[0]
        logit_node = sess.graph.get_operation_by_name(output_node_logit).outputs[0]
        return sess,input_node,logit_node




