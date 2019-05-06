# create by fanfan on 2019/3/26 0026
import tensorflow as tf
import tqdm
import os
from sklearn.metrics import classification_report,f1_score
from category.tf_models import constant
import six
import json
import copy





class ClassifyConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
               vocab_size,
                num_tags=None,
               hidden_size=256,
               embedding_size=256,
               dropout_prob=0.7,
               initializer_range=0.02,
                learning_rate=0.0001,
                 max_sentence_length=50,
                 evaluate_every_steps=20,
                 min_freq= 3,
                 batch_size=300
                 ):

        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_size = embedding_size
        self.max_sentence_length = max_sentence_length
        self.learning_rate = learning_rate
        self.evaluate_every_steps = evaluate_every_steps
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.initializer_range =initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = ClassifyConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




class BaseClassifyModel(object):
    def __init__(self,classify_config):
        self.num_tags = classify_config.num_tags
        self.vocab_size = classify_config.vocab_size
        self.embedding_size = classify_config.embedding_size
        self.hidden_size = classify_config.hidden_size
        self.max_sentence_length = classify_config.max_sentence_length
        self.learning_rate = classify_config.learning_rate

        self.output_path = classify_config.output_path

        self.evaluate_every_steps = classify_config.evaluate_every_steps
        self.dropout_prob = classify_config.dropout_prob


    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def create_model(self,input_x,dropout,already_embedded=False,real_sentence_length=None):
        if already_embedded:
            input_embeddings = input_x
        else:
            with tf.variable_scope('embeddings_layer'):
                word_embeddings = tf.get_variable('word_embeddings', [self.vocab_size, self.embedding_size])
                input_embeddings = tf.nn.embedding_lookup(word_embeddings, input_x)

        with tf.variable_scope('classify_layer'):
            output_layer = self.classify_layer(input_embeddings,dropout)
        #logits = tf.nn.softmax(output_layer,name=output_node_logit)
        logits = tf.identity(output_layer, name=constant.OUTPUT_NODE_LOGIT)
        return logits

    def train(self,inputX,inputY):
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dropout = tf.placeholder_with_default(self.dropout_prob,(), name='dropout')

            logits = self.create_model(inputX,dropout)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            with tf.variable_scope('loss'):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=inputY))
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                grads_and_vars = optimizer.compute_gradients(loss)
                trainOp = optimizer.apply_gradients(grads_and_vars,globalStep)

            predict = tf.argmax(logits,axis=1,output_type=tf.int64,name=constant.OUTPUT_NODE_NAME)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,inputY),dtype=tf.float32))


            with tf.variable_scope('summary'):
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("accuracy", accuracy)
                summary_op = tf.summary.merge_all()

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            sess.run(tf.global_variables_initializer())
            steps = 0

            tf_save_path = os.path.join(self.output_path,'tf')

            best_f1 = 0
            for _ in tqdm.tqdm(range(40000), desc="steps",miniters=10):
                sess_loss,predict_var,steps,_,train_y_var = sess.run(
                    [loss,  predict,globalStep,trainOp,inputY])

                if steps % self.evaluate_every_steps == 0:
                    f1_val = f1_score(train_y_var,predict_var,average='micro')
                    print("current step:%s ,loss:%s , f1 :%s" % (steps,sess_loss,f1_val))


                    if f1_val > best_f1:
                        saver.save(sess, tf_save_path, steps)
                        print("new best f1: %s ,save to dir:%s" % (f1_val, self.output_path))
                        best_f1 = f1_val


            saver.save(sess, tf_save_path, steps)
            print("save to dir:%s" % self.output_path)


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

        input_node = sess.graph.get_operation_by_name(constant.INPUT_NODE_NAME).outputs[0]
        logit_node = sess.graph.get_operation_by_name(constant.OUTPUT_NODE_NAME).outputs[0]
        return sess,input_node,logit_node




