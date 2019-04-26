# create by fanfan on 2019/4/18 0018
import os
import tqdm
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf
from tensorflow.contrib import layers
from sklearn.metrics import f1_score

from ner.tf_models import constant
import six
import json
import copy


class NerConfig(object):
    """Configuration for `NerModel`."""
    def __init__(self,
                 vocab_size,
                 num_tags=None,
                 hidden_size=256,
                 embedding_size=256,
                 dropout_prob=0.7,
                 max_seq_length=512,
                 learning_rate=0.0001,
                 evaluate_every_steps=20,
                 batch_size=128
                 ):
        """Constructs NerConfig.

        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.evaluate_every_steps = evaluate_every_steps
        self.batch_size = batch_size




    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = NerConfig(vocab_size=None)
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



class BasicNerModel():
    def __init__(self,ner_config):
        self.num_tags = ner_config.num_tags
        self.vocab_size = ner_config.vocab_size
        self.embedding_size = ner_config.embedding_size
        self.hidden_size = ner_config.hidden_size
        self.max_seq_length = ner_config.max_seq_length
        self.learning_rate = ner_config.learning_rate

        self.output_path = ner_config.output_path

        self.evaluate_every_steps = ner_config.evaluate_every_steps



        with tf.variable_scope('crf_layer'):
            self.trans = tf.get_variable('transitions',
                                         shape=[self.num_tags,self.num_tags],
                                         initializer=initializers.xavier_initializer())

    def crf_layer_loss(self,logits,labels,seq_lens):
        """
                        calculate crf loss
                        :param project_logits: [1, num_steps, num_tags]
                        :return: scalar loss
                        """
        with tf.variable_scope('crf_layer'):
            log_likelihood,self.trans = crf.crf_log_likelihood(logits,tag_indices=labels,sequence_lengths=seq_lens,transition_params=self.trans)

        return tf.reduce_mean(-log_likelihood)

    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def create_model(self,input_x,dropout,already_embedded=False,real_sentence_length=None):
        if not already_embedded:
            with tf.variable_scope('embeddings_layer'):
                word_embeddings = tf.get_variable('word_embeddings', [self.vocab_size, self.embedding_size])
                input_embeddings = tf.nn.embedding_lookup(word_embeddings, input_x)
            self.real_sentence_length = self.get_setence_length(input_x)
        else:
            input_embeddings = input_x
            self.real_sentence_length = real_sentence_length

        model_output = self.model_layer(input_embeddings,dropout,self.real_sentence_length)
        logits = self.project_layer(model_output)
        return logits

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
                hidden_output = layers.fully_connected(model_outputs,self.hidden_size,activation_fn=tf.tanh)

            # project to score of tags
            with tf.variable_scope('logits'):
                pred = layers.fully_connected(hidden_output,self.num_tags,activation_fn=None)

            return tf.reshape(pred,[-1,self.max_seq_length,self.num_tags])



    def train(self,input_x,input_y):
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dropout = tf.placeholder(dtype=tf.float32, name='dropout')

            logits = self.create_model(input_x, dropout)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            with tf.variable_scope('loss'):
                loss = self.crf_layer_loss(logits,input_y,self.real_sentence_length)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                grads_and_vars = optimizer.compute_gradients(loss)
                trainOp = optimizer.apply_gradients(grads_and_vars, globalStep)

            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=self.trans, sequence_length=self.real_sentence_length)
            pred_ids = tf.identity(pred_ids,name=constant.OUTPUT_NODE_NAME)
            with tf.variable_scope('summary'):
                tf.summary.scalar("loss", loss)
                summary_op = tf.summary.merge_all()

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            sess.run(tf.global_variables_initializer())
            steps = 0

            tf_save_path = os.path.join(self.output_path, 'tf')
            try:
                best_f1 = 0
                for _ in tqdm.tqdm(range(40000), desc="steps", miniters=10):
                    sess_loss, predict_var, steps, _,real_sentence,input_y_val  = sess.run(
                        [loss, pred_ids, globalStep, trainOp,self.real_sentence_length,input_y],
                        feed_dict={dropout: 0.8}
                    )

                    if steps % self.evaluate_every_steps == 0:
                        train_labels = []
                        predict_labels = []
                        for train_, predict_,len_ in zip(input_y_val, predict_var,real_sentence):
                            train_labels += train_[:len_].tolist()
                            predict_labels += predict_[:len_].tolist()
                        f1_val = f1_score(train_labels, predict_labels, average='micro')
                        print("current step:%s ,loss:%s ,f1 :%s" % (steps, sess_loss,f1_val))

                        if f1_val > best_f1:
                            saver.save(sess, tf_save_path, steps)
                            print("new best f1: %s ,save to dir:%s" % (f1_val,self.output_path))
                            best_f1 = f1_val
            except tf.errors.OutOfRangeError:
                print("training end")
