# create by fanfan on 2019/4/24 0024
import tensorflow as tf
from third_models.bert import modeling as bert_modeling
from ner.tf_models.bilstm import BiLSTM
from ner.tf_models.idcnn import IdCnn
from ner.tf_models import constant
from tensorflow.contrib import crf
from sklearn.metrics import f1_score
import tqdm
import os


class BertNerModel(object):
    def __init__(self,ner_config,bert_config):
        self.ner_config = ner_config
        self.bert_config = bert_config


    def create_model(self,input_ids, input_mask, segment_ids,is_training,dropout,use_one_hot_embeddings=False):
        bert_model_layer = bert_modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
        embedding_input_x = bert_model_layer.get_sequence_output()

        if self.ner_config.ner_type == "idcnn":
            self.ner_model = IdCnn(self.ner_config)
        else:
            self.ner_model = BiLSTM(self.ner_config)
        real_sentece_length = self.ner_model.get_setence_length(input_ids)
        logits = self.ner_model.create_model(embedding_input_x,dropout,already_embedded=True,real_sentence_length=real_sentece_length)
        return logits



    def train(self, input_ids, input_mask, segment_ids,label_ids):
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dropout = tf.placeholder(dtype=tf.float32, name='dropout')

            logits = self.create_model(input_ids, input_mask,segment_ids,is_training=True,dropout=dropout)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            with tf.variable_scope('loss'):
                loss = self.ner_model.crf_layer_loss(logits, label_ids, self.ner_model.real_sentence_length)
                optimizer = tf.train.AdamOptimizer(self.ner_config.learning_rate)

                grads_and_vars = optimizer.compute_gradients(loss)
                trainOp = optimizer.apply_gradients(grads_and_vars, globalStep)

            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=self.ner_model.trans,
                                         sequence_length=self.ner_model.real_sentence_length)
            pred_ids = tf.identity(pred_ids, name=constant.OUTPUT_NODE_NAME)
            with tf.variable_scope('summary'):
                tf.summary.scalar("loss", loss)
                summary_op = tf.summary.merge_all()

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            sess.run(tf.global_variables_initializer())
            steps = 0

            tf_save_path = os.path.join(self.ner_config.output_path, 'tf')
            #try:
            best_f1 = 0
            for _ in tqdm.tqdm(range(40000), desc="steps", miniters=10):
                sess_loss, predict_var, steps, _, real_sentence, input_y_val = sess.run(
                    [loss, pred_ids, globalStep, trainOp, self.ner_model.real_sentence_length, label_ids],
                    feed_dict={dropout: 0.8}
                )

                if steps % self.ner_config.evaluate_every_steps == 0:
                    train_labels = []
                    predict_labels = []
                    for train_, predict_, len_ in zip(input_y_val, predict_var, real_sentence):
                        train_labels += train_[:len_].tolist()
                        predict_labels += predict_[:len_].tolist()
                    f1_val = f1_score(train_labels, predict_labels, average='micro')
                    print("current step:%s ,loss:%s ,f1 :%s" % (steps, sess_loss, f1_val))

                    if f1_val > best_f1:
                        saver.save(sess, tf_save_path, steps)
                        print("new best f1: %s ,save to dir:%s" % (f1_val, self.ner_config.output_path))
                        best_f1 = f1_val
            #except tf.errors.OutOfRangeError:
                #print("training end")