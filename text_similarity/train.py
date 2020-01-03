# create by fanfan on 2019/12/13 0013
import sys
sys.path.append('/data/python_project/nlp_research')
import tensorflow as tf
from text_similarity import data_prepare
from text_similarity.esim import esim_model
from text_similarity.bimpm import bimpm_model
from text_similarity.paircnn import paircnn_model
from text_similarity.abcnn import abcnn_model
from text_similarity import config


import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn import metrics
import os

conf = config.Config()
current_path = os.path.dirname(os.path.abspath(__file__))
data_reader = data_prepare.Data_Prepare()

output_path = os.path.join(current_path,'output')
if not os.path.exists(output_path):
    os.mkdir(output_path)
vocab_path = os.path.join(output_path,'vocab.txt')
model_savepath = os.path.join(output_path,conf.save_path)


class TrainModel(object):
    def pre_processing(self):
        train_texta, train_textb, train_tag = data_reader.readfile(current_path + '/data/train.txt')
        data = []
        data.extend(train_texta)
        data.extend(train_textb)
        data_reader.build_vocab(data, vocab_path)

        self.vocab,vocab_list = data_prepare.load_vocab(vocab_path)
        train_texta_list = []
        for sentence in train_texta:
            train_texta_list.append(data_prepare.pad_sentence(sentence,conf.max_sentence_len,self.vocab))
        train_textb_list = []
        for sentence in train_textb:
            train_textb_list.append(data_prepare.pad_sentence(sentence, conf.max_sentence_len, self.vocab))

        train_texta_embedding = np.array(train_texta_list)
        train_textb_embedding = np.array(train_textb_list)

        dev_texta, dev_textb, dev_tag = data_reader.readfile(current_path + '/data/dev.txt')
        dev_texta_list = []
        for sentence in dev_texta:
            dev_texta_list.append(data_prepare.pad_sentence(sentence, conf.max_sentence_len, self.vocab))
        dev_textb_list = []
        for sentence in dev_textb:
            dev_textb_list.append(data_prepare.pad_sentence(sentence, conf.max_sentence_len, self.vocab))
        dev_texta_embedding = np.array(dev_texta_list)
        dev_textb_embedding = np.array(dev_textb_list)
        return train_texta_embedding, train_textb_embedding, np.array(train_tag), \
               dev_texta_embedding, dev_textb_embedding, np.array(dev_tag)

    def get_batches(self, texta, textb, tag):
        num_batch = int(len(texta) / conf.Batch_Size)
        for i in range(num_batch):
            a = texta[i*conf.Batch_Size:(i+1)*conf.Batch_Size]
            b = textb[i*conf.Batch_Size:(i+1)*conf.Batch_Size]
            t = tag[i*conf.Batch_Size:(i+1)*conf.Batch_Size]
            yield a, b, t

    def get_length(self, trainX_batch):
        # sentence length
        lengths = []
        for sample in trainX_batch:
            count = 0
            for index in sample:
                if index != 0:
                    count += 1
                else:
                    break
            lengths.append(count)
        return lengths


    def model_restore(self, sess, tf_saver,model_path):
        '''
        模型恢复或者初始化
        :param sess: 
        :param tf_saver: 
        :return: 
        '''
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("restor model")
            tf_saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("init model")
            sess.run(tf.global_variables_initializer())

    def trainModel(self):
        train_texta_embedding, train_textb_embedding, train_tag, \
        dev_texta_embedding, dev_textb_embedding, dev_tag = self.pre_processing()
        # 定义训练用的循环神经网络模型

        if conf.model_name == 'esim':
            with tf.variable_scope('esim_model', reuse=None):
                # esim model
                model = esim_model.ESIM(True, seq_length=len(train_texta_embedding[0]),
                                        class_num=len(train_tag[0]),
                                        vocabulary_size=len(self.vocab),
                                        embedding_size=conf.embedding_size,
                                        hidden_num=conf.hidden_num,
                                        l2_lambda=conf.l2_lambda,
                                        learning_rate=conf.learning_rate)
        elif conf.model_name == 'bimpm':
            with tf.variable_scope('bimpm_model', reuse=None):
                # bimpm model
                model = bimpm_model.BIMPM(True, seq_length=len(train_texta_embedding[0]),
                                        class_num=len(train_tag[0]),
                                        vocabulary_size=len(self.vocab),
                                        embedding_size=conf.embedding_size,
                                        hidden_num=conf.hidden_num,
                                        l2_lambda=conf.l2_lambda,
                                        learning_rate=conf.learning_rate)
        elif conf.model_name == 'paircnn':
            with tf.variable_scope('paircnn', reuse=None):
                # paircnn model
                model = paircnn_model.PairCNN(True, seq_length=len(train_texta_embedding[0]),
                                        class_num=len(train_tag[0]),
                                        vocabulary_size=len(self.vocab),
                                        embedding_size=conf.embedding_size,
                                        hidden_num=conf.hidden_num,
                                        l2_lambda=conf.l2_lambda,
                                        learning_rate=conf.learning_rate)
        elif conf.model_name == 'abcnn':
            model = abcnn_model.ABCNN(True, seq_length=len(train_texta_embedding[0]),
                                          class_num=len(train_tag[0]),
                                          vocabulary_size=len(self.vocab),
                                          embedding_size=conf.embedding_size,
                                          hidden_num=conf.hidden_num,
                                          l2_lambda=conf.l2_lambda,
                                          learning_rate=conf.learning_rate)
        else:
            raise ValueError("model name is not right!")


        # 训练模型
        with tf.Session() as sess:
            saver = tf.train.Saver()
            self.model_restore(sess,saver,model_savepath)
            best_f1 = 0.0
            for time in range(conf.epoch):
                print("training " + str(time + 1) + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                model.is_trainning = True
                loss_all = []
                accuracy_all = []
                for texta, textb, tag in tqdm(
                        self.get_batches(train_texta_embedding, train_textb_embedding, train_tag)):
                    feed_dict = {
                        model.text_a: texta,
                        model.text_b: textb,
                        model.y: tag,
                        model.dropout_keep_prob: conf.dropout_keep_prob,
                        model.a_length: np.array(self.get_length(texta)),
                        model.b_length: np.array(self.get_length(textb))
                    }
                    _, cost, accuracy = sess.run([model.train_op, model.loss, model.accuracy], feed_dict)
                    loss_all.append(cost)
                    accuracy_all.append(accuracy)

                print("第" + str((time + 1)) + "次迭代的损失为：" + str(np.mean(np.array(loss_all))) + ";准确率为：" +
                      str(np.mean(np.array(accuracy_all))))

                def dev_step():
                    """
                    Evaluates model on a dev set
                    """
                    loss_all = []
                    accuracy_all = []
                    predictions = []
                    for texta, textb, tag in tqdm(
                            self.get_batches(dev_texta_embedding, dev_textb_embedding, dev_tag)):
                        feed_dict = {
                            model.text_a: texta,
                            model.text_b: textb,
                            model.y: tag,
                            model.dropout_keep_prob: 1.0,
                            model.a_length: np.array(self.get_length(texta)),
                            model.b_length: np.array(self.get_length(textb))
                        }
                        dev_cost, dev_accuracy, prediction = sess.run([model.loss, model.accuracy,
                                                                       model.prediction], feed_dict)
                        loss_all.append(dev_cost)
                        accuracy_all.append(dev_accuracy)
                        predictions.extend(prediction)
                    y_true = [np.nonzero(x)[0][0] for x in dev_tag]
                    y_true = y_true[0:len(loss_all)*conf.Batch_Size]
                    f1 = f1_score(np.array(y_true), np.array(predictions), average='weighted')
                    print('分类报告:\n', metrics.classification_report(np.array(y_true), predictions))
                    print("验证集：loss {:g}, acc {:g}, f1 {:g}\n".format(np.mean(np.array(loss_all)),
                                                                      np.mean(np.array(accuracy_all)), f1))
                    return f1

                model.is_trainning = False
                f1 = dev_step()

                if f1 > best_f1:
                    best_f1 = f1
                    saver.save(sess, model_savepath)
                    print("Saved model success\n")

if __name__ == '__main__':
    train = TrainModel()
    train.trainModel()