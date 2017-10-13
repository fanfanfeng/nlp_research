# create by fanfan on 2017/10/11 0011
from ner.tv import ner_setting
import tensorflow as tf
from ner.tv import data_util
import jieba
import numpy as np
from collections import OrderedDict
import  time
from tensorflow.contrib import crf

class Meta_Load():
    def __init__(self):
        self.sess = tf.Session()
        self.load_model(self.sess)
        self._init__tensor()
        self.word2id_dict = data_util.load_word2id()




    def _init__tensor(self):
        self.dropout_tensor = self.sess.graph.get_operation_by_name("dropout").outputs[0]
        self.input_x_tensor = self.sess.graph.get_operation_by_name("inputs").outputs[0]

        self.length_tensor = self.sess.graph.get_operation_by_name('word2vec_embedding/Cast').outputs[0]
        self.logit_tensor = self.sess.graph.get_operation_by_name("Reshape").outputs[0]
        self.tran_tensor = self.sess.graph.get_operation_by_name('crf_loss/transitions').outputs[0]

    def predict(self,text):
        words_list = list(jieba.cut(text))
        words = " ".join(words_list)
        tokens = [int(self.word2id_dict[token]) for token in words.split(" ") if token != "" and token in self.word2id_dict]

        if len(tokens) < ner_setting.max_document_length:
            tokens = tokens + [0] * (ner_setting.max_document_length - len(tokens))
        else:
            tokens = tokens[:ner_setting.max_document_length]

        feed_dict = {}
        feed_dict[self.dropout_tensor] = 1.0
        feed_dict[self.input_x_tensor] = np.array([tokens])

        fetch_dict = [self.tran_tensor,self.length_tensor,self.logit_tensor]
        crf_trans_matrix,lengths,scores = self.sess.run(fetch_dict,feed_dict=feed_dict)
        paths = []
        for score, length in zip(scores, lengths):
            score = score[:length]
            path, _ = crf.viterbi_decode(score, crf_trans_matrix)
            paths.append(path[:length])

        fenchiResult = {
            "Command": "",
            "Person": "",
            "Place": "",
            "Language": "",
            "Time": "",
            "Episode": "",
            "MajorNoun": "",
            "Category": "",
        }
        for word, seg_id in zip(words_list, paths[0]):
            if seg_id == 1:
                fenchiResult["Commandf"] += word
            elif seg_id == 2:
                fenchiResult["Command"] += word + " "
            elif seg_id == 3:
                fenchiResult["Person"] += word
            elif seg_id == 4:
                fenchiResult["Person"] += word + " "
            elif seg_id == 5:
                fenchiResult["Place"] += word
            elif seg_id == 6:
                fenchiResult["Place"] += word + " "
            elif seg_id == 7:
                fenchiResult["Language"] += word
            elif seg_id == 8:
                fenchiResult["Language"] += word + " "
            elif seg_id == 9:
                fenchiResult["Time"] += word
            elif seg_id == 10:
                fenchiResult["Time"] += word + " "
            elif seg_id == 11:
                fenchiResult["Episode"] += word
            elif seg_id == 12:
                fenchiResult["Episode"] += word + " "
            elif seg_id == 13:
                fenchiResult["MajorNoun"] += word
            elif seg_id == 14:
                fenchiResult["MajorNoun"] += word + " "
            elif seg_id == 15:
                fenchiResult["Category"] += word
            elif seg_id == 16:
                fenchiResult["Category"] += word + " "

        return fenchiResult

    def load_model(self,sess):
        with tf.gfile.FastGFile(ner_setting.train_model_bi_lstm+"weight_seq2seq.pb",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with sess.graph.as_default():
                tf.import_graph_def(graph_def,name="")




def predict():

    model_obj = Meta_Load()
    while True:
        text = input("请输入句子")
        print(model_obj.predict(text))





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

