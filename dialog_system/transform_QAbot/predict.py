# create by fanfan on 2017/10/11 0011

import tensorflow as tf
from dialog_system.transform_QAbot import data_process
from dialog_system.transform_QAbot import data_utils
import jieba
import numpy as np
from collections import OrderedDict
import  time

class Meta_Load():
    def __init__(self):
        self.sess = tf.Session()
        self.load_model(self.sess)
        self._init__tensor()
        data_processer = data_process.NormalData("", output_path="output/")
        self.vocab, self.vocab_list =  data_processer.load_vocab_and_intent()
        self.max_sentence_length = 50




    def _init__tensor(self):
        self.input = self.sess.graph.get_operation_by_name('input').outputs[0]
        self.target = self.sess.graph.get_operation_by_name('target').outputs[0]

        self.decoder_tensor = self.sess.graph.get_operation_by_name("predict").outputs[0]

    def predict(self,text):
        input_ids_sentence = data_utils.pad_sentence(text.split(" "), self.max_sentence_length,self.vocab)
        output_ids_sentence = data_utils.pad_sentence("",self.max_sentence_length,self.vocab)


        answer = ""

        for i in range(self.max_sentence_length):
            if  1 == 0:
                output_ids_sentence = data_utils.pad_sentence(answer,self.max_sentence_length,self.vocab)
            feed_dict = {}
            feed_dict[self.input] = np.array([input_ids_sentence])
            feed_dict[self.target] = np.array([output_ids_sentence])

            predicts = self.sess.run(self.decoder_tensor, feed_dict)
            an1 = answer
            answer = self.pred_next_string(predicts[0])
            print("answer:" + " ".join(answer))
            if answer == an1:
                break
        return answer




    def load_model(self,sess):
        with tf.gfile.FastGFile("output/transform.pb",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with sess.graph.as_default():
                tf.import_graph_def(graph_def,name="")


    def pred_next_string(self,token_list):
        answer = []
        for word_id in token_list:
            if word_id < 3:
                break
            else:
                answer.append(self.vocab_list[word_id])
        return answer

def predict():

    model_obj = Meta_Load()
    while True:
        text = input("请输入句子:")
        print(model_obj.predict(text))





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

