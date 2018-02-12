# create by fanfan on 2017/10/11 0011
from category.tv import classfication_setting
import tensorflow as tf
from category.tv import data_util
import jieba
import numpy as np
from collections import OrderedDict
import  time

class Meta_Load():
    def __init__(self):
        start = time.time()

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        ))
        self.load_model(self.sess)
        self._init__tensor()
        end = time.time()
        print("time Cost load model:%f",end - start)
        self.word2id_dict = data_util.load_word2id()


    def _init__tensor(self):
        self.drouput_tensor = self.sess.graph.get_operation_by_name("dropout").outputs[0]
        self.input_x_tensor = self.sess.graph.get_operation_by_name("Placeholder").outputs[0]

        self.logit_tensor = self.sess.graph.get_operation_by_name("softmax_layer/logits").outputs[0]

    def predict(self,text):
        words = list(jieba.cut(text))
        words = " ".join(words)
        print(words)
        tokens = [int(self.word2id_dict[token]) for token in words.split(" ") if token != "" and token in self.word2id_dict]
        print(tokens)
        if len(tokens) < classfication_setting.max_document_length:
            tokens = tokens + [0] * (classfication_setting.max_document_length - len(tokens))
        else:
            tokens = tokens[:20]
        tokens.reverse()

        feed_dict = {}
        feed_dict[self.drouput_tensor] = 1.0
        feed_dict[self.input_x_tensor] = np.array([tokens])

        logit_result = self.sess.run(self.logit_tensor,feed_dict=feed_dict)
        result = { classfication_setting.index2label[i]: value for i, value in enumerate(logit_result[0])}
        order_dict = OrderedDict(sorted(result.items(), key=lambda t: t[1], reverse=True))
        print(order_dict.items())

    def load_model(self,sess):
        with tf.gfile.FastGFile(classfication_setting.graph_model_bi_lstm+"weight_classify.pb",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name="")




def predict():

    model_obj = Meta_Load()
    while True:
        text = input("请输入句子")
        model_obj.predict(text)





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

