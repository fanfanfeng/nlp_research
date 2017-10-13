# create by fanfan on 2017/7/7 0007
import tensorflow as tf
import numpy as np
from category.tv import  classfication_setting
from category.tv import bi_lstm_model
import jieba
from category.tv import data_util
from collections import OrderedDict
import time
def predict():

    index2label = {i: l.strip() for i, l in enumerate(classfication_setting.label_list)}

    word2id_dict = data_util.load_word2id()

    start = time.time()
    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = bi_lstm_model.Bi_lstm()
        model.restore_model(sess)
        end = time.time()
        print("time cost load model:",end - start)
        text = ""#input("请输入句子：")
        while text:
            words = list(jieba.cut(text))
            words = " ".join(words)
            print(words)
            tokens = [int(word2id_dict[token]) for token in words.split(" ") if token != "" and token in word2id_dict]

            if len(tokens) < classfication_setting.max_document_length:
                tokens = tokens + [0] * (classfication_setting.max_document_length - len(tokens))
            else:
                tokens = tokens[:20]
            tokens.reverse()
            x_test = np.array([tokens])
            predict_one,logit = model.predict(sess,x_test)
            result = {index2label[i]: value for i, value in enumerate(logit[0])}
            order_dict = OrderedDict(sorted(result.items(), key=lambda t: t[1], reverse=True))
            print(order_dict.items())
            text = input("请输入句子：")







if __name__ == '__main__':
    for i in range(10):
        predict()


