# create by fanfan on 2017/12/1 0001
from sklearn.metrics import classification_report
from category.tv import classfication_setting
from category.tv import graph_restore
import tensorflow as tf
import numpy as np
from category.tv import bi_lstm_model_attention
from category.tv import bi_lstm_attention_and_cnn_model
from category.tv import data_util
import time

predict_label = classfication_setting.label_list
def predict():
    data_set = data_util.read_data(classfication_setting.tv_data_path,test=True)
    input_x = data_set['input_x']
    input_y = data_set['input_y']

    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = bi_lstm_model_attention.Bi_lstm()
        model.restore_model(sess)
        predict,_ = model.predict(sess,input_x)
        print(classification_report(input_y,predict,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],target_names=predict_label))







if __name__ == '__main__':
    predict()


