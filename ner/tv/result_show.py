# create by fanfan on 2017/12/1 0001
from sklearn.metrics import classification_report
from ner.tv import ner_setting
import tensorflow as tf
import numpy as np
from ner.tv import blstm_crf
from ner.tv import data_util
import time

predict_label = [key for key in ner_setting.tag_to_id.keys()]
def predict():
    data_set = data_util.read_data(ner_setting.tv_data_path,test=True)
    input_x = data_set['input_x']
    input_y = data_set['input_y']

    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        model = blstm_crf.Model()
        model.model_restore(sess)
        predict,real_inputs_y = model.predict(sess,input_x,input_y)

        make_hole_predict = []
        for i in predict:
            make_hole_predict.extend(i)
        make_hole_real = []
        for i in real_inputs_y:
            make_hole_real.extend(i)
        print(classification_report(make_hole_real,make_hole_predict,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],target_names=predict_label))







if __name__ == '__main__':
    predict()


