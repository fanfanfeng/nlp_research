# create by fanfan on 2017/8/28 0028
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
import config
import data_utils
import model
import jieba
import numpy as np

def predict():
    with tf.Session() as sess:
        model_obj = model.Seq2SeqModel('decode')
        model_obj.batch_size = 1
        model_obj.model_restore(sess)

        vocab_path = config.source_vocabulary
        vocab,vocab_list = data_utils.initialize_vocabulary(vocab_path)

        while True:
            question = input("输入：")
            if question == "" or question == 'exit':
                break
            sentence =" ".join(list(jieba.cut(question)))
            token_ids_sentence = data_utils.sentence_to_token_ids(sentence,vocab)
            predicted_sentence = model_obj.predict(sess,np.array([token_ids_sentence]),np.array([len(token_ids_sentence)]),vocab_list)
            print("输出:",predicted_sentence)

if __name__ == '__main__':
    predict()