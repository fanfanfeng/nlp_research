# create by fanfan on 2017/8/28 0028
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
from seq2seq.seq2seq_dynamic import config,model_new
import data_utils
import model
import jieba
import numpy as np

def predict():
    with tf.Session() as sess:
        model_obj = model_new.Seq2SeqModel(config,'decode')
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
            if config.beam_with >1:
                predicted_sentence = model_obj.predict_beam_search(sess, np.array([token_ids_sentence]),
                                                       np.array([len(token_ids_sentence)]), vocab_list)
            else:
                predicted_sentence = model_obj.predict(sess,np.array([token_ids_sentence]),np.array([len(token_ids_sentence)]),vocab_list)
            print("输出:",predicted_sentence)


from collections import deque
_PAD = '_PAD'
def make_context_text(contex_one,context_two,sentence):
    total_list = []
    context_one_list = contex_one.strip().split(" ")
    context_one_list = [item for item in context_one_list if item!=""]
    if len(context_one_list) <= 10:
        context_one_list = context_one_list + ( 10 - len(context_one_list))*[_PAD]
    else:
        context_one_list = context_one_list[:10]
    total_list += context_one_list

    context_two_list = context_two.strip().split(" ")
    context_two_list = [item for item in context_two_list if item != ""]
    if len(context_two_list) <= 10:
        context_two_list = context_two_list + (10 - len(context_two_list)) * [_PAD]
    else:
        context_two_list = context_two_list[:10]
    total_list += context_two_list

    sentence_list = sentence.strip().split(" ")
    sentence_list = [item for item in sentence_list if item != ""]
    #if len(sentence_list) <= 10:
        #sentence_list = sentence_list + (10 - len(sentence_list)) * [_PAD]
    #else:
        #sentence_list = sentence_list[:10]
    total_list += sentence_list

    return " ".join(total_list)


def predict_with_context():
    d = deque(maxlen=3)
    d.append("")
    d.append("")

    with tf.Session() as sess:
        model_obj = model_new.Seq2SeqModel(config,'train')
        model_obj.batch_size = 1
        model_obj.model_restore(sess)


        vocab_path = config.source_vocabulary
        vocab,vocab_list = data_utils.initialize_vocabulary(vocab_path)

        while True:
            question = input("输入：")
            if question == "" or question == 'exit':
                break

            sentence =" ".join(list(jieba.cut(question)))
            d.append(sentence)
            sentence = make_context_text(*list(d))
            token_ids_sentence = data_utils.sentence_to_token_ids(sentence,vocab)
            if config.beam_with >1:
                predicted_sentence = model_obj.predict_beam_search(sess, np.array([token_ids_sentence]),
                                                       np.array([len(token_ids_sentence)]), vocab_list)
            else:
                predicted_sentence = model_obj.predict(sess,np.array([token_ids_sentence]),np.array([len(token_ids_sentence)]),vocab_list)
            print("输出:",predicted_sentence)
            d.append(predicted_sentence)


if __name__ == '__main__':
    predict()