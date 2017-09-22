# create by fanfan on 2017/8/28 0028
import tensorflow as tf
from src.chatbot.simple_seq2seq import config
from src.chatbot.simple_seq2seq import data_utils
from src.chatbot.simple_seq2seq import seq2seq_model
import os
import jieba

def predict():
    with tf.Session() as sess:
        model = seq2seq_model.Seq2SeqModel.model_create_or_restore(forward_only=True,session=sess)
        model.batch_size = 1

        vocab_path = os.path.join(config.data_dir,"vocab%d.in" % config.vocabulary_size)
        vocab,rec_vocab = data_utils.initialize_vocabulary(vocab_path)

        while True:
            question = input("输入：")
            if question == "" or question == 'exit':
                break
            sentence = " ".join(list(jieba.cut(question)))
            predicted_sentence = model.get_predicted_sentence(sentence,vocab,rec_vocab,model,sess)
            print("输出:",predicted_sentence)

if __name__ == '__main__':
    predict()