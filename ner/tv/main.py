# create by fanfan on 2017/7/26 0026
from ner.tv import ner_setting
from ner.tv import data_util
import tensorflow as tf
from ner.tv import blstm_crf
import jieba

def train():
    data_manager = data_util.BatchManager(ner_setting.tv_data_path, ner_setting.batch_size)

    with tf.Session() as sess:
        model = blstm_crf.Model()
        model.model_restore(sess)

        if 1:
            output_tensor = []
            output_tensor.append(model.trans.name.replace(":0", ""))
            output_tensor.append(model.lengths.name.replace(":0", ""))
            output_tensor.append(model.logits.name.replace(":0",""))
            output_tensor.append(model.dropout.name.replace(":0",""))

            output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_tensor)
            with tf.gfile.FastGFile(os.path.join(ner_setting.train_model_bi_lstm, "weight_ner.pb"),
                                    'wb') as gf:
                gf.write(output_graph_with_weight.SerializeToString())

        for epoch in range(model.train_epoch):
            print("start epoch {}".format(str(epoch)))
            average_loss = 0
            for train_inputs,train_labels in data_manager.train_iterbatch():
                step,loss = model.run_step(sess,train_inputs,train_labels,True)
                average_loss += loss
                if step % ner_setting.show_every == 0:
                    average_loss = average_loss / ner_setting.show_every
                    print("iteration:{} step:{},NER loss:{:>9.6f}".format(epoch,  step, average_loss))
                    average_loss = 0

                if step % ner_setting.valid_every == 0:
                    total_accuracy = 0
                    total_batch = 0
                    for test_inputs,test_labels  in data_manager.test_iterbatch():
                        accuracy = model.test_accuraty(sess,test_inputs,test_labels)
                        total_accuracy += accuracy
                        total_batch +=1
                    if  total_batch !=0 :
                        mean_accuracy = total_accuracy/total_batch
                    else:
                        mean_accuracy = 0
                    print("iteration:{},NER accuracy:{:>9.6f}".format(epoch, mean_accuracy))
                    model.saver.save(sess, model.model_save_path, global_step=step)


import pickle
import os
import numpy as np
def predict(text):
    if os.path.exists(ner_setting.word2id_path):
        word2id_dict = data_util.load_word2id()
        words_list = list(text)
        words_list_id = [word2id_dict[i] for i in words_list]
        text_len = len(words_list_id)
        if text_len < ner_setting.max_document_length:
            words_list_id += [0] * (ner_setting.max_document_length - text_len)

        inputs = np.array(words_list_id).reshape([1, ner_setting.max_document_length])

        with tf.Session() as sess:
            model = blstm_crf.Model()
            model.model_restore(sess)

            fenchiResult = {
                "Command":"",
                "Person":"",
                "Place":"",
                "Language":"",
                "Time":"",
                "Episode": "",
                "MajorNoun": "",
                "Category": "",
            }

            path = model.predict(sess,inputs)
            for word,seg_id in zip(words_list,path[0]):
                if seg_id == 1:
                    fenchiResult["Command"] += word
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






if __name__ == '__main__':
    state = 0
    if state == 0:
        train()
    else:
        text = u'想看郭德纲的电影东风破'
        result = predict(text)
        print("分词前：" + text)
        print("分词后：" )
        for key in result:
            print("{} : {}".format(key,result[key]))
