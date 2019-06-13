# create by fanfan on 2017/10/11 0011
from dialog_system.attention_seq2seq import data_utils
from dialog_system.attention_seq2seq.params import Params,TestParams
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
import os
import tensorflow as tf
import jieba
import numpy as np




class Meta_Load():
    def __init__(self,params):
        self.sess = tf.Session()
        self.params = params
        self.load_model(self.sess)
        self._init__tensor()
        self.vocab, self.vocab_list = self.load_vocab_and_intent()

    def load_vocab_and_intent(self):
        vocab_list = []
        with open(os.path.join(self.params.output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())

        return {key: index for index, key in enumerate(vocab_list)}, vocab_list


    def _init__tensor(self):
        self.encoder_inputs = self.sess.graph.get_operation_by_name('input').outputs[0]
        self.decoder_tensor = self.sess.graph.get_operation_by_name("predicts").outputs[0]


    def predict(self,text):
        words_list = list(jieba.cut(text))

        token_ids_sentence,_ = data_utils.pad_sentence(words_list,self.params.max_seq_length,self.vocab)

        feed_dict = {}
        feed_dict[self.encoder_inputs] = np.array([token_ids_sentence])

        predicts = self.sess.run(self.decoder_tensor, feed_dict)

        outputs = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        if self.params.beam_with <= 1:
            outputs.append(self.ids2words(predicts[0],self.vocab_list))
        else:
            for i in range(self.params.beam_with):
                ids = predicts[0][:,i]
                outputs.append(self.ids2words(ids,self.vocab_list))

        # Forming output sentence on natural language
        output_sentence = "\n".join(outputs)
        return output_sentence


    def ids2words(self,seq, ids2words):
        words = []
        for w in seq:
            if w == data_utils.EOS_ID or w == data_utils.PAD_ID:
                break
            words.append(ids2words[w])
        return ' '.join(words)



    def load_model(self,sess):
        with tf.gfile.FastGFile(os.path.join(self.params.output_path,"transform.pb"),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with sess.graph.as_default():
                tf.import_graph_def(graph_def,name="")




def predict():
    params = TestParams()
    model_obj = Meta_Load(params)
    while True:
        text = input("请输入句子:")
        print(model_obj.predict(text))





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

