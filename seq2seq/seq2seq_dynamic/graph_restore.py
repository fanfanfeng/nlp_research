# create by fanfan on 2017/10/11 0011
from seq2seq.seq2seq_dynamic import config
import tensorflow as tf
from seq2seq.seq2seq_dynamic import data_utils
import jieba
import numpy as np
from collections import OrderedDict
import  time

class Meta_Load():
    def __init__(self):
        self.sess = tf.Session()
        self.load_model(self.sess)
        self._init__tensor()
        vocab_path = config.source_vocabulary
        self.vocab, self.vocab_list = data_utils.initialize_vocabulary(vocab_path)




    def _init__tensor(self):
        self.encoder_inputs_length_tensor = self.sess.graph.get_operation_by_name('encoder_inputs_length').outputs[0]
        self.encoder_inputs_tensor = self.sess.graph.get_operation_by_name('encoder_inputs').outputs[0]


        self.keep_prob_tensor = self.sess.graph.get_operation_by_name("keep_prob").outputs[0]

        self.decoder_tensor = self.sess.graph.get_operation_by_name("decoder/ExpandDims").outputs[0]

    def predict(self,text):
        words_list = list(jieba.cut(text))
        words = " ".join(words_list)

        token_ids_sentence = data_utils.sentence_to_token_ids(words, self.vocab)

        feed_dict = {}
        feed_dict[self.keep_prob_tensor] = 1.0
        feed_dict[self.encoder_inputs_tensor] = np.array([token_ids_sentence])
        feed_dict[self.encoder_inputs_length_tensor] = np.array([len(token_ids_sentence)])

        predicts = self.sess.run(self.decoder_tensor, feed_dict)

        outputs = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        for token in predicts[0]:
            selected_token_id = int(token)
            if selected_token_id == data_utils.EOS_ID or selected_token_id == data_utils.PAD_ID:
                break
            else:
                outputs.append(selected_token_id)

        # Forming output sentence on natural language
        output_sentence = " ".join([self.vocab_list[output] for output in outputs])
        return output_sentence




    def load_model(self,sess):
        with tf.gfile.FastGFile(config.model_dir+"weight_seq2seq.pb",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with sess.graph.as_default():
                tf.import_graph_def(graph_def,name="")




def predict():

    model_obj = Meta_Load()
    while True:
        text = input("请输入句子:")
        print(model_obj.predict(text))





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

