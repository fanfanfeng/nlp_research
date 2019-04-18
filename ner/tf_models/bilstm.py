# create by fanfan on 2019/4/17 0017
import tensorflow as tf

class BiLSTM():
    def __init__(self,config):
        self.num_hidden = config.numHidden
        self.num_tags = config.numTags
        self.max_seq_len = config.max_seq_len

        


