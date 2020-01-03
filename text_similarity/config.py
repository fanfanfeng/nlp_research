import os
class Config(object):

    def __init__(self):
        self.embedding_size = 200  # 词向量维度
        self.hidden_num = 150  # 隐藏层规模
        self.l2_lambda = 0.0001
        self.learning_rate = 0.0001
        self.dropout_keep_prob = 0.5
        self.attn_size = 200
        self.K = 2

        self.epoch = 20
        self.Batch_Size = 100

        self.max_sentence_len = 20

        self.model_name="abcnn" # bimpm,esim, paircnn ,abcnn

    @property
    def save_path(self):
        return os.path.join(self.model_name,"model.ckpt")
