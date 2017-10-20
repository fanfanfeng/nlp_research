# create by fanfan on 2017/10/17 0017

vocab_pkl = r'data/vocab.pkl'

pos_data_path = r'data/pos_t.txt'
neg_data_path = r'data/neg_t.txt'


data_dir = r'data/'


max_doc_len = 30
max_sentence_len = 50
batch_size = 12
epochs = 100
rnn_size = 300
word_attention_size = 300
sent_attention_size = 300
char_embedding_size = 300
model_save_dir = 'Model/han_model/'
vocab_size = 6790
keep_prob = 0.5
learning_rate = 0.001
grad_clip = 5.0

#tensorboard log文件夹
graph_log_path = 'graph/'