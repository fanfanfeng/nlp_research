__author__ = 'fanfan'

###### generator setting #######
embedding_dim = 32 # embedding dimension
hidden_dim = 32 # hidden state dimension of lstm cell
seq_length = 20 # sequence length
start_token = 0
per_epoch_num = 120 # supervise (maximum likelihood estimation) epochs
random_seed = 88
batch_size = 64
learning_rate = 0.01
reward_gamma = 0.95
temperature = 1.0
grad_clip = 5.0
vocab_size = 5000