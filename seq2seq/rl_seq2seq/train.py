# create by fanfan on 2017/12/27 0027
import tensorflow as tf
from seq2seq.rl_seq2seq import  data_utils
from seq2seq.rl_seq2seq import setting

def train():
    tran_batch_manager = data_utils.BatchManager("data/train.txt.id40000.in", setting.batch_size)
    test_batch_manager = data_utils.BatchManager("data/test.txt.id40000.in", setting.batch_size)


if __name__ == '__main__':
    train()