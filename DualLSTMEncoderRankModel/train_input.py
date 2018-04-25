# create by fanfan on 2018/4/19 0019
from DualLSTMEncoderRankModel import data_util
from DualLSTMEncoderRankModel import config
from DualLSTMEncoderRankModel.data_process.corpus import xiaohuangjidata
import numpy as np
from sklearn.model_selection import train_test_split
import random

import os

def data_to_token_ids(data_path,target_path,vocabulary_path,tokenizer=None):
    """
        #根据生成的词汇库，将data_path里面的句子全部转换为id的形式，并生成新的文件
      Args:
        data_path: 原始数据地址
        target_path: 处理后数据保存位置
        vocabulary_path: 词库文件的路径
        tokenizer: 分词性
      """
    if not os.path.exists(target_path):
        print('生成转换为id以后的训练数据')
        vocab_dict,_ = data_util.load_vocabulary_from_file(vocabulary_path)
        with open(data_path,mode='r',encoding='utf-8') as data_file:
            with open(target_path,mode='w',encoding='utf-8') as tokens_file:
                count = 0
                for line in data_file:
                    line = line.strip()
                    count +=1
                    if count % 10000 == 0:
                        print("已经处理了%d 行" % count)
                    seq_list = line.split("###")

                    seqids_list = []
                    for seq in seq_list:
                        token_ids = data_util.sentence_to_token_ids(seq,vocab_dict,tokenizer)
                        if len(token_ids) < config.max_seq_len:
                            token_ids += [data_util.PAD_ID] * (config.max_seq_len - len(token_ids))
                        else:
                            token_ids = token_ids[:config.max_seq_len]
                        seqids_list += token_ids
                    tokens_file.write(" ".join([str(tok) for tok in seqids_list]) + '\n')

    else:
        print("{} 文件已经存在，无需重新生成".format(target_path))


def prepare_data_for_model_trainning():
    """加载训练数据，生成词库，生成相应的文件
      """
    # 将原始语料进行处理，变成一问一答的形式
    if not os.path.exists(config.corpus_processed_path):
        xiaohuangjidata.load_no_processed_xiaohuangji(config.corpus_data_path, config.corpus_processed_path)

    train_data_path = config.corpus_processed_path

    # 根据config生成词库文件
    data_util.create_vocabulary(config.vocabulary_path,train_data_path)

    # 根据词库生成相应的id形式文件
    data_to_token_ids(train_data_path,config.corpus_to_id_path,config.vocabulary_path)


class Batch:
    def __init__(self):
        # nice summary of various sequence required by chatbot training.
        self.query_seqs = []
        self.response_seqs = []
        # self.xxlength：encoding阶段调用 dynamic_rnn 时的一个 input_argument
        self.query_length = []
        self.response_length = []

class BatchManager():
    def __init__(self,data_path):
        self.data_path = data_path
        self.total_data = self.load_data()
        self.total_length = self.total_data.shape[0]

        #创建训练集，测试集
        self.make_train_valid_test()
        #创建样本对于top 集合
        self._sampleNegative()



    def load_data(self):
        total_data = []
        with open(self.data_path,'r',encoding='utf-8') as fread:
            for line in fread:
                item = []
                line = line.strip().split(" ")
                if len(line) != 40:
                    assert ValueError("len not equal 40")
                item.append(line[:20])
                item.append(line[20:])
                total_data.append(item)
        return np.array(total_data)

    def make_train_valid_test(self):
        shuffle_index = np.random.permutation(np.arange(self.total_length))

        self.total_data = self.total_data[shuffle_index]
        self.train_length = int(self.total_length * .8)
        self.valid_length = int(self.total_length * .1)
        self.test_length = self.total_length - self.train_length - self.valid_length

        self.train_data = self.total_data[:self.train_length]
        self.valid_data = self.total_data[self.train_length : self.train_length + self.valid_length]
        self.test_data = self.total_data[self.train_length + self.valid_length:]

    def _sampleNegative(self):
        """ 
        对于 valid_data/test_data 里面的每一个数据，生成 [ ranksiez -1]个反例子，为后面调用
        Recall@top k做准备

        """
        def npSampler(n):
            """Generate [n x ranksize-1] np.array
            """
            neg = np.zeros(shape = (n, config.ranksize-1))
            for i in range(config.ranksize-1):
                neg[:,i] = np.arange(n)
                np.random.shuffle(neg[:,i])
            findself = neg - np.arange(n).reshape([n, 1])
            findzero = np.where(findself==0)
            for (r, c) in zip(findzero[0], findzero[1]):
                x = np.random.randint(n)
                while x == r:
                    x = np.random.randint(n)
                neg[r, c] = x
            return neg.astype(int)

        self.valid_neg_idx = npSampler(self.valid_length)
        self.test_neg_idx = npSampler(self.test_length)


    def _createBatch(self, samples):
        """创建train的Batch
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]
            sample[0] = [int(item) for item in sample[0]]
            sample[1] = [int(item) for item in sample[1]]

            batch.query_seqs.append(sample[0])
            batch.response_seqs.append(sample[1])
            batch.query_length.append(len([int(item) for item in sample[0] if item!=0]))
            batch.response_length.append(len([int(item) for item in sample[1] if item!=0]))
        return batch


    def _createEvalBatch(self, samples, dataset, neg_responses):
        """
        创建valid或者test的batch
        """
        batch = Batch()
        batchSize = len(samples)
        for i in range(batchSize):
            sample = samples[i]
            sample[0] = [int(item) for item in sample[0]]
            sample[1] = [int(item) for item in sample[1]]
            batch.query_seqs.append(sample[0])
            batch.query_length.append(len([int(item) for item in sample[0] if item!=0]))

            batch.response_seqs.append(sample[1])
            batch.response_length.append(len(sample[1]))

            for j in range(config.ranksize-1):
                sample = dataset[neg_responses[i][j]]
                sample[1] = [int(item) for item in sample[1]]
                batch.response_seqs.append(sample[1])
                batch.response_length.append(len([int(item) for item in sample[1] if item!=0]))
        return batch


    def get_training_batches(self):
        """
        返回 training batch
        """
        random.shuffle(self.train_data)
        batches = []
        def genNextSamples():
            for i in range(0, self.train_length, config.batch_size):
                yield self.train_data[i:min(i + config.batch_size, self.train_length)]
        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    def get_test_batches(self):
        """
        返回test batch
        """
        if self.test_neg_idx == []:
            self._sampleNegative()

        batches = []
        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.test_length, config.batch_size):
                yield (self.test_data[i:min(i + config.batch_size, self.test_length)],
                       self.test_neg_idx[i:min(i + config.batch_size, self.test_length), :])

        for (samples, neg_responses) in genNextSamples():
            batch = self._createEvalBatch(samples, self.test_data, self.test_neg_idx)
            batches.append(batch)
        return batches

    def get_valid_batches(self):
        """
        返回valid batch
        """
        if self.valid_neg_idx == []:
            self._sampleNegative()

        batches = []
        def genNextSamples():
            for i in range(0, self.valid_length, config.batch_size):
                yield (self.valid_data[i:min(i + config.batch_size, self.valid_length)],
                       self.valid_neg_idx[i:min(i + config.batch_size, self.valid_length), :])

        for (samples, neg_responses) in genNextSamples():
            batch = self._createEvalBatch(samples,self.valid_data, self.valid_neg_idx)
            batches.append(batch)
        return batches

    def get_predict_batch_response(self):
        batch = Batch()
        predictingResponseSeqs = []
        predictingResponseLength = []

        for sample in self.total_data:
            sample = sample[1]
            sample = [int(item) for item in sample]
            predictingResponseSeqs.append(sample)
            predictingResponseLength.append(len([item for item in sample if item !=0]))
            break
        batch.response_seqs = predictingResponseSeqs
        batch.response_length = predictingResponseLength
        return batch



if __name__ == '__main__':
    BatchManager(config.corpus_to_id_path)