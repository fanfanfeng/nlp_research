# create by fanfan on 2017/8/25 0025
import os
import re
import sys
from tensorflow.python.platform import gfile
import numpy as np

from seq2seq.seq2seq_dynamic.movie_name import config

_PAD = '_PAD'
_GO = '_GO'
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#去除各种特殊符号
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

def basic_tokenizer(sentence):
    words = sentence.strip().split(" ")
    return [w for w in words if re.split(_WORD_SPLIT,w)]

#根据训练集，生成词汇库
def create_vocabulary(vocabulary_path,data_path,max_vocab_size,tokenizer = None):
    """Create vocabulary file (if it does not exist yet) from data-en file.

      Data file is assumed to contain one sentence per line. Each sentence is
      tokenized and digits are normalized (if normalize_digits is set).
      Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
      We write it to vocabulary_path in a one-token-per-line format, so that later
      token in the first line gets id=0, second line gets id=1, and so on.

      Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data-en file that will be used to create vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data-en sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
      """
    if not gfile.Exists(vocabulary_path):
        print("从训练集 %s 中生成 词汇库 %s" % (data_path,vocabulary_path))
        vocab = {}
        with gfile.GFile(data_path,mode='rb') as f:
            count = 0
            for line in f:
                line = line.decode('utf-8',errors="ignore")
                count +=1
                if count % 10000 == 0:
                    print("处理了数据：%d" % count)

                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            vocab_list = _START_VOCAB + sorted(vocab,key=vocab.get,reverse=True)
            print("过滤器，词库大小：%d" % len(vocab_list))
            if len(vocab_list) > max_vocab_size:
                vocab_list = vocab_list[:max_vocab_size]

            with gfile.GFile(vocabulary_path,mode='wb') as vocab_write:
                for w in vocab_list:
                    vocab_write.write((w + "\n").encode())

#根据vocabulary_path生成 vocab列表
def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

      We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
      will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
      also return the reversed-vocabulary ["dog", "cat"].

      Args:
        vocabulary_path: path to the file containing the vocabulary.

      Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).

      Raises:
        ValueError: if the provided vocabulary_path does not exist.
      """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path,mode='r') as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict((x,y) for (y,x) in enumerate(rev_vocab))
        return  vocab,rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found",vocabulary_path)

#根据vocab，将一句话转换为id形式
def sentence_to_token_ids(sentence,vocabulary,tokenizer=None):
    """Convert a string to list of integers representing token-ids.

     For example, a sentence "I have a dog" may become tokenized into
     ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
     "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

     Args:
       sentence: a string, the sentence to convert to token-ids.
       vocabulary: a dictionary mapping tokens to integers.
       tokenizer: a function to use to tokenize each sentence;
         if None, basic_tokenizer will be used.
       normalize_digits: Boolean; if true, all digits are replaced by 0s.

     Returns:
       a list of integers, the token-ids for the sentence.
     """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)

    return [vocabulary.get(w,UNK_ID) for w in words]

#根据生成的词汇库，将data_path里面的句子全部转换为id的形式，并生成新的文件
def data_to_token_ids(data_path,target_path,vocabulary_path,tokenizer=None):
    """Tokenize data-en file and turn into token-ids using given vocabulary file.

      This function loads data-en line-by-line from data_path, calls the above
      sentence_to_token_ids, and saves the result to target_path. See comment
      for sentence_to_token_ids on the details of token-ids format.

      Args:
        data_path: path to the data-en file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
      """
    if not gfile.Exists(target_path):
        vocab,_ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path,mode='rb') as data_file:
            with gfile.GFile(target_path,mode='wb') as tokens_file:
                count = 0
                for line in data_file:
                    line = line.decode(errors="ignore")
                    count +=1
                    if count % 10000 == 0:
                        print("已经处理了%d 行" % count)

                    token_ids = sentence_to_token_ids(line,vocab,tokenizer)
                    tokens_file.write((" ".join([str(tok) for tok in token_ids]) + "\n").encode())

#为模型训练准备好数据
def prepare_data_for_model():
    """Get dialog data into data_dir, create vocabularies and tokenize data-en.

      Returns:
        A tuple of 3 elements:
          (1) path to the token-ids for chat training data-en-set,
          (2) path to the token-ids for chat development data-en-set,
          (3) path to the chat vocabulary file
      """
    # Get dialog data-en to the specified directory.
    train_path = os.path.join(config.data_dir,config.train_file)
    test_path = os.path.join(config.data_dir,config.test_file)

    vocab_path = os.path.join(config.data_dir,'vocab%d.in' % config.vocabulary_size)
    create_vocabulary(vocab_path,train_path,config.vocabulary_size)

    train_ids_path = train_path + (".id%d.in" %config.vocabulary_size)
    data_to_token_ids(train_path,train_ids_path,vocab_path)

    test_ids_path = test_path + (".id%d.in" %config.vocabulary_size)
    data_to_token_ids(test_path,test_ids_path,vocab_path)
    return (train_ids_path,test_ids_path,vocab_path)



def pad_sentence_batch(sentence_batch, pad_int,max_sentence):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def read_data(tokenized_data_path,max_size=15):
    """Read data-en from source file and put into buckets.

      Args:
        source_path: path to the files with token-ids.
        max_size: maximum number of lines to read, all other will be ignored;
          if 0 or None, data-en files will be read completely (no limit).

      """
    data_set = dict()
    encode =[]
    encode_length = []
    decode =[]
    decode_length = []
    with gfile.GFile(tokenized_data_path,mode='r') as fh:
        source,target = fh.readline(),fh.readline()
        counter = 0
        while source and target:
            counter += 1
            if counter % 10000 == 0:
                print("已经读取了%d 行" % counter)

            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            #target_ids.append(EOS_ID)

            if len(source_ids) < max_size and  len(target_ids) < max_size :

                encode_length.append(len(source_ids))
                decode_length.append(len(target_ids) )
                #对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
                encode.append(source_ids)
                #encode.append(list(reversed(source_ids + [PAD_ID] * (max_size - len(source_ids)))))
                decode.append(target_ids)
                #decode.append([GO_ID] + target_ids  + [PAD_ID] * (max_size - len(target_ids))+ [EOS_ID])
            source,target = fh.readline(),fh.readline()

    data_set['encode'] = np.asarray(encode)
    data_set['encode_lengths'] = np.asarray(encode_length)

    data_set['decode'] = np.asarray(decode)
    data_set['decode_lengths'] = np.asarray(decode_length)
    return data_set

class BatchManager():
    def __init__(self,data_path,batch_size):
        data_set = read_data(data_path)
        self.encode = data_set['encode']
        self.encode_lengths = data_set['encode_lengths']
        self.decode = data_set['decode']
        self.decode_lengths = data_set['decode_lengths']
        self.total = len(self.encode)
        self.batch_size = batch_size
        self.num_batch = int(self.total/self.batch_size)

    def shuffle(self):
        shuffle_index = np.random.permutation(np.arange(self.total))
        self.encode = self.encode[shuffle_index]
        self.encode_lengths = self.encode_lengths[shuffle_index]
        self.decode = self.decode[shuffle_index]
        self.decode_lengths = self.decode_lengths[shuffle_index]

    def pad_sentence_batch(self,sentence_batch, pad_int):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

        参数：
        - sentence batch
        - pad_int: <PAD>对应索引号
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return np.array([sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch])

    def iterbatch(self):
        self.shuffle()
        for i in range(self.num_batch + 1):
            if i == self.num_batch:
                encode = self.pad_sentence_batch(self.encode[i * self.batch_size:].tolist(),PAD_ID)
                decode = self.pad_sentence_batch(self.decode[i * self.batch_size:].tolist(),PAD_ID)
                data = {"encode": encode, "decode": decode,
                        "encode_lengths": self.encode_lengths[i * self.batch_size:],
                        "decode_lengths": self.decode_lengths[i * self.batch_size:]}
            else:
                encode = self.pad_sentence_batch(self.encode[i * self.batch_size:(i + 1) * self.batch_size].tolist(), PAD_ID)
                decode = self.pad_sentence_batch(self.decode[i * self.batch_size:(i + 1) * self.batch_size].tolist(), PAD_ID)
                data = {"encode": encode,
                        "decode": decode,
                        "encode_lengths": self.encode_lengths[i * self.batch_size:(i + 1) * self.batch_size],
                        "decode_lengths": self.decode_lengths[i * self.batch_size:(i + 1) * self.batch_size]}
            yield data






def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


if __name__ == '__main__':
    prepare_data_for_model()