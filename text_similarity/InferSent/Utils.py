# create by fanfan on 2019/12/4 0004
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import numpy as np
import os
from datetime import timedelta
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import pickle
import time
import json

UNKNOWN = '<UNK>'
PADDING = '<PAD>'
CATEGORIE_ID = {'entailment':0,'neutral':1,'contradiction':2}

def lazy_property(function):
    attribute = '_cache_'+ function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self,attribute):
            with tf.variable_scope(function.__name__):
                setattr(self,attribute,function(self))
        return getattr(self,attribute)


def print_shape(varname,var):
    '''
    :param varname: tensor name 
    :param var: tensor variable
    :return: 
    '''
    print('{0}:{1}'.format(varname,var.get_shape()))


def init_embeddings(vocab,embedding_dims):
    rng = np.random.RandomState(None)
    random_init_embeddings = rng.normal(size=(len(vocab),embedding_dims))
    return random_init_embeddings.astype(np.float32)

def load_embeddings(path,vocab):
    with open(path,'rb') as fin:
        _embeddings,_vocab = pickle.load(fin)
    embedding_dims = _embeddings.shape[1]
    embeddings = init_embeddings(vocab,embedding_dims)
    for word,id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    return embeddings.astype(np.float32)

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings,axis=1).reshape((-1,1))
    return embeddings/norms

def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value

        totalParams += variableParams
    return totalParams

def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds=int(round(diff)))

def build_vocab(dataPath,vocabPath,threshold =0,lowercase=True):
    cnt = Counter()
    with open(dataPath,mode='r',encoding='utf-8') as iF:
        for line in iF:
            try:
                if lowercase:
                    line = line.lower()
                tempLine = line.strip().split('||')
                l1 = tempLine[1][:-1]
                l2 = tempLine[2][:-1]
                words1 = l1.split(" ")
                for word in list(words1):
                    cnt[word] += 1

                words2 = l2.split(" ")
                for word in list(words2):
                    cnt[word] += 1
            except:
                pass

    cntDict = [item for item in cnt.items() if item[1] >= threshold]
    cntDict = sorted(cntDict,key=lambda  d:d[1],reverse=True)
    wordFreq = ["||".join([word,str(freq)]) for word,freq in cntDict]
    with open(vocabPath,mode='w',encoding='utf-8') as oF:
        oF.write('\n'.join(wordFreq) + '\n')
    print("Vocabulary is stored in : {}".format(vocabPath))

def load_vocab(vocabPath,threshold = 0):
    vocab = {}
    index = 2
    vocab[PADDING] = 0
    vocab[UNKNOWN] = 1
    with open(vocabPath,encoding='utf-8') as f:
        for line in f:
            items = [v.strip() for v in line.split("||")]
            if len(items) != 2:
                print("Wrong format:",line)
                continue
            word,freq = items[0],int(items[1])
            if freq >= threshold:
                vocab[word] = index
                index += 1
    return vocab

def sentence2Index(dataPath,vocabDict,maxLen =100,lowercase=True):
    s1List,s2List,labelList = [],[],[]
    s1Mask,s2Mask =[],[]

    with open(dataPath,mode='r',encoding='utf-8') as f:
        for line in f:
            try:
                l,s1,s2 = [v.strip() for v in line.strip().split("||")]
                if lowercase:
                    s1,s2 = s1.lower(),s2.lower()

                s1 = [v.strip() for v in s1.split()]
                s2 = [v.strip() for v in s2.split()]

                if len(s1) > maxLen:
                    s1 = s1[:maxLen]

                if len(s2) > maxLen:
                    s2 = s2[:maxLen]

                if l in CATEGORIE_ID:
                    labelList.append([CATEGORIE_ID[l]])
                    s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
                    s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
                    s1Mask.append(len(s1))
                    s2Mask.append(len(s2))
            except:
                ValueError("Input Data Value Error!")
    s1Pad,s2Pad = pad_sequences(s1List,maxLen,padding='post'),pad_sequences(s2List,maxLen,padding='post')
    enc = OneHotEncoder(sparse=False)
    labelList = enc.fit_transform(labelList)
    s1Mask = np.asarray(s1Mask,np.int32)
    s2Mask = np.asarray(s2Mask,np.int32)
    labelList = np.asarray(labelList,np.int32)
    return s1Pad,s1Mask,s2Pad,s2Mask,labelList


def next_batch(premise,premise_mask,hypothesis,hypothesis_mask,y,batchSize=64,shuffle=True):
    sampleNums = len(premise)
    batchNums = int((sampleNums - 1)/batchSize) + 1
    if shuffle:
        indices = np.random.permutation(np.arange(sampleNums))
        premise = premise[indices]
        premise_mask = premise_mask[indices]
        hypothesis = hypothesis[indices]
        hypothesis_mask = hypothesis_mask[indices]
        y = y[indices]


    for i in range(batchNums):
        startIndex = i * batchSize
        endIndex = min((i + 1)*batchSize,sampleNums)
        yield (premise[startIndex:endIndex],premise_mask[startIndex:endIndex],
               hypothesis[startIndex:endIndex],hypothesis_mask[startIndex:endIndex],
               y[startIndex:endIndex])

def convert_data(jsonPath,txtPath):
    fout = open(txtPath,'w')
    with open(jsonPath) as fin:
        i = 0
        cnt = {key :0 for key in CATEGORIE_ID.keys()}
        cnt['-'] = 0
        for line in fin:
            text = json.loads(line)
            cnt[text['gold_label']] += 1
            print("||".join([text['gold_label'],text['sentence1'],text['sentence2']]),file=fout)

            i+= 1
            if i % 10000 == 0:
                print(i)


    for key,value in cnt.items():
        print("#{0}:{1}".format(key,value))

    print("Source data has been converted from \"{0}\" to \"{1}\"".format(jsonPath,txtPath))

def print_log(*args,**kwargs):
    print(*args)
    if len(kwargs) > 0 :
        print(*args,**kwargs)
    return None

def print_args(args,log_file):
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log('-------------------- HYPER PARAMETERS -------------------',file = log_file)
    for a in argsList:
        print_log("%s:%s" % (a[0],str(a[1])))
    print("----------------------------------------",file=log_file)
    return None

if __name__ == '__main__':
    if os.path.exists('SNLI/clear data/'):
        os.makedirs('SNLI/clean data/')

    convert_data('SNLI/raw data/snli_1.0_train.jsonl', 'SNLI/clean data/train.txt')
    convert_data('SNLI/raw data/snli_1.0_dev.jsonl', 'SNLI/clean data/dev.txt')
    convert_data('SNLI/raw data/snli_1.0_test.jsonl', 'SNLI/clean data/test.txt')

    build_vocab('SNLI/clean data/train.txt', 'SNLI/clean data/vocab.txt')




