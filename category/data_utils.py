# create by fanfan on 2019/4/10 0010

import jieba
import json
import os
import sys
import tqdm

_START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
if 'win' in sys.platform:
    user_dict_path = r'E:\nlp-data\jieba_dict\dict_modify.txt'
else:
    user_dict_path = r'/data/python_project/rasa_corpus/jieba_dict/dict_modify.txt'

jieba.load_userdict(user_dict_path)




def pad_sentence(sentence, max_sentence,vocabulary):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')
    PAD_ID = vocabulary.get('_PAD')
    sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
    if len(sentence_batch_ids) > max_sentence:
        sentence_batch_ids = sentence_batch_ids[:max_sentence]
    else:
        sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))

    if max(sentence_batch_ids) == 0:
        print(sentence_batch_ids)
    return sentence_batch_ids




