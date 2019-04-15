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


def load_rasa_data(file_name):
    sentents = []
    intentions = []
    with open(file_name,'r',encoding='utf-8') as fr:
        data = json.load(fr)
        for item in data['rasa_nlu_data']['common_examples']:
            sentents.append(item['text'])
            intentions.append(item['intent'])
    return sentents,intentions




def create_vocab_dict(file_or_folder,min_freq=3,output_path=None):
    files = []
    if os.path.isfile(file_or_folder):
        files.append(file_or_folder)
    else:
        for file in os.listdir(file_or_folder):
            files.append(os.path.join(file_or_folder,file))


    vocab = {}
    intent = []
    for file in tqdm.tqdm(files):
        sentences,intentions = load_rasa_data(file)
        for sentence in sentences:
            real_tokens = jieba.cut(sentence)
            for word in real_tokens:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

            intent += set(intentions)
    vocab = {key: value for key, value in vocab.items() if value >= min_freq}
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_dict = {key:index for index,key in enumerate(vocab_list)}
    intent = list(set(intent))

    if output_path != None:
        with open(os.path.join(output_path,"vocab.txt"),'w',encoding='utf-8') as fwrite:
            for word in vocab_list:
                fwrite.write(word + "\n")

        with open(os.path.join(output_path,'label.txt'),'w',encoding='utf-8') as fwrite:
            for itent in intent:
                fwrite.write(itent + "\n")

    return vocab_dict,vocab_list,list(set(intent))


def load_vocab_and_intent(output_path):
    vocab_list = []
    intent = []

    with open(os.path.join(output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
        for word in fread:
            vocab_list.append(word.strip())

    with open(os.path.join(output_path, 'label.txt'), 'r', encoding='utf-8') as fread:
        for itent in fread:
            intent.append(itent.strip())

    return {key:index for index,key in enumerate(vocab_list)},vocab_list,intent

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




