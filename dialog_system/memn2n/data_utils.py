# create by fanfan on 2018/8/28 0028
import os
import re
import numpy as np
def load_task(data_dir,task_id,only_supporting=False):
    assert task_id >0 and task_id <21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir,f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file,only_supporting)
    test_data = get_stories(test_file,only_supporting)
    return train_data,test_data


def get_stories(file,only_supporting=False):
    with open(file) as fread:
        return parse_stories(fread.readlines(),only_supporting=only_supporting)


def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.lower()
        nid,line = line.split(' ',1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q,a,supporting = line.split('\t')
            q = tokenize(q)
            a = [a]

            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                supporting = map(int,supporting.split())
                substory = [story[i-1] for i in supporting]
            else:
                substory = [x for x in story if x ]
            data.append((substory,q,a))
            story.append("")
        else:
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data



def tokenize(sentence):
    return [x.strip() for x in re.split('(\W+)?',sentence) if x.strip()]

def vectorize_data(data,word_idx,sentence_size,memory_size):
    S = []
    Q = []
    A = []
    for story,query,answer in data:
        ss = []
        for i ,sentence in enumerate(story,1):
            ls = max(0,sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)
        ss = ss[::-1][:memory_size][::-1]
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        lm = max(0,memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0,sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1)
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)

    return np.array(S),np.array(Q),np.array(A)

