# create by fanfan on 2018/8/28 0028
from dialog_system.memn2n import config
from dialog_system.memn2n import data_utils
from sklearn import  metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import  numpy as np
import pandas as pd
from functools import reduce
from itertools import chain
from dialog_system.memn2n.memn2n import MemN2N

task_ids = [2]#range(1,2)
train,test = [],[]
for i in task_ids:
    train_data_onetask,test_data_onetask = data_utils.load_task(config.data_dir,i)
    train.append(train_data_onetask)
    test.append(test_data_onetask)

total_data = train + test
total_data = list(chain.from_iterable(total_data))
vocab = sorted(reduce(lambda x,y:x| y, (set(list(chain.from_iterable(s)) + q + a ) for s,q,a in total_data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len,(s for s,_,_ in total_data)))
mean_story_size = int(np.mean([len(s) for s,_,_ in total_data]))
sentence_size = max(map(len,chain.from_iterable(s for s,_,_ in total_data)))
query_size = max(map(len,(q for _ ,q,_ in total_data)))
memory_size = min(config.memory_size,max_story_size)

for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1
sentence_size = max(query_size,sentence_size)
sentence_size += 1

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

trainS = []
valS = []
trainQ = []
valQ = []
trainA = []
valA = []
for task in train:
    S,Q,A = data_utils.vectorize_data(task,word_idx,sentence_size,memory_size)
    ts,vs,tq,vq,ta,va = train_test_split(S,Q,A,test_size=0.1)
    trainS.append(ts)
    trainQ.append(tq)
    trainA.append(ta)
    valS.append(vs)
    valQ.append(vq)
    valA.append(va)

trainS = reduce(lambda a,b:np.vstack((a,b)),(x for x in trainS))
trainQ = reduce(lambda a,b:np.vstack((a,b)),(x for x in trainQ))
trainA = reduce(lambda a,b:np.vstack((a,b)),(x for x in trainA))
valS = reduce(lambda a,b:np.vstack((a,b)),(x for x in valS))
valQ = reduce(lambda a,b:np.vstack((a,b)),(x for x in valQ))
valA = reduce(lambda a,b:np.vstack((a,b)),(x for x in valA))

testS,testQ,testA = data_utils.vectorize_data(list(chain.from_iterable(test)),word_idx,sentence_size,memory_size)
n_train = trainS.shape[0]
n_val = valS.shape[0]
n_test = testS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

batch_size  = config.batch_size

batches = zip(range(0,n_train - batch_size, batch_size),range(batch_size,n_train,batch_size))
batches = [(start,end) for start,end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, config.embedding_size, session=sess,
                   hops=config.hops, max_grad_norm=config.max_grad_norm)
    for i in range(1, config.epochs+1):
        # Stepped learning rate
        if i - 1 <= config.anneal_stop_epoch:
            anneal = 2.0 ** ((i - 1) // config.anneal_rate)
        else:
            anneal = 2.0 ** (config.anneal_stop_epoch // config.anneal_rate)
        lr = config.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if i % config.evaluation_interval == 0:
            train_accs = []
            for start in range(0, n_train, int(n_train/20)):
                end = start + int(n_train/20)
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                acc = metrics.accuracy_score(pred, train_labels[start:end])
                train_accs.append(acc)

            val_accs = []
            for start in range(0, n_val, int(n_val/20)):
                end = start + int(n_val/20)
                s = valS[start:end]
                q = valQ[start:end]
                pred = model.predict(s, q)
                acc = metrics.accuracy_score(pred, val_labels[start:end])
                val_accs.append(acc)

            test_accs = []
            for start in range(0, n_test, int(n_test/20)):
                end = start + int(n_test/20)
                s = testS[start:end]
                q = testQ[start:end]
                pred = model.predict(s, q)
                acc = metrics.accuracy_score(pred, test_labels[start:end])
                test_accs.append(acc)

            print('-----------------------')
            print('Epoch', i)
            print('Total Cost:', total_cost)
            print()
            t = 1
            for t1, t2, t3 in zip(train_accs, val_accs, test_accs):
                print("Task {}".format(t))
                print("Training Accuracy = {}".format(t1))
                print("Validation Accuracy = {}".format(t2))
                print("Testing Accuracy = {}".format(t3))
                print()
                t += 1
                break
            print('-----------------------')

        # Write final results to csv file
        if i == config.epochs:
            print('Writing final results to {}'.format(config.output_file))
            df = pd.DataFrame({
            'Training Accuracy': train_accs,
            'Validation Accuracy': val_accs,
            'Testing Accuracy': test_accs
            }, index=range(1, 21))
            df.index.name = 'Task'
            df.to_csv(config.output_file)





