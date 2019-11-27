# create by fanfan on 2019/8/9 0009
from dialog_system.MultiTurnResponseSelection.sequential_matching_network import SequentialMatchingNetwork
from dialog_system.MultiTurnResponseSelection.params import Params
from dialog_system.MultiTurnResponseSelection import utils
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

params = Params()
def train(countinue_train = False, previous_modelpath = "model"):
    with open(params.response_file, 'rb') as f:
        actions = pickle.load(f)
    with open(params.history_file,'rb' )as f:
        history,true_utt = pickle.load(f)
    history,history_len = utils.multi_sequences_padding(history,params.max_sentence_len)


    true_utt_len = np.array(utils.get_sequences_length(true_utt,maxlen=params.max_sentence_len))
    true_utt = np.array(pad_sequences(true_utt,padding='post',maxlen=params.max_sentence_len))
    actions_len = np.array(utils.get_sequences_length(actions,maxlen=params.max_sentence_len))
    actions = np.array(pad_sequences(actions,padding='post',maxlen=params.max_sentence_len))
    history,history_len = np.array(history),np.array(history_len)






    model = SequentialMatchingNetwork()
    model.build_model()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    mergerd = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('output2',sess.graph)

        if countinue_train == False:
            sess.run(init)
        else:
            saver.restore(sess,previous_modelpath)



    low = 0
    epoch = 1
    while epoch < 10:
        n_sample = min(low + params.batch_size,history.shape[0]) - low
        negative_indices = [np.random.randint(0,actions.shape[0],n_sample) for _ in range(params.negative_samples)]

        negs = [actions[negative_indices[i]:i] for i in range(params.negative_samples)]
        negs_len = [actions_len[negative_indices[i]] for i in range(params.negative_samples)]
        feed_dict = {
            model.utterance_ph: np.concatenate([history[low:low + n_sample]] * (params.negative_samples + 1),axis=0),
            model.all_utterance_len_ph:np.concatenate([history_len[low:low+n_sample]] * (params.negative_samples + 1),axis=0),
            model.response_ph:np.concatenate([true_utt[low:low + n_sample]] + negs,axis=0),
            model.response_len:np.concatenate([true_utt_len[low:low + n_sample]] + negs_len,axis=0)
        }











if __name__ == '__main__':
    train()