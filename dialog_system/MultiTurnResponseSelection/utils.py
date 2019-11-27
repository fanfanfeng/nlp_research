# create by fanfan on 2019/8/9 0009

from keras.preprocessing.sequence import pad_sequences


def multi_sequences_padding(all_sequences,max_sentence_len = 50):
    max_num_utterance = 10
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences,maxlen=max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]

        sequences = pad_sequences(sequences,padding='post',maxlen=max_sentence_len)
        padded_sequences.append(sequences)

    return padded_sequences,sequences_length






def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length