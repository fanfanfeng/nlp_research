# create by fanfan on 2019/7/17 0017
import sys
class Params():
    max_num_utterance = 10
    negative_samples = 1
    max_sentence_len = 50
    word_embedding_size = 200
    rnn_units = 200
    total_words = 434511
    batch_size = 40
    if 'win' in sys.platform:
        embedding_file = r"E:\nlp-data\MultiTurnResponseSelection\embedding.pkl"
        evaluate_file = r"E:\nlp-data\MultiTurnResponseSelection\Evaluate.pkl"
        response_file = r"E:\nlp-data\MultiTurnResponseSelection\responses.pkl"
        history_file = r"E:\nlp-data\MultiTurnResponseSelection\utterances.pkl"
    else:
        embedding_file = r"E:\nlp-data\MultiTurnResponseSelection\embedding.pkl"
        evaluate_file = r"E:\nlp-data\MultiTurnResponseSelection\Evaluate.pkl"
        response_file = r"E:\nlp-data\MultiTurnResponseSelection\responses.pkl"
        history_file = r"E:\nlp-data\MultiTurnResponseSelection\utterances.pkl"