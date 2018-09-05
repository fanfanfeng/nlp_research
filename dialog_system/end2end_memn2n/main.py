# create by fanfan on 2018/9/5 0005
import tensorflow as tf
from dialog_system.end2end_memn2n.model import MemN2n
from dialog_system.end2end_memn2n.data_reader import read_data,pad_data,depad_data
from dialog_system.end2end_memn2n import config
import os
import pprint
import numpy as np
pp = pprint.PrettyPrinter()
def main():
    word2idx = {}
    max_words = 0
    max_sentences = 0

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    train_stories, train_questions, max_words, max_sentences = read_data(
        '{}/qa{}_train.txt'.format(config.data_dir, config.babi_task), word2idx, max_words, max_sentences)
    valid_stories, valid_questions, max_words, max_sentences = read_data(
        '{}/qa{}_valid.txt'.format(config.data_dir, config.babi_task), word2idx, max_words, max_sentences)
    test_stories, test_questions, max_words, max_sentences = read_data(
        '{}/qa{}_test.txt'.format(config.data_dir, config.babi_task), word2idx, max_words, max_sentences)

    pad_data(train_stories, train_questions, max_words, max_sentences)
    pad_data(valid_stories, valid_questions, max_words, max_sentences)
    pad_data(test_stories, test_questions, max_words, max_sentences)

    with tf.Session() as sess:
        model = MemN2n(sess,len(word2idx),max_words,max_sentences)
        model.build_model()
        if config.is_test:
            model.run(valid_stories,valid_questions,test_stories,test_questions)
        else:
            model.run(train_stories,train_questions,valid_stories,valid_questions)

        if config.is_test:
            predictions, target = model.predict(train_stories, train_questions)
            index = 25
            idx2word = dict(zip(word2idx.values(), word2idx.keys()))
            depad_data(train_stories, train_questions)

            question = train_questions[index]['question']
            answer = train_questions[index]['answer']
            story_index = train_questions[index]['story_index']
            sentence_index = train_questions[index]['sentence_index']

            story = train_stories[story_index][:sentence_index + 1]

            story = [list(map(idx2word.get, sentence)) for sentence in story]
            question = list(map(idx2word.get, question))
            prediction = [idx2word[np.argmax(predictions[index])]]
            answer = list(map(idx2word.get, answer))

            print('Story:')
            pp.pprint(story)
            print('\nQuestion:')
            pp.pprint(question)
            print('\nPrediction:')
            pp.pprint(prediction)
            print('\nAnswer:')
            pp.pprint(answer)
            print('\nCorrect:')
            pp.pprint(prediction == answer)

if __name__ == '__main__':
    main()
