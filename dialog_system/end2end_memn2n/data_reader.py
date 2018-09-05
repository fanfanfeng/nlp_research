# create by fanfan on 2018/8/31 0031
import os

def read_data(fname,word2id,max_words,max_sentences):
    # stories[story_ind] = [[sentence1], [sentence2], ..., [sentenceN]]
    # questions[question_ind] = {'question': [question], 'answer': [answer], 'story_index': #, 'sentence_index': #}
    stories = {}
    questions = {}

    if len(word2id) == 0:
        word2id['<null>'] = 0


    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise Exception("[!] Data {file} not found".format(file = fname))

    for line in lines:
        words = line.split()
        max_words = max(max_words,len(words))

        if words[0] == '1':
            story_ind = len(stories)
            sentence_ind = 0
            stories[story_ind] = []

        if '?' in line:
            is_question = True
            question_ind = len(questions)
            questions[question_ind] = {'question':[],'answer':[],'story_index':story_ind,'sentence_index':sentence_ind}
        else:
            is_question = False
            sentence_ind = len(stories[story_ind])

        sentence_token_list = []
        for k in range(1,len(words)):
            w = words[k].lower()
            if ('.' in w) or ('?' in w):
                w = w[:-1]

            if w not in word2id:
                word2id[w] = len(word2id)

            if not is_question:
                sentence_token_list.append(w)
                if '.' in words[k]:
                    stories[story_ind].append(sentence_token_list)
                    break
            else:
                sentence_token_list.append(w)
                if '?' in words[k]:
                    answer = words[k+1].lower()
                    if answer not in word2id:
                        word2id[answer] = len(word2id)


                    questions[question_ind]['question'].extend(sentence_token_list)
                    questions[question_ind]['answer'].append(answer)
                    break



        max_sentences = max(max_sentences,sentence_ind + 1)

    for idx,context in stories.items():
        for i in range(len(context)):
            temp = list(map(word2id.get,context[i]))
            context[i] = temp

    for idx,value in questions.items():
        temp1 = list(map(word2id.get,value['question']))
        temp2 = list(map(word2id.get,value['answer']))
        value['question'] = temp1
        value['answer'] = temp2


    return stories,questions,max_words,max_sentences


def pad_data(stories,questions,max_words,max_sentences):
    for idx,context in stories.items():
        for sentence in context:
            while len(sentence) < max_words:
                sentence.append(0)

        while len(context) < max_sentences:
            context.append([0] * max_words)

    for idx,value in questions.items():
        while len(value['question']) < max_words:
            value['question'].append(0)

def depad_data(stories, questions):
    for idx, context in stories.items():
        for i in range(len(context)):
            if 0 in context[i]:
                if context[i][0] == 0:
                    temp = context[:i]
                    context = temp
                    break
                else:
                    index = context[i].index(0)
                    context[i] = context[i][:index]

    for idx, value in questions.items():
        if 0 in value['question']:
            index = value['question'].index(0)
            value['question'] = value['question'][:index]