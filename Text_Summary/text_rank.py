# create by fanfan on 2019/11/26 0026
import jieba
from utils import text_util
from text_similarity.bm25 import BM25
class TextRank(object):
    def __init__(self,docs):
        self.docs = docs
        self.bm25 = BM25(docs)
        self.D = len(docs)
        self.d = 0.85
        self.weight = []
        self.weight_sum = []
        self.vertex = []
        self.max_iter = 200
        self.min_diff = 0.001
        self.top = []


    def text_rank(self):
        for cnt,doc in enumerate(self.docs):
            scores = self.bm25.simall(doc)
            self.weight.append(scores)
            self.weight_sum.append(sum(scores) - scores[cnt])
            self.vertex.append(1.0)


        for _ in range(self.max_iter):
            m = []
            max_diff = 0
            for i in range(self.D):
                m.append(1 - self.d)
                for j in range(self.D):
                    if j == i or self.weight_sum[j] == 0:
                        continue

                    # TextRank的公式
                    m[-1] += (self.d * self.weight[j][i]/self.weight_sum[j]*self.vertex[j])

                if abs(m[-1] - self.vertex[i]) > max_diff:
                    max_diff = abs(m[-1] - self.vertex[i])
            self.vertex = m
            if max_diff <= self.min_diff:
                break

        self.top = list(enumerate(self.vertex))
        self.top = sorted(self.top,key=lambda x:x[1],reverse=True)


    def top_index(self,limit):
        return list(map(lambda x:x[0],self.top))[:limit]

    def top(self,limit):
        return list(map(lambda x:self.docs[x[0]],self.top))[:limit]


if __name__ == '__main__':
    text = '''
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
    它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
    因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
    所以它与语言学的研究有着密切的联系，但又有重要的区别。
    自然语言处理并不是一般地研究自然语言，
    而在于研制能有效地实现自然语言通信的计算机系统，
    特别是其中的软件系统。因而它是计算机科学的一部分。
    '''
    sents = text_util.get_sentences(text)
    doc = []
    for sent in sents:
        words = list(jieba.cut(sent))
        words = text_util.filter_stop(words)
        doc.append(words)
    print(doc)

    rank = TextRank(doc)
    rank.text_rank()
    for index in rank.top_index(3):
        print(sents[index])

