# 自然语言处理一些技术mark


## BERT-BiLSTM-CRF-NER
   利用谷歌bert模型训练ner,融合bert_as_service作为一个服务
   参考[BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)


## Segment 分词
### segment_tried.py
        构建字典树，然后查询在字典树里面的实体
### bpe.py
        基于byte pair 原理，自动构建词库，然后根据词库分词
        网络上的bpe 包有：[bheinzerling/bpemb](https://github.com/bheinzerling/bpemb)  [soaxelbrooke/python-bpe](https://github.com/soaxelbrooke/python-bpe)

## LangueModel 语言模型
### Word2vec skip
### ELMO
       blog:    https://blog.csdn.net/sinat_26917383/article/details/81913790
                https://blog.csdn.net/jeryjeryjery/article/details/81183433
       参考 [ELMo-chinese](https://github.com/Rokid/ELMo-chinese)
### BERT
       参考[google-research/bert](https://github.com/google-research/bert)


## Text Summary 文本关键字提取以及总结
### TextRank4Z
        基于text rank算法计算关键字


## utils 工具类以及一些小工具实现
    -- filter.py
        敏感词过滤的几种实现+某1w词敏感词库，[textfilter](https://github.com/observerss/textfilter)

