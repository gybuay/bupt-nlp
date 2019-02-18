# bupt-nlp
- #### `Chinese word segmentation`

  `设计一个中文分词器，对所给的中文语料数据进行分词。通过所给的PKU语料数据，其中已做好分词与词性标注，对该语料进行训练，得到一个中文分词器，并对原始语料进行分词，与正确的分词结果进行比较，用Precision、Recall和F Measure方法进行评估。`

- #### `N-gram Language Models`

  `设计一个典型的N-gram语言模型。用所给的已分词数据一部分训练得到该模型，用另一部分数据测试该模型。最后用Word Perplexity评价该模型。本文设计的是采用Lidstone（add-Delta）平滑稀疏数据的unigram、bigram和trigram，通过对不同n在不同的Delta下的word perplexity值进行评价，从而选出不同的n对应的最优Delta。`

- #### `Detecting Sentiment Polarity`

  `设计一个情感分类器，对未知情感极性的句子进行正确分类。通过所给的情感词典数据（其中已做好情感极性、强弱标注），还有已经分好极性的句子，进行训练得到一个基于朴素Bayes的情感分类器。使用该情感分类器对未知极性的句子进行分类，与正确的分类结果进行比较，用Precision、Recall方法进行评估。`

