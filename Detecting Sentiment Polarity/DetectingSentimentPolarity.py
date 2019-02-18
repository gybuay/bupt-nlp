import codecs
import re
import random
from math import log
def SplitTrainAndTest(file):#句子数据处理
    rec = codecs.open(file,encoding='gb18030',errors='ignore')
    filt = re.compile('\\W+')
    clean_rec = []
    cnt = 0
    for i in rec:
        seg = filt.split(i)
        clean_rec.append(seg)
        cnt+=1
    #以上清洗数据
    test_index = random.sample(range(cnt), int(cnt * 0.2))
    test_data = []
    for i in range(len(test_index)):
        test_data.append(clean_rec[test_index[i]])
    for i in test_data:
        if i in clean_rec:
            clean_rec.remove(i)
    train = clean_rec
    for i in train:
        if '' in i:
            i.remove('')
    for i in test_data:
        if '' in i:
            i.remove('')
    #以上分测试集与训练集
    # train = clean_rec[0:4001]
    # test_data = clean_rec[4001:]
    return train,test_data

def GetPolarDict(file):#词典数据处理
    rec = codecs.open(file,encoding='gb18030',errors='ignore')
    filt = re.compile('=\\w+')
    polar = {}
    for i in rec:
        seg = filt.findall(i)
        type = seg[0].replace('=','')
        len = seg[1].replace('=','')
        word = seg[2].replace('=','')
        pos = seg[3].replace('=', '')
        stemmed = seg[4].replace('=','')
        polarity = seg[5].replace('=','')
        polar[word] = {}
        polar[word]['type'] = type
        polar[word]['len'] = len
        polar[word]['pos'] = pos
        polar[word]['stemmed'] = stemmed
        polar[word]['polarity'] = polarity
    return polar

def PolarScore(polar):#词强弱转换为得分
    for i in polar.keys():
        tot = 0
        type = polar[i]['type']
        polarity = polar[i]['polarity']
        if type == 'strongsubj':
            if polarity == 'positive':
                tot += 2
            elif polarity == 'negative':
                tot -= 2
            else:
                tot = 0
        elif type == 'weaksubj':
            if polarity == 'positive':
                tot += 1
            elif polarity == 'negative':
                tot -= 1
            else:
                tot = 0
        polar[i]['score'] = tot

def PureDictClassifier(test_pos,test_neg,polar,notlist):
    #基于情感词典的分类器
    ture_pos = 0
    false_pos = 0
    ture_neg = 0
    false_neg = 0
    for i in test_pos:#对正面情感进行检测
        score = 0
        cnt_not = 0
        for word in i:
            if word in polar:
                score += polar[word]['score']
            if word in notlist:
                cnt_not += 1
        if cnt_not % 2 == 1:
            score = -score
        #以上得到单个句子的情感得分
        if score < 0:#若判定为负面情感则错误
            false_pos += 1
        else:
            ture_pos += 1

    for i in test_neg:#对负面情感进行检测
        score = 0
        cnt_not = 0
        for word in i:
            if word in polar:
                score += polar[word]['score']
            if word in notlist:
                cnt_not += 1
        if cnt_not % 2 == 1:
            score = -score
        #以上得到单个句子的情感得分
        if score > 0:#若判定为正面情感则错误
            false_neg += 1
        else:
            ture_neg += 1

    print("Based on sentiment dict:")
    prec_pos = ture_pos / (ture_pos + false_neg)
    prec_neg = ture_neg / (ture_neg + false_pos)
    recall_pos = ture_pos / len(test_pos)
    recall_neg = ture_neg / len(test_neg)
    print("Precision(pos):%f" % prec_pos)
    print("Precision(neg):%f" % prec_neg)
    print("Recall(pos):%f" % recall_pos)
    print("Recall(neg):%f" % recall_neg)

def BayesTrainer(train_pos,train_neg,polar,polar_list,delta=1):
    #Bayes训练模型
    bayes_pos = [0 for i in range(len(polar))]#pos分类向量
    bayes_neg = [0 for i in range(len(polar))]#neg分类向量
    p_pos = len(train_pos) / (len(train_pos) + len(train_neg))
    p_neg = len(train_neg) / (len(train_pos) + len(train_neg))
    #统计词在情感词典中出现的频次
    for i in train_pos:
        for j in i:
            if j in polar_list:
                bayes_pos[polar_list.index(j)] += 1

    for i in train_neg:
        for j in i:
            if j in polar_list:
                bayes_neg[polar_list.index(j)] += 1

    for i in range(len(bayes_pos)):
        bayes_pos[i] = log((bayes_pos[i] + delta) / (len(train_pos) + len(polar) * delta))
    for i in range(len(bayes_neg)):
        bayes_neg[i] = log((bayes_neg[i] + delta) / (len(train_neg) + len(polar) * delta))

    return bayes_pos,bayes_neg,p_pos,p_neg

def testTrans(test,polar,polar_list):
    ans = []
    for sent in test:
        tmp = [0 for i in range(len(polar))]
        for i in sent:
            if i in polar_list:
                tmp[polar_list.index(i)] += 1
        ans.append(tmp)
    return ans

def BayesClassifier(bayes_pos,bayes_neg,p_pos,p_neg,test_pos,test_neg):
    ture_pos = 0
    false_pos = 0
    ture_neg = 0
    false_neg = 0
    for i in test_pos:
        p_to_pos = sum(map(lambda a_b:a_b[0]*a_b[1],zip(i,bayes_pos))) + log(p_pos)#测试语料与分类向量点积
        p_to_neg = sum(map(lambda a_b:a_b[0]*a_b[1],zip(i,bayes_neg))) + log(p_neg)#测试语料与分类向量点积
        if p_to_pos > p_to_neg:
            ture_pos += 1
        else:
            false_pos += 1

    for i in test_neg:
        p_to_pos = sum(map(lambda a_b:a_b[0]*a_b[1],zip(i,bayes_pos))) + log(p_pos)
        p_to_neg = sum(map(lambda a_b:a_b[0]*a_b[1],zip(i,bayes_neg))) + log(p_neg)
        if p_to_neg > p_to_pos:
            ture_neg += 1
        else:
            false_neg += 1

    print("Based on BayesClassifier:")
    prec_pos = ture_pos / (ture_pos + false_neg)
    prec_neg = ture_neg / (ture_neg + false_pos)
    recall_pos = ture_pos / len(test_pos)
    recall_neg = ture_neg / len(test_neg)
    print("Precision(pos):%f" % prec_pos)
    print("Precision(neg):%f" % prec_neg)
    print("Recall(pos):%f" % recall_pos)
    print("Recall(neg):%f" % recall_neg)



if __name__ == '__main__':
    train_pos,test_pos = SplitTrainAndTest('rt-polarity.pos')
    train_neg,test_neg = SplitTrainAndTest('rt-polarity.neg')
    polar = GetPolarDict('subjclueslen1-HLTEMNLP05.tff')
    PolarScore(polar)
    polar_list = [i for i in polar.keys()]
    #notlist = ['nothing','t','barely','no','not']
    notlist = []
    PureDictClassifier(test_pos,test_neg,polar,notlist)
    bayes_pos,bayes_neg,p_pos,p_neg=BayesTrainer(train_pos,train_neg,polar,polar_list)
    test_vpos = testTrans(test_pos,polar,polar_list)
    test_vneg = testTrans(test_neg,polar,polar_list)
    BayesClassifier(bayes_pos,bayes_neg,p_pos,p_neg,test_vpos,test_vneg)