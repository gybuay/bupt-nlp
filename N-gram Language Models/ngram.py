import codecs
import re
from math import log
#import matplotlib.pyplot as plt 利用matplotlib画图寻找最优delta

#测试Unigram
def Unigram(test,cnt_unigram,delta):
    V = len(cnt_unigram)
    total = 0
    info = 0.0
    cnt = 0
    for i in cnt_unigram.values():
        total += i
    for i in test:
        if i == 'flag':
            continue
        cnt += 1
        uni = lambda i : 0 if i not in cnt_unigram else cnt_unigram[i]
        p = (uni(i) + delta)/(total+V*delta)
        info += log(p,2)
    exp = (-info)/cnt
    wp = 2 ** exp
    print('wp(Unigram)=%f' % wp)
    return wp

#测试Bigram
def Bigram(test,cnt_bigram,cnt_unigram,delta):
    test = [(test[i],test[i+1]) for i in range(len(test)-1)]
    V = len(cnt_unigram)
    info = 0.0
    cnt = 0
    for i in test:
        cnt += 1
        bi = lambda i : 0 if i not in cnt_bigram else cnt_bigram[i]
        uni = lambda i : 0 if i[0] not in cnt_unigram else cnt_unigram[i[0]]
        p = float(bi(i) + delta)/(uni(i) + V*delta)
        info += log(p,2)
    exp = (-info)/cnt
    wp = 2 ** exp
    print('wp(Bigram)=%f' % wp)
    return wp

#测试Trigram
def Trigram(test,cnt_trigram,cnt_bigram,cnt_unigram,delta):
    test = [(test[i],test[i+1],test[i+2]) for i in range(len(test)-2)]
    V = len(cnt_unigram)
    info = 0.0
    cnt = 0
    for i in test:
        cnt += 1
        tri = lambda i : 0 if i not in cnt_trigram else cnt_trigram[i]
        bi = lambda i : 0 if (i[0],i[1]) not in cnt_bigram else cnt_bigram[(i[0],i[1])]
        p = float(tri(i) + delta)/(bi(i) + V*delta)
        info += log(p, 2)
    exp = (-info) / cnt
    wp = 2 ** exp
    print('wp(Trigram)=%f' % wp)
    return wp

if __name__ == '__main__':
    #提取训练集与测试集
    rec = codecs.open("nlpdata.txt",encoding='gb18030')
    cnt = 0
    date = re.compile('\d+-\d+-\d+-\d+/m')
    note = re.compile('\{\w+\}')
    wrap = re.compile('\r\n')
    clean_rec = []
    for i in rec:
        clean = date.sub('',i)
        clean = note.sub('',clean)
        clean = wrap.sub('',clean)
        #print(clean)
        clean_rec.append(clean)
        cnt+=1
    train = clean_rec[:int(cnt*0.8)]
    test = clean_rec[int(cnt*0.8):]

    #unigram and bigram替换开始标志
    train = ' '.join(train)
    test = ' '.join(test)
    train0 = train
    test0 = test
    punctuation = ['，/wd','。/wj','；/wf','！/wt','：/wm']
    for i in punctuation:
        train0 = train0.replace(i,'flag')
        test0 = test0.replace(i,'flag')
    train1 = train0.split()
    train1.insert(0,'flag')
    test1 = test0.split()
    test1.insert(0,'flag')

    #trigram替换开始标志
    train0 = train
    test0 = test
    for i in punctuation:
        train0 = train0.replace(i,'flag flag')
        test0 = test0.replace(i,'flag flag')
    train2 = train0.split()
    train2.insert(0,'flag')
    train2.insert(0,'flag')
    test2 = test0.split()
    test2.insert(0,'flag')
    test2.insert(0,'flag')

    #构造unigram模型
    seg_unigram = train1
    cnt_unigram = {}#频次统计
    for i in train1:
        if i in cnt_unigram:
            cnt_unigram[i] += 1
        else:
            cnt_unigram[i] = 1

    #构造bigram模型
    seg_bigram = [(train1[i],train1[i+1]) for i in range(len(train1)-1)]
    cnt_bigram = {}#频次统计
    for i in seg_bigram:
        if i in cnt_bigram:
            cnt_bigram[i] += 1
        else:
            cnt_bigram[i] = 1

    #构造trigram模型
    seg_trigram = [(train2[i],train2[i+1],train2[i+2]) for i in range(len(train2)-2)]
    cnt_trigram = {}#频次统计
    for i in seg_trigram:
        if i in cnt_trigram:
            cnt_trigram[i] += 1
        else:
            cnt_trigram[i] = 1

    #查找最优delta
    # deltas = [i/1000000.0 for i in range(90,100)]
    # uni = []
    # bi = []
    # tri = []
    # for delta in deltas:
    #     print('delta=%f' % delta)
    #     uni.append(Unigram(test1,cnt_unigram,delta))
    #     bi.append(Bigram(test1,cnt_bigram,cnt_unigram,delta))
    #     tri.append(Trigram(test2,cnt_trigram,cnt_bigram,cnt_unigram,delta))
    # plt.plot(deltas,uni,label='unigram')
    # plt.plot(deltas,bi,label='bigram')
    # plt.plot(deltas,tri,label='trigram')
    # plt.xlabel('delta')
    # plt.ylabel('Word Perplexity')
    # plt.legend()
    # plt.show()

    #得到不同模型对应的最优delta,输出结果
    deltas = [0.000092,0.002,1]
    for delta in deltas:
        print('delta=%f' % delta)
        if delta == 0.000092:
            print('Trigram Best')
        elif delta == 0.002:
            print('Bigram Best')
        else:
            print('Unigram Best')
        Unigram(test1,cnt_unigram,delta)
        Bigram(test1,cnt_bigram,cnt_unigram,delta)
        Trigram(test2,cnt_trigram,cnt_bigram,cnt_unigram,delta)
        print('\n')