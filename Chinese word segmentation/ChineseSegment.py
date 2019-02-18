import codecs
import re
import random
#from sklearn.model_selection import train_test_split

def SplitTrainAndTest():
    rec = codecs.open("nlpdata.txt",encoding='gb18030')
    cnt = 0

    wrap = re.compile('\r\n')
    date = re.compile('\d+-\d+-\d+-\d+')
    tag = re.compile('/[a-zA-Z]+')
    note = re.compile('{\w+}')
    left0 = re.compile('\[')
    right0 = re.compile('\][a-zA-z]+')
    clean_rec = []
    for i in rec:
        clean = wrap.sub('', i)
        clean = date.sub('',clean)
        clean = left0.sub('',clean)
        clean = right0.sub('',clean)
        clean = tag.sub('',clean)
        clean = note.sub('',clean)
        cnt += 1
        clean_rec.append(clean)
    #以上数据清洗
    # target = [i for i in range(len(clean_rec))]
    # train_x,test_x,nonsense_y,nonsen_y = train_test_split(clean_rec,target,test_size=0.2)
    # print(len(dataset))
    # return train_x,test_x
    test_index = random.sample(range(cnt), int(cnt * 0.2))
    test_data = []
    for i in range(len(test_index)):
        test_data.append(clean_rec[test_index[i]])
    for i in test_data:
        if i in clean_rec:
            clean_rec.remove(i)
    clean_rec = ' '.join(clean_rec)
    test_data = ' '.join(test_data)
    train = clean_rec.split()
    test = test_data.split()
    waiting_test = ''
    for i in test:
        waiting_test += i
    #以上分测试集与训练集
    return train,test,waiting_test

def MaxLen(t):
    maxlen = 0
    for i in t:
         if maxlen < int(len(i)):
             maxlen = int(len(i))
    return maxlen

def CountTrain(t):
    cnt = {}
    for i in t:
        if i not in cnt:
            cnt[i] = 1
    return cnt

def FMM(waiting_test,test_data,train_count,maxlen):
    wlen = len(waiting_test)
    left = 0
    right = 0
    segment = []
    word = ''
    seg_point = 0
    cnt = 0
    test = '/'.join(test_data)
    test_point = 0

    while left != (wlen - 1):
        for i in range(maxlen):
            right = left + maxlen - i
            if right >= wlen:
                right = wlen - 1
            if waiting_test[left:right] in train_count:
               word = waiting_test[left:right]
               segment.append(word)
            elif left + 1 == right:
                word = waiting_test[left]
                segment.append(word)
            else:
                continue
            left = right
            if word == test_data[seg_point]:
                cnt += 1
            for j in range(len(word)):
                test_point += 1
                if test[test_point] == '/':
                    seg_point += 1
                    test_point += 1
            break

    p = float(cnt)/len(segment)
    r = float(cnt)/len(test_data)
    f = 2*p*r/(p+r)
    print('Precision=%f' % p)
    print('Recall=%f' % r)
    print('F measure=%f' % f)
    fp0 = codecs.open('test_segment.txt',encoding='gb18030',mode='w')
    fp0.write('/'.join(segment))
    fp1 = codecs.open('test_origin.txt',encoding='gb18030',mode='w')
    fp1.write(test)
    fp0.close()
    fp1.close()

if __name__ == '__main__':
    train_data,test_data,waiting_test = SplitTrainAndTest()
    maxlen = MaxLen(train_data)
    train_count = CountTrain(train_data)
    FMM(waiting_test,test_data,train_count,maxlen)
