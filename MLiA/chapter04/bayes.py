import glob
import random
import re

import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set()
    vocabSet.update(*dataSet)
    return sorted(list(vocabSet))

def setOfWords2Vec(vocabList: list, inputSet):
    returnVec= [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word ({}) is not in my Vocabulary!'.format(word))
    else:
        return returnVec

def trainNB0(trainMatrix, trainCategory):
    # numTrainDocs -> record count
    # numWords -> record length
    numTrainDocs, numWords = np.array(trainMatrix).shape
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i, trainDoc in enumerate(trainMatrix):
        if trainCategory[i] == 1:
            p1Num += trainDoc
            p1Denom += sum(trainDoc)
        else:
            p0Num += trainDoc
            p0Denom += sum(trainDoc)
    else:
        p0Vect = p0Num / p0Denom
        p1Vect = p1Num / p1Denom
        return p0Vect, p1Vect, pAbusive

def trainNB1(trainMatrix, trainCategory):
    numTrainDocs, numWords = np.array(trainMatrix).shape
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # why 2?
    p0Denom = 2.0
    p1Denom = 2.0
    print('#' * 64)
    for i, trainDoc in enumerate(trainMatrix):
        print('trainDoc: {}'.format(trainDoc))
        print('class: {}'.format(trainCategory[i]))
        if trainCategory[i] == 1:
            # p1Num是spam
            p1Num += trainDoc
            p1Denom += sum(trainDoc)
        else:
            # p0Num是ham
            p0Num += trainDoc
            p0Denom += sum(trainDoc)
    else:
        print('p0Num: {}'.format(p0Num))
        print('p1Num: {}'.format(p1Num))
        p0Vect = np.log(p0Num / p0Denom)
        p1Vect = np.log(p1Num / p1Denom)
        return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    print('vec2Classify: {}'.format(vec2Classify))
    print('p0Vec: {}'.format(p0Vec))
    print('p1Vec: {}'.format(p1Vec))
    print('p1sum: {}'.format(sum(vec2Classify * p1Vec)))
    print('p0sum: {}'.format(sum(vec2Classify * p0Vec)))
    # 麻蛋，这个地方+写错了，搞了一个晚上！！！
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    print('p1: {}'.format(p1))
    print('p0: {}'.format(p0))
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word ({}) is not in my Vocabulary!'.format(word))
    else:
        return returnVec


def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = list()
    classList = list()
    fullText = list()
    for d, c in {'ham': 0, 'spam': 1}.items():
        for p in glob.glob('email/{}/*.txt'.format(d)):
            print('start reading {}'.format(p))
            wordList = textParse(open(p).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(c)
    else:
        print('number of docList: {}'.format(len(docList)))
        vocabList = createVocabList(docList)
    print('vocabList: {}'.format(vocabList))
    trainMat = list()
    trainClasses = list()
    for inputSet, trainClass in zip(docList, classList):
        trainMat.append(setOfWords2Vec(vocabList, inputSet))
        trainClasses.append(trainClass)
    else:
        p0V, p1V, pSpam = trainNB1(trainMat, trainClasses)
    print('p0V: {}'.format(p0V))
    print('p1V: {}'.format(p1V))
    print('pSpam: {}'.format(pSpam))
    errorCount = 0.0
    testCount = 0.0
    for i in sorted(list(set(random.randint(0, 49) for i in range(10)))):
        print('i: {}'.format(i))
        wordVector = setOfWords2Vec(vocabList, docList[i])
        print('wordVector: {}'.format(wordVector))
        classRet = classifyNB(np.array(wordVector), p0V, p1V, pSpam)
        classRes = classList[i]
        print('classRet: {}'.format(classRet))
        print('classRes: {}'.format(classRes))
        if classRet != classRes:
            errorCount += 1
        testCount += 1
    else:
        print('the error rate is: {}'.format(errorCount / testCount))


