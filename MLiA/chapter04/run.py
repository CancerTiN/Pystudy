import numpy as np

from MLiA.chapter04 import bayes

def test0():
    print('#' * 64)
    listOPosts, listClasses = bayes.loadDataSet()
    print('listOPosts: {}'.format(listOPosts))
    print('listClasses: {}'.format(listClasses))
    myVocabList = bayes.createVocabList(listOPosts)
    print('myVocabList: {}'.format(myVocabList))
    for wordList in listOPosts:
        ret = bayes.setOfWords2Vec(myVocabList, wordList)
        print('returnList: {}'.format(ret))
    print('#' * 64)

def test1():
    print('#' * 64)
    listOPosts, listClasses = bayes.loadDataSet()
    myVocabList = bayes.createVocabList(listOPosts)
    trainMat = [bayes.setOfWords2Vec(myVocabList, postinDoc) for postinDoc in listOPosts]
    print('trainMat: {}'.format(trainMat))
    print('#' * 64)
    p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
    print('p0V: {}'.format(p0V))
    print('p1V: {}'.format(p1V))
    print('pAb: {}'.format(pAb))
    print('#' * 64)

def test2():
    print('#' * 64)
    listOPosts, listClasses = bayes.loadDataSet()
    print('listOPosts: {}'.format(listOPosts))
    print('listClasses: {}'.format(listClasses))
    myVocabList = bayes.createVocabList(listOPosts)
    print('myVocabList: {}'.format(myVocabList))
    trainMat = [bayes.setOfWords2Vec(myVocabList, postinDoc) for postinDoc in listOPosts]
    print('trainMat: {}'.format(trainMat))
    p0V, p1V, pAb = bayes.trainNB1(trainMat, listClasses)
    print('p0V: {}'.format(p0V))
    print('p1V: {}'.format(p1V))
    print('pAb: {}'.format(pAb))
    print('#' * 64)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(bayes.setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc: {}'.format(thisDoc))
    print('{} classified as {}'.format(testEntry, bayes.classifyNB(thisDoc, p0V, p1V, pAb)))
    print('#' * 64)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bayes.setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc: {}'.format(thisDoc))
    print('{} classified as {}'.format(testEntry, bayes.classifyNB(thisDoc, p0V, p1V, pAb)))
    print('#' * 64)

def test3():
    print('#' * 64)
    bayes.spamTest()
    print('#' * 64)

if __name__ == '__main__':
    test2()
