import math
import operator
import pickle

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = dict()
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    else:
        shannonEnt = 0.0
    for key, value in labelCounts.items():
        prob = float(value) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    else:
        return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = list()
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(reducedFeatVec)
    else:
        return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    # print('baseEntropy is ({})'.format(baseEntropy))
    bestInfoGain = 0.0
    bestFeature = -1
    for axis in range(numFeatures):
        featList = [example[axis] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, axis, value)
            subProp = float(len(subDataSet)) / len(dataSet)
            subEntropy = calcShannonEnt(subDataSet)
            newEntropy += subProp * subEntropy
        else:
            infoGain = baseEntropy - newEntropy
            # print('feature ({}) harvests ({}) infoGain'.format(axis, infoGain))
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = axis
                minEntropy = newEntropy
    else:
        # print('feature ({}) gives the minimum entropy ({})'.format(bestFeature, minEntropy))
        return bestFeature


def majorityCnt(classList):
    classCount = dict()
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    else:
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        ret = classList[0]
        print('type1 return ({})'.format(ret))
        return ret
    if len(dataSet[0]) == 1:
        ret = majorityCnt(classList)
        print('type2 return ({})'.format(ret))
        return ret
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: dict()}
    labels.pop(bestFeat)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    print('bestFeatLabel: {}'.format(bestFeatLabel))
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
        print('({}) -> ({})'.format(value, myTree[bestFeatLabel][value]))
    else:
        print('type3 return ({})'.format(myTree))
        return myTree

def classify(inputTree: dict, featLabels: list, testVec: list):
    firstStr, secondDict = list(inputTree.items())[0]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    if isinstance(secondDict[key], dict):
        subTree = secondDict[key]
        classLabel = classify(subTree, featLabels, testVec)
    else:
        classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    pickle.dump(inputTree, open(filename, 'wb'))

def grabTree(filename):
    return pickle.load(open(filename, 'rb'))


