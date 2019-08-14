import math
import operator

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
            reducedFeatVec = featVec[:axis] + featVec[axis+1:]
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
    print('+' * 64)
    print('dataSet: {}'.format(dataSet))
    print('labels: {}'.format(labels))
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        print('type1 return from ({})'.format(classList))
        print('=' * 64)
        return classList[0]
    if len(dataSet[0]) == 1:
        print('type2 return from ({})'.format(classList))
        print('=' * 64)
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: dict()}
    print('bestFeat: {}'.format(bestFeat))
    print('bestFeatLabel: {}'.format(bestFeatLabel))
    labels.pop(bestFeat)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
        print('myTree: {}'.format(myTree))
    else:
        print('-' * 64)
        return myTree
