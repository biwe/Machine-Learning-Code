import numpy as np


def loadDataSet():
    """create data"""
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """create vocabulary list, which is never repeated"""
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """transform text input to vector. Corresponding element in vocabulary is set to 1
    if the word exist in text.
    Note: len(returnVec) = len(vocabList)"""
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    """This is a naive bayes classifier --- multi-variate Bernoulli event model."""
    num_y0 = sum(trainCategory)  # sample number of y = 0
    num_y1 = len(trainCategory) - sum(trainCategory)
    py_1 = num_y1/(num_y0 + num_y1)  # p(y=1)
    # +1 for numerator and +2 for Denominator is laplace smooth
    phi_x1_y1 = (np.sum(trainMatrix[trainCategory == 1, :], axis=0) + 1)/(num_y1 + 2)
    phi_x1_y0 = (np.sum(trainMatrix[trainCategory == 0, :], axis=0) + 1)/(num_y1 + 2)
    p0Vect = np.log(phi_x1_y0)  # just for convenience of calculate
    p1Vect = np.log(phi_x1_y1)

    return p0Vect, p1Vect, py_1


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """Note: here we just consider the case of element exist.
    namely, p(xi=1|y)"""
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)  # element-wise mult
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


listOposts, listClasses = loadDataSet()
myVocablist = createVocabList(listOposts)
# create empty matrix to contain the train vector
trainMat = []
for postinDoc in listOposts:
    trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
trainMat = np.array(trainMat)
listClasses = np.array(listClasses)
# train the model
p0Vect, p1Vect, py_1 = trainNB0(trainMat, listClasses)


# test data
test_txt = ['my', 'dog', 'has', 'stupid',  'garbage']
test_vect = np.array(setOfWords2Vec(myVocablist, test_txt))
result = classifyNB(test_vect, p0Vect, p1Vect, py_1)

print(result)
