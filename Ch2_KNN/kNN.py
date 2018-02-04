# encoding=utf-8
import numpy as np
import operator
#import matplotlib.pyplot as plt
from os import listdir

# 加载数据
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k近邻
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 处理数据格式
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 存特征
    returnMat = np.zeros((numberOfLines, 3))
    # 存标签
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        #按下面注释的方式写的话会报下表溢出，所以往空列表里添加元素要用append
        #classLabelVector[index] = listFromLine[-1]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值，防止数据过大过小影响分类器
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# 测试分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio*m)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs: m], 3)
        print "The classifier came back with: %d, the real answer is: %d" % \
              (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "The total error rate is: %f" % (errorCount/float(numTestVecs))

# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    videoGames = float(raw_input("percentage of time spent playing video games:"))
    flyMiles = float(raw_input("frequent flier miles earned per year:"))
    iceCream = float(raw_input("liters of ice cream consumed per year:"))
    inArr = np.array([flyMiles, videoGames, iceCream])
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 把输入标准化
    norm_inArr = (inArr - minVals)/ranges
    # 找norm_inArr的kNN
    classifierResult = classify0(norm_inArr, normMat, datingLabels, 3)
    print "You will probably like this person " + resultList[classifierResult-1] + "."

# 实现手写数字识别系统
# 把32*32的图像转换为1*1024的行向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# 手写数字识别
def handwritingClassTest():
    # 获得文件夹里所有文件的名字
    trainingFileList = listdir('trainingDigits')
    # 存储各文件对应的数字分类标签
    hwLabels = []
    # 训练集里一共有多少文件
    m = len(trainingFileList)
    # 训练样本矩阵
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 获得带格式的文件名
        fileNameStr = trainingFileList[i]
        # 取得无后缀的文件名
        fileName = fileNameStr.split('.')[0]
        # 取得该文件的标签
        fileLabel = int(fileName.split('_')[0])
        # 把该标签添加到标签列表里
        hwLabels.append(fileLabel)
        # 把文件转化为向量，并添加到训练样本矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # 处理测试集
    # 测试集文件名列表
    testFileList = listdir('testDigits')
    # 测试集文件个数
    mTest = len(testFileList)
    # 统计分类错误个数
    errorCount = 0.0
    for i in range(mTest):
        # 预测当前文件的标签，和正确的标签对比，求出分类正确率
        # 带后缀的文件名
        testFileNameStr = testFileList[i]
        # 把文件转换成向量
        inX = img2vector('testDigits/%s' % testFileNameStr)
        # 获得当前文件的正确分类标签(去掉后缀、取下划线前半截)
        testFileName = testFileNameStr.split('.')[0]
        testFileLabel = int(testFileName.split('_')[0])
        # 预测当前文件的分类标签
        predictedLabel = int(classify0(inX, trainingMat, hwLabels, 3))
        print "The classifier came back with: %d, while the real answer is: %d" % \
              (predictedLabel, testFileLabel)
        if predictedLabel != testFileLabel:
            errorCount += 1.0
    # 计算分类错误率
    errorRate = errorCount/float(mTest)
    print "\nThe total number of errors is: %d" % errorCount
    print "The total error rate is: %f" % errorRate


if __name__ == "__main__":
    # 约会对象测试
    # classifyPerson()
    # 手写数字识别
    handwritingClassTest()
