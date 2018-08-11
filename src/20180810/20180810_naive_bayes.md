
# 基于贝叶斯决策理论的分类方法

### 1 概述

> 我们称之为“朴素”，是因为整个形式化过程只做最原始、最简单的假设。

* 优点：在数据较少的情况下仍然有效，可以处理多类别问题
* 缺点：对于输入数据的准备方式较为敏感
* 适用数据类型：标称型数据

### 2  使用朴素贝叶斯进行文档分类

1. 收集数据：可以使用任何方法。本章使用RSS源。
2. 准备数据：需要数值型或者布尔型数据。
3. 分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
4. 训练算法：计算不同的独立特征的条件概率。
5. 测试算法：计算错误率。
6. 使用算法：一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。


```python
from numpy import *

# 将词表转换为向量
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec
```


```python
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
myVocabList
```




    ['flea',
     'dalmation',
     'is',
     'stop',
     'buying',
     'garbage',
     'take',
     'help',
     'him',
     'quit',
     'food',
     'to',
     'problems',
     'I',
     'maybe',
     'has',
     'posting',
     'my',
     'ate',
     'please',
     'licks',
     'how',
     'park',
     'not',
     'dog',
     'love',
     'so',
     'steak',
     'cute',
     'mr',
     'worthless',
     'stupid']




```python
setOfWords2Vec(myVocabList, listOPosts[0])
```




    [1,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     1,
     0,
     1,
     0,
     1,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0]



下面讨论概率计算问题。由贝叶斯公式：

<center>$p(c_i|\boldsymbol{w})=\frac{p(\boldsymbol{w}|c_i)p(c_i)}{p(\boldsymbol{w})}$</center>

其中$\boldsymbol{w}$是一个向量，由多数值组成。

所以计算概率的伪代码如下：

```
计算每个类别中的文档数目
对每篇训练文档：
    对每个类别：
        如果词条出现在文档中 → 增加该词条的计数值
        增加所有词条的计数值
    对每个类别：
        对每个词条：
            将该词条的数目除以总词条数目得到条件概率
    返回每个类别的条件概率
```


```python
# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0
    p1Denom = 2.0               #change to 2.0
    for i in range(numTrainDocs):
        # 向量相加
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    # 对每个元素做除法
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive
```


```python
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
```


```python
print(pAb) # 任意文档是侮辱性文档的概率
```

    0.5



```python
# 构建完整的分类器

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
```


```python
testingNB()
```

    ['love', 'my', 'dalmation'] classified as:  0
    ['stupid', 'garbage'] classified as:  1



```python
# 朴素贝叶斯词袋模型

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
```

### 3 示例：使用朴素贝叶斯过滤垃圾邮件

1. 收集数据：提供文本文件。
2. 准备数据：将文本文件解析成词条向量。
3. 分析数据：检查词条确保解析的正确性。
4. 训练算法：使用我们之前建立的trainNB0()函数。
5. 测试算法：使用classifyNB()，并且构建一个新的测试函数来计算文档集的错误率。
6. 使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。


```python
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    
    # 导入并解析文本文件
    for i in range(1,26):
        wordList = textParse(open('../data/email/spam/%d.txt' % i, encoding='gbk').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../data/email/ham/%d.txt' % i, encoding='gbk').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)          #create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    
    # 随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
#             print("classification error",docList[docIndex])
    errorRate = float(errorCount)/len(testSet)
    print('the error rate is: ', errorRate)
    return errorRate
    #return vocabList,fullText
```


```python
errorRate = 0
for i in range(10):
    errorRate += spamTest() / float(10)
print('avg error rate: ', errorRate)
```


    the error rate is:  0.2
    the error rate is:  0.0
    the error rate is:  0.0
    the error rate is:  0.0
    the error rate is:  0.0
    the error rate is:  0.0
    the error rate is:  0.0
    the error rate is:  0.0
    the error rate is:  0.1
    the error rate is:  0.0
    avg error rate:  0.03

