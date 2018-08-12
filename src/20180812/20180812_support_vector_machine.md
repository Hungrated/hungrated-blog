
# 支持向量机（Support Vector Machine, SVM）

### 1 概述

简单点讲，SVM 就是个分类器，它用于回归的时候称为SVR（Support Vector Regression），SVM和SVR本质上都一样。

SVM的目的：**寻找到一个超平面使样本分成两类，并且间隔最大**。而我们求得的$w$就代表着我们需要寻找的超平面的系数。即

<center>$max\frac{1}{\left \| w \right \|}, s.t., y_i(w^Tx_i + b) \geq  1, i = 1, ..., n$</center>

### 2 SVM优化问题基本描述

目标函数

<center>$d=\frac{|w^Tx+\gamma|}{\left \| w \right \|}$</center>

优化目标是使$d$最大化。所有支持向量上的样本点，都满足

<center>$|w^Tx+\gamma|=1$</center>

化简得

<center>$d=\frac{1}{\left \| w \right \|}$</center>

我们只关心支持向量上的点，随后我们求解$d$的最大化问题变成了$\|w\|$的最小化问题。进而$\|w\|$的最小化问题等效于

<center>$min\frac{1}{2}\left\|w\right\|^2$</center>

等效是为了在进行最优化的过程中对目标函数求导时比较方便，但不影响最优化问题最后的求解。我们将最终的目标函数和约束条件放在一起进行描述：

<center>$min\frac{1}{2}\left\|w\right\|^2,$</center>

<center>$s.t., y_i(w^Tx_i + b) \geq  1, i = 1, ..., n$</center>

这就是支持向量机的基本型。

* 优点：泛化错误率低，计算开销不大，结果易解释
* 缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题
* 适用数据类型：数值型和标称型数据

### 3 SVM的一般流程

1. 收集数据：可以使用任意方法。
2. 准备数据：需要数值型数据。
3. 分析数据：有助于可视化分隔超平面。
4. 训练算法：SVM的大部分时间都源自训练，该过程主要实现两个参数的调优。
5. 测试算法：十分简单的计算过程就可以实现。
6. 使用算法：几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改。


```python
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))


'''
Non-Kernel Versions below
'''

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas
```


```python
dataArr, labelArr = loadDataSet('../data/SVMTestSet.txt')
labelArr
```




    [-1.0,
     -1.0,
     1.0,
     -1.0,
     1.0,
     1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     1.0,
     -1.0,
     1.0,
     1.0,
     -1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     1.0,
     -1.0,
     -1.0,
     1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     -1.0,
     1.0,
     -1.0,
     -1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     -1.0,
     1.0,
     1.0,
     -1.0,
     -1.0,
     1.0,
     1.0,
     -1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     1.0,
     -1.0,
     1.0,
     -1.0,
     -1.0,
     1.0,
     1.0,
     1.0,
     -1.0,
     1.0,
     1.0,
     -1.0,
     -1.0,
     1.0,
     -1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     1.0,
     -1.0,
     1.0,
     1.0,
     1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0]




```python
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
```

    L==H
    iter: 0 i:1, pairs changed 1
    iter: 0 i:2, pairs changed 2
    iter: 0 i:3, pairs changed 3
    L==H
    iter: 0 i:5, pairs changed 4
    j not moving enough
    iter: 0 i:18, pairs changed 5
    L==H
    L==H
    j not moving enough
    iter: 0 i:28, pairs changed 6
    iter: 0 i:29, pairs changed 7
    iter: 0 i:30, pairs changed 8
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iter: 0 i:52, pairs changed 9
    j not moving enough
    iter: 0 i:55, pairs changed 10
    j not moving enough
    L==H
    j not moving enough
    L==H
    j not moving enough
    iter: 0 i:99, pairs changed 11
    iteration number: 0
    L==H
    iter: 0 i:1, pairs changed 1
    j not moving enough
    iter: 0 i:3, pairs changed 2
    j not moving enough
    L==H
    iter: 0 i:10, pairs changed 3
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    iter: 0 i:31, pairs changed 4
    iter: 0 i:33, pairs changed 5
    L==H
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iter: 0 i:62, pairs changed 6
    iter: 0 i:66, pairs changed 7
    iter: 0 i:67, pairs changed 8
    L==H
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    L==H
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iter: 0 i:46, pairs changed 1
    j not moving enough
    L==H
    iter: 0 i:55, pairs changed 2
    L==H
    L==H
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    L==H
    iter: 0 i:94, pairs changed 3
    j not moving enough
    iteration number: 0
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:52, pairs changed 1
    j not moving enough
    L==H
    iter: 0 i:62, pairs changed 2
    L==H
    j not moving enough
    j not moving enough
    iter: 0 i:95, pairs changed 3
    iter: 0 i:96, pairs changed 4
    iteration number: 0
    L==H
    iter: 0 i:1, pairs changed 1
    j not moving enough
    j not moving enough
    iter: 0 i:10, pairs changed 2
    j not moving enough
    L==H
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:33, pairs changed 3
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    L==H
    L==H
    L==H
    j not moving enough
    iter: 0 i:52, pairs changed 4
    j not moving enough
    j not moving enough
    L==H
    iter: 0 i:62, pairs changed 5
    L==H
    j not moving enough
    j not moving enough
    iteration number: 0
    L==H
    iter: 0 i:1, pairs changed 1
    iter: 0 i:5, pairs changed 2
    L==H
    j not moving enough
    L==H
    L==H
    j not moving enough
    iter: 0 i:29, pairs changed 3
    j not moving enough
    j not moving enough
    iter: 0 i:43, pairs changed 4
    j not moving enough
    iter: 0 i:54, pairs changed 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    iter: 0 i:8, pairs changed 1
    j not moving enough
    L==H
    L==H
    j not moving enough
    iter: 0 i:31, pairs changed 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:62, pairs changed 3
    L==H
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    iter: 0 i:52, pairs changed 1
    j not moving enough
    j not moving enough
    iter: 0 i:87, pairs changed 2
    L==H
    L==H
    L==H
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:13, pairs changed 1
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iter: 0 i:39, pairs changed 2
    iter: 0 i:43, pairs changed 3
    j not moving enough
    iter: 0 i:54, pairs changed 4
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    iter: 0 i:5, pairs changed 1
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:43, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iter: 1 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    L==H
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:24, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    iter: 0 i:0, pairs changed 1
    j not moving enough
    iter: 0 i:8, pairs changed 2
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:29, pairs changed 3
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 5 i:54, pairs changed 1
    j not moving enough
    L==H
    L==H
    j not moving enough
    j not moving enough
    iteration number: 0
    L==H
    j not moving enough
    L==H
    L==H
    j not moving enough
    L==H
    iter: 0 i:24, pairs changed 1
    iter: 0 i:27, pairs changed 2
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    iter: 0 i:5, pairs changed 1
    j not moving enough
    L==H
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    L==H
    iter: 0 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    iter: 11 i:0, pairs changed 1
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    iter: 6 i:17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    iter: 7 i:54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 4 i:52, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 5
    iter: 5 i:8, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    iter: 3 i:17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    iter: 1 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    L==H
    iter: 6 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    iter: 13 i:17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i:54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    iter: 6 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    iter: 12 i:17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    iter: 4 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 4 i:54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 3 i:55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    iter: 4 i:52, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    L==H
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 16 i:54, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 22
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 23
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 24
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 25
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 26
    j not moving enough
    iter: 26 i:29, pairs changed 1
    j not moving enough
    j not moving enough
    iter: 26 i:55, pairs changed 2
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    iter: 12 i:17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    iter: 0 i:55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    L==H
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    L==H
    j not moving enough
    iteration number: 4
    j not moving enough
    L==H
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    L==H
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    L==H
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    L==H
    j not moving enough
    iteration number: 11
    j not moving enough
    L==H
    j not moving enough
    iteration number: 12
    j not moving enough
    L==H
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    L==H
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    L==H
    j not moving enough
    iteration number: 18
    j not moving enough
    L==H
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    L==H
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 22
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 23
    j not moving enough
    L==H
    j not moving enough
    iteration number: 24
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 25
    j not moving enough
    L==H
    j not moving enough
    iteration number: 26
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 27
    j not moving enough
    L==H
    j not moving enough
    iteration number: 28
    j not moving enough
    L==H
    j not moving enough
    iteration number: 29
    j not moving enough
    L==H
    j not moving enough
    iteration number: 30
    j not moving enough
    L==H
    j not moving enough
    iteration number: 31
    j not moving enough
    L==H
    j not moving enough
    iteration number: 32
    j not moving enough
    L==H
    j not moving enough
    iteration number: 33
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 34
    j not moving enough
    L==H
    j not moving enough
    iteration number: 35
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 36
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 37
    j not moving enough
    L==H
    j not moving enough
    iteration number: 38
    j not moving enough
    L==H
    j not moving enough
    iteration number: 39
    j not moving enough
    L==H
    j not moving enough
    iteration number: 40



```python
b
```




    matrix([[-3.61815721]])




```python
alphas[alphas > 0]
```




    matrix([[0.10242867, 0.24268542, 0.00960676, 0.33550732]])


