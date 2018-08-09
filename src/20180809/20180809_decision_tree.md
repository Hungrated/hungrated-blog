
# 决策树（Decision Tree）相关算法

### 1 概念

决策树是一种树形结构，每个内部节点表示一个属性判断，每条分支代表一个判断结果的输出，每个叶子节点代表一种分类结果。决策树的生成属于监督学习。

* 优点：计算复杂度不高，输出结果易于理解，对中间值缺失不敏感，可以处理不相关特征数据
* 缺点：可能会产生过度匹配问题
* 适用数据类型：数值型和标称型

### 2 决策树的构造

#### 2.1 使用信息论划分数据集

划分数据集的大原则是：将无序的数据变得更加有序。在构造决策树时，我们需要解决的第一个问题就是，当前数据集上哪个特征在划分数据分类时起决定性作用。因此必须评估每个特征。

测试完成后，原始数据集被划分为几个数据子集，它们分布在第一个决策点所有分支上，接着递归划分，直到数据子集内的数据均属于同一类型为止。

```
# 检测数据集中的每个子项是否属于同一分类：
If so return 类标签；
Else
    寻找划分数据集的最好特征
    划分数据集
    创建分支节点
        for 每个划分的子集
            调用函数createBranch并增加返回结果到分支节点中
    return 分支节点
```

#### 2.2 决策树的一般流程

1. 收集数据：可以使用任何方法
2. 准备数据：决策树构造算法只适用于标称型数据，因此数值型数据必须离散化
3. 分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期
4. 训练算法：构造树的数据结构
5. 测试算法：使用经验树计算错误率
6. 使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义

#### 2.3 信息增益与熵

在划分数据集之前之后信息发生的变化称为信息增益，知道如何计算信息增益，我们就可以
计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择。

熵定义为信息的期望值，在明晰这个概念之前，我们必须知道信息的定义。如果待分类的事
务可能划分在多个分类之中，则符号$x_i$的信息定义为

<center>$l(x_i) = -log_2p(x_i)$</center>
  
其中$p(x_i)$是选择该分类的概率。

为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值，通过下面的公式得到：
  
<center>$H = -\sum_{i=1}^{n}p(x_i)log_2p(x_i)$</center>

其中$n$是分类的数目。

下面编程实现构造决策树（ID3算法）。


```python
from math import log

'''

计算给定数据集的香农熵

@param data 待计算数据集

'''
def calc_shannon_ent(data):
    num_entries = len(data)
    label_counts = {}
    
    # 为所有可能分类创建字典
    for feat_vec in data:
        cur_label = feat_vec[-1]   
        if cur_label not in label_counts.keys():
            label_counts[cur_label] = 0
        label_counts[cur_label] += 1
        
    # 计算香农熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent
```


```python
'''

按照给定特征划分数据集

@param data 待划分的数据集
@param axis 划分数据集的特征
@param value 需要返回的特征的值

'''

def split_data_set(data, axis, value):
    ret_data_set = []
    for feat_vec in data:
        if feat_vec in data:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set
```


```python
'''

创建数据集

'''

def create_data_set():
    data = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data, labels
```


```python
'''

选择最好的数据集划分方式

@param data 数据集

'''

def choose_best_split_feature(data):
    num_features = len(data[0]) - 1
    base_entropy = calc_shannon_ent(data)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data, i, value)
            prob = len(sub_data_set) / float(len(data))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy + new_entropy
        if info_gain > best_info_gain:            
            best_info_gain = info_gain
            best_feature = i
    return best_feature
```

> 函数choose_best_split_feature中调用的数据需要满足一定的要求：第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签


```python
'''

计算出现次数最多的分类名称

@param class_list 类别列表

'''

import operator

def majority_cnt(class_list):
    class_count={}
    for vote in class_list:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]
```


```python
'''

创建决策树

@param dataSet 数据集
@param labels 标签集

'''

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majority_cnt(classList)
    bestFeat = choose_best_split_feature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(split_data_set(dataSet, bestFeat, value),subLabels) 
    return myTree
```


```python
data, labels = create_data_set()
data2 = split_data_set(data, 0, 1)
print(data)
print(data2)
```

    [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    [[1, 'yes'], [1, 'yes'], [0, 'no'], [1, 'no'], [1, 'no']]



```python
calc_shannon_ent(data)
```




    0.9709505944546686




```python
choose_best_split_feature(data)
```




    0




```python
tree = createTree(data, labels)
```


```python
tree
```




    {'no surfacing': {0: {'flippers': {0: 'no', 1: 'no'}},
      1: {'flippers': {0: 'no', 1: 'no'}}}}


