# NNI学生项目2020

## Task 1.3.1

ID: 14 Name: 中科大飙车队

### 任务描述

跑通 NNI [Feature Engineering Sample](https://github.com/SpongebBob/tabular_automl_NNI)

### 代码分析

我们只对作者给出的第一个简单的例子分析，针对其他数据集的情况可以作简单的类比即可。

#### 数据预处理

原文的数据集是一个二分类问题，数据量也不算大。有1999个样本，每个样本的数据维度是40维。

原文中的数据有一些特点：一个是有部分的缺失值，一个是有不少特征是非数值的特征。对于这两个情况，作者对于缺失值的情况采取了Pandas中的fillna来解决问题，对于非数值特征的情况，采用了sklearn.preprocessing的LabelEncoder进行处理，LabelEncoder是一种对于非数值特征简单编码为数值特征的方法。主要的实现在model.py当中。

#### 机器学习算法

原文中采用的机器算法是LightGBM，是一种基于决策树的集成学习算法，特点是训练速度很快，效率很高，即使采用NNI来自动调参也会比较节省时间。主要的实现在model.py当中。

#### 特征选取

原生的NNI并不支持高阶的特征组合和选取，只有两种比较简单的工具GradientFeatureSelector和GBDTSelector。而该项目的作者没有选用原生NNI的特征选取工具，而是做了自己的实现。

我们首先介绍作者组合特征的方式，在const.py中作者使用了多种组合的方式，包括count,crosscount,aggregate_{min,max,mean,median,var},nunique,histstat,target,embedding等组合方式。

> count: 计算某一列中的每个特征出现的次数并作为样本新的特征
>
> crosscount:将某两列或多列的组合特征当成单个特征$x_i'=(x_j,x_k)$，计算出现的次数并作为样本新的特征
>
> nunique: 考虑某列的元素是否为唯一的
>
> aggregate: 把两列中的元素合并到一起，每个样本仅由一行来代表，计算min max等对应的统计量
>
> embedding:在多类别的特征上做，当作自然语言做编码
>
> histstat在直方图上得到聚类的结果

作者在search_space.json里面也只使用了count,aggregate,crosscount这三种方法。

### 环境配置

​	在Windows下开展实验

1. 配置NNI基础环境。原始仓库中要求NNI为0.9.1版本，实际上配置1.5最新版本也可以。

2. 通过git clone的方式下载代码包

3. 配置需要的其他安装包。

   ```clike
   pip install lightgbm
   ```

   ```
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim
   ```
   
4. 使用NNI命令运行程序

   ```
   nnictl create --config config.yml
   ```

*一些不算坑的小坑*

- 在服务器上配置，队友们都会发现各种各样奇怪的问题

- 程序运行开始会报错，有可能是config.yml的这里出现问题

  ```python
  trial:
    command: python3 main.py #如果出现bug可以试着改为python
    codeDir: .
    gpuNum: 0
  ```

### 实现效果



### 结果分析

