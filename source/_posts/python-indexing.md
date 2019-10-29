---
title: Numpy的广播Broadcasting和奇妙索引Fancy Indexing
tags: [python]
categories: programming
date: 2019-10-20
---

参考文献：
[Computation on Arrays: Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)
[Indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)

本文就是对上面两篇参考文献的翻译理解。

# 广播Broadcasting
广播的原则：
（1）如果两个数组的形状不同，那么形状小的那个数组填充成另一个数组的形状（注意是用形状1来向左填充，具体见下方示例）。
（2）如果两个数组的形状相同，但某一维度上数目不匹配，那么在这一维度上形状为1的数组扩展成另一个数组的形状。
（3）如果两个数组的形状相同，在某一维度上数目不匹配，但在这一维度上形状都不为1，那么报错。

## Example 1
我们想把下面两个数组相加：
```python
M = np.ones((2, 3))
a = np.arange(3)
```
两者的形状是：
```python
M.shape = (2, 3)
a.shape = (3,)
```
那么，根据规则1，需要先填充a的形状为：
```python
M.shape -> (2, 3)
a.shape -> (1, 3)
```
根据规则2，在第一维上，两者形状不匹配，那么继续扩展：
```python
M.shape -> (2, 3)
a.shape -> (2, 3)
```
因此M和a相当于：
```python
M -> array([[1., 1., 1.], [1., 1., 1.]])
a -> array([[0, 1, 2], [0, 1, 2]])
```
那么，两者相加即为：
```python
>>> M + a
array([[1., 2., 3.],
       [1., 2., 3.]])
```

## Example 2
这个例子是两个数组都要广播的情形。
```python
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
```
那么，两者的shape是：
```python
a.shape = (3, 1)
b.shape = (3,)
```
根据规则1，需要将b的shape填充为：
```python
a.shape -> (3, 1)
b.shape -> (1, 3)
```
然后根据规则2，需要将两者都扩展为：
```python
a.shape -> (3, 3)
b.shape -> (3, 3)
```
那么，实际两者扩展成：
```python
a -> array([[0, 0, 0],[1, 1, 1], [2, 2, 2]])
b -> array([[0, 1, 2],[0, 1, 2], [0, 1, 2]])
```
所以，两者相加为：
```python
>>> a + b
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
```
## Example 3
这个例子是两个例子不兼容的情形。
```python
M = np.ones((3, 2))
a = np.arange(3)
```
两者的形状为：
```python
M.shape = (3, 2)
a.shape = (3,)
```
根据规则1，要将a填充：
```python
M.shape -> (3, 2)
a.shape -> (1, 3)
```
根据规则2，要将a的第一维扩展：
```python
M.shape -> (3, 2)
a.shape -> (3, 3)
```
此时就会触发规则3，即在第二维上形状不相同，但又都不为1：
```python
>>> M + a
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
```
此时你可能想着如果不是向左填充形状1，而是向右填充，那么问题就可以解决了。但广播的规则规定了只能向左填充，这是为了防止出现混乱。其实此时可以人为手动将a的形状改为(3,1)，即使用np.newaxis，因为：
```python
>>> a[:, np.newaxis].shape
(3, 1)
```
所以：
```python
>>> M + a[:, np.newaxis]
array([[1., 1.],
       [2., 2.],
       [3., 3.]])
```
上述例子是针对于加号操作符，实际上上面的广播也适用于任意二元操作符。

# 奇妙索引Fancy Indexing
Fancy Indexing是指传递一个索引序列，然后一次性得到多个索引元素。
## 一维数组的索引
生成一个一维数组x，
```python
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)
```
返回结果：
```python
[51 92 14 71 60 20 82 86 74 74]
```
（1）一维索引
```python
ind = [3, 7, 4]
x[ind]
```
返回：
```python
array([71, 86, 60])
```
（2）多维索引
```python
ind = np.array([[3, 7],
                [4, 5]])
x[ind]
```
返回：
```python

array([[71, 86],
       [60, 20]])
```
因此，这里可以看出来，返回的array的shape是索引数组的shape，而不是被索引array的shape。
## 多维数组的索引
生成一个多维数组X：
```python
X = np.arange(12).reshape((3, 4))
X
```
返回：
```python
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
```
对于传给X的索引，第一个参数表示行row，第二个参数表示列column：
```python
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]
```
返回：
```python
array([ 2,  5, 11])
```
即分别取的是X[0, 2]、X[1, 1]和X[2, 3]的值。
注意，这个返回数组的shape仍然是索引数组的shape，而不是原始数组的shape。
再来一个多维的索引数组：
```python
X[row[:, np.newaxis], col]
```
返回：
```python
array([[ 2,  1,  3],
       [ 6,  5,  7],
       [10,  9, 11]])
```
可以逐个看一下第一个参数和第二个参数：
```python
>>> row[:, np.newaxis]
array([[0],
       [1],
       [2]])
>>> col
array([2, 1, 3])
```
两个的维度不同，那么就符合前面所说的广播的原则，即其实索引是这样的：
```python
X[[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[2, 1, 3], [2, 1, 3], [2, 1, 3]]]
```
仍然提醒一句，返回数组的shape就是索引数组被广播后的shape。
## 组合索引
上面的Fancy Indexing可以和其他的索引方式进行组合。
### Fancy Indexing + slice
如果索引中有切片操作slice，或者ellipsis、newaxis等，可以看做是一个多维切片：
```python
X[1:, [2, 0, 1]]
```
返回：
```python
array([[ 6,  4,  5],
       [10,  8,  9]])
```
### Fancy Indexing + mask
```python
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]
```
返回：
```python
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
```
