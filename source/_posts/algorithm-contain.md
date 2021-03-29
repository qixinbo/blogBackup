---
title:  算法赏析——判断某点是否在某区域内
tags: [algorithm]
categories: algorithm
date: 2021-3-28
---

# 问题描述
给定一个多边形区域，怎样判断某个点是否在该区域内？
如下图所示的蓝色多边形框，判断某点是否在该框内。
![area](https://user-images.githubusercontent.com/6218739/112778360-f66fe100-9076-11eb-9ec4-5e197aebdebd.png)

# 定义域
先写出该蓝色框的坐标序列：
```python
import numpy as np
import matplotlib.pyplot as plt

poly = np.array([(0,0),(1,0),(0.7,0.7),(1,1),(0,1),(0.5,0.5),(0,0)])
```
注意，该坐标序列是首尾相接的。
然后，定义出任意数量、任意位置的随机点：
```python
pts = np.random.rand(80).reshape((40,2))
```
这里给出了40个随机点。

# 解法
为了简单起见，以随机点[0.3, 0.4]为例：
```python
o = np.array([(0.3, 0.4)])
```
## 计算点与区域边界点形成的向量
```python
vs = poly - o
```
可以看作是以随机点为中心，区域边界点指向该中心的向量。
此例中，vs的值就是：
```python
[[-0.3 -0.4]
 [ 0.7 -0.4]
 [ 0.4  0.3]
 [ 0.7  0.6]
 [-0.3  0.6]
 [ 0.2  0.1]
 [-0.3 -0.4]]
```
## 计算点与区域边界点的距离
然后计算该向量的绝对长度：
```python
ls = np.linalg.norm(vs, axis=1)
```
ls的值为：
```python
[0.5        0.80622577     0.5        0.92195445   0.67082039    0.2236068  0.5]
```

## 计算相邻向量的外积
```python
cs = np.cross(vs[:-1], vs[1:])
```
这一步是计算以随机点为中心所形成的向量中，两个相邻向量所形成的外积。
可以详细看看numpy的cross函数的两个输入：
```python
[[-0.3 -0.4]
 [ 0.7 -0.4]
 [ 0.4  0.3]
 [ 0.7  0.6]
 [-0.3  0.6]
 [ 0.2  0.1]]
```
和
```python
[[ 0.7 -0.4]
 [ 0.4  0.3]
 [ 0.7  0.6]
 [-0.3  0.6]
 [ 0.2  0.1]
 [-0.3 -0.4]]
```
即第一个输入是排除了vs的最后一个向量，而第二个输入是排除了vs的第一个向量，这样两者一交错，就是在cross时计算的就是两个相邻向量的外积。
外积的概念可以参见[维基百科](https://zh.wikipedia.org/zh-cn/%E5%8F%89%E7%A7%AF)
![cross](https://user-images.githubusercontent.com/6218739/112794912-4d3ae200-909a-11eb-8fa4-cda6b524613e.png)
而numpy的具体cross函数的计算方式见[这里](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)。
因为这里输入的两个向量都是二维的，因此计算出来的虽然应该仍然是个向量，但这里只返回它的z轴长度（In cases where both input vectors have dimension 2, the z-component of the cross product is returned）。
因此，cs的值为：
```python
[ 0.4   0.37  0.03  0.6  -0.15 -0.05]
```
这里比较重要的是数值的符号。

## 计算相邻向量的内积
```python
dot = (vs[:-1]*vs[1:]).sum(axis=1)
```
这一步是计算以随机点为中心所形成的向量中，两个相邻向量所形成的内积。
内积的概念可以参见[维基百科](https://zh.wikipedia.org/zh-cn/%E7%82%B9%E7%A7%AF)。
![dot](https://user-images.githubusercontent.com/6218739/112796213-42814c80-909c-11eb-98bd-a62fd9a24ba1.png)
里面重要的一点：从代数角度看，先对两个数字序列中的每组对应元素求积，再对所有积求和，结果即为点积。从几何角度看，点积则是两个向量的长度与它们夹角余弦的积。这两种定义在笛卡尔坐标系中等价。

## 计算相邻向量的长度乘积
```python
ls = ls[:-1] * ls[1:]
```
这句简单，就是相邻两个向量的长度的乘积。

## 计算相邻向量的角度
```python
ang = np.arccos(dot/ls) * np.sign(cs)
```
该行有两部分组成：
（1）前一部分就是两个向量的点积除以这两个向量的长度乘积。由点积的定义可知，这样的除法得到了角度的余弦值。这样求反余弦后，就可以得到两个向量之间的角度。
（2）第二部分就是对上面的角度赋予符号，这个符号的正负是通过相邻向量的外积来得到的。

## 计算角度之和
```python
ang.sum() > np.pi
```
对上述角度求和，然后判断其大小。
这里就是整个算法的点睛之笔，假设一个人站在了这个随机点上，然后他在原地转圈：
（1）如果随机点在区域内部，那么这个人转一圈，其转过的角度就是2*pi；
（2）如果随机点在区域外部，那么这个人没法转一个完整的圈，而是转一个角度，然后又转回来，因此最终转过的角度就是0。
所以就可以根据这个角度之和来判断随机点与区域的关系。

将以上函数封装成一个统一的函数：
```python
def contain(poly, o):
    vs = poly - o
    ls = np.linalg.norm(vs, axis=1)
    cs = np.cross(vs[:-1], vs[1:])
    dot = (vs[:-1]*vs[1:]).sum(axis=1)
    ls = ls[:-1] * ls[1:]
    ang = np.arccos(dot/ls) * np.sign(cs)
    return ang.sum() > np.pi
```

# 可视化
```python
msk = np.array([contain(poly, i) for i in pts])

plt.plot(*poly.T)
plt.plot(*pts[msk].T, 'go')
plt.plot(*pts[~msk].T, 'ro')
plt.show()
```
结果为：
![vis](https://user-images.githubusercontent.com/6218739/112799536-11574b00-90a1-11eb-812e-21cdbdf4f45e.png)

