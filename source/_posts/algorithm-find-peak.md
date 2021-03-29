---
title:  算法赏析——寻找线条的转折点
tags: [algorithm]
categories: algorithm
date: 2021-3-27
---

# 问题描述
图像中有一条线，如何判断这条线的转折点？
比如下面一张图：
![test](https://user-images.githubusercontent.com/6218739/112800422-35fff280-90a2-11eb-8a43-45d6a4682e3a.png)
目的是找到图中的三个转折点。

# 解法
## 找到轮廓线
```python
img = cv2.imread('test.png', 0)
conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
xs, ys = conts[:,:,0], conts[:,:,1]
```
这一步实际作用是通过寻找轮廓线，从像素类型的位图中提取有意义的这条线的坐标序列，即矢量序列。
同时将横坐标和纵坐标分别提取出来。

## 高斯模糊
```python
gxs = ndimg.gaussian_filter(xs, 15, mode='wrap')
gys = ndimg.gaussian_filter(ys, 15, mode='wrap')
```
对横纵坐标分别做高斯模糊，相当于对一维数据做高斯模糊，同时注意上面的轮廓线寻找到的序列是首尾相连，要用到wrap这个模式。

## 新旧坐标对比
```python
ds = ((xs-gxs)**2+(ys-gys)**2)**0.5
```
将高斯模糊后的坐标与之前的坐标进行对比，用标准差来衡量差距大小。

## 寻找局部极大值
```python
maxds = ndimg.maximum_filter(ds, 100, mode='wrap')
idx = np.where((ds > ds.std()*3) & (ds==maxds))[0]
```
这个地方首先使用一个极大值滤波，然后再通过两个判断条件：是否大于标准差的3倍以及同时等于局部极大值。
这样就找到了局部极大值点所在的位置。
当然也可以直接用那种寻找局部极值的算法，但不如这种“极大值滤波+大于某个阈值”的方法来得简单直接。

#可视化
```python
ax = plt.subplot(211)
ax.plot(xs, ys)
ax.plot(gxs, gys)
plt.plot(xs[idx], ys[idx], 'ro')

ax = plt.subplot(212)
ax.plot(ds)
ax.plot(maxds)
ax.plot(ds/ds*ds.std()*3)
ax.plot(idx, ds[idx], 'ro')
plt.show()
```
将结果可视化出来：
![vis_peak](https://user-images.githubusercontent.com/6218739/112804493-115a4980-90a7-11eb-8992-f2f8fb80b3b9.png)

