---
title: Mathematica版GetData——用Mathematica提取图片中的数据点
tags: [mathematica]
categories: programming
date: 2016-6-12
---

阅读文献时常常会遇到只给图像却没有原始数据的情形，此时如果想要提取数据，就得借助相应软件，目测用的最多的就是[GetData](http://www.getdata-graph-digitizer.com/)，这个是个商业软件，还有个好用的基于web的开源软件[WebPlotDigitizer](http://arohatgi.info/WebPlotDigitizer/app/)。这里我们基于Mathematica写一套能用于提取图片中数据点的代码。
参考文献(基本思路参考SE上的这个问题，但具体取点和去点方式不同)：
[Recovering data points from an image](http://mathematica.stackexchange.com/questions/1524/recovering-data-points-from-an-image)

Attention：这里的版本强烈依赖于Mathematica的版本，此处使用的是10.4版本，目测应该使用10以上版本，因为低版本中不会出现工具提示条。

# 实际坐标系与图像坐标系的对应
首先使用工具提示条中的“坐标工具”提取已知点的图像坐标，然后选择“复制坐标”并保存(注意保存的数据中的点的先后顺序)。
如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mma-getdata1.png)
利用这些图像中的坐标与实际坐标建立对应关系：
```cpp
trans = FindGeometricTransform[{{0, 1.2}, {0, 1.1}, {4, 0.82}, {2, 0.82}},
      {{20.630372492836678`, 146.88825214899714`}, {20.630372492836678`, 111.8166189111748`}, 
       {153.1805157593123`, 14.33810888252151`}, {86.1318051575931`, 14.33810888252151`}},TransformationClass->"Affine"][[2]];
```
注意是实际坐标在前，图像坐标在后。此次提取后注意找一个未知点验证一下，防止选择的变换方式出错。

# 提取曲线颜色
要正确识别出图像中要提取的曲线，必须先让程序知道该曲线的颜色，即RGB值。这里依然使用工具提示条的“坐标工具”，提取曲线上一点后，选择“复制颜色值”。这里仅提取了一个点的RGB值，很难覆盖整条曲线，所以再设置一个忍量，使得与此RGB值相近的点都可以被识别：
```cpp
objRGB = {{103, 125, 174}};
tolRGB = 40;
rangeRGB = Flatten[{objRGB - tolRGB, objRGB + tolRGB}, 1]/255.0;
img1 = ImageApply[
  If[#[[1]] > rangeRGB[[1, 1]] && #[[1]] < 
      rangeRGB[[2, 1]] && #[[2]] > rangeRGB[[1, 2]] && #[[2]] < 
      rangeRGB[[2, 2]] && #[[3]] > rangeRGB[[1, 3]] && #[[3]] < 
      rangeRGB[[2, 3]], {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}] &, img]
```
此忍量的值可视情形调节。以上策略就是当某点的RGB值在忍量之内时，就变为(0.0,0.0,0.0)，即黑色，否则则为白色(1.0,1.0,1.0)。
效果如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mma-getdata2.png)

# 选区和去噪点 
从上图可以看出，识别出的曲线中含有一些噪点，如果不处理掉就会影响结果。此时需要对图片进行精修，使其干净无污染。具体方法是采用工具提示条的“掩模工具”(话说10版本的MMA工具提示条真是逆天的存在)，将要选择的区域勾勒出来，然后选择“逆掩模为一个图”，之所以是“逆掩模”，是因为要将这部分选出来而不是去除，也不选择“逆掩模为一个图像”，经测试选择“图”而不是“图像”，分辨率要更高。
逆掩模如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mma-getdata6.png)
将此逆掩模的图放入下面代码中的Masking参数中即可：
```cpp
curve = ImageApply[{1.0, 1.0, 1.0} &, img1, Masking -> 此处是那个选区]
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mma-getdata3.png)
可以看出噪点已经被去除。

# 提取数据点的位置
这一步就是提取上面曲线中的黑点的位置，代码为：
```cpp
curvLoc = Reverse /@ Position[ImageData[curve, DataReversed -> True], {0., 0., 0.}];
```
注意这里需要注意Position取得的位置是(1,1)在左上角，而图像坐标则是(1,1)在左下角，所以需要进行一系列变换，具体的变换规则如示意图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mma-getdata4.jpg)

# 作图并与解析解作对比
实际要提取的曲线是有具体的表达式的，此处将提取出的数据与解析表达式对比：
```cpp
Show[ListPlot[trans@curvLoc],Plot[x^((x-2)^2 E^-x)+E^-x, {x, 0, 10}, PlotStyle->Red]]
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mma-getdata5.png)
可以看出效果还不错。

目前还是以代码的形式操作，以后没准能有图形界面？
以上。
