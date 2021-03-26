---
title: 用Mathematica将图片背底变成透明
tags: [mathematica]
categories: computer vision 
date: 2016-4-26
---

2016-4-26更新：新增针对彩色图片的处理。

有时候需要将图片的背底变成透明，从而适应各种不同背景(尤其不是白色背景)，使得图片能更融入到背景中去。专业中也遇到这样一个问题，需要将两个枝晶轮廓在一起比对看是否重合。将背底变为透明色，通常可以使用PowerPoint的"设置透明色"这一功能，但对于对比很明显的图片效果较好，对比不明显的图片则偏差很大，且不能人为操纵。这里采用Mathematica，仅用几行代码即可实现。

对于黑白图片，或可以将图片转为黑白的情形：
Mathematica源码为：
```cpp
img1 = Binarize[img];
img2 = ColorConvert[img1, "Grayscale"];
img3 = ImageApply[{#, Boole[# < .1]} &, img2]
```
程序思路：
先将导入的img图片通过Binarize函数二值化成为img1,再将img1的颜色空间转化为Grayscale灰度，这一步很重要，否则下一步无法进行。最后使用ImageApply函数对img2的每个像素添加一个alpha通道，具体为判断像素是否为黑色，若是则alpha通道为1;若否则alpha通道为0。

对于彩色图片，想要将某一颜色设为透明：
Mathematica源码为：
```cpp
imgB1 = SetAlphaChannel[imgB, 1];
imgB2 = ImageApply[
  If[(#[[1]] + #[[2]] + #[[3]])/3 < 0.9, #, {#[[1]], #[[2]], #[[3]],0}] &, imgB1]
```
程序思路：
首先应用SetAlphaChannel对导入的图片增加alpha通道，然后应用ImageApply对每个像素进行判断，这里的判断条件需要针对要研究的对象进行调节，如果不满足设为透明的条件，则保留原像素值;若满足，则保留像素的RGB值，但修改其alpha通道为0。


注意：输出图片的格式需要选择支持alpha通道的格式，如png，而不能选择jpg。

