---
title: 用Mathematica统计区域所占面积比
tags: [mathematica]
categories: computer vision 
date: 2016-4-24
---
本文参考了百度Mathematica吧的[这个帖子](http://tieba.baidu.com/p/2222457786)。
统计图片中某个区域的面积可以有多种方法和多种软件，比如ImageJ(多谢柏爷推荐~)、ImageMagick(参见[这篇教程](http://www.ps314.com/photoshop/jiqiao/20131119/161.html))、PhotoShop(参见[这篇教程](http://www.ps314.com/photoshop/jiqiao/20131119/160.html))。这里采用Mathematica，仅用少于10行代码即可实现。

Mathematica源码为：

```cpp
img2 = Binarize[img];
totalPixels = Times @@ ImageDimensions[img2];
blackPixels = Count[ImageData[img2], a_ /; a < 0.5, {2}];
blackAreaFraction = blackPixels/totalPixels // N
```

程序思路：
先将导入的img图片通过Binarize函数二值化成为img2,再计算img2总的像素数totalPixels。针对图片的黑色部分，通过判断每个像素点的数值是否小于0.5而统计相应的数量。最后黑色部分像素数除以总像素数就得到区域的面积比。

这是一个基础版程序，有一个缺陷是其导入的图片必须是去边且去除杂质以后的图片，这意味着首先要用PhotoShop或截图软件等将图片裁剪并去掉杂色。

更高级的功能可以在此基础程序上继续添加：
- 去边：使用Mathematica的ImageCrop函数
- 去杂色：使用DeleteSmallComponents将一些小的杂色像素用背景像素替代。

