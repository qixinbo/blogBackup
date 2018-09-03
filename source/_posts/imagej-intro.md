---
title: ImageJ 用户指南-- 1.简介和安装
tags: [ImageJ]
categories: programming
date: 2018-9-1
---

# 开篇说明
ImageJ是一个优秀的开源图像处理工具，本系列是对ImageJ的官方[User Guide](https://imagej.nih.gov/ij/docs/guide/user-guide.pdf)的学习。

# 简介
ImageJ是一个基于Java平台的图像处理和分析工具（它的发行版Fiji也提供了其他语言，如Python，的开发接口，Fiji与ImageJ的关系，就跟Ubuntu和Linux的关系一样，即Fiji是ImageJ和它的很多插件的集合发行版），由美国国立卫生研究院NIH所创立和开发。因为它基于Java平台，所以通吃各大平台，如Windows、Mac OS和Linux，只要有Java运行环境即可安装和运行。
ImageJ的功能有：
- 显示、编辑、分析、处理、保存和打印8位、16位和32位（这些是色彩深度，简单解释见[这里](https://blog.csdn.net/panshun888/article/details/78278104)）
- 支持多线程，所以可以并行读取图片文件
- 支持像素操作，如创建图片直方图等，支持标准的图片处理功能，如对比度调节、锐化、平滑、边缘检测和中值滤波等
- 支持几何变换，如缩放、旋转、翻转等
- 强大的插件系统，可以任意定制自己想要的功能，同时拥有极好的插件生态

# 安装
ImageJ因为发展了多年，有很多版本，乍一看很容易弄混。最开始是1997年开发的ImageJ1，即ImageJ 1.x这些版本号的软件，目前也在活跃开发中。但目前所说的ImageJ，是指的ImageJ2，它是对ImageJ1的一个完全重写，更加便于二次开发等，同时它也保持了对ImageJ1的兼容性，所以以前的插件和宏都能在新的ImageJ上运行。
这里建议直接安装Fiji，它是ImageJ和常用插件的一个综合发行版。Fiji的进一步功能有：
- 更多的功能，如[图像配准](https://zh.wikipedia.org/wiki/%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86)、[图像拼接](https://zh.wikipedia.org/zh-cn/%E5%BD%B1%E5%83%8F%E6%8B%BC%E6%8E%A5)、[图像分割](https://zh.wikipedia.org/zh-cn/%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2)、特征提取和三维可视化
- 支持多种脚本语言，如BeanScript、Clojure、Jython、Python、Ruby等
- 便利的插件升级系统，可以追踪和提示插件是否有更新及后续安装

# 相关软件
还有很多其他相对大型的软件基于ImageJ来开发，如：
- [Bio7](https://bio7.org/)
- [BoneJ](http://bonej.org/)
- [TrakEM2](https://imagej.net/TrakEM2)

建议遇到问题时参考一下这些软件能不能有启发。

