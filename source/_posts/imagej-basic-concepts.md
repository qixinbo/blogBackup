---
title: ImageJ 用户指南 -- 2. 基本概念
tags: [ImageJ]
categories: computer vision 
date: 2018-9-1
---

# 本章说明
这一章主要介绍ImageJ的一些基本概念。

# 快捷键
ImageJ的快捷键在菜单中都有提示，且区分大小写，比如大写的A表示"Shift+A"。同时还要注意查看"Edit-Options-Misc-Require control key for shortcuts"是否勾选，如果未勾选，则快捷键不用按ctrl。
查看所有的快捷键：Plugins-Shortcuts-List shortcuts
ImageJ中所有的操作都是在目前激活的图片上进行的，即最前面的图片。在任意图片上按Enter，都会激活ImageJ的主窗口。
查找宏、命令、脚本和插件：小写的l。
ImageJ的撤销和重做：
- 因为缓存的限制，Edit-Undo仅能撤销最近一次的图像操作，如果Edit-Options-Memory& Threads中的Keep multiple undo buffers勾选后，撤销操作可以应用在多张图片中。
- File-Revert可将图片设置为最近一次保存的状态。
- 对于Selections选择，Edit-Selection-Restore Selection可以用来恢复所有错选操作
- ImageJ的redo重做是Process-Repeat Command，即再次执行上一次命令

# 图片类型和格式
图片是一个二维网格，长宽分别是像素的个数，即像素是图片的最小的单元。网格上的数值是像素的强度，它代表了像素的这个属性的强弱，比如一个灰度图片，强度越大，图片越白。另一方面，像素的强度用多少位二进制数来表示，就是图片的色彩深度，叫做bit，它代表了像素被编码的精度。比如一个2-bit的图片，它只能表示4种强度，即00(黑)、01(灰)、10(灰)、11(白)，而一个8-bit的图片，则可以表示256种灰度值。再比如一张RGB图，它能分别表示红蓝绿三个通道的256种值，因此它是24-bit的。RGB图也可以是32-bit的，即再加上一个表示透明度的8-bit通道。

如果不使用第三方插件，ImageJ可以打开如下格式：TIFF、GIF、JPEG、PNG、DICOM、BMP、PGM和FITS。

# Stacks
ImageJ可以在单一窗口中显示多个时间或空间相关的图片。这些图片集称为“Stacks”，这些图片称为"Slices"。在Stacks中，原先的二维的像素变成了一个体素Voxel，即在三维空间中的网格上的强度值。
在一个stack中的所有的slices都必须有相同的尺寸的色彩深度。
一个文件夹中的图片可以通过拖拽进入ImageJ窗口或者File-Import-Image Sequence来形成一个Stack。
创建一个新的Stack：File-New-Image，将Slices这一项设为大于1的数即可。
Image-Stacks中包含了常用的对stack进行的操作。大多数ImageJ的过滤器是对Stack中的所有Slices进行操作。

# 彩色图片
ImageJ主要用以下三种方式来处理彩色图片：
## 伪彩色图片
伪彩色图片实际是一张单通道的灰度图片，然后通过一个查找表lookup table (LUT)来为它分配颜色。
## 真彩色图片
真彩色图片有色彩空间的概念，常用的是RGB空间，以及HSB、YUV等。HSB就是色度、饱和度和亮度，这种颜色空间在处理颜色信息时特别有用。
色彩空间的转换在Image-Type中。
色彩空间中的色彩分割在:Image-Adjust-Color Threshhold。
## 组合图片
这种图片将各个通道都分开，所以可以对单个通道进行操作。Image-Color-Channel Tool

# 选区
选区Selections，也就是画出ROI (Regions of Interest)。尽管ImageJ可以同时显示多个ROIs，但一次只能激活一个ROI。
区域选择时，选择好的ROI可以进行以下操作：
- 测量：Analyze-Measure
- 绘制：Edit-Draw
- 填充：Edit-Fill
- 滤波：Process-Filters再选择子菜单

选区时初始颜色是ImageJ默认九种颜色中的一个，一旦创建后，就可以使用Edit-Selection-Property来设定。

# Overlay
Overlay是最好的对图片做注释的方式，因为它不会改变像素值。
Overlay可以看成是一个不可见的ROI管理器。
可以把多个ROIs放进一个Overlay中，这样就可以来回调用多个ROI，方法是Image-Overlay-Add Selection。
也可以把Overlay转成ROI管理器：Image-Overlay-To ROI Manager。

# 三维图片
原生的ImageJ对三维ROI支持不是很好，但有很多插件可以辅助：
- 3D Filters
- 3D Object Counter
- 3D Viewer
- Simple Neurite Tracer
- TrakEM2

# 设置和默认值
ImageJ的配置文件是IJ\_prefs.txt。
