---
title: ImageJ 用户指南 -- 6. 菜单栏之Edit
tags: [ImageJ]
categories: programming
date: 2018-9-4
---

# 本章说明
这里详解Edit菜单的功能。
# Edit
## Undo
撤销上一步操作。
## Cut
将当前选区中的内容复制到内部剪贴板，然后用当前背景色填充选区。
## Copy
将当前选区中的内容复制到内部剪贴板，如果没有选区，则复制整个图片。
## Copy to system
将当前选区中的内容复制到系统剪贴板。
## Paste
将内部剪贴板的内容（如果为空，则是系统剪贴板）粘贴到当前图片。
## Paste Control
粘贴以后，使用该菜单控制图片的粘贴方式。
## Clear
将选区中的内容清除，换成当前的背景色。Backspace和Del是该操作的快捷键。
## Clear Outside
将当前选区以外的区域清除，换成背景色。
## Fill
将当前前景色填充进当前选区。
## Draw
使用当前的前景色和线宽绘制当前选区的轮廓。使用Edit-Options-Colors设定前景色和背景色，使用Edit-Options-Line Width设定线宽。
## Invert
对当前选区或当前图片进行取反。对于8-bit和RGB图片，Invert总是使用$min=0$和$max=255$。对于16-bit和32-bit的图片，分别使用真实的最小和最大值。
## Selection
### Select All
创建一个与图片相同大小的矩形选区
### Select None
使当前图片的选区失效
### Restore Selection
恢复之前的寻去到它原先的位置。该命令可以用于在不同图片之间传递选区，也可以用于恢复之前不小心删除的ROI。
在不同图片之间传递ROI的方式有：
- 激活有当前选区的图片，然后激活要传递选区的图片，然后Edit-Selection-Restore Selection
- 使用ROI管理器
- 使用Analyze-Tools-Synchronize Windows

### Fit Spline
用一个三次样条(cubic spline)曲线拟合一个多边形polygon或多线polyline选区。
### Fit Circle
用一个圆circle拟合一个多点（至少三个点）或区域。不支持复合选区。如果是一个非闭合选区（比如点或线），拟合算法用的是基于Netwon的Pratt拟合；如果是一个闭合选区，该命令就是创建一个与该选区面积相同、重心相同的圆。
### Fit Ellipse
用一个椭圆拟合一个选区，该椭圆与原始选区有相同的面积、取向和重心。
### Interpolate
把当前选区转化成一个亚像素的ROI。
### Convex Hull
将多边形选区转成它的凸包（凸包可看成紧紧套在选区的各个角点上的橡胶带）。
### Make Inverse
反选选区，将原先选区的“内部”变成了“外部”。
### Create Mask
创建一个新的名为“Mask”的8-bit图片，内部的像素是255，外部是0。默认下该图片的LUT是反的，所以黑色是255，白色是0，除非Process-Binary-Options中的Black Background勾选。
### Create Selection
从一个做过阈值处理的图片或一个二值mask中创建一个选区。
### Properties
打开一个对话框，使得用户设置画笔颜色Stroke color和画笔宽度，或者设置填充颜色。注意，选区只能被填充或绘制轮廓，不能两者同时设定。
也能通过勾选List coordiantes显示选区的XY坐标。
注意，该命令仅对当前活动选区有效。而ROI管理器的Properties（在Analyze-Tools-ROI Manager）对多个ROI有效。
### Rotate
旋转选区
### Enlarge
通过设定特定数目的像素来扩大或缩小选区。
该项设为0可以讲一个复合选区转为一个多边形选区。
### Make Band
基于当前的选区形成一个条带，即可视为在当前选区上长出了一个条带。
### Specify
打开一个对话框，允许用户定义一个矩形或椭圆形选区。可以定义大小和位置。
### Straighten
该命令可以把图片中的弯曲的对象变直，比如图片中有条弯曲的河，通过该命令将该河拉直并提取成一张新的图片。该对象必须提前用分段直线工具标示出来。
### To Bounding Box
将一个非矩形的寻去转为完全包含它的最小的矩形。
### Line to Area
将一个线段选择转为一个选区。
### Area to Line
将一个选区转为它的轮廓。
### Image to Selection
创建一个图像选区ImageROI。
### Add to Manager
将当前选区加入ROI管理器。

## Options
使用该命令来改变ImageJ的用户偏好设置。
### Line Width
改变线宽，用来改变Line Selections的线宽和Edit-Draw的线宽。
### Input/Output
改变某些输入和输出的设置，比如JPEG的质量、table的后缀名、Results Table的选项。
### Fonts
改变字体，改变Text Tool的文本显示和Image-Stacks-Label。
### Plots
使用该对话框来控制ImageJ所产生的各种Plots的显示形式，如Image-Stacks-Plot Z-axis Profile、Analyze-Plot Profile。
比如坐标轴的长度、y轴的范围、是否绘制网格线等。
### Rounded Rect Tool
设置圆角矩阵选择工具的属性。
### Arrow Tool
设置箭头工具的属性
### Point Tool
设置点工具的属性
### Wand Tool
设置魔棒工具的属性
### Colors
设置前景色、背景色和选区工具的颜色。
### Appearance
控制图片怎样被显示，工具条怎样显示更好，以及设置菜单字体尺寸等。
### Conversions
控制图片怎样从一种格式转换为另一种格式，比如转换过程中是否缩放，RGB怎样转成灰度等。
### Memory & Threads
设置ImageJ可用的最大内存，以及当处理stack时线程数目。
### Proxy Settings
修改Java虚拟机的代理。
### Complier
设置所编译的插件的Java版本。
### DICOM
设置与DICOM图片相关的参数。
### Misc
设置其他的一些选项，有：
- Divide by zero value：设置当除以0时怎样处理，默认是infinity无穷大，也可以设置max（最大的正值）和NaN（不是一个数字）。
- Use pointer cursor：如果勾选了，ImageJ将会使用一个箭头指针，而不是默认的交叉十字类型的指针。
- Hide "Process Stack?" dialog：勾选后，ImageJ将不会显示询问是否处理所有的slices，而是直接仅仅处理当前slice。
- Require control/command key for shortcuts：勾选后，按快捷键时需要按下Ctrl。
- Move isolated plugins to Misc. menu：可以有效降低Plugin菜单的大小，防止一直显示到屏幕底部
- Run single instance listener：勾选后，ImageJ将会使用sockets来阻止多个实例开启。
- Debug mode：勾选后，ImageJ将会把调试信息显示在Log窗口中。

### Reset
将会在ImageJ退出后，删除"IJ\_pref.txt"这个文件，然后在ImageJ重启后使用所有参数的默认值。

