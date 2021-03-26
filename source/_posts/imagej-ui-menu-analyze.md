---
title: ImageJ 用户指南 -- 9. 菜单栏之Analyze
tags: [ImageJ]
categories: computer vision
date: 2018-9-9
---

# 本章说明
这里详解Analyze菜单的功能。
# Analyze
## Measure
基于当前选择，在Results Table中计算和显示区域统计、线长、角度或者点坐标等信息。具体的测量操作可以在下方的Set Measurment对话框中进行指定。
## Analyze Particles
在二值图片或阈值处理过的图片上，对对象进行计算和测量。它是通过扫描图片或选区直到找到对象的边缘，然后用魔棒工具将对象的轮廓画出来，使用上面的Measure命令计算测量。
- Size：给定一个面积范围，如果particle的尺寸面积在该范围之外，其将被忽略。如果标度过图片，则使用真实单位所形成的物理面积，否则使用像素的平方做单位。
- Circularity：球形度范围，在此范围以外的particle将被忽略。
- Show：决定在分析之后怎样显示结果。Nothing：图片或Overlay都不显示，注意，如果该particle analyzer测量到的particles数目为0以及Show选择Nothing，那么就会显示一个空白图片；Outline：显示一张含有用数字标示的particle的轮廓的8-bit图片；Bare Outlines：8-bit图仅显示轮廓，不显示标签；Masks：一张8-bit图片，包含particles的对轮廓的填充；Ellipses：8-bit图片，包括最近似的椭圆；Count Masks：16-bit图片，包含particle的对轮廓的填充，同时用与particle number相对应的灰度值显示；Overlay Outlines：在overlay中显示particle的轮廓，删除之前的overlay；Overlay Masks：在overlay中显示particle的轮廓的填充，删除之前的overlay。
- Display Results：勾选后，每个particle的测量结果将在Results Table中显示
- Clear Results：勾选后，Results Table中的之前的结果将被清除
- Summarize：勾选后，将在一个Summary的表格中显示particle的个数、总面积、平均尺寸、面积分数和Set Measurements中的所有参数的平均值。
- Add to Manager：勾选后，测量到的particles都将添加进ROI管理器。
- Exclude on Edges：勾选后，碰到图片或选区边缘的particle将被忽略。
- Include Holes：勾选后，内部的孔洞将被作为每个Particle的内部区域，即ImageJ将会仅通过外边界来寻找每个Particle，内部的。不勾选此项，将会通过泛洪填充来寻找对象，然后会在Particle中排除孔洞。
- Record Starts：该选项允许插件和宏使用doWand函数来重新创建边界，CircularParticles宏展示了使用方法。
- In situ Show：勾选后，原始图片将被新图片替代，该选项对上面的overlay无效。

## Summarize
对于Results Table中的每一列，计算这一项的均值、标准差、最小和最大值。
## Distribution
从Results Table的选定列中创建该列数据的频率直方图。
## Label
该命令使用Results Table的行数来对当前的选区进行标注。
## Clear Results
清除结果
## Set Measurements
使用该对话框来指定Analyze-Measure、ROI管理器的Measure和Analyze-Analyze Particles怎样进行测量。对于阈值处理的图片，如果勾选了Limit to Threshold，则可以仅对高亮的像素点进行测量。
这些选项分成了两类：第一类是控制输出到Results Table中的测量的类型有哪些；第二类是怎样测量。
第一类的18个选项有：
- Area：面积，如果下面的Analyze-Set scale用来进行空间标度，那么面积就是真实面积，否则用像素面积
- Mean gray value：当前选区的平均灰度值。对于灰度图，就是所有灰度值加起来除以总个数；对于RGB图，使用之前介绍过的转换法则将每个像素转为灰度值；
- Standard deviation：灰度值的标准差。
- Modal gray value：出现频率最大的灰度值，即直方图中的高峰
- Min & Max gray level：最小和最大灰度值
- Centroid：中心点，即图片或选区中的所有像素点的XY坐标的平均
- Center of mass：这是用亮度加权的XY坐标点的平均。
- Perimeter：选区的外边界的长度。
- Bounding rectangle：包住选区的最小矩形。使用矩形的左上角的坐标及长宽表示。
- Fit ellipse：用椭圆来拟合选区，使用椭圆的主轴和次轴和角度来表示。如果上面的Centroid勾选后，椭圆的中心店也显示出来。注意，如果Analyze-Set Scale中的Pixel Aspect Ratio不勾选，那么ImageJ不能计算主轴和次轴的长度。
- Shape descriptors：计算和显示以下形状因子：Circularity球形度、Aspect ratio长宽比、Roundness和Solidity。
- Feret's dismeter：在选区边缘上两点之间的最大距离
- Integrated density：像素值的总和，它等于Area和Mean Gray Value的乘积。
- Median：像素值的平均值
- Skewness：均值的三次矩
- kurtosis：均值的四次矩
- Area Fraction：面积分数，对于阈值处理过的图片，它是红色高亮的像素的分数；对于非阈值处理过的图片，它是非零像素的分数。
- Stack position：在stack或hyperstack中的位置：slice、channel和frame。

第二类的选项是控制怎样测量：
- Limit to threshold：勾选后，仅阈值范围内的像素被测量
- Display level：勾选后，图片名字和slice的序号会在Results Table中记录。
- Invert Y coordinates：勾选后，XY的原点变成窗口的左下角，而不是默认的左上角。
- Scientific notation：勾选后，用科学计数法显示结果
- Add to Overlay：勾选后，所测量的ROI自动添加进Overlay
- Redirect to：从该菜单中选择要统计的图片，这使得可以在一张图片中的统计同样应用于另一张图片的相应区域。
- Decimal places：显示小数点的位数。

## Set Scale
使用该对话框来定义空间比例，从而使得测量能用真实单位显示，比如$mm$和$\mu m$。
在使用该命令之前，先用一个直线选区工具在已知距离上进行划线，然后再调用该对话框，在Known Distance和Unit中填入真实距离及单位即可。
如果将Pixel Aspect Ratio设为非1，还可以支持水平和垂直两个方向上不同的空间比例。
当勾选Global后，该比例将会应用于所有的当前session已打开的图片中。
## Calibrate
功能是使用不同的函数来拟合像素值和灰度值之间的关系。

## Histogram
计算和显示当前图片或选区的灰度值的分布直方图。
X轴是可能的灰度图，Y轴是该灰度值的像素个数。X轴下方的LUT用来显示图片的显示范围。再下方会显示总的像素个数、灰度值的平均值、标准差、最小、最大和modal值。
点击list或copy来存储直方图数据。点击Log来显示一个对数坐标的直方图。点击live可以在浏览stack或移动ROI时见识直方图的变化。

## Plot Profile
显示沿着一条线或一个矩形选区的像素值的强度的变化曲线。为了得到多个选区的作图，可以使用ROI管理的Multi Plot命令。
其他类型的选区，可以先运行Edit-Selction-Area to Line将其转化为直线选择。

## Surface Plot
在一个灰度图或伪彩色图上显示一个三维的像素值的图。作图是基于现有的矩形选区或整个图片。

## Gels
使用该命令来分析一个一维的电泳凝胶。

## Tools
该菜单提供了多种图像分析插件。
### Save XY Coordinates
将当前图片的所有非背景像素点的XY坐标值和像素值写入一个文本文件中。背景假设为图片左上角的像素点的值。对于灰度图，每行写入三个值，用空格分割。对于RGB图，每行写入五个值。坐标系的原点是在图片的左下角。
### Fractal Box Count
估计一个二值图片的分形维度。
### Analyze Line Graph
该命令使用上面的Particle Analyzer来提取线图的坐标值。这个功能跟GetData软件一样，但明显不如专业的GetData好用。
### Curve Fitting
曲线拟合。这块还是使用专业的软件吧。。
### ROI Manager
可以用来管理多个ROI。
### Scale Bar
绘制一个带标注的空间比例尺。
### Calibration Bar
绘制一个带标注的色度条。
### Synchronize WIndows
在多个窗口上同步鼠标移动和输入，使得某个图片上绘制的ROI能够复制到其他同步窗口中。
