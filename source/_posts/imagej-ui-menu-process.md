---
title: ImageJ 用户指南 -- 8. 菜单栏之Process
tags: [ImageJ]
categories: programming
date: 2018-9-8
---

# 本章说明
这里详解Process菜单的功能。
# Process
## Smooth
对当前图片或选区进行模糊处理。该过滤器将每个像素值设为它的3*3邻居的平均值。
## Sharpen
对当前图片或选区进行锐化，即增加对比度和强调细节，但有可能对噪点进行了加强。该滤波器使用下面的权重因子：
$$
\begin{matrix}
-1 & -1 & -1 \\\
-1 & 12 & -1 \\\
-1 & -1 & -1
\end{matrix}
$$
## Find Edges
使用Sobel边缘检测器来高粱像素值强度的剧烈变化。使用下面的两个3*3的卷积核来产生垂直和水平的梯度。最终的图片是计算两个梯度的平方的和的平方根得到：
$$
\begin{matrix}
1 & 2 & 1 \quad \quad & 1 & 0 & -1 \\\
0 & 0 & 0 \quad \quad & 2 & 0 & -2 \\\
-1 & -2 & -1 \quad \quad & 1 & 0 & -1 
\end{matrix}
$$
## Find Maxima
计算当前图片的局部最大值，然后创建一个含最大值等形式的二值图片。对于RGB图片，挑选的是luminance的最大值，luminance是根据Edit-Options-Conversions中的平均或加权平均计算出来的。
- Noise Tolerance：如果最大值没有比周围的数值大这个tolerance，那么该最大值会被忽略。即，设置了一个最大值减去该tolerance的阈值，超过该阈值的区域才被分析。
- Output Type：Single Points：图片是每个最大值所对应的点；Maxima Within Tolerance：每个最大值周围在Tolerance范围之内的所有点；Segmented Particles：假定每个最大值都属于一个颗粒，然后使用一个泛洪算法将图片分割，与之对比的是，Process-Binary-Watershed使用的是欧拉空间距离；Point Selection：在每个最大值点上显示一个多点选区；List：在Results窗口中显示每个最大值点的坐标；Count：显示最大值的数目。
- Exclude Edge Maxima：排除边缘上的最大值点
- Above Lower Threshold：仅适用于阈值处理过的图片，仅寻找在阈值下界以上的最大值，图片的阈值上界被忽略。如果在Output Type中选择Segmented Particles，那么在阈值下界以下的区域处理成背景。
- Light Backgroud：如果图片背景要比要寻找的对象亮，则勾选。

该命令不适用于stacks，但FindStackMaxima宏可以作用于stack的所有图片。

## Enhance Contrast
通过使用histogram stretching或histogram equalization来增强图片对比。该命令不会改变像素值，只要Normalize、Equalize Histogram等不勾选。
- Saturated Pixels：决定图片中要饱和处理（即转成黑色或白色）的像素的个数。增大这个值会增加对比度。
- Normalize：勾选后，ImageJ将会重新计算像素值使得范围等于该图片类型的最大范围，或者对于浮点图片，范围是0-1.0。比如，对于8-bit图片，该最大范围是0-255，对于16-bit图片，范围是0-65535。对于RGB图，该项不显示。
- Equalize Histogram：勾选后，ImageJ将会使用histogram equalization来加强图片。勾选此项后，上面两项将失效。
- Use stack histogram：勾选后，ImageJ将会使用整体的stack的histogram，而不是单个slice的histogram。

## Noise
### Add noise
对图片增加随机噪声，噪声服从均值为0、标准差为25的高斯分布。
### Add Specified Noise
增加一个均值为0、手动输入标准差的高斯噪声。
### Salt and Pepper
通过随机替换2.5%的黑色像素和2.5%的白色像素来增加椒盐噪声。该命令仅适用于8-bit图片。
### Despeckle
这是一种中值滤波器，它将像素值替换为它周围3*3的像素点的均值。中值滤波器对于去掉上面的椒盐噪声很有用。
### Remove Outliers
如果一个像素点离它周围点的均值超过一定数值，该命令就会去除这个像素点。该命令对CCD相机的hot pixels或dead pixels很有用。
- Radius：决定计算均值的范围（单位是像素）。使用Process-Filter-Show Circular Masks来显示半径是如何转变为面积的。
- Threshold：决定阈值（单位是像素）
- Which Outliers：决定比均值更亮还是更暗的像素点去除。

### Remove NANs
该滤波器将32-bit图片中的NaN像素点替换成由Radius定义的圆形kernel区域内部的像素点的平均值。

## Shadows
创建阴影效果，使得光看起来从不同的方向照来。这些操作实际上是使用了不同的3*3的卷积核。

## Binary
创建或处理二值图片，图片里仅有两个值，ImageJ处理成0和255，也有软件处理成0和1。这里假设对象是黑色的，背景是白色的，除非Process-Binary-Options中的Black Background被勾选。
### Make Binary
将一张图片转化为黑白二值图片。如果之前使用Image-Adjust-Threshold设置了阈值，就会跳出一个对话框使设置怎样处理阈值以外和以内的像素。如果没有设置阈值，就会分析当前选区或整个图片的直方图，然后自动设置阈值进行二值化。如果是Stack，就会显示Convert to Mask对话框。注意，对于未经过阈值处理的图片和stack，Make Binary和Convert to Mask表现类似。
### Convert to Mask
将图片转为黑白二值图片。该mask有一个反转的LUT（即白色是0，黑色是255），除非在Process-Binary-Options中勾选了Black Background。效果跟上面的Make Binary近似。
### Erode
在二值图片中在图像边缘去除像素，在非阈值化的图片上使用Filters-Minimum来腐蚀灰度图。
### Dilate
在二值图片中在图像边缘增加像素，在非阈值化的图片上使用Filters-Maximum来膨胀灰度图。
### Open
开操作，即先腐蚀后膨胀。这将平滑对象及去除独立的像素点。
### Close
闭操作，即先膨胀后腐蚀。这将平滑对象及填充小洞。
### Outline
在二值图片中在前景图片中产生一个像素宽的轮廓。
### Fill Holes
填充小洞（4个相连的背景色的元素）。
### Skeletonize
在二值图片中对对象边缘不断地去除像素点知道形成一个单像素宽的形状。
### Distance Map
从二值图片中产生一个欧氏距离映射EDM。每一个前景像素被设为等于其离最近的背景像素的距离。下面的Ultimate Points、Watershed和Voronoi操作都是基于EDM算法。
该命令的输出类型需要在Binary-Options中设定，注意当选择Overwrite或8-bit output时，大于255的距离会被设为255。
### Ultimate Points
产生极限腐蚀点，这种点是上面EDM的最大值。
### Watershed
泛洪分割可以自动分割两个碰撞的颗粒。它首先计算欧氏距离映射EDM，然后找到极限腐蚀点。接着尽可能膨胀每一个极限腐蚀点，直到达到颗粒的边缘或者达到另一个正在膨胀的极限腐蚀点。泛洪分割对重叠不严重的平滑凸包对象的分割很有用。
在Edit-Options-Misc中开启debug模式后，该命令可以创建一个泛洪算法怎样工作的动画。
### Voronoi
将图片分割成与两个最近邻颗粒的边界有相等距离的一系列的点连成的线。因此，每个颗粒的Voronoi包含了与该颗粒更近的所有点。当颗粒是单个的点时，这个过程称为Voronoi镶嵌或称Dirichlet镶嵌。
在输出中，在Voronoi胞内部的值是0，分割线上的点的像素值等于两个最近邻颗粒的距离。
### Options
指定Binary命令的有关设置：
- Iterations：指定腐蚀、膨胀、开、闭操作的迭代次数，迭代过程可以被Esc打断。
- Count：指定腐蚀或膨胀时在边缘上去除或添加的像素的临近背景像素个数。
- Black Background：指定背景为黑色。
- Pad edges when eroding：勾选后，不会在图片的边缘进行腐蚀，该选项也会影响闭操作。
- EDM output：决定输出类型。
- Do：预览一下上述设置的影响。

## Math
该菜单对当前图片或选区上的每个像素加减乘除一个常数。
### Add
相加一个常数。对于8-bit图片，大于255的结果被置为255；对于16-bit图片，大于65535的结果被置为65535。
### Subtract
减去一个常数。对于8-bit和16-bit图片，小于0的结果被置为0。
### Multiply
乘以一个常数。对于8-bit图片，大于255的结果被置为255；对于16-bit图片，大于65535的结果被置为65535。
### Divide
除以一个常数。对于非32-bit的图片，忽略除以0的操作；对于32-bit图片，如果源像素分别是正值、负值或零，那么默认除以0的结果是正无穷、负无穷和NaN。可以Edit-Options-Misc重新定义除以0的结果。
### And
与一个特定的二进制常数进行逐位与运算
### OR
与一个特定的二进制常数进行逐位或运算
### XOR
与一个特定的二进制常数进行逐位异或运算
### Min
如果像素值小于某特定常数，则该像素值被替换为该常数
### Max
如果像素值大于某特定常数，则该像素值被替换为该常数
### Gamma
对每一个像素值施加$f(p)=(p/255)^\gamma *255$，其中$\gamma$在0.1和5.0之间。对于RGB图片，该函数作用于所有的3个通道，对于16-bit图片，图片的最小和最大值将代替255用于缩放。
### Set
用特定值来填充图片或选区。
### Log
对于8-bit图片，对图片或选区中的每个像素施加$f(p)=\ln (p)*255/ \ln(255)$；对于RGB图片，该函数作用于三个通道；对于16-bit图片，图片的最小和最大值将代替255；对于float型图片，不进行缩放。如果想计算$\log_{10}$，则对该结果乘以0.4343。
### Exp
对当前图片或选区进行指数变换
### Square
对当前图片或选区进行平方变换
### Square Root
对当前图片或选区进行平方根变换
### Reciprocal
对当前图片或选区进行倒数变换
### NaN Background
将32-bit浮点型图片的非阈值的像素设为NaN。对于浮点型图片，Image-Adjust-Threshold的Apply就是执行的该命令。
### Abs
产生当前图片或选区的绝对值，仅对32-bit浮点型图片或signed 16-bit图片有效。
### Macro
可以自定义算术运算。

## FFT
该菜单支持频域显示、编辑和处理，基于二维快速哈特利变换FHT。三维的FHT可以通过3D Fast Hartley Transform插件来实现。
### FFT
进行傅里叶变换，显示功率谱。测量的点的极坐标由Anayze-Measure所记录。如果鼠标在当前频谱窗口上悬停，那么它的位置是通过极坐标显示。
### Inverse FFT
进行逆向傅里叶变换。
### Redisplay Power Spectrum
从频谱图片中重新计算功率谱。
### FFT Options
显示快速傅里叶变换的选项。
### Bandpass Filter
去除高频和低频。
### Custom Filter
使用用户自定义的空间域图片作为滤波器。
### FD Math
对两张图片进行convolve或deconvolve。
### Swap Quadrants
交换象限。

## Filters
该菜单包含五花八门的滤波器。
### Convolve
使用填入文本区域的kernel进行空间卷积。
一个kernel就是一个矩阵，它的中心是源像素，其他的元素是该像素的邻居。通过对像素点乘以相应的kernel中的系数然后相加得到结果。对kernel的尺寸没有限制，但它必须是方形，且必须是奇数宽度。
勾选Normalize Kernel可以使得每个系数都除以所有系数的和，从而保持图片的亮度。
### Gaussian Blur
该过滤器使用一个高斯函数进行卷积，从而实现平滑效果。
### Gaussian Blur 3D
计算一个三维高斯低通滤波。
### Median
将像素替换为周围点的平均像素值，从而实现降噪效果。
### Mean
怎么感觉跟上面的Median是一个意思呢。。
### Minimum
将像素替换为周围点的最小值，从而实现灰度腐蚀。
### Maximum
将像素替换为周围点的最大值，从而实现灰度膨胀。
### Unsharp Mask
通过从原图片中提出一个模糊的版本，从而锐化和加强边缘。
### Variance
将每个像素替换为邻居的方差，从而高亮边缘。
### Show Circular Masks
产生一个包含上面Median、Mean、Minimum、Maximum和Variance滤波器使用的圆形mask产生的事例。

## Batch
包含批量处理一系列图片的命令。
Batch命令是非递归的，即命令是施加在当前Input文件夹的所有图片上，但不作用于它的子文件夹，除非使用BatchProcessFolders宏中定义了目录遗传树。
关于批处理有三个重要提醒：
- 文件很容易被覆盖，因为批处理器总是静默地覆盖有同样名称的已有文件；
- 目标Output文件夹应该有足够的硬盘空间来存储所创建的图片；
- 对于非原始格式的图片，批处理操作会被那个读取该文件格式的插件或库所影响。

### Convert
在指定文件夹中批量转换或调整文件尺寸。
- Input：选择源文件夹
- Output：选择目标文件夹
- Output Format：选择输出图片的格式
- Interpolation：如果Scale Factor不设为1，那么将会使用重采样方法。
- Scale Factor：是否缩放。

### Macro
运行指定文件夹中的一个宏，最近使用的宏存储在/ImageJ/macros/batchmacro.ijm文件中，可以在重启时记忆住。
- Input：选择要处理的图片所在的文件夹
- Output：选择目标文件夹。如果为空，源文件不会被存储
- Output format：指定输出格式
- Add Macro Code：下拉菜单中包含了一些宏片段，可以组合起来形成一个宏。其他的代码可以粘贴进下面的编辑器中。之前写的宏可以通过下面的Open导入。
- Test：用Input文件夹中的第一张图片进行测试
- Open：导入之前写的宏
- Save：保存组装好的宏

### Virtual Stack
该命令与上面的macro的界面相同，允许操作virtual stack。

### Image Calculator
对两张图片进行逻辑或算术运算，Image1可以是stack，或者Image1和Image2同时是stacks。如果两者都是stacks，那么都是有相同数目的slice。两张图片不一定有相同的文件类型或尺寸。
- Operation：选择13种操作中的一种
- Create New Window：勾选后，就会创建一个新的图片，如果不勾选，则结果作用在Image1上。
- 32-bit Result：勾选后，源图片在操作前会转换为32-bit float型

## Subtract background
去除平滑的连续的背景，基于“rolling ball”算法。想象一个二维灰度图有一个第三维度，其值是每个点的像素值的大小，一个有特定半径的球在这个表面下面滚动，碰到该图的点就是要去除的背景。
- Rolling Ball Radius：抛物线的曲率半径。
- Light Background：允许处理明亮背景、对象深色的情形。
- Separate colors：仅适用于RGB图像，如果未勾选，则操作仅影响亮度，而不对灰度和饱和度进行操作。
- Create background (Don't subtract)：勾选后，输出不再是扣除了背景的图片，而是背景本身。
- Sliding Paraboloid：勾选后，球被一个有相同曲率的的抛物面所替代。
- Disable Smoothing：为了计算背景，图片会先用一个3*3的最大值滤波器进行滤波，从而去除异常值和噪点的影响。勾选后，使用原始值进行操作。

## Repeat Command
重复之前的命令。忽略Edit-Undo和File-Open这两个命令。
