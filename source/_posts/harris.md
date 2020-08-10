---
title: (转载)Harris角点检测原理
tags: [Image]
categories: computational material science 
date: 2020-8-5
---

原文在[这里](https://senitco.github.io/2017/06/18/image-feature-harris/)，中间增加了一些额外的内容辅助理解。

角点检测(Corner Detection)也称为特征点检测，是图像处理和计算机视觉中用来获取图像局部特征点的一类方法，广泛应用于运动检测、图像匹配、视频跟踪、三维建模以及目标识别等领域中。

# 局部特征
不同于HOG、LBP、Haar等基于区域(Region)的图像局部特征，Harris是基于角点的特征描述子，属于feature detector，主要用于图像特征点的匹配(match)，在SIFT算法中就有用到此类角点特征；而HOG、LBP、Haar等则是通过提取图像的局部纹理特征(feature extraction)，用于目标的检测和识别等领域。无论是HOG、Haar特征还是Harris角点都属于图像的局部特征，满足局部特征的一些特性。主要有以下几点：
- 可重复性(Repeatability)：同一个特征可以出现在不同的图像中，这些图像可以在不同的几何或光学环境下成像。也就是说，同一物体在不同的环境下成像(不同时间、不同角度、不同相机等)，能够检测到同样的特征。
- 独特性(Saliency)：特征在某一特定目标上表现为独特性，能够与场景中其他物体相区分，能够达到后续匹配或识别的目的。
- 局部性(Locality)；特征能够刻画图像的局部特性，而且对环境影响因子(光照、噪声等)鲁棒。
- 紧致性和有效性(Compactness and efficiency)；特征能够有效地表达图像信息，而且在实际应用中运算要尽可能地快。
 

相比于考虑局部邻域范围的局部特征，全局特征则是从整个图像中抽取特征，较多地运用在图像检索领域，例如图像的颜色直方图。
除了以上几点通用的特性外，对于一些图像匹配、检测识别等任务，可能还需进一步考虑图像的局部不变特征。例如尺度不变性(Scale invariance)和旋转不变性(Rotation invariance)，当图像中的物体或目标发生旋转或者尺度发生变换，依然可以有效地检测或识别。此外，也会考虑局部特征对光照、阴影的不变性。

# Harris角点检测
特征点在图像中一般有具体的坐标，并具有某些数学特征，如局部最大或最小灰度、以及某些梯度特征等。角点可以简单的认为是两条边的交点，比较严格的定义则是在邻域内具有两个主方向的特征点，也就是说在两个方向上灰度变化剧烈。如下图所示，在各个方向上移动小窗口，如果在所有方向上移动，窗口内灰度都发生变化，则认为是角点；如果任何方向都不变化，则是均匀区域；如果灰度只在一个方向上变化，则可能是图像边缘。
![image](https://user-images.githubusercontent.com/6218739/89362741-360d4580-d701-11ea-8779-9d993f5e7dce.png)

对于给定图像$I(x,y)$（即图像强度）和固定尺寸的邻域窗口，计算窗口平移前后各个像素差值的平方和，也就是自相关函数：
$$
E(u,v)=\Sigma_x\Sigma_yw(x,y)[I(x+u,y+v)-I(x,y)]^2
$$
其中，窗口加权函数$w(x,y)$可取均值函数或者高斯函数，如下图所示：
![image](https://user-images.githubusercontent.com/6218739/89362908-97351900-d701-11ea-8d13-57cc49354cec.png)
根据泰勒展开，可得到窗口平移后图像的一阶近似（梯度乘以位移，注意$I_x$表示x方向的梯度）：
$$
I(x+u,y+v)\approx I(x,y)+I_x(x,y)u+I_y(x,y)v
$$
因此，$E(u,v)$可化为：
$$
E(u,v) \approx \Sigma_{x,y}w(x,y)[I_x(x,y)u+I_y(x,y)v]^2=\left[u,v\right] M(x,y) \left[ \begin{matrix} u\\ v\end{matrix} \right]
$$
其中：
$$
M(x,y)=\Sigma_{x,y} w \left[ \begin{matrix} I_x^2& I_xI_y \\ I_xI_y & I_y^2\end{matrix} \right] = \left[ \begin{matrix} A& C\\ C& B\end{matrix} \right]
$$
因此，$M$就是偏导数矩阵。
可以有多个角度来理解这个矩阵：
（1）几何角度：
$E(u,v)$可表示为一个二次项函数：
$$
E(u,v)=Au^2+2Cuv+Bv^2
$$
其中：
$$
A=\Sigma_{x,y} w I_x^2, B = \Sigma_{x,y} w I_y^2, C=\Sigma_{x,y} w I_x I_y
$$

二次项函数本质上是一个椭圆函数，椭圆的曲率和尺寸可由$M(x,y)$的特征值$\lambda_1,\lambda_2$决定，椭圆方向由$M(x,y)$的特征向量决定，椭圆方程和其图形分别如下所示：
![image](https://user-images.githubusercontent.com/6218739/89366840-3a8a2c00-d70a-11ea-86af-28329b0e68fa.png)

（2）线性代数角度：
首先来点线性代数中特征值和特征向量的基本知识：
对于一个给定的方阵，它的特征向量经过这个方阵的线性变换后，得到的新向量与原来的特征向量保持在同一条直线上，但其长度或方向也许会改变，这个长度的缩放比例就是特征值。
![image](https://user-images.githubusercontent.com/6218739/89378163-d7a68e00-d725-11ea-84fb-06bdb29339d9.png)
注意：方阵代表了对向量的变换，而不是向量代表了对方阵的变换。对于方阵所产生的变换效果，就可以分解为特征向量和特征值的效果：特征向量代表了旋转，特征值代表了缩放。因此，对于任一向量，如果对其施加了方阵这一变换，就有可能使其旋转和缩放；特别地，对于特征向量这一向量，施加方阵后，就只会缩放，而不会旋转。
通过矩阵相似对角化分解，可以得到：
$$
A=PBP^{-1}
$$
其中，$B$为对角阵，里面是特征值，决定了缩放；$P$的列向量是单位化的特征向量，并且互相正交，决定了旋转。
![image](https://user-images.githubusercontent.com/6218739/89378770-20ab1200-d727-11ea-8b4e-fb3c4183a5a1.png)

一些参考文章：
[矩阵特征值与特征向量和相似对角化](https://www.jianshu.com/p/a2ef1b585b03)
[如何理解矩阵特征值和特征向量？](https://www.matongxue.com/madocs/228.html)

有了上面的背景，先考虑角点的边界和坐标轴对齐的这种特殊情况，如下图所示，在平移窗口内，只有上侧和左侧边缘，上边缘$I_y$很大而$I_x$很小，左边缘$I_x$很大而$I_y$很小，所以矩阵$M$可化简为（即没有旋转）：
$$
M=\left[ \begin{matrix} \lambda_1& 0\\ 0& \lambda_2\end{matrix} \right]
$$
![image](https://user-images.githubusercontent.com/6218739/89379432-61eff180-d728-11ea-98e0-23361456bed1.png)
当角点边界和坐标轴没有对齐时，可对角点进行旋转变换，将其变换到与坐标轴对齐，这种旋转操作可用矩阵的相似对角化来表示，即：
$$
M=X\Sigma X^T = X \left[ \begin{matrix} \lambda_1& 0\\ 0& \lambda_2\end{matrix} \right] X^T
$$
![image](https://user-images.githubusercontent.com/6218739/89379537-88159180-d728-11ea-9c0f-59f4d5d071ad.png)

再回过头来重新看一下$M$矩阵：
$$
M(x,y)=\Sigma_{x,y} w \left[ \begin{matrix} I_x^2& I_xI_y \\ I_xI_y & I_y^2\end{matrix} \right] = \left[ \begin{matrix} A& C\\ C& B\end{matrix} \right]
$$
时刻注意，式中的$I_x$是梯度，是导数，是灰度强度的差别。
对于矩阵$M$，可以将其和协方差矩阵类比，协方差表示多维随机变量之间的相关性，协方差矩阵对角线的元素表示的是各个维度自身的方差，而非对角线上的元素表示的是各个维度之间的相关性，在PCA(主成分分析)中，将协方差矩阵对角化，使不同维度的相关性尽可能的小（相关性为0时就是非对角线元素为0），并取特征值较大的维度，来达到降维的目的。而这里的矩阵M中的对角线元素是灰度强度在某一方向上的梯度的平方，而非对角线上的元素则是灰度在两个不同方向上的梯度的乘积，所以可以将矩阵$M看成是一个二维随机分布的协方差矩阵，通过将其对角化，一方面可以得到两个正交的特征向量，另一方面也可以求取矩阵的两个特征值（与两个方向上的梯度直接相关），并根据这两个特征值来判断角点。

更多地关于PCA的补充知识：
![20200807152900_1](https://user-images.githubusercontent.com/6218739/89623622-80d8ba00-d8c7-11ea-9520-728ae9fc1f88.jpg)

一些参考文章：
[深度学习的预处理：从协方差矩阵到图像白化](https://zhuanlan.zhihu.com/p/45140262)
[PCA （主成分分析）详解 （写给初学者）](https://my.oschina.net/gujianhan/blog/225241#OSC_h2_1)
[图像空间域分析之图像统计特征](https://cggos.github.io/computervision/image-process-moments.html)

![image](https://user-images.githubusercontent.com/6218739/89386440-a6cd5580-d733-11ea-8456-b46674c64b32.png)
在判断角点时，无需具体计算矩阵$M$的特征值，而使用下式近似计算角点响应值。
![image](https://user-images.githubusercontent.com/6218739/89386536-c9f80500-d733-11ea-9245-75a24c616c6b.png)
式中，$detM$为矩阵$M$的行列式，$traceM$为矩阵$M$的迹，$\alpha$为一常数，通常取值为0.04~0.06。

# 算法实现
![image](https://user-images.githubusercontent.com/6218739/89386732-0d527380-d734-11ea-83fb-93a791fe7f36.png)
