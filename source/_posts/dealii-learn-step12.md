---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 12
tags: [deal.II]
categories: simulation
date: 2016-9-17
---

# 引子
本例主要讲解MeshWorker框架和间断Galerkin方法，即DG。
- 用间断Galerkin法离散线性对流方程
- 使用MeshWorker::loop()来组装系统矩阵

本例主要关心的就是间断Galerkin法的循环，这相当复杂，因为必须分辨边界、常规内部边和有悬点的内部边。MeshWorker框架能够对所有的单元和边进行标准循环，它将分辨过程隐藏在了内部。
使用MeshWorker需要手动做两件事：一是针对特定问题写内部积分器，二是从该命名空间中选择类，然后将它们组合起来来解决问题。

要求解的问题是线性对流方程：
$$
\nabla\cdot \left({\mathbf \beta} u\right)=0 \qquad\mbox{in }\Omega,
$$
边界条件是：
$$
u=g\quad\mbox{on }\Gamma\_-,
$$
这是入流边界，定义是：
$$
\Gamma\_-:=[{\bf x}\in\Gamma, {\mathbf \beta}({\bf x})\cdot{\bf n}({\bf x})<0]
$$

这个方程是之前的Step9的守恒版本。
具体取值为：$\Omega=[0,1]^2$，${\mathbf \beta}=\frac{1}{|x|}(-x\_2, x\_1)$代表一个环形逆时针流场，在${\bf x}\in\Gamma\_-^1:=[0,0.5]\times[0]$上$g=1$，而在${\bf x}\in \Gamma\_-\setminus \Gamma\_-^1$上$g=0$。
这里使用迎风间断Galerkin方法。
从这里就完全看不懂了。留坑待填。

