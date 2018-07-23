---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 48
tags: [deal.II]
categories: computational material science
date: 2016-11-29
---

# 引子
本例提供了一个框架来应用MatrixFree类，既包括求解非线性偏微分方程过程，同时演示MatrixFree类怎样处理“constraints”以及如何在分布式节点上并行。这个算例显示基于单元的运算在六面体单元的二阶或更高阶插值上要比稀疏矩阵-向量乘法快得多，能达到后者10倍的浮点运算速率。
使用MatrixFree类，可以不用组装一个大型的稀疏矩阵，其运算都是基于单元。这里的并行也充分利用了现代超算机器的架构，分为三个层级：
- 不同节点之间使用MPI并行
- 单个节点内使用“动态任务规划”进行线程并行
- 单个核心内使用处理器的向量单元进行显式向量化并行

很多通用有限元包，比如deal.II、libMesh等都将有限元计算和线性代数分开，然后依赖迭代型求解器的稀疏矩阵-向量乘法的内核，比如直接应用特定线性代数包PETSc或Trilinos等。这个算例想要挑战这种传统的将线性代数和有限元组装分开的思路。

具体来说，就是不组装一个全局的稀疏矩阵，仅仅存储参考单元形函数信息、自由度的枚举、从参考单元到真实单元的变换。通常情况下这个方法能够显著降低存储需求，在数值结算操作上比系数矩阵稍有增大，但有时甚至会更少。降低内存需求会提高浮点运算速率，因为稀疏矩阵运算通常受限于内存带宽，而不是浮点运算。尽管可以将稀疏矩阵优化成比如只运算非零元素的模式，但此时的浮点运算速率很少能超过峰值运算的2%-20%。
除了稀疏矩阵-向量乘法较差的计算性能，对于一个d维的(p-1)次的有限单元来说，矩阵中每排的非零元素的数目与p的d次方成正比，这样对于很高次的方法的开销就很大。如果我们将有限元算子分割成用参考单元的形函数值及其导数表示的函数计算和积分步，那么就可以通过一个张量乘法一次就对一个维度上的形函数信息同时计算，这使得计算复杂度变为每个自由度计算d^2p此，这在文献中称为sum-factorization方法。
之前的有限元方法应用中通常存储单元矩阵及其指标，这些单元矩阵再组成总体刚度矩阵，这是一种“一个元素一个元素”的存储方法。然后这种方法增大了内存消耗量。最近，不明确存储矩阵、基于单元的方法已经用在GPU运算和一些应用导向的软件包如SPEC-FEM 3D中。

# 算法
这里还是选择一个通用问题：算子A对一个向量u作用。
如果用了MPI并行，那么算法就是：
(1)从其他MPI进程中导入向量值，用于当前MPI进程在自己所拥有的单元上的计算。
```cpp
update_ghost_values
```
(2)对当前拥有的单元进行循环(在当前MPI进程中进行线程并行)
(2.1)取得当前单元上向量的局部值
(2.2)通过积分计算当前单元上矩阵与向量的值
(2.3)将当前单元的贡献叠加到全局目标向量中
(3)与另一个MPI进程所拥有的单元交换所计算的信息
```cpp
compress
```
第(2.2)步可以通过显式地构建单元矩阵A来实现，这就是通常的存储矩阵的所有元素的方法。为了避免这种显式地存储，就可以用无矩阵方式(还有另外一种方法，是FEniCs软件包中用的方式，但其局限于线性算子和简单几何上，即雅各比变换是常量，所以不如下面更通用)：
在单元上通过积分计算“算子作用在向量上”：计算有限元函数u在所有积分点上的值和/或它的导数，然后用所有与此单元相关的试函数来测试。
下面是对方法的具体分析。

## 局部积分方法
为了方便说明，这里特化算子A为变系数的拉普拉斯算子：
$$
-\nabla \cdot K(x) \nabla
$$
其中，K是d乘d的对称矩阵。
相应的有限元弱形式为：
$$
(\nabla \phi\_j, K\nabla u^h), \qquad j=1,...,n
$$
其中，$u^h(x)=\sum\_{i=1}^n \phi\_i (x)u^{(i)}$是全局有限元函数u的离散插值，${\phi\_j,j=1,...,n}$是试探函数。n是总自由度个数。
再定义参考单元上的基函数$\hat{\phi}\_j(\hat{x}),j=1,...,p^d$。对应于全局的积分点$x\_q$，其在参考单元上的对应的积分点就是$\hat{x}\_q$。即在参考单元上的对应量都加上个冒，其中$(p-1)$是有限单元的“度”，即p是每个方向上自由度的个数，d是维度。
，在某个单元k上，在积分点$x\_q$上，有限元函数u的插值就是：
$$
u\_k^h(x\_q)=\sum\_{i=1}^{p^d}\hat{\phi}\_i (\hat{x}\_q)u\_k^{(i)}
$$
在新方法中，显式地计算该局部有限元插值的梯度：
$$
\nabla u\_k^h(x\_q)=\sum\_{i=1}^{p^d}J\_k^{-T}(\hat{x}\_q) \hat{\nabla}\hat{\phi}\_i (\hat{x}\_q)u\_k^{(i)}=J\_k^{-T}(\hat{x}\_q) \sum\_{i=1}^{p^d} \hat{\nabla}\hat{\phi}\_i (\hat{x}\_q)u\_k^{(i)}
$$
其中，$\nabla$代表真实坐标系中的梯度，$\hat{\nabla}$代表参考单元上的梯度，$J\_k^{-T}(\hat{x}\_q)$是从参考单元到真实单元的逆转置的雅各比矩阵。注意：上面先求梯度，即先求和，再施加几何变换。
目标向量$v\_k=A\_k u\_k$的每一个分量i都是一个积分，通过数值积分来获得：
$$
\begin{split}
v\_k^{(i)}&=\sum\_q (\nabla \phi\_i (x\_q)^T K(x\_q)\nabla u^h(x\_q))w\_q |\text{det}J\_K(\hat{x}\_q)| \\\
&=\sum\_q (\hat{\nabla}\hat{\phi}\_i (\hat{x}\_q)^T (J\_k^{-T}(\hat{x}\_q))^T K(x\_q)\nabla u^h (x\_q))w\_q |\text{det}J\_k (\hat{x}\_q)|
\end{split}
$$

因此，对于拉普拉斯算子的局部算法可以表述为：
(1)计算参考单元上所有积分点上的梯度值，即第一个公式的求和部分
(2)在每个积分点上：
- 施加雅各比变换，从而根据第一个公式得到真实空间上的离散函数u的梯度
- 乘以变系数矩阵
- 再乘以雅各比行列式和积分权重
- 再次施加雅各比矩阵变换

(3)乘以参考单元上形函数的梯度，然后遍历所有积分点求和

这种算法相比于传统的矩阵组装算法有如下特点：
- 将梯度的计算分成两部分：在参考单元上计算梯度，然后再施加雅各比变换。而不是直接在真实空间中计算形函数梯度。
- 参考单元上的计算对所有单元都是相同的，所以很多单元可以组合在一块同时计算
- 对于张量积单元，可以应用下面的“sum-factorization”算法，其相比于传统的直接运算方法(对所有积分点和局部基函数嵌套循环)有更小的复杂度

## FEEvaluation类
上面给出了在每个单元上的局部积分算法，这些基于单元的操作全部在FEEvaluation类中实现：
- 读取源向量，然后再组装回全局目标向量中
- 使用“sum-factorization”算法计算参考单元上的函数值、梯度及二阶导数，并对其积分
- 应用雅各比变换、与变系数向量相乘等。

# 问题描述
这里要求解Sine-Gordon方程：
$$
\begin{split}
u\_{tt} &= \Delta u -\sin(u) \quad\mbox{for}\quad (x,t) \in \Omega \times (t\_0,t\_f],\\\
{\mathbf n} \cdot \nabla u &= 0 \quad\mbox{for}\quad (x,t) \in \partial\Omega \times (t\_0,t\_f],\\\ 
u(x,t\_0) &= u\_0(x).
\end{split}
$$

这里采用显式的时间步进离散方法：
$$
(v,u^{n+1}) = (v,2 u^n-u^{n-1} - (\Delta t)^2 \sin(u^n)) - (\nabla v, (\Delta t)^2 \nabla u^n)
$$

如果使用Gauss-Lobatto单元，将会产生一个对角质量矩阵M，那么更新方程就变成：
$$
U^{n+1}=M^{-1}L(U^n,U^{n-1})
$$
其中，非线性算子$L(U^n,U^{n-1})$就是上式右端项。

# 注意点
## 返回类型
FEEvaluation取得某个自由度和某个积分点上的函数值的返回类型都是value_type类型，其定义为：
```cpp
typedef FEEvaluationAccess<dim,n_components_,Number> BaseClass;
typedef typename BaseClass::value_type value_type;
```
而此处用到的FEEvaluationAccess类是：
```cpp
class FEEvaluationAccess<dim,1,Number>
typedef VectorizedArray<Number> value_type;
```
所以，此处得到返回值的类型都是VectorizeArray类型的。

同理，对于返回的梯度类型：
```cpp
typedef FEEvaluationAccess<dim,n_components_,Number> BaseClass;
typedef typename BaseClass::gradient_type gradient_type;

typedef Tensor<1,dim,VectorizedArray<Number> > gradient_type;
```
即，返回的梯度是一个一阶张量，即矢量，它有dim个分量。

## 梯度计算
真实单元中的函数值和梯度与参考单元中的不同，两者要进行矩阵变换，同时梯度之间的变换还要再多一道，这在FEEvaluation的submit_gradient函数中能够明确地体现出来：
```cpp
this->gradients_quad[0][d][q_point] = (grad_in[d] *
  this->cartesian_data[0][d] *
  JxW);
```
而在计算函数值时则不需要加中间的数值：
```cpp
this->values_quad[0][q_point] = val_in * JxW;
```
