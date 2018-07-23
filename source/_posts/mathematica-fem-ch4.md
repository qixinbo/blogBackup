---
title: 利用Mathematica进行有限元编程(三)：三角形单元分析
tags: [mathematica]
categories: programming
date: 2016-8-19
---

本文是对[Mathematica有限元分析与工程应用](https://www.amazon.cn/Mathematica%E6%9C%89%E9%99%90%E5%85%83%E5%88%86%E6%9E%90%E4%B8%8E%E5%B7%A5%E7%A8%8B%E5%BA%94%E7%94%A8-%E9%B2%8D%E5%9B%9B%E5%85%83/dp/B00328IIOC)一书的学习笔记。

三角形单元适用于具有复杂和不规则边界形状的问题，是一种常见的网格离散方式。
# 双线性三角形单元
## 局部坐标系
之前的杆单元和桁架元的局部坐标系容易建立，即建在其自身上即可。而三角形单元的局部坐标系显然不能这样建立，其常采用一套无量纲的自然坐标系——面积坐标，如下图所示：
![](http://7xrm8i.com1.z0.glb.clouddn.com/fem-4-1.jpeg)
三角形123内部有一任意点P，P与顶点1、2、3组成3个子三角形，每个子三角形的面积与总面积之比记为$L\_i$，即P点的面积坐标为$(L\_1,L\_2,L\_3)$。
因为三个坐标相加为1，所以仅有两个独立的自然坐标，所以可以简记为：
$$(\xi,\eta,1-\xi-\eta)$$
注意到：面积坐标还有如下特点，在顶点1时，$\xi=1$，其他坐标为0。其他顶点亦同，见上图三个顶点处的坐标值。因此，面积坐标还有形函数的功能。即：
$$
N\_1=L\_1=\xi, N\_2=L\_2=\eta, N\_3=L\_3=1-\xi-\eta
$$
写成矩阵形式为:
$$
ShapeN=
\begin{bmatrix}
\xi \\\
\eta \\\
1-\xi-\eta
\end{bmatrix}
$$
在局部坐标系下形函数对$\xi,\eta$的偏导数为：
$$
DeriveN=
\begin{bmatrix}
\partial ShapeN^T/\partial\xi \\\
\partial ShapeN^T/\partial\eta
\end{bmatrix}
$$
代码为：
```cpp
ShapeN = {xi, eta, 1 - xi - eta};
DeriveN = {D[ShapeN, xi], D[ShapeN, eta]}
```
结果为：
```cpp
{{1, 0, -1}, {0, 1, -1}}
```
偏导数在不同坐标系下的变换：
$$
\begin{bmatrix}
\partial N\_1/\partial\xi \\\
\partial N\_1/\partial\eta
\end{bmatrix}
=
\begin{bmatrix}
\partial x/\partial\xi & \partial y/\partial\xi \\\
\partial x/\partial\eta & \partial y/\partial\eta
\end{bmatrix}
\begin{bmatrix}
\partial N\_1/\partial x \\\
\partial N\_1/\partial y
\end{bmatrix}
=J
\begin{bmatrix}
\partial N\_1/\partial x \\\
\partial N\_1/\partial y
\end{bmatrix}
$$
其中，J为Jacobian矩阵。在上式中，当局部坐标系中明确给定函数$N\_1$时，等式的左侧可以求出。同时，当基于局部坐标系给出x和y的显式表达时，基于局部坐标系也可以给出Jacobian矩阵的显式表达。具体过程如下：
根据等参元的性质，基于局部坐标给出的标准形函数与整体坐标的形函数完全形同，则：
$$
\begin{equation}
\begin{split}
x&=N\_1 x\_1+N\_2 x\_2 +N\_3 x\_3 \\\
y&=N\_1 y\_1+N\_2 y\_2 +N\_3 y\_3 
\end{split}
\end{equation}
$$
所以，Jacobian矩阵为：
$$
J=
\begin{bmatrix}
\partial x/\partial\xi & \partial y/\partial\xi \\\
\partial x/\partial\eta & \partial y/\partial\eta
\end{bmatrix}
=
\begin{bmatrix}
\sum\frac{\partial N\_i}{\partial\xi}x\_i & \sum\frac{\partial N\_i}{\partial\xi}y\_i \\\
\sum\frac{\partial N\_i}{\partial\eta}x\_i & \sum\frac{\partial N\_i}{\partial\eta}y\_i
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial N\_1}{\partial\xi} & \frac{\partial N\_2}{\partial\xi} & \frac{\partial N\_3}{\partial\xi} \\\
\frac{\partial N\_1}{\partial\eta} & \frac{\partial N\_2}{\partial\eta} & \frac{\partial N\_3}{\partial\eta} 
\end{bmatrix}
\begin{bmatrix}
x\_1 & y\_1 \\\
x\_2 & y\_2 \\\
x\_3 & y\_3
\end{bmatrix}
$$
具体到这里使用的面积坐标，则有：
$$
\begin{equation}
\begin{split}
x&=N\_1 x\_1+N\_2 x\_2 +N\_3 x\_3=\xi(x\_1-x\_3)+\eta(x\_2-x\_3)+x\_3 \\\
y&=N\_1 y\_1+N\_2 y\_2 +N\_3 y\_3=\xi(y\_1-y\_3)+\eta(y\_2-y\_3)+y\_3 
\end{split}
\end{equation}
$$
所以：
$$
J=
\begin{bmatrix}
x\_1-x\_3 & y\_1-y\_3 \\\
x\_2-x\_3 & y\_2-y\_3
\end{bmatrix}
$$

Jacobian矩阵的逆变换为：
$$
\begin{bmatrix}
\partial/\partial x \\\
\partial/\partial y
\end{bmatrix}
=J^{-1}
\begin{bmatrix}
\partial/\partial\xi \\\
\partial/\partial\eta
\end{bmatrix}
$$
而Jacobian行列式则为：
$$
|J|=
\begin{vmatrix}
\partial x/\partial\xi & \partial y/\partial\xi \\\
\partial x/\partial\eta & \partial y/\partial\eta
\end{vmatrix}
=\frac{\partial x}{\partial\xi}\frac{\partial y}{\partial\eta}-\frac{\partial y}{\partial\xi}\frac{\partial x}{\partial\eta}
$$
在求解刚度矩阵时，所对应的积分中要用到此行列式。积分过程中的变量和区域需要进行变换：
$$
dxdy=|J|d\xi d\eta
$$

Jacobian矩阵的逆矩阵和行列式则可以通过Inverse和Det命令求得。

## 应力应变矩阵
对平面应力问题，应力应变矩阵D为：
$$
D=\frac{E}{1-\nu^2}
\begin{bmatrix}
1 & \nu & 0 \\\
\nu & 1 & 0 \\\
0 & 0 & (1-\nu)/2
\end{bmatrix}
$$
对平面应变问题，应力应变矩阵D为：
$$
D=\frac{E}{(1+\nu)(1-2\nu)}
\begin{bmatrix}
1-\nu & \nu & 0 \\\
\nu & 1-\nu & 0 \\\
0 & 0 & (1-2\nu)/2
\end{bmatrix}
$$

## 应变位移矩阵
应变、位移的关系为：
$$
\epsilon=
\begin{bmatrix}
\frac{\partial}{\partial x} & 0 \\\
0 & \frac{\partial}{\partial y} \\\
\frac{\partial}{\partial y} & \frac{\partial}{\partial x}
\end{bmatrix}
\begin{bmatrix}
u \\\
v
\end{bmatrix}
$$
代入位移函数表达式，可得：
$$
\epsilon=Ba=
\begin{bmatrix}
B\_1 & B\_2 & B\_3 
\end{bmatrix}
\begin{bmatrix}
a\_1 \\\
a\_2 \\\
a\_3
\end{bmatrix}
$$
其中应变位移矩阵的一部分为：
$$
B1=
\begin{bmatrix}
\frac{\partial N\_1}{\partial x} & 0 \\\
0 & \frac{\partial N\_1}{\partial y} \\\
\frac{\partial N\_1}{\partial y} & \frac{\partial N\_1}{\partial x}
\end{bmatrix}
$$

## 单元刚度矩阵
根据最小势能原理，求得单元刚度矩阵表达式：
$$
K=\int\_0^1\int\_0^{1-\eta}B^TDBt|J|d\xi d\eta
$$


## 模块分析
### 建立单元刚度矩阵
```cpp
GenerateLinearTriangKm[cord_] := Module[{Bmatrix, DeriveN, J, km},
      n1 = xi; n2 = eta; n3 = 1 - xi - eta;
  bdv = {{1, 0, -1}, {0, 1, -1}};(*这里是局部坐标系下的偏导数*)
  J = bdv.cord;(*求解Jacobian矩阵*)
  bdv = Simplify[Inverse[J].bdv];(*转换成整体坐标系下的偏导数*)
  Bmatrix = {{bdv[[1, 1]], 0, bdv[[1, 2]], 0, bdv[[1, 3]], 0}, {0, 
              bdv[[2, 1]], 0, bdv[[2, 2]], 0, bdv[[2, 3]]}, {bdv[[2, 1]], 
              bdv[[1, 1]], bdv[[2, 2]], bdv[[1, 2]], bdv[[2, 3]], bdv[[1, 3]]}};
  Dmatrix = 
     ee/(1 - nu*nu) {{1, nu, 0}, {nu, 1, 0}, {0, 0, (1 - nu)/2}};
  km = h Integrate[
       Det[J] (Transpose[Bmatrix].Dmatrix.Bmatrix), {xi, 0, 1}, {eta, 0,
                  1 - xi}]
                    ]
```
### 组装刚度矩阵
```cpp
AssembleLinearTriangKm[p1_, p2_, p3_, m_] := 
 Module[{f}, f = {p1, p2, p3};
  For[j = 1, j <= 3, j++, For[k = 1, k <= 3, k++,
      GlobalK[[2 f[[j]], 2 f[[k]]]] += m[[2 j, 2 k]];
      GlobalK[[2 f[[j]] - 1, 2 f[[k]]]] += m[[2 j - 1, 2 k]];
      GlobalK[[2 f[[j]], 2 f[[k]] - 1]] += m[[2 j, 2 k - 1]];
      GlobalK[[2 f[[j]] - 1, 2 f[[k]] - 1]] += m[[2 j - 1, 2 k - 1]];
      ]];
  GlobalK]
```
这里注意矩阵带不带Matrixform在计算时有很大区别。
注意这里的循环次数为3,是因为每个单元有3个节点。

# 二次三角形单元
二次三角形单元就是在每条边上还各有一个节点，如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/fem-4-2.jpeg)
具体分析过程跟双线性三角形单元相同，只不过形函数更加复杂。且在组装总刚时循环系数为6。
