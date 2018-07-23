---
title: 利用Mathematica进行有限元编程(二)：桁架元分析
tags: [mathematica]
categories: programming
date: 2016-8-18
---

本文是对[Mathematica有限元分析与工程应用](https://www.amazon.cn/Mathematica%E6%9C%89%E9%99%90%E5%85%83%E5%88%86%E6%9E%90%E4%B8%8E%E5%B7%A5%E7%A8%8B%E5%BA%94%E7%94%A8-%E9%B2%8D%E5%9B%9B%E5%85%83/dp/B00328IIOC)一书的学习笔记。

# 桁架元的特点 
平面桁架元是既有局部坐标又有整体坐标的二维有限元，因此比起之前的杆单元，需要多一步坐标变换。
桁架元示意图如下：
![](http://7xrm8i.com1.z0.glb.clouddn.com/fem-3-1.jpeg)
指定整体坐标系为X-Y，局部坐标系为x-y。则两者之间的转换关系为：
$$
\begin{equation}
\begin{split}
U\_{Xi}&=u\_{xi}cos\theta,U\_{Yi}=u\_{xi}sin\theta \\\
U\_{Xj}&=u\_{xj}cos\theta,U\_{Yj}=u\_{xj}sin\theta 
\end{split}
\end{equation}
$$
即：
$$
U=Tu
$$
其中：
$$
U=
\begin{bmatrix}
U\_{Xi} \\\
U\_{Yi} \\\
U\_{Xj} \\\
U\_{Yj} 
\end{bmatrix}
,T=
\begin{bmatrix}
cos\theta & 0 \\\
sin\theta & 0 \\\
0 & cos\theta \\\
0 & sin\theta 
\end{bmatrix}
,u=
\begin{bmatrix}
u\_{xi} \\\
u\_{xj}
\end{bmatrix}
$$
局部坐标系下的有限元方程为：
$$
f=Ku
$$
为了把有限元方程从局部坐标系变换到整体坐标系，可通过转换矩阵：
$$
u=T^{-1}U, f=T^{-1}F
$$
所以：
$$
F=TKT^{-1}U
$$
又因为转换矩阵T满足如下关系(可实际计算验证一下)：
$$
T^{T}T=I
$$
所以：
$$
F=TKT^TU=\overline{K}U
$$
所以整体坐标系的刚度矩阵与局部坐标系的刚度矩阵关系为：
$$
\overline{K}=TKT^T
$$
由于局部坐标系下的单元刚度矩阵为：
$$
K=
\begin{bmatrix}
k & -k \\\
-k & k
\end{bmatrix}
$$
其中$k=\frac{EA}{L}$。那么整体坐标系下的单刚为：
$$
\overline{K}=\frac{EA}{L}
\begin{bmatrix}
C^2 & CS & -C^2 & -CS \\\
CS & S^2 & -CS & -S^2 \\\
-C^2 & -CS & C^2 & CS \\\
-CS & -S^2 & CS & S^2
\end{bmatrix}
$$
其中$C=cos\theta,S=sin\theta$。

# 模块分析：
## 建立单元刚度矩阵(经过了坐标变换)
```cpp
TrussElementKm[EE_, AA_, LL_, theta_] := Module[{},
      x = theta*Pi/180;
  w1 = Cos[x]^2;
    w2 = Sin[x]^2;
      w3 = Sin[x]*Cos[x];
        y = EE*AA/
             LL*{{w1, w3, -w1, -w3}, {w3, w2, -w3, -w2}, {-w1, -w3, w1, 
                       w3}, {-w3, -w2, w3, w2}};
  y]
```
## 组装整体刚度矩阵
```cpp
AssembleSpringKm[p1_, p2_, m_] := Module[{j, k}, f = {p1, p2};
  For[j = 1, j <= 2, j++, For[k = 1, k <= 2, k++,
      GlobalK[[2 f[[j]], 2 f[[k]]]] += m[[2 j, 2 k]];
    GlobalK[[2 f[[j]] - 1, 2 f[[k]]]] += m[[2 j - 1, 2 k]];
        GlobalK[[2 f[[j]], 2 f[[k]] - 1]] += m[[2 j, 2 k - 1]];
            GlobalK[[2 f[[j]] - 1, 2 f[[k]] - 1]] += m[[2 j - 1, 2 k - 1]];
                ]];
                  GlobalK]
```
这里的组装与之前的杆单元不同，注意此处每个节点上有两个自由度，但总体原则还是将单刚的每个元素叠加到总刚的对应位置上，只是自由度的多少决定了每个单刚的矩阵块的大小，所以得乘以适当的系数。
比如平面刚架元，其既考虑轴向变形，也考虑弯曲变形，每个节点上有三个自由度，其总刚组装时的系数同时变化，如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/fem-3-2.jpeg)
