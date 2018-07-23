---
title: 利用Mathematica进行有限元编程(一)：杆单元分析
tags: [mathematica]
categories: programming
date: 2016-8-17
---

本文是对[Mathematica有限元分析与工程应用](https://www.amazon.cn/Mathematica%E6%9C%89%E9%99%90%E5%85%83%E5%88%86%E6%9E%90%E4%B8%8E%E5%B7%A5%E7%A8%8B%E5%BA%94%E7%94%A8-%E9%B2%8D%E5%9B%9B%E5%85%83/dp/B00328IIOC)一书的学习笔记。

这里的杆单元是总体坐标与局部坐标一致的一维有限元，因此不需要坐标变换，更易分析。
# 线性杆单元的形函数
线性杆单元的位移函数是坐标x的一次函数，可用以下代码求解：
```cpp
Clear[u1, u2];
u = a1 + a2 x;
solut = Solve[{u1 == (u /. x -> 0), u2 == (u /. x -> 1)}, {a1, a2}];
u = u /. solut[[1]]
```
其中u1和u2是杆端点x=0和x=1处的位移，结果为：
```cpp
u1 + (-u1 + u2) x
```
形函数则可以通过取系数函数Coefficient获得：
```cpp
NShape = {Coefficient[u, u1], Coefficient[u, u2]}
```
结果为：
```cpp
{1 - x, x}
```
# 二次杆单元的形函数
二次杆单元的位移函数是坐标x的二次函数，求解代码为：
```cpp
Clear[u1, u2, u3, a1, a2, a3];
u = a1 + a2 x + a3 x^2;
solut = Solve[{u1 == (u /. x -> 0), u2 == (u /. x -> 1/2), 
        u3 == {u /. x -> 1}}, {a1, a2, a3}];
u = u /. solut[[1]]
```
结果为：
```cpp
u1 + (-3 u1 + 4 u2 - u3) x + 2 (u1 - 2 u2 + u3) x^2
```
其中u1、u2和u3分别是杆端点x=0、杆中点x=0.5、杆端点x=1处的位移。
那么形函数就是：
```cpp
NShape = Coefficient[u, {u1, u2, u3}]
```
结果为：
```cpp
{1 - 3 x + 2 x^2, 4 x - 4 x^2, -x + 2 x^2}
```
# 单元刚度矩阵
一次杆单元的刚度矩阵可以类比弹簧元的刚度矩阵。
设弹簧的刚度系数为k，节点i，j的位移分别为$u_i$和$u_j$，受到的力分别为$F_i$，$F_j$。易知：
$$
\begin{equation}
\begin{split}
F\_i &= ku\_i-ku\_j \\\
F\_j &= ku\_j-ku\_i
\end{split}
\end{equation}
$$
写成矩阵形式为：
$$
\begin{bmatrix}
k & -k \\\
-k & k
\end{bmatrix}
\begin{bmatrix}
u\_i \\\
u\_j
\end{bmatrix}
=
\begin{bmatrix}
F\_i \\\
F\_j
\end{bmatrix}
$$
所以单元刚度矩阵为：
$$
k\_{ij}=
\begin{bmatrix}
k & -k \\\
-k & k
\end{bmatrix}
$$
同理，一次杆单元的刚度矩阵为：
$$
k\_{ij}=\frac{EA}{L}
\begin{bmatrix}
1 & -1 \\\
-1 & 1
\end{bmatrix}
$$
二次杆单元的刚度矩阵可通过能量法求解。应用能量法的示意图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/fem-2-1.jpeg)
应变能为：
$$
U=\int\_\Omega \frac{1}{2}\sigma\epsilon dV=\int\_0^L\frac{1}{2}EA\epsilon^2dx
=\int\_0^L\frac{1}{2}EAa^T\frac{dN^T}{dx}\frac{dN}{dx}adx=\frac{1}{2}a^TKa
$$
其中：
$$
\sigma=E\epsilon; \epsilon=\frac{du}{dx}=\frac{dN}{dx}a, a=[u\_1,u\_2,u\_3]^T
$$
所以单元刚度矩阵为：
$$
K=EA\int\_{0}^L[\frac{dN}{dx}]^T[\frac{dN}{dx}]dx
$$
代入上面的形函数，那么求解代码为：
```cpp
dNx = L D[NShape, x];
K = EA Integrate[Transpose[{dNx}].{dNx}, {x, 0, 1}]/L;
K // MatrixForm
```
注意：代码中的L这个系数是因为使用了之前位于0到1上的形函数，如果直接采用0到L，则形函数的具体形式会发生改变。
输出结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/chapter-2.jpeg)

# 模块分析
## 模块1:建立单元刚度矩阵
以下分别建立了弹簧元、线性杆单元、二次杆单元的单元刚度矩阵：
```cpp
GenerateSpringKm[k_] := 
 Module[{y}, y = k*{{1, -1}, {-1, 1}}; y]
GenerateLinearRodKm[k_] := 
 Module[{y}, y = EE*AA/L*{{1, -1}, {-1, 1}}; y]
GenerateQuadraticRodKm[k_] := 
 Module[{y}, y = EE*AA/L*{{7, 1, -8}, {1, 7, -8}, {-8, -8, 16}}; y]
```
## 模块2:组装整体刚度矩阵
以弹簧元为例，单独的一个弹簧元有两个节点，其单元刚度矩阵是2行2列的矩阵。对于含有n个节点的弹簧元系统，则是n行n列的整体刚度矩阵。组装过程为：假设某个弹簧元的节点的整体编号为i和j，其单刚中的四个元素要分别找到其对应的整体编号位置，然后叠加上去，即11对应ii，12对应ij，21对应ji，22对应jj。代码为：
```cpp
GlobalK = 0 IdentityMatrix[3];
AssembleSpringKm[p1_, p2_, m_] := Module[{j, k}, f = {p1, p2};
  For[j = 1, j <= 2, j++,
     For[k = 1, k <= 2, k++,
         GlobalK[[f[[j]], f[[k]]]] += m[[j, k]];
    ]];
      GlobalK]
```
代码中p1和p2是节点的整体编号(输入时要注意节点号的顺序)，m是该单元的单刚，GlobalK是整刚。
二次杆单元的整体刚度矩阵则为：
```cpp
AssembleQuadraticRod[p1_, p2_, p3_, m_] := 
 Module[{i, j, k}, f = {p1, p2};
  For[i = 1, i <= 3, i++,
     For[j = 1, j <= 3, j++,
         For[k = 1, k <= 3, k++,
              GlobalK[[f[[j]], f[[k]]]] += m[[j, k]];
     ]]];
       GlobalK]
```
建立好整体刚度矩阵后，代入边界条件，即可求出位移量，比如：
```cpp
U = {u1x, u2x, u3x};
u1x = u3x = 0;
F2x = 17.5;
Load = {F1x, F2x, F3x};
tmp = Sort[Join[U, Load]];
unknown = Drop[tmp, 3];
solut = Solve[GlobalK.U == Load, unknown]
```
这里使用的边界条件是编号1和3上的位移为0,2上的力为17.5，然后利用Solve求解。
## 模块3:得到单元应力
已知单元的位移向量u(在整体坐标系下的值)，求单元上的力：
```cpp
SpringElementForce[k_, u_, i_] := Module[{force},
      force = k.u;
  Print["The spring force of the ele", i, " is ", force, "kN"];
    ];
RodElementForce[k_, u_, AA_, i_] := Module[{force},
      force = k.u;
  sigma = force/AA;
    Print["The force of the ele", i, " is ", force, "kN, ", 
       "the stress is ", sigma, " Pa."];
  ]
```
i是单元编号，u是位移向量，k是单刚。

