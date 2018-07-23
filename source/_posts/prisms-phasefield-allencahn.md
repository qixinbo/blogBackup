---
title: PRISMS-PF学习手册——AllenCahn方程求解
tags: [deal.II, prisms]
categories: computational material science
date: 2016-12-1
---

# 概述
[PRISMS](http://www.prisms-center.org/)是“PRedictive Integrated Structural Materials Science”的缩写，是美国能源部旗下的一个软件创新中心，致力于开发用于多尺度结构材料模拟的开源计算工具。该中心目前已释放出多个开源软件，基于包括相场模型、晶体塑性模型等。这里是对PhaseField的学习。源码在[这里](https://github.com/prisms-center/phaseField)。

几点说明：
- 最好是从最早的v0.8学起，v0.8是最早公开的代码，程序结构简单，耦合程度低，便于学习。
- PRISMS-PF的版本与其所基于的deal.II版本需要对应，v0.8基于deal.II的v8.2.1，v1.0基于v8.4.1。这是因为deal.II跨版本升级时会弃用某些类，比如8.2.1中的FEEvaluationGL类在8.3中就不再支持。
- 目前测试gcc版本不能低于4.8.0，如4.6.3就会出现编译错误。

# 文件
v0.8主要包括三个文件：
- CMakeLists.txt: cmake所需的编译文件，用于产生makefile文件
- main.cc: 包含初始条件和程序入口main函数 (v0.8中的边界条件都是no-flux，v1.0中加入了多种条件)
- parameters.h: 用于输入各种参数
## 输入参数
parameter.h中输入的参数包含两大类：适用于所有模型的通用参数和与模型相关的参数。
参数都使用#define来定义。
### 通用参数
- 维数：problemDIM
- x方向长度：spanX
- y方向长度：spanY (如果维数小于2,则不使用)
- z方向长度：spanZ (如果维数小于3,则不使用)
- 网格细化次数：refineFactor。每个方向上有2的refineFactor次方个单元
- 有限元空间的插值次数：finiteElementDegree。此时都设为1,即使用Lagrange单元
- 时间步长：dt
- 模拟步数：numIncrements。最终时刻则等于dt乘以numIncrements
- 是否输出：writeOutput。设定布尔值true则为输出
- 每隔多少步输出：skipOutputSteps。如果writeOutput为真时，初始条件先输出。
### 与模型相关的参数
这些参数包括常系数、体积自由能和余量。其中，体积自由能及其导数是相场变量(在代码中通常用n或c表示)的函数，v0.8中都是多项式自由能，非多项式的自由能也能添加，但v0.8不支持。
这里以Allen-Cahn方程为例：
- 迁移率系数：Mn
- 梯度能系数的平方：Kn
- 体积自由能的一阶导数：fnV
- 余量：rnV和rnxV

## 初始条件
如前所述，初始条件放在main.cc文件中设置。v0.8使用了两种类型的初始条件：在平均值附近的随机噪声(比如初始浓度的微小起伏)或位置的解析函数(比如相场的双曲正切分布)。

# 程序解析
## 头文件
包含三种头文件：所用的deal.II的头文件、参数头文件和方程求解方法的头文件：
```cpp
#include "../../include/dealIIheaders.h"
#include "parameters.h"
#include "../../src/AC.h"
```
## 初始条件
以下使用的是在初始浓度基础上加一个随机噪声小量，实际就是对InitialConditionN类的virtual函数value进行了函数实现。InitialConditionN类在AC.h中定义，是对Function类的Public继承，它的构造函数用srand函数对下面的rand函数撒种子，这个srand函数这里设置成不同进程有不同的seed值。
```cpp
template <int dim>
double InitialConditionN<dim>::value (const Point<dim> &p, const unsigned int /* component */) const
{
  //return the value of the initial order parameter field at point p 
  return  0.5+ 0.2*(0.5 - (double)(std::rand() % 100 )/100.0);
}
```

## 求解过程
生成AllenCahnProblem类的一个对象，并执行run函数：
```cpp
AllenCahnProblem<problemDIM> problem;
problem.run ();
```
AllenCahnProblem类Public继承自Subscriptor类。
### 创建网格
首先根据不同的维度创建初始网格，网格大小就是从原点到之前参数中给的位置点。然后使用triangulation全局细化，细化次数是之前参数中的refineFactor，所以每个方向上的网格个数是2的refineFactor次方。

### 构建系统
调用该类的setup_system()成员函数：
先是自由度分配，因为这里还牵扯到多进程的自由度的分配，使用了很多高级语句，具体可以参考Step-40的用法。
再调用该类的setup_matrixfree()成员函数，创建该问题所依赖的数据体data，并初始化，然后建立质量矩阵，因为这里使用了GL单元，所以该矩阵其实是对角的，它的类型是一个存储对角元素的容器。
然后施加初始条件。
### 求解
先是调用updateRHS()函数更新右端项，其调用ComputeRHS来计算右端项。
这块是程序的主要部分，也是主要修改部分，涉及AC方程的分解：
(1)右端项中涉及试探函数的函数值的部分：
```cpp
valN.submit_value(rnV,q);
```
(2)右端项中涉及试探函数的梯度的部分：
```cpp
valN.submit_gradient(constNx*rnxV,q);
```

# 结果
结果如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/AC_outAnimation.gif)
