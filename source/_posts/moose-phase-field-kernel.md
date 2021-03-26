---
title: MOOSE相场模块的内核模型
tags: [MOOSE, phasefield]
categories: simulation 
date: 2017-1-11
---

# 引子
moose的相场模块提供了通用的求解相场模型的算法，其通式采用的是自由能泛函的原始形式，只要用户要求解的模型满足这里内置的方程形式，那么用户仅需要提供自由能的导数和材料参数，就可以迅速进行模拟。比如教程中的调幅分解模型。
如果用户不能知道相场的原始形式，也可以自己开发模型，比如用于模拟枝晶生长的Kobayashi模型，就不满足上述规则，此时可以自己创建模型。

# 牛顿迭代法
moose内核模型的表达形式是将原来的控制方程中的右端项都移动到左端，得到这样的形式：
$$
R_i(u)=0, \qquad i=1,\ldots, N
$$
其中，N是分量的个数。
如果写成有限元函数的形式：
$$
R\_i(u\_h)=0, \qquad i=1,\ldots, N
$$
其中：
$$
u \approx u\_h = \sum\_{j=1}^N u\_j \phi\_j
$$
即，原来的连续的场变量由节点上的离散的系数来代替。

## 单变量求根
现在仅考虑单变量情形：
$$
f(x)=0
$$
moose采用Newton迭代法来对其求根，该方法的基本思想是：将非线性方程的求根问题，转化成某个线性方程的求根。比如，对该方程的左端项进行泰勒展开，且只取其线性主部：
$$
0=f(x)=f(x\_0)+f'(x\_0)(x-x\_0) 
$$
那么，解就是：
$$
x\_1=x\_0-\frac{f(x\_0)}{f'(x\_0)}
$$
这样就得到根的新的近似值。进一步迭代：
$$
x\_{k+1}=x\_k-\frac{f(x\_k)}{f'(x\_k)}
$$
当前后两步解的差值达到某个精度时，就认为该解就是方程的根。

## 多变量求根
回到有限元法中，因为场函数由分布在节点上的有限元函数值所代替，所以此时变量就有多个。那么，对应于单变量情形，可得出多变量时的表达式：
$$
\mathbf{J}(\vec{u}\_0) (\vec{u}\_1-\vec{u}\_0) = -\vec{R}(\vec{u}\_0)
$$
其中，$\mathbf{J}(\vec{u}\_0)$是当前迭代步的雅各比矩阵。上方带箭头的变量表示其是一个矢量，矢量大小就是节点的个数。
通用的迭代格式就是：
$$
\mathbf{J}(\vec{u}\_k) (\vec{u}\_{k+1}-\vec{u}\_k) = -\vec{R}(\vec{u}\_k)
$$
其中雅各比矩阵的具体形式为：
$$
J\_{ij}(\vec{u}\_k) = \frac{\partial R\_i(\vec{u}\_k)}{\partial u\_j}
$$
在求雅各比矩阵时，有两个基本公式是非常重要的：
$$
\frac{\partial u\_h}{\partial u\_j} =
      \sum\_k\frac{\partial }{\partial u\_j}(u\_k \phi\_k) = \phi\_j
    \qquad
\frac{\partial (\nabla u\_h)}{\partial u\_j} =
      \sum\_k \frac{\partial }{\partial u\_j}(u\_k \nabla \phi\_k)=\nabla \phi\_j
$$

## 线性迭代和非线性迭代
首先将
$$
\mathbf{J}(\vec{u}\_0) (\vec{u}\_1-\vec{u}\_0) = -\vec{R}(\vec{u}\_0)
$$
等效成：
$$
Ax=b
$$
线性迭代：对于某一个Newton迭代步，比如上面的k+1步，Ax=b是一个大型线性方程组。在求解这个线性方程组的过程中A和b保持不变，仅是不断地迭代x，使其收敛，这个过程叫线性迭代Linear Iteration，每步迭代的残差称为线性残差Linear Residual，第i步迭代的残差表达式为\rhoi=Axi-b。如果设置print_linear_residuals=true，那么MOOSE就会打印该残差向量的范数。

JFNK算法：在上面的线性迭代过程中，想要高效地求解x，这里使用Krylov子空间，将Ax=b转换为Kxi+1=Kxi+b-Axi的迭代形式来求解，具体的算法可以是GMRES或共轭梯度法等。这个算法中不明确地需要Jacobi矩阵，仅仅需要J对向量的作用。下面有详细介绍。

非线性迭代：也是在一个Newton迭代步中，针对的是总的残差R的迭代，从而使其收敛，进入下一个时间步，该残差称为非线性残差Nonlinear Residuals。

那么总的思路就是：对于某个时间步，首先给定一个R的初值R0，然后计算出一个x0，从而对x进行迭代，直到x收敛到给定线性迭代精度，这时候也就计算出了新的R值，即R1，如果R1和R0不满足非线性迭代精度要求，那么将R1再代入Ax=b中，再次对x迭代，直到再次达到线性迭代精度，再计算出R2，再判定是否达到非线性迭代精度。  

## 牛刀小试
对于一个经典的对流-扩散方程：
$$
-\nabla\cdot k\nabla u + \vec{\beta} \cdot \nabla u = f
$$
其残差向量的第i个分量为：
$$
R\_i(u\_h) = (\nabla\psi\_i, k\nabla u\_h)-\langle\psi\_i, k\nabla u\_h\cdot \hat{n} \rangle + (\psi\_i, \vec{\beta} \cdot \nabla u\_h)-(\psi\_i, f)
$$
雅各比矩阵的某个元素为：
$$
J\_{ij}(u\_h)= (\nabla\psi\_i,\frac{\partial k}{\partial u\_j}\nabla u\_h) +(\nabla\psi\_i, k \nabla \phi\_j ) - \langle\psi\_i, \frac{\partial k}{\partial u\_j}\nabla u\_h\cdot \hat{n} \rangle - \langle\psi\_i, k\nabla \phi_j\cdot \hat{n} \rangle + (\psi\_i, \frac{\partial \vec{\beta}}{\partial u\_j} \cdot\nabla u\_h) + (\psi\_i, \vec{\beta} \cdot \nabla \phi\_j) - (\psi\_i, \frac{\partial f}{\partial u\_j})
$$
注意，这里假定扩散系数、对流速度、源项等都是变量，所以形式会比较复杂。
尤其是对于多个方程相互耦合，或材料属性比较复杂的情形，雅各比矩阵的计算会很困难。

## 链式法则
有时候，可以应用链式法则来简化一下，比如：
$$
\frac{\partial f}{\partial u\_j} = \frac{\partial f}{\partial u\_h} \frac{\partial u\_h}{\partial u\_j}=\frac{\partial f}{\partial u\_h} \phi\_j
$$
如果f的表达式已知，比如$f(u) = \sin(u)$，那么：
$$
\frac{\partial f}{\partial u\_j} = \cos(u\_h) \phi\_j
$$

## JFNK算法
JFNK算法，全称是Jacobian Free Newton Krylov methods，是数值求解非线性问题的一种先进方法，其核心思想是将Newton非线性迭代法嵌入到Krylov空间法求解线性代数方程组的过程中，其显著优点是避免了传统Newton迭代法中的Jacobian矩阵生成环节，有利于降低内存占用率，缩短计算时长。
JFNK算法将Newton迭代法与Krylov空间法结合的方式是将Newton迭代法中的Jacobian矩阵$\mathbf{J}$与Krylov空间法的解向量$\vec{v}$之间进行向量积操作，其近似为：
$$
\mathbf{J}\vec{v} \approx \frac{\vec{R}(\vec{u} + \epsilon\vec{v}) - \vec{R}(\vec{u})}{\epsilon}
$$
这个算法的优点有：
- 无需计算偏导数来得到雅各比矩阵
- 无需直接计算雅各比矩阵
- 无需空间来存储雅各比矩阵

# 调幅分解所用的Cahn-Hilliard方程
这里将含有四阶导数的原CH方程，拆分成两个，这样每个都只含有二阶导数，易于求解，两个方程的变量分别是化学势和浓度。

## 化学势的残差
其表达式及其求解内核分为两个：
### 第一项
$$
(\frac{\partial c}{\partial t},\psi)
$$
该项中变量是化学势，耦合的变量是浓度，使用的内核是CoupledTimeDerivative。
```cpp
Real
CoupledTimeDerivative::computeQpResidual()
{
 return _test[_i][_qp] * _v_dot[_qp];
}

Real
CoupledTimeDerivative::computeQpJacobian()
{
 return 0.0;
}

```
### 第二项
$$
(M\nabla\mu,\nabla \psi)
$$
该项中变量是化学势，使用的内核是SplitCHWRes，实际使用的是SplitCHWResBase：

```cpp
template<typename T>
Real
SplitCHWResBase<T>::computeQpResidual()
{
 return _mob[_qp] * _grad_u[_qp] * _grad_test[_i][_qp];
}

template<typename T>
Real
SplitCHWResBase<T>::computeQpJacobian()
{
 return _mob[_qp] * _grad_phi[_j][_qp] * _grad_test[_i][_qp];
}
```

## 浓度的残差
残差表达式为：
$$
(\nabla c, \nabla(\kappa \psi))+((\frac{\partial f\_{loc}}{\partial c}+\frac{\partial E\_d}{\partial c} - \mu ), \psi) 
$$
变量是浓度，还需耦合化学势$\mu$，使用的内核是SplitCHParsed，实际使用的内核是SplitCHCRes：
```cpp
 Real
 SplitCHCRes::computeQpResidual()
 {
  Real residual = SplitCHBase::computeQpResidual(); //(f_prime_zero+e_prime)*_test[_i][_qp] from SplitCHBase
 
  residual += -_w[_qp] * _test[_i][_qp];
  residual += _kappa[_qp] * _grad_u[_qp] * _grad_test[_i][_qp];
 
  return residual;
 }
 
 Real
 SplitCHCRes::computeQpJacobian()
 {
  Real jacobian = SplitCHBase::computeQpJacobian(); //(df_prime_zero_dc+de_prime_dc)*_test[_i][_qp]; from SplitCHBase
 
  jacobian += _kappa[_qp] * _grad_phi[_j][_qp] * _grad_test[_i][_qp];
 
  return jacobian;
 }
``` 

注意，在computeQpResidual函数中，可以很容易地找出第一项和第三项的计算过程，但对于第二项，可能不容易发现，其实第二项的计算放在了最前面：

```cpp
Real residual = SplitCHBase::computeQpResidual(); //(f_prime_zero+e_prime)*_test[_i][_qp] from SplitCHBase
```
然后：
```cpp
Real
SplitCHBase::computeQpResidual()
{
 Real f_prime_zero = computeDFDC(Residual);
 Real e_prime = computeDEDC(Residual);

 Real residual = (f_prime_zero + e_prime) *_test[_i][_qp];

 return residual;
}
```
然后computeDFDC和computeDEDC就是计算自由能密度和其他能量对浓度的一阶导数，即：
```cpp
Real
SplitCHParsed::computeDFDC(PFFunctionType type)
{
 switch (type)
 {
 case Residual:
 return _dFdc[_qp];

 case Jacobian:
 return _d2Fdc2[_qp] * _phi[_j][_qp];
 }

 mooseError("Internal error");
}
```
实际上，
```cpp
_dFdc(getMaterialPropertyDerivative<Real>("f_name", _var.name())),
_d2Fdc2(getMaterialPropertyDerivative<Real>("f_name", _var.name(), _var.name()))
```
注意取一阶导数和二阶导数时，是由后面的参数个数来控制。

具体的自由能形式则是在输入文件中输入：
```cpp
[Materials]
  [./local_energy]
    # Defines the function for the local free energy density as given in the
    # problem, then converts units and adds scaling factor.
    type = DerivativeParsedMaterial
    block = 0
    f_name = f_loc
    args = c
    constant_names = 'A   B   C   D   E   F   G  eV_J  d'
    constant_expressions = '-2.446831e+04 -2.827533e+04 4.167994e+03 7.052907e+03
                            1.208993e+04 2.568625e+03 -2.354293e+03
                            6.24150934e+18 1e-27'
    function = 'eV_J*d*(A*c+B*(1-c)+C*c*log(c)+D*(1-c)*log(1-c)+
                E*c*(1-c)+F*c*(1-c)*(2*c-1)+G*c*(1-c)*(2*c-1)^2)'
  [../]
[]
```

# 枝晶生长所用的Kobayashi模型

## 相场的残差
包括以下几项：

### 第一项
$$
(\frac{\partial w}{\partial t},\psi)
$$
内核使用TimeDerivative：
```cpp
#include "TimeDerivative.h"
#include "Assembly.h"

// libmesh includes
#include "libmesh/quadrature.h"

template<>
InputParameters validParams<TimeDerivative>()
{
 InputParameters params = validParams<TimeKernel>();
 params.addParam<bool>("lumping", false, "True for mass matrix lumping, false otherwise");
 return params;
}

TimeDerivative::TimeDerivative(const InputParameters & parameters) :
 TimeKernel(parameters),
 _lumping(getParam<bool>("lumping"))
{
}

Real
TimeDerivative::computeQpResidual()
{
 return _test[_i][_qp]*_u_dot[_qp];
}

Real
TimeDerivative::computeQpJacobian()
{
 return _test[_i][_qp]*_phi[_j][_qp]*_du_dot_du[_qp];
}

void
TimeDerivative::computeJacobian()
{
 if (_lumping)
 {
 DenseMatrix<Number> & ke = _assembly.jacobianBlock(_var.number(), _var.number());

 for (_i = 0; _i < _test.size(); _i++)
 for (_j = 0; _j < _phi.size(); _j++)
 for (_qp = 0; _qp < _qrule->n_points(); _qp++)
 ke(_i, _i) += _JxW[_qp] * _coord[_qp] * computeQpJacobian();
 }
 else
 TimeKernel::computeJacobian();
}
```

### 第二项
注意将残差都移动到方程左端，注意符号变化。
$$
(-L\epsilon\epsilon\prime\frac{\partial w}{\partial y},\frac{\partial \psi}{\partial x})+(L\epsilon\epsilon\prime\frac{\partial w}{\partial x},\frac{\partial \psi}{\partial y})
$$
这两项使用内核ACInterfaceKobayashi1.h，其实际使用KernelGrad：
```cpp
RealGradient
ACInterfaceKobayashi1::precomputeQpResidual()
{
 // Set modified gradient vector
 const RealGradient v(- _grad_u[_qp](1), _grad_u[_qp](0), 0);

 // Define anisotropic interface residual
 return _eps[_qp] * _deps[_qp] * _L[_qp] * v;
}
...
void
KernelGrad::computeResidual()
{
 DenseVector<Number> & re = _assembly.residualBlock(_var.number());
 _local_re.resize(re.size());
 _local_re.zero();

 const unsigned int n_test = _test.size();
 for (_qp = 0; _qp < _qrule->n_points(); _qp++)
 {
 RealGradient value = precomputeQpResidual() * _JxW[_qp] * _coord[_qp];
 for (_i = 0; _i < n_test; _i++) // target for auto vectorization
 _local_re(_i) += value * _grad_test[_i][_qp];
 }

 re += _local_re;

 if (_has_save_in)
 {
 Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);
 for (const auto & var : _save_in)
 var->sys().solution().add_vector(_local_re, var->dofIndices());
 }
}
```

### 第三项
$$
(L\epsilon^2\nabla w,\nabla \psi)
$$
注意，内核中不是直接使用该形式，该形式是自由能对相场参量的一阶导数，实际程序中使用的也是一阶导数，所以需要输入的是自由能，即该形式的积分。
其使用的内核是ACInterfaceKobayashi2，实际也是KernelGrad：
```cpp
RealGradient
ACInterfaceKobayashi2::precomputeQpResidual()
{
 // Set interfacial part of residual
 return _eps[_qp] * _eps[_qp] * _L[_qp] * _grad_u[_qp];
}
...
KernelGrad::computeResidual()
{
 DenseVector<Number> & re = _assembly.residualBlock(_var.number());
 _local_re.resize(re.size());
 _local_re.zero();

 const unsigned int n_test = _test.size();
 for (_qp = 0; _qp < _qrule->n_points(); _qp++)
 {
 RealGradient value = precomputeQpResidual() * _JxW[_qp] * _coord[_qp];
 for (_i = 0; _i < n_test; _i++) // target for auto vectorization
 _local_re(_i) += value * _grad_test[_i][_qp];
 }

 re += _local_re;

 if (_has_save_in)
 {
 Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);
 for (const auto & var : _save_in)
 var->sys().solution().add_vector(_local_re, var->dofIndices());
 }
}
```
(4)
$$
(-Lw(1-w)(w-0.5+m),\psi)
$$
使用的内核是AllenCahn，实际间接使用了ACBulk，最终使用KernelValue：
```cpp
...
_dFdEta(getMaterialPropertyDerivative<Real>("f_name", _var.name())),
_d2FdEta2(getMaterialPropertyDerivative<Real>("f_name", _var.name(), _var.name())),
...
Real
AllenCahn::computeDFDOP(PFFunctionType type)
{
 switch (type)
 {
 case Residual:
 return _dFdEta[_qp];

 case Jacobian:
 return _d2FdEta2[_qp] * _phi[_j][_qp];
 }

 mooseError("Internal error");
}
...
template<typename T>
Real
ACBulk<T>::precomputeQpResidual()
{
 // Get free energy derivative from function
 Real dFdop = computeDFDOP(Residual);

 // Set residual
 return _L[_qp] * dFdop;
}
...
void
KernelValue::computeResidual()
{
 DenseVector<Number> & re = _assembly.residualBlock(_var.number());
 _local_re.resize(re.size());
 _local_re.zero();

 const unsigned int n_test = _test.size();
 for (_qp = 0; _qp < _qrule->n_points(); _qp++)
 {
 Real value = precomputeQpResidual() * _JxW[_qp] * _coord[_qp];
 for (_i = 0; _i < n_test; _i++) // target for auto vectorization
 _local_re(_i) += value * _test[_i][_qp];
 }

 re += _local_re;

 if (_has_save_in)
 {
 Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);
 for (const auto & var : _save_in)
 var->sys().solution().add_vector(_local_re, var->dofIndices());
 }
}
```
## 温度场的残差
由以下几项构成：

### 第一项
$$
(\frac{\partial T}{\partial t},\psi)
$$
所以使用的内核是TimeDerivative。

### 第二项
$$
(\nabla T,\nabla \psi)
$$
所以使用的内核是Diffusion。

### 第三项
$$
(-K\frac{\partial w}{\partial t},\psi)
$$
使用的内核是CoefCoupledTimeDerivative，实际就是CoupledTimeDerivative再乘以一个系数：
```cpp
Real
CoefCoupledTimeDerivative::computeQpResidual()
{
 return CoupledTimeDerivative::computeQpResidual() * _coef;
}
Real
CoupledTimeDerivative::computeQpResidual()
{
 return _test[_i][_qp] * _v_dot[_qp];
}
```
