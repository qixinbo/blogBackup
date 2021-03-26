---
title: 多物理场面向对象模拟环境MOOSE学习手册
tags: [MOOSE]
categories: simulation
date: 2017-1-15
---

# 引子
MOOSE，全名是Multiphysics Object Oriented Simulation Environment，是一套求解多物理场耦合的工程问题的框架。其设计规范，采用面向对象的编程范式，非常易于扩展和维护，而且尽可能地隐藏问题背后的计算数学问题，比如自适应网格算法、并行计算技术等，使得用户能够着眼于自己所要研究的科学问题。
# 特点
其有如下特点：
1. 与维度无关的编程，用户代码不需要考虑维度问题
2. 基于有限元，可以是连续有限元，也可以是间断有限元
3. 多物理场耦合，隐藏编程细节
4. 非规则网格，有多种形状：三角形、四边形、六边形、六面体、八面体等，可读入和输出多种形式。
5. 自适应网格
6. 并行计算
7. 高次插值
8. 内置后处理
等等。

# 安装
提前安装好的软件：
- gcc4.8(系统默认自带)
- g++4.8(可以直接命令行安装，也可以使用新立得，如果默认源不匹配，要注意更新源和切换源，如果必要，也要安装build-essential)
- gfortran4.8 (moose中有fortran程序，如waterStreamEos模块)
- python相关库：python-dev、python-numpy、python-yaml、python-matplotlib、python-vtk
- git
- mpich3.2，不推荐openmpi，会出现以下问题：
```cpp
no underlying compiler was specified in the wrapper compiler data file (e.g. mpicc-wrapper-data.txt)
```
- hypre 最好是装上，不装也可以，程序也能运行，但会出现以下问题：
```cpp
Unable to find requested PC type hypre
```
- PETSc3.7.4(注意记录安装完成后产生的路径和架构信息)，其参数可为：
```cpp
--download-fblaslapack --with-mpi-dir=/usr/local/mpich --download-hypre
```

然后，按照官网教程来：
1. 克隆源码库
```cpp
mkdir ~/projects
cd ~/projects
git clone https://github.com/idaholab/moose.git
cd ~/projects/moose
git checkout master
```
2. 编译libMesh
```cpp
cd ~/projects/moose
scripts/update_and_rebuild_libmesh.sh
```
3. 测试
```cpp
cd ~/projects/moose/test
make -j8
./run_tests -j8
```

# 创建程序
要创建自己的程序，先去github上克隆官方给出的一个stork库，然后在自己的github里重命名该库，官方建议用动物名命名，然后再将自己的库克隆下来：
```cpp
git clone https://github.com/<username>/<app_name>.git
```
注意，克隆下来的库的位置要与moose库同级，原因可以查看后面生成的Makefile文件，其会在与当前库相同的路径下寻找moose库，就是下面这句话：
```cpp
MOOSE_DIR        ?= $(shell dirname `pwd`)/moose
```
从上面的“问号等于”赋值语句来看，也可以任意放置该库位置，只需手动指定一下MOOSEDIR这个环境变量即可。
这时候克隆下来的库还是很凌乱的，不知道其组织结构是什么，再运行“创建新程序”这个脚本：
```cpp
cd <app_name>
./make_new_application.py
```
这时候整个代码结构就焕然一新，通过查看git状态就可以知道该脚本干了什么：
```cpp
#	renamed:    Makefile.app -> Makefile
#	deleted:    Makefile.module
#	modified:   README.md
#	modified:   doc/doxygen/Doxyfile
#	new file:   include/base/StarfishApp.h
#	deleted:    include/base/StorkApp.h
#	deleted:    make_new_application.py
#	deleted:    make_new_module.py
#	renamed:    run_tests.app -> run_tests
#	deleted:    run_tests.module
#	new file:   src/base/StarfishApp.C
#	new file:   src/base/StarfishApp.C.module
#	deleted:    src/base/StorkApp.C.app
#	deleted:    src/base/StorkApp.C.module
#	modified:   src/main.C
#	modified:   unit/Makefile
#	modified:   unit/run_tests
#	modified:   unit/src/main.C
```
然后，再make一下，就可以得到可执行文件。用tests中的输入文件测试一下：
```cpp
./starfish-opt -i /home/qixinbo/MyProjects/starfish/tests/kernels/simple_diffusion/simple_diffusion.i
```
后续就是创建自己的输入文件。

# 输入文件
输入文件是“分级块状”形式，比如：
```cpp
[Variables]
  active = 'u'
  [./u]
    order = FIRST
    family = LAGRANGE
  [../]
  [./v]
    order = FIRST
    family = LAGRANGE
  [../]
[]
[Kernels]
  ...
[]
```
可以看出，方括号括起来的就是一个块，且块里还可以包含块，这就是分级块状结构。
在moose的[帮助文档](http://mooseframework.com/docs/syntax/moose/)中，已经很直观地列出了各种块的结构。需要注意的是其颜色和符号表示：加粗的名称表示其是一个块，红色参数表示该参数必须人为指定。输入文件可以手动书写，也可以在Peacock孔雀中设定。Peacock是一个图形前端，使用户可以创建或修改输入文件，执行计算和可视化计算结果。Peacock中黑色字体的块可以双击以编辑，蓝色字体的块可以右键单击进行添加，出来编辑对话框后，最上面可以选择总的类型选择，Peacock中必须设定的字段是橘黄色填充，对应moose帮助文档中的红色参数。

重要的“块”有网格、变量、内核模型、边界条件、求解器、输出等。通常，用户程序需要自己创建网格，设定自己的变量及其有限元插值方式，编写自己的内核模型，当然对简单问题也可以直接调用moose的内核模型，设定边界条件，选择求解器，设置输出格式等。

# 创建私有对象
若想创建自己的程序，比如编写自己的物理模型或边界条件等，只需继承一个已有的moose对象，然后对其进行扩展。比如编写自己的内核模型，需要在本地程序中在头文件和源文件路径中都创建kernals路径，并分别创建相应的头文件和源文件，然后需要在这两个文件中做以下事情：
## 定义有效参数和继承已有的moose对象：
在头文件中：
```cpp
class Convection;

template<>
InputParameters validParams<Convection>();

class Convection : public Kernel
...
```
这里的Convection就是新定义的内核，其父类是moose的Kernel类。
每一个moose派生的对象都必须指定一个validParams函数，它必须先取出父类中的参数，然后再增加额外的参数。即在源文件中：
```cpp
template<>
InputParameters validParams<Convection>()
{
  InputParameters params = validParams<Kernel>();  // Must get from parent
  params.addRequiredParam<RealVectorValue>("velocity", "Velocity Vector");
  params.addParam<Real>("coefficient", "Diffusion coefficient");
  return params;
}
```
这样，在输入文件中就需要指定该对象所需要的参数。可以用
```cpp
./ex02-opt --dump [optional search string]
```
来具体查看所需参数。

添加函数的语法为：
```cpp
addRequiredParam<Real>("required_const", "doc");
addParam<int>("count", 1, "doc"); // default supplied
addParam<unsigned int>("another_num", "doc");
addRequiredParam<std::vector<int> >("vec", "doc");
```
第一个参数是参数名，输入文件中要跟其相同，最后一个参数是注释。
对于非required的参数，可以设置其缺省值或取值范围，如：
```cpp
addParam<RealVectorValue>("direction", RealVectorValue(0,1,0), "doc");
addRangeCheckedParam<Real>("temp", "temp>=300 & temp<=330", "doc");
```

## 注册该对象
即在具体问题的app源文件中注册一下：
```cpp
#include "ExampleConvection.h" 
void
ExampleApp::registerObjects(Factory & factory)
{
// Register any custom objects you have built on the MOOSE Framework
registerKernel(ExampleConvection); // <- registration
}
```
## 使用该对象
在输入文件中使用该对象，比如这里的内核模型：
```cpp
[Kernels]
  [./diff]
    type = Diffusion
    variable = convected
  [../]
  [./conv]
    type = ExampleConvection
    variable = convected
    velocity = '0.0 0.0 1.0'
  [../]
[]
```

# 网格
网格块默认是读入网格文件，即默认：
```cpp
type = FileMesh
```
可以读入很多种格式的网格文件，比如libMesh、Tecplot、GMSH、Abaqus等。所以对于复杂形状的几何体，可以使用专业的网格生成软件，再由moose读入。

使用GMSH画网格时，注意按照“点-线-面-体”的步骤生成几何体，不要漏了某一步骤。
同时，Attention：还要生成"Physical Groups"，将面加上标识，用以设置边界条件，同时还要将体加上标识，否则在peacock中不能正确识别整个三维结构。

还可以是内置生成的网格，即：
```cpp
type = GeneratedMesh
```
可以用来创建一些简单的形状，如立方体等。
边界标识为：
```cpp
In 1D, left = 0, right = 1
In 2D, bottom = 0, right = 1, top = 2, left = 3
In 3D, back = 0, bottom = 1, right = 2, top = 3, left = 4, front = 5
```
计算域的维度、长高宽、每个方向上单元个数等都可以人为指定：
```cpp
[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 100
  ny = 100
  xmax = 60
  ymax = 60
[]
```
还有其他的网格类型，比如EBSDMesh、ImageMesh等。
# 内核模型对象
一个内核Kernel是一个物理模型，表示偏微分方程中的一个或多个算子。
一个内核必须重载computeQpResidual函数，可以视情况重载computeQpJacobian和computeQpOffDiagJacobian函数。
内核的常用数据成员：
```cpp
_u,_grad_u: 该kernel所操作的变量的值和梯度
_phi,_grad_phi: 基函数在积分点上的值和梯度
_test,_grad_test: 试探函数在积分点上的值和梯度 
_q_point: 当前积分点的xyz坐标
_i,_j: 分别是试探函数和有限元插值函数的当前形函数
_qp: 当前积分点标识
_current_elem: 指向当前单元的指针
```
在伽辽金有限元法中，试探函数和有限元函数的基函数相同。
MOOSE中默认把控制方程写成右端项为0，即将所有项都移到左端，称为残差向量Residual Vector。

对于时间发展方程，moose的内核是TimeDerivative。常用的数据成员是：
```cpp
_u_dot: u对时间的导数
_du_dot_du: u对时间的导数对u的导数，用于计算雅各比矩阵
```

# 初始条件对象
时间相关的问题需要设置初始条件，如果使用ExodusII网格，还可以将之前模拟中的变量的值作为初始条件。
首先在头文件中创建有效参数和继承已有moose对象：
```cpp
#ifndef EXAMPLEIC_H
#define EXAMPLEIC_H

#include "InitialCondition.h"

class ExampleIC;
template<>
# InputParameters validParams<ExampleIC>();

class ExampleIC : public InitialCondition
{
public:
# ExampleIC(const std::string | name,
          InputParameters parameters);

virtual Real value(const Point | p);

private:
  Real _coefficient;
};

#endif //EXAMPLEIC_H
```
父类是moose的InitialCondition类，派生类必须重载value函数，可以选择性重载gradient函数。
然后在源文件中：
```cpp
#include "ExampleIC.h"

template<>
# InputParameters validParams<ExampleIC>()
{
  InputParameters params = validParams<InitialCondition>();
  params.addRequiredParam<Real>("coefficient", "A coef");
  return params;
}

# ExampleIC::ExampleIC(const std::string | name,
                     InputParameters parameters):
  InitialCondition(name, parameters),
  _coefficient(getParam<Real>("coefficient"))
{}

# Real
# ExampleIC::value(const Point | p)
{
  // 2.0 * c * |x|
  return 2.0*_coefficient*std::abs(p(0));
}
```
然后，注册该初始条件对象：
```cpp
#include "ExampleIC.h"
...
registerInitialCondition(ExampleIC);
```
在输入文件中使用时可以有多种形式：
第一种：直接使用ICs块：
```cpp
[ICs]
  [./mat_1]
    type = ExampleIC
    variable = u
    coefficient = 2.0
    block = 1
  [../]

  [./mat_2]
    type = ExampleIC
    variable = u
    coefficient = 10.0
    block = 2
  [../]
```
第二种：在variable块下使用：
```cpp
[Variables]
  [./diffused]
    order = FIRST
    family = LAGRANGE
    # Use the initial Condition block underneath the variable
    # for which we want to apply this initial condition
    [./InitialCondition]
      type = ExampleIC
      coefficient = 2.0;
    [../]
  [../]
[]
```
如果是简单的初始条件，设置不需要额外创建该对象，直接：
```cpp
[Variables]
  active = 'u'

  [./u]
    order = FIRST
    family = LAGRANGE
    # For simple constant ICs
    initial_condition = 10
  [../]
```
第三种：导入之前求解的解：
```cpp
[Variables]
  active = 'u'

  [./u]
    order = FIRST
    family = LAGRANGE
    # For reading a solution
    # from an ExodusII file
    initial_from_file_var = diffused}
    initial_from_file_timestep = 2
  [../]
```

# 边界条件对象
边界条件的结构与内核模型类似，不同是，有些边界条件在边界上积分，其继承moose的IntegratedBC类，而有些边界条件不在边界上积分，即Dirichlet边界条件，其继承NodalBC类。
对于边界条件可以使用的内部函数有：
对于积分的边界条件：
```cpp
_u, _grad_u: 此边界条件所操作的值和梯度
_phi, _grad_phi: 基函数的值和梯度
_test, _grad_test: 试探函数的值和梯度
_q_point: XYZ坐标
_i, _j: 形函数索引
_qp: 当前积分点索引
_normals: 法向量
_boundary_id: 边界标识
_current_elem: 指向当前单元的指针
_current_side: 当前单元的边的标识
```
对于非积分的边界条件：
```cpp
_u
_qp
_boundary_id
_current_node: 指向当前节点的指针
```
不同的边界可以有不同的边界条件，所以相互之间可以耦合使用：
```cpp
coupledValue()
coupledValueOld()
coupledValueOlder()
coupledGradient()
coupledGradientOld()
coupledGradientOlder()
coupledDot()
```

周期性边界条件对于守恒物理量在近似无限大的计算域中的演化非常有用。moose对于边界条件有非常强大的支持：可以设置不同维度、网格自适应、对于特定变量施加、设置任意变换矢量来设定周期性(可以与坐标系平行也可以成角度)。

# 材料属性对象
首先还是创建有效参数和继承moose对象：
```cpp
template<>
InputParameters validParams<ExampleMaterial>();
class ExampleMaterial : public Material
{
public:
  ExampleMaterial(const InputParameters & parameters);
protected:
  virtual void computeQpProperties() override;
...
}
这里的父类是moose的Material类，必须重载computeQpProperties函数。
Material类定义了材料属性后，其他对象通过getMaterialProperty函数来获取它们：
```cpp
ExampleConvection::ExampleConvection(const InputParameters & parameters) :
Kernel(parameters),
// Retrieve a gradient material property to use for the convection
// velocity
_velocity(getMaterialProperty<RealGradient>("convection_velocity"))
{}
```

# 后处理对象
一个后处理对象是对所得的解进行加工再处理的过程，比如得到每条边上的平均通量，结果是一个标量值。
其何时计算取决于这个参数：
```cpp
execute_on = timestep_end
execute_on = linear
execute_on = nonlinear
execute_on = timestep_begin
execute_on = custom
```
其可以限制在计算域的特定区域、边或节点的特定集合上:
1. 作用在单元上：可以是整个计算域上的单元，也可是通过设定一个或多个block标识来限制子区域，继承自ElementPostprocessor类;
2. 作用在节点上：可以是整个计算域上的节点，也可通过设定一个或多个boundary标识来限制点集，继承自NodalPostprocessor类;
3. 作用在边上：必须设定一个或多个boundary标识来决定在哪个边上计算，继承自SidePostprocessor类;
4. 通用设置：根据自己需求设置，继承自GeneralPostprocessor。

一些内置的后处理模块有：
```cpp
ElementIntegral 单元积分, ElementAverageValue 单元平均值
SideIntegral 边积分, SideAverageValue 边平均值
ElementL2Error 单元L2误差, ElementH1Error 单元H1误差
NumDOFs 自由度个数, NumNodes 节点个数, NumVars 变量个数
```
上述这些都可以继承并扩展。
后处理结果可以输出在屏幕上，也能写入一个CSV或Tecplot文件中，或者作为全局数据输出到Exodus文件中

# 求解器对象
有两种主要的求解器类型：稳态和瞬态。moose提供很多内置求解器，也可以自己创建。
控制求解器的常用参数：
```cpp
l_tol 	   线性忍量
l_max_its  最大线性迭代次数
nl_rel_tol 非线性相对忍量
nl_max_its 最大非线性迭代次数
```
对于瞬态问题，还有一些与时间相关的设置：
```cpp
dt          起始时间步长(后续可以变时间步长)
num_steps   时间步数
start_time  起始时间
end_time    终止时间
scheme      时间积分算法
```
moose提供了多种时间积分算法，包括下面的隐式时间积分算法：
```cpp
Backward Euler (缺省值)
BDF2
Crank-Nicolson
Implicit-Euler
Implicit Midpoint (implemented as two-stage RK method)
Diagonally-Implicit Runge-Kutta (DIRK) methods of order 2 and 3.
```
以及显式时间积分算法：
```cpp
Explicit-Euler
Various two-stage explicit Runge-Kutta methods (Midpoint, Heun, Ralston, TVD)
```
以上算法都支持自适应时间步进。
