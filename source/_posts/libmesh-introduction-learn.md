---
title: libMesh的5个Introduction教学例子学习 
tags: [libmesh,linux]
categories: programming
date: 2017-6-18
---

# Intro 1 
第一个例子演示了怎样创建一个Mesh Object。
头文件：
```cpp
#include "libmesh/libmesh.h"  //用来初始化库
#include "libmesh/mesh.h"  //包含基本的mesh功能
using namespace libMesh;  //所有的东西都在libMesh命名空间中
```
入口函数代码： 
```cpp
LibMeshInit init (argc, argv); // 初始化，这是必要的一步，因为libMesh可能依赖其他的比如MPI、PETSc等函数库
```
因为该程序的用法是：
```cpp
./ex1 -d DIM input_mesh_name [-o output_mesh_name]
```
所有这里会有一个参数个数的判断，如果少于4个参数，就会报错，提示正确的用法：
```cpp
libmesh_error_msg("Usage: " << argv[0] << " -d 2 in.mesh [-o out.mesh]");
```
以下是对mesh object的操作：
```cpp
Mesh mesh(init.comm());  // 在默认通信子上创建一个Mesh
mesh.read(argv[3]);  //读入网格文件
mesh.print_info();  //在屏幕上输出网格信息
mesh.write(argv[5]);  //输出网格文件
```
# Intro 2
第二个例子演示了怎样创建一个标量方程。同时引入了PETSc，默认情形下方程数据都存入PETSc vector中。
头文件：
```cpp
#include <iostream> // C++头文件
//实现网格功能的基本头文件
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h" // 定义多种生成网格的组件
#include "libmesh/equation_systems.h" // 定义方程系统
// 定义简单的稳态系统
#include "libmesh/linear_implicit_system.h" 
#include "libmesh/transient_system.h"
#include "libmesh/explicit_system.h"

using namespace libMesh;
``` 
源码：
```cpp
Mesh mesh(init.comm()); // 创建mesh
MeshTools::Genertion::build_square (mesh,5,5); // 划分成5×5的方形grid
```
这样创建的网格就有5*5=25个单元elements，6*6=36个节点nodes。
```cpp
EquationSystems eqution_systems (mesh); // 创建方程，需要将mesh传递给equation
equation_systems.add_system<TransientLinearImplicitSystem>("Simple System"); // 往方程系统中添加一个方程，并命名
equation_systems.get_system("Simple System").add_variable("u", FIRST); // 给方程添加变量
equation_systems.add_system<ExplicitSystem>("Complex System"); // 再次添加一个方程
equation_systems.get_system("Complex System").add_variable("c", FIRST); // 添加c变量
equation_systems.get_system("Complex System").add_variable("T", FIRST); // 添加T变量
equation_systems.get_system("Complex System").add_variable("dv", SECOND, MONONIAL); // 添加dv变量

equation_systems.init(); // 初始化方程系统的数据结构

mesh.print_info(); // 在屏幕上打印网格信息

equation_systems.print_info(); // 在屏幕上打印方程系统信息

equation_systems.write(argv[1],WRITE); // 将方程信息输出到文件中
equation_systems.clear(); // 清除方程的数据结构
equaiton_systems.read(argv[1],READ); // 读入刚才的文件, 这三步可用于周期性地存储结果，并接着计算
```
# Intro 3
这个例子演示了怎样求解Possion方程。同样展示了矩阵组装、解析解对比、单元迭代等功能。

头文件：
```cpp
#include "libmesh/fe.h" // 定义有限元对象
#include "libmesh/quadrature_gauss.h" // 定义高斯积分规则

// 定义有限元刚度矩阵和解向量
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"

#include "libmesh/dof_map.h" // 定义自由度映射
```

源码：
```cpp
MeshTools::Generation::build_square(mesh, 15,15,-1.0,1.0,-1.0,1.0,QUAD9); // 设定生成网格的计算域、网格密度和类型
EquationSystems equation_systems(mesh);
equation_systems.add_system<LinearImplicitSystem>("Possion");
equation_systems.get_system("Possion").add_variable("u", SECOND);

// 指定一个指向矩阵组装功能的指针，具体的组装方法见函数实现
equation_systems.get_system("Possion").attatch_assemble_function (assemble_possion); 

// 求解Possion方程，这将组装线性系统，并调用数值求解器，对于PETSc来说，可以在命令行中指定求解器 -ksp_type cg
equation_systems.get_system("Possion").solve();
```
具体的组装方法：
```cpp
const MeshBase & mesh = es.get_mesh(); // 得到mesh对象的const引用
const unsigned int dim = mesh.mesh_dimension(); // 得到dimension
LinearImplicitsystem & system = es.get_system<LinearImplicitsystem>("Possion"); // 得到方程系统的引用
const DofMap & dof_map = system.get_dof_map(); // 得到自由度映射
FEType fe_type =  dof_map.variable_type(0); // 得到系统的第一个且这里是唯一一个变量的有限单元类型的引用
UniquePtr<FEBase> fe (FEBase::build(dim,fe_type)); // 基于有限单元类型创建一个有限元对象
QGauss qrule (dim, FIFTH); // 使用5次高斯积分准则
fe -> attatch_quadrature_rule (&qrule); //告诉有限元对象使用该高斯积分准则

UniquePtr<FEBase> fe_face (FEBase::build(dim,fe_type)); // 创建一个特殊的有限元对象，用于边界积分
QGauss qface (dim-1, FIFTH); // 边界上的积分准则的dimension比单元的小1
fe_face -> attatch_quadrature_rule (&qface); //告诉有限元对象使用该高斯积分准则

const std::vector<Real> & JxW = fe->get_JxW(); // 得到Jacobian矩阵与积分权重的乘积
const std::vector<Point> & q_point = fe -> get_xyz(); // 得到单元积分点的坐标
const std::vector<std::vector<Real> > & phi = fe -> get_phi(); // 得到积分点上的形函数
const std::vector<std::vector<RealGradient> > & dphi = fe -> get_dphi(); // 得到积分点上的形函数梯度

DenseMatrix<Number> Ke; // 定义存储单元刚度矩阵的数据结构
DenseVector<Number> Fe; // 定义存储单元右端项向量的数据结构

std::vector<dof_id_type> dof_indices; // 存储自由度的全局标识

// 下面是对所有单元进行循环，计算单元刚度矩阵和右端项对全局的贡献
MeshBase::const_element_iterator el = mesh.active_local_elements_begin(); // 起始单元
const MeshBase::const_element_iterator end_el = mesh.active_local_element_end();  // 结束单元

for (; el != end_el; ++el) // for循环
{ const Elem * elem = *el; // 存储一下当前单元的指针，便于后续的语法表示
dof_map.dof_indices(elem, dof_indices); // 得到当前单元的全局自由度标识，从而知道该单元往哪个全局自由度上做贡献
fe->reinit(elem); // 计算当前单元的特定信息，包括积分点的位置和形函数及其梯度。
Ke.resize(dof_indices.size(),dof_indices.size()); // 在累加之前先置0
Fe.resize(dof_indices.size()); //在累加之前先置0

// 然后在该单元的所有积分点上进行循环
for (unsigned int qp=0; qp < qrule.n_points(); qp++){
// 下面是构建单元刚度矩阵，包括一个嵌套循环
for (std::size_t i=0; i< phi.size(); i++)
  for(std::size_t j=0; j< phi.size(); j++)
    {
      Ke(i,j)+=JxW[qp]*(dphi[i][qp]*dphi[j][qp]);
    }
// 下面是构建单元右端项
for (std::size_t i=0; i < phi.size(); i++)
Fe(i) += JxW[qp]*fxy*phi[i][qp]; //fxy是源项
}

// 还要施加边界条件，该部分详见原代码

system.matrix->add_matrix(Ke, dof_indices); // 将单元刚度矩阵的贡献叠加到全局上
system.rhs->add_vector(Fe, dof_indices); // 将单元右端项的贡献叠加导全局上
}
```

# Intro 4
该例子是基于例子3，展示了通过很小的改动来实现与维度无关的编程。
同时引出PerfLog类来查看性能，从而确定瓶颈所在以实现优化。
头文件：
```cpp
#include "libmesh/perf_log.h" // 是一个性能日志组件
#include "libmesh/dirichlet_boundaries.h" // 施加Dirichlet边界条件
#include "libmesh/getpot.h" // GetPot命令行解析
```
源代码：
```cpp
GetPot command_line (argc, argv); // 创建一个GetPot对象来解析命令行

int dim = 2;
if(command_line.search(1, "-d")) 
  dim = command_line.next(dim);  // 读取命令行中设定的dim

// 根据不同的dimension来生成网格
MeshTools::Generation::build_cube (mesh,
                                   ps,
                                   (dim>1) ? ps : 0,
                                   (dim>2) ? ps : 0,
                                   -1., 1.,
                                   -halfwidth, halfwidth,
                                   -halfheight, halfheight,
                                   (dim==1)    ? EDGE2 :
                                   ((dim == 2) ? QUAD4 : HEX8));

perf_log.push("elem init"); // 开始记录形函数初始化过程
perf_log.pop("elem init"); // 结束记录形函数初始化过程，push和pop成对出现，否则会报错
```
# Intro 5
第五个例子展示了怎样方便调整积分准则。
源码：
```cpp
QuadratureType quad_type = INVALID_Q_RULE; // 定义积分类型
quad_type = static_cast<QuadratureType>(std::atoi(argv[2])); // 从命令行中读取积分类型
```

