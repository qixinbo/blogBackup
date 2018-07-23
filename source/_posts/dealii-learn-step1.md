---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 1
tags: [deal.II]
categories: computational material science
date: 2016-8-2
---

# 简介
[deal.II](dealii.org)是一款开源的求解偏微分方程的有限元软件，它有如下几个[特点](https://en.wikipedia.org/wiki/Deal.II)：
- 使用C++编写
- 有多种单元类型
- 可以大规模并行
- 可以自适应网格
- 文档和范例齐全
- 与其他库有良好的接口

# 安装
deal.II最新版本为8.4.1,可从官网上下载源码，解压后进入源文件目录安装：
```cpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir ../deal.II
make install
make test
```
如果期间需要其他依赖如cmake、doxygen等，自行安装好即可。

# 编译运行
deal.II的文档和example特别完备，刚开始接触可以它多达54个例子进行。
具体入口在[这里](https://dealii.org/8.4.1/doxygen/deal.II/Tutorial.html)。
编译运行命令为：
```cpp
cmake .
make
make run
```
第一条命令用来创建Makefile文件，指明程序所依赖的文件、怎样编译和运行。此命令应该能找到之前安装deal.II后的库文件，如果不能找到，需要人为指定路径：
```cpp
cmake -DDEAL_II_DIR=/path/to/installed/deal.II .
```
第二条命令将源文件编译成可执行文件，第三条是运行该可执行文件。其实可以省略第二条命令。后面的example遵循同样的命令。

# 第一个教学实例——step-1
```cpp
#include <deal.II/grid/tria.h>
```
此头文件声明了Triangulation类模板，其用途是生成各种单元。如Triangulation<1,1>表示一维线段，Triangulation<1,2>或<2,3>表示二维中的曲线和三维中的面，通常用于边界单元中。
```cpp
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
```
这两个头文件一个是存取器，一个是迭代器，用于对单元的循环遍历。
```cpp
#include <deal.II/grid/grid_generator.h>
```
该头文件声明了GridGenerator命名空间，用于生成标准网格
```cpp
#include <deal.II/grid/manifold_lib.h>
```
本例中用到了非直线的边以及双线性四边形，所以还要导入一些类来预先描述流形，比如描述球、圆柱等。
```cpp
#include <deal.II/grid/grid_out.h>
```
该头文件声明了GridOut类，用于生成多种格式的数据，如dx、gnuplot、msh、eps、svg、vtk等。
```cpp
#include <iostream>
#include <fstream>
#include <cmath>
```
以上是c++的一些标准库，用于字符和文件的输入输出，以及作开方、取绝对值等数学库。

```cpp
using namespace dealii;
```
为了防止命名冲突，该包的函数和类都包含在dealii这个命名空间中。

```cpp
void first_grid ()
{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (4);
```
首先创建一个Triangulation类的对象，这里只有一个参数是2，表明该对象是作用在二维上。然后用超立方体形状的一个单元来填充给定的triangulation。这里因为是二维，就是一个正方形。正方形的大小是默认的从0到1的边长，可以在参数中人为设定。还有一个参数colorize是设定边界标识，这里默认都是0。
此时整个网格只有一个单元，然后将网格加密4次。
```cpp
std::ofstream out ("grid-1.eps");
GridOut grid_out;
grid_out.write_eps (triangulation, out);
std::cout << "Grid written to grid-1.eps" << std::endl;
}
```
然后输出，这里使用的是eps格式。

第一个函数first_grid生成一个由一个单元网格加密四次后生成的4的4次方=256个正方形单元的网格。
![](http://7xrm8i.com1.z0.glb.clouddn.com/dealii-step-1-1.jpeg)

然后是第二个函数：
```cpp
GridGenerator::hyper_shell (triangulation,center,inner_radius,outer_radius,10);
```
生成一个三维的壳体或二维的环。注意：这里的边界是弯曲的，而默认情形下边界都是直线。这时需要指定“manifold indicator”来指明边界，告诉在哪个地方细化。
如果不指明manifold indicator，那么就只会得到在周向为10个cell，然后加密3次：
```cpp
Triangulation<2> triangulation;
const Point<2> center (1,0);
const double inner_radius = 0.5,
outer_radius = 1.0;
GridGenerator::hyper_shell (triangulation,
center, inner_radius, outer_radius,
10);
triangulation.refine_global (3);
```
结果如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/hypershell-nothing.png)
对边界进行特定标识后，效果会好一些：
```cpp
Triangulation<2> triangulation;
const Point<2> center (1,0);
const double inner_radius = 0.5,
outer_radius = 1.0;
GridGenerator::hyper_shell (triangulation,
center, inner_radius, outer_radius,
10);
const HyperShellBoundary<2> boundary_description(center);
triangulation.set_boundary (0, boundary_description);
triangulation.refine_global (3);
```
结果如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/hypershell-boundary-only.png)
可以看出此时能较好地表现出内外边界的圆形特征，但仍然能从切线上的弯折分辨出最初的10个cell。这可以通过不光对边界上的线，而是对所有的线都指定indicator来优化：
```cpp
Triangulation<2> triangulation;
const Point<2> center (1,0);
const double inner_radius = 0.5,
outer_radius = 1.0;
GridGenerator::hyper_shell (triangulation,
center, inner_radius, outer_radius,
10);
const SphericalManifold<2> boundary_description(center);
triangulation.set_manifold (0, boundary_description);
Triangulation<2>::active_cell_iterator
cell = triangulation.begin_active(),
endc = triangulation.end();
for (; cell!=endc; ++cell)
cell->set_all_manifold_ids (0);
triangulation.refine_global (3);
```
效果如图：
![](http://7xrm8i.com1.z0.glb.clouddn.com/hypershell-all.png)

之前设定的hyper_shell初始周向网格为10个cell，如果设置为3,且只对边界上的cell设定indicator：
```cpp
Triangulation<2> triangulation;
const Point<2> center (1,0);
const double inner_radius = 0.5,
outer_radius = 1.0;
GridGenerator::hyper_shell (triangulation,
center, inner_radius, outer_radius,
3); // four circumferential cells
const HyperShellBoundary<2> boundary_description(center);
triangulation.set_boundary (0, boundary_description);
triangulation.refine_global (3);
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/hypershell-boundary-only-3.png)
可以看出细化效果很差，但即使初始为3,也可以通过对全部的cell设定indicator来优化：
```cpp
Triangulation<2> triangulation;
const Point<2> center (1,0);
const double inner_radius = 0.5,
outer_radius = 1.0;
GridGenerator::hyper_shell (triangulation,
center, inner_radius, outer_radius,
3); // three circumferential cells
const SphericalManifold<2> boundary_description(center);
triangulation.set_manifold (0, boundary_description);
Triangulation<2>::active_cell_iterator
cell = triangulation.begin_active(),
endc = triangulation.end();
for (; cell!=endc; ++cell)
cell->set_all_manifold_ids (0);
triangulation.refine_global (3);
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/hypershell-all-3.png)

以上分析过程见[这里](https://dealii.org/8.4.1/doxygen/deal.II/group__manifold.html)。

在step-1的second_grid函数中，将上述细化更进一步，不再是全局细化，而是局部细化，因此首先是要得到能指向每个cell的指针，可以想象一个triangulation是所有cell的集合，而cell在其中并不是一个序列，因此这里不能直接用指针，而是用迭代器iterator，从第一个cell开始，遍历所有cell。不过这里没有使用遍历所有cell的迭代器，而是使用只遍历active cell的迭代器active_cell_iterator，active cell是没有children的cell，其后来将被细化：
```cpp
Triangulation<2>::active_cell_iterator
cell = triangulation.begin_active(),
endc = triangulation.end();
for (; cell!=endc; ++cell)
```
然后在这个for循环中再遍历每个cell的所有顶点：
```cpp
for (unsigned int v=0;
v < GeometryInfo<2>::vertices_per_cell;
++v)
```
然后从这些顶点中通过判断该顶点与圆心的距离找到属于内边界的顶点，从而标记该顶点所在的cell，用于后续的细化：
```cpp
const double distance_from_center
= center.distance (cell->vertex(v));
if (std::fabs(distance_from_center - inner_radius) < 1e-10)
{
cell->set_refine_flag ();
break;
}
```
然后开始执行细化操作：
```cpp
triangulation.execute_coarsening_and_refinement ();
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-1.grid-2r2.png)

注意：当函数结束，开始销毁所创建的对象时，按相反顺序进行，因为定义的manifold object是在triangulation之后，所以manifold object先销毁，此时将会报错，因此必须先释放它：
```cpp
triangulation.set_manifold (0);
```
另一种简单的方法是在triangulation之前定义manifold object。

second_grid给了我们提示，可以设定不同的细化条件，比如设定当cell的中心的y坐标大于0时才加密：
```cpp
for (; cell!=endc; ++cell)
if (cell->center()[1] > 0)
cell->set_refine_flag ();
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/dealii-step-1-2.jpeg)

# 最后
deal.II的作者建议尽早学会使用一个debugger！

# 补充
## 产生的信息
此例最简化的版本是：
```cpp
void first_grid()
{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(0);

  std::ofstream out("grid-1.vtk");
  GridOut grid_out;
  grid_out.write_vtk(triangulation,out);
}
```
产生的vtk文件是：
```cpp
# vtk DataFile Version 3.0
#This file was generated by the deal.II library on 2016/11/7 at 15:38:30
ASCII
DATASET UNSTRUCTURED_GRID

POINTS 4 double
0 0 0
1 0 0
0 1 0
1 1 0

CELLS 1 5
4	0	1	3	2

CELL_TYPES 1
9

POINT_DATA 4
SCALARS level double 1
LOOKUP_TABLE default
0 0 0 0 
SCALARS manifold double 1
LOOKUP_TABLE default
-1 -1 -1 -1 
SCALARS material double 1
LOOKUP_TABLE default
0 0 0 0 
SCALARS subdomain double 1
LOOKUP_TABLE default
0 0 0 0 
SCALARS level_subdomain double 1
LOOKUP_TABLE default
0 0 0 0 
```
说明生成的数据的特点是：
(1)拓扑结构：只有四个点，且其构成四边形
(2)属性：该网格上有五个属性值，包括level、manifold、material、subdomain和level_subdomain。
具体的样子可以通过ParaView查看。

注意，如果将全局优化次数改为1，那么节点个数就变成16个，点的编号也相应变化，虽然点变成了16个，但实际只有9个，因为有些是同一个坐标点有多个编号，相应的上面的每个属性值的个数也变成了16个，单元个数还是实际个数，不存在重合现象，即此时是4个。改为全局细化两次后，节点有64个，单元有16个。

## cell和vertex分布
deal.II对cell的个数计数有两种：一种是active_cells，它们就是网格中呈现的单元，表示不能再继续细化的单元;另一种是cells，它们不仅包含active_cells，还包含之前细化过的单元。
对单元作循环，统计个数，可以看出两者的区别：
```cpp
Triangulation<2>::cell_iterator
  cell = triangulation.begin(),
  endc = triangulation.end();

Triangulation<2>::active_cell_iterator
  active_cell = triangulation.begin_active(),
  active_endc = triangulation.end();
```
以全局细化两次为例，active_cells共有16个，而cells共有21个。

```cpp
for(;cell!=endc;cell++)
{
  for(unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; v++)
    std::cout << "the point at " << cell->vertex(v)[0] << " "
              << "and "          << cell->vertex(v)[1] << std::endl;
}
```
上面就是将顶点的坐标输出，其结果跟vtk文件是相同的。