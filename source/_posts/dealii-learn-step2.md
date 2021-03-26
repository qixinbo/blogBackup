---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 2 
tags: [deal.II]
categories: simulation
date: 2016-8-4
---

# 引子
在step1中创建了网格，下面就是在网格上定义自由度。此例中使用一阶线性有限元，其自由度的个数与网格的顶点数相关。后面的例子将展示更高次的单元，其上面的自由度与顶点、边、面及cell都有关。
自由度可以理解为形函数中的系数个数，因为它们是未知的，所以称之为未知量或自由度。
定义网格上的自由度很简单，因为deal.II已经内置该功能了，唯一要做的是创建有限元空间。

# 头文件
```cpp
#include <deal.II/dofs/dof_handler.h>
```
该头文件将自由度与顶点、线、cell联系起来。
```cpp
#include <deal.II/fe/fe_q.h>
```
该头文件包含双线性有限元的描述，即只在顶点上有自由度，在边上和cell内部无自由度。
```cpp
#include <deal.II/dofs/dof_tools.h>
```
该头文件包含对自由度的操作工具。
```cpp
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>
```
# 产生网格
这里用了step-1中的方法，只不过这里将网格triangulation作为参数返回，同时将manifold object声明为static，防止其过早销毁：
```cpp
void make_grid (Triangulation<2> &triangulation)
{
    const Point<2> center (1,0);
    const double inner_radius = 0.5,
          outer_radius = 1.0;
    GridGenerator::hyper_shell (triangulation,
            center, inner_radius, outer_radius,
            5 );
    static const SphericalManifold<2> manifold_description(center);
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold (0, manifold_description);
    for (unsigned int step=0; step<3; ++step)
    {
        Triangulation<2>::active_cell_iterator
            cell = triangulation.begin_active(),
                 endc = triangulation.end();
        for (; cell!=endc; ++cell)
            for (unsigned int v=0;
                    v < GeometryInfo<2>::vertices_per_cell;
                    ++v)
            {
                const double distance_from_center
                    = center.distance (cell->vertex(v));
                if (std::fabs(distance_from_center - inner_radius) < 1e-10)
                {
                    cell->set_refine_flag ();
                    break;
                }
            }
        triangulation.execute_coarsening_and_refinement ();
    }
}
```
# 创建DoFHandler
目前为止，只创建了一个网格，包含几何信息(顶点的位置)和拓扑信息(顶点怎样连成线，线连成cell，cell之间怎样连接)。为了执行数值运算，还需要一些逻辑信息，比如将自由度赋给顶点，创建矩阵和矢量，用来描述网格上的场量。
首先描述自由度是如何分布的。这里使用类模板FE_Q来创建拉格朗日单元，它的成员函数需要一个参数来描述单元的多项式次数，此处是1,表明是双线性单元，也就意味着自由度只在顶点上。如果参数是3,那么意味着是双三次单元，自由度分布为：每个顶点上一个，每条边上两个，每个cell内有四个。
示意图为：
对于Q1单元：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjknj5y7uj30ej0ha0t7.jpg)
对于Q2单元：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjknsl84nj30f90jujs4.jpg)
对于Q3单元：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjko5pvifj30890833yi.jpg)
每种单元上形函数的模样可以参见[这里](https://dealii.org/8.4.1/doxygen/deal.II/classFE__Q.html)。
通过创建一个有限元对象，然后用DoFHandler分配自由度：
```cpp
static const FE_Q<2> finite_element(1);
dof_handler.distribute_dofs (finite_element);
```
将自由度分配到每个顶点上去后，不是很容易直接可视化来看到它们，但这也不重要，因为一般情况下自由度的标号是随机的。
与网格每个顶点对应的还有形函数。注意：形函数仅在它们对应的顶点上为1，在其他顶点上则为0。那么也只相邻顶点形成的矩阵不为0，由于顶点的标号是随机的，那么总矩阵应该是稀碎的。
首先创建一个结构来存储非0元素的位置。这个类是SpasityPattern，但它有一些缺点，因为它需要事先估计每排最多有多少个，这会造成不必要的内存浪费。因此这里换用DynamicSparsityPattern这个类，传入的参数是矩阵的大小：
```cpp
DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
        dof_handler.n_dofs());
```
然后根据自由度分布给出非零元素的位置：
```cpp
DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern);
```
然后将DynamicSparsityPattern的信息传回SpasityPattern：
```
SparsityPattern sparsity_pattern;
sparsity_pattern.copy_from (dynamic_sparsity_pattern);
```
然后存储到文件：
```cpp
std::ofstream out ("sparsity_pattern1.svg");
sparsity_pattern.print_svg (out);
```
结果为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjkqcgh74j30480483yd.jpg)
其中每个小红方块都是矩阵中的一个非0元素。

# 对自由度重新编号
上面的结果可以看出，非0元素离对角线很远。对于有些算法，如不完全LU分解和Gauss-Seidel预条件子，这样的分布不好，因此需要改进。
注意：对于矩阵中非0的元素(i，j)，对应的形函数i和j必须相交，而此时其所在的顶点需要相邻，因此，同一个cell内顶点的编号不能差太多才行。这可以通过一种简单的步进方法实现：首先给定一个顶点标识为0,然后对它的邻居连续标号。这里使用的是Cuthill_Mckee提出的方法：
```cpp
DoFRenumbering::Cuthill_McKee (dof_handler);
DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
        dof_handler.n_dofs());
DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern);
SparsityPattern sparsity_pattern;
sparsity_pattern.copy_from (dynamic_sparsity_pattern);
```
结果为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjkqt98kkj3048048gli.jpg)


