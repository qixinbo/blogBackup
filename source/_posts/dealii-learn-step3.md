---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 3
tags: [deal.II, c++]
date: 2016-8-25
categories: computational material science
---

# 引子
这是使用有限元法进行具体计算的第一个算例，求解的是一个简化的Possion方程，其在边界上为0，而右端项不为0，即：
$$
\begin{equation}
\begin{split}
-\Delta u &=f \qquad in \qquad \Omega; \\\
u &=0 \qquad on \qquad \partial \Omega
\end{split}
\end{equation}
$$
求解域是单位正方形$\Omega=[0,1]^2$，其上的网格划分在step1和step2中已经涉及。
这里也仅仅计算特例$f(x)=1$，更一般的情形详见step4。

# 推导
首先需要得到上述方程的弱形式，即在方程两侧左乘一个测试函数$\phi$并在计算域上积分(注意是左乘，而不是右乘)：
$$
-\int\_{\Omega}\phi\Delta u=\int\_{\Omega}\phi f
$$
然后利用高斯散度公式进行分部积分，可得：
$$
\int\_{\Omega}\nabla\phi\cdot\nabla u-\int\_{\partial\Omega}\phi\vec n\cdot\nabla u =
\int\_\Omega\phi f
$$
同时，因为测试函数必须满足相同的边界条件，即在边界上$\phi=0$，所以上式变为：
$$
(\nabla \phi,\nabla u)=(\phi,f)
$$
这里应用了通常的简写形式：$(a,b)=\int\_\Omega ab$。
那么问题就变成了找到在所在空间上对于所有的测试函数上式都成立的函数$u$。当然在计算机上我们不能找到这样的函数的一个通用形式，事实上是寻找一个近似表达式$u\_h(x)=\sum\_jU\_j\phi\_j(x)$，其中$U\_j$是需要确定的未知系数(即自由度)，而$\phi\_j(x)$是有限单元的形函数。同样可以对试探函数作这样的处理，其实之前的形式中已经把试探函数取为了形函数。另外，由于可以任意选择试验函数，因此可以将除了j点之外的所有的$\phi\_j$设置为0。
为了确定形函数，需要作如下工作：
- 形函数所在的网格。在step1和step2中已经有关于产生网格的方法。
- 描述形函数形式的参考单元。比如二维下的双线性单元。
- 枚举网格上所有自由度的DofHandler对象。
- 描述由参考单元的形函数获得真实单元的形函数的映射，默认deal.II使用线性映射。

通过上述几步就得到了形函数的集合，那么就可以得到离散形式的弱形式，即：
$$
(\nabla \phi\_i,\nabla u\_h)=(\phi\_i,f) \qquad i=0...N-1
$$
对于每个试验函数都可以得到一个方程，所有方程就可以生成一个线性代数系统。将$u\_h$的表达式代入，可得：
$$
(\nabla \phi\_i,\nabla \sum\_jU\_j\phi\_j)=(\phi\_i,f) \qquad i=0...N-1
$$
整理得：
$$
\sum\_j(\nabla\phi\_i,\nabla\phi\_j)U\_j=(\phi\_i,f) \qquad i=0...N-1
$$
展开形式为：
$$
\begin{bmatrix}
(\nabla\phi\_0,\nabla\phi\_0) & (\nabla\phi\_0,\nabla\phi\_1) & \cdots & (\nabla\phi\_0,\nabla\phi\_j) \\\
(\nabla\phi\_1,\nabla\phi\_0) & (\nabla\phi\_1,\nabla\phi\_1) & \cdots & (\nabla\phi\_1,\nabla\phi\_j) \\\
. \\\
(\nabla\phi\_{N-1},\nabla\phi\_0) & (\nabla\phi\_{N-1},\nabla\phi\_1) & \cdots & (\nabla\phi\_{N-1},\nabla\phi\_j) 
\end{bmatrix}
\begin{bmatrix}
u\_0 \\\
u\_1 \\\
. \\\
u\_{j}
\end{bmatrix}
=
\begin{bmatrix}
(\phi\_0,f) \\\
(\phi\_1,f) \\\
. \\\
(\phi\_j,f)
\end{bmatrix}
$$

采用矩阵形式：
$$
AU=F
$$
所以，矩阵的元素$A\_{ij}=(\nabla\phi\_i,\nabla\phi\_j)$。
右端项的元素$F\_i=(\phi\_i,f)$。$\sum\_j$代表矩阵运算中每个方程的相加操作。
注意：以上都是对节点的循环，对单元的循环则需要对上述矩阵进行分块，比如如果0号、1号、2号节点在单元0上，那么左上角的九个量就形成一个子矩阵。上述矩阵整体可称为整体刚度矩阵，但单元刚度矩阵则需要是对单元进行循环得到的分块矩阵，即单元刚度矩阵必须建立在单元上。
上面两个元素都含有积分，因此还需要求解积分，数学上的积分是连续的，但有限元必须离散到每个单元上来求解。通常使用如下方法：在每个单元上取一系列积分点，积分由函数在这些积分点上的值的和来代替，即：
$$
\begin{equation}
\begin{split}
A^K\_{ij} &= \int\_K \nabla\phi\_i \cdot \nabla \phi\_j \approx \sum\_q \nabla\phi\_i(x^K\_q) \cdot \nabla \phi\_j(x^K\_q) w\_q^K, \\\
F^K\_i &= \int\_K \phi\_i f \approx \sum\_q \phi\_i(x^K\_q) f(x^K\_q) w^K\_q
\end{split}
\end{equation}
$$
其中，$x^k\_q$是单元K上的第q个积分点，$w^k\_q$是该积分点的权重。
注意：积分必须在单元上进行，所以必须是对单元的循环。
再次Attention！！：这样，循环的过程就是这样：先是对单元循环，找到当前的单元，然后对单元上的积分点进行循环，目的就是为了能计算积分，再对自由度循环，找到相互作用的i、j，计算了刚度矩阵后放入合适的位置。

附录：从参考单元向真实单元的映射过程
映射就是一个变换$x=F\_K(\widehat{x})$，它将参考单元$[0,1]^{dim}$中的点$\widehat{x}$映射到真实网格单元$K\subset R^{spacedim}$中的点$x$。
一些复杂映射通常需要这种映射的Jacobian矩阵，即$J(\widehat{x})=\widehat{\nabla}F\_K(\widehat{x})$。比如，如果$dim=spacedim=2$，那么：
$$
J(\widehat{x})
\begin{bmatrix}
\frac{\partial x}{\partial\widehat{x}} & \frac{\partial x}{\partial\widehat{y}} \\\
\frac{\partial y}{\partial\widehat{x}} & \frac{\partial y}{\partial\widehat{y}}
\end{bmatrix}
$$
1. 标量函数的映射
形函数是定义在参考单元中，它通过以下映射到真实单元中：
$$
\phi(x)=\phi(F\_K(\widehat{x}))=\widehat{\phi}(\widehat{x})
$$

2. 积分的映射
积分的映射关系如下：
$$
\int\_K u(x)dx = \int\_{\widehat K} \widehat u(\widehat{x}) \left|\text{det}J(\widehat{x})\right| d\widehat x.
$$
将上述积分由在积分点上的离散形式表示：
$$
\int\_K u(x)dx \approx \sum\_{q} \widehat u(\widehat{x}\_q) \underbrace{\left|\text{det}J(\widehat{x}\_q)\right| w\_q}\_{=: \text{JxW}_q}.
$$
其中，$JxW$是一个象形词，代表Jacobian行列式乘以积分权重，数学上来说，积分点上的$Jxw\_q$也就承担了原来积分中的$dx$的作用，它们在编程中可以由FEValues::JxW()函数获得。

3. 矢量、微分和矢量的梯度的映射
矢量场和微分(即标量场的梯度)统一记为$v$，矢量场的梯度记为$T$，它们的映射形式为：
$$
v(x)=A(\widehat{x})\widehat{v}(\widehat{x}), \qquad 
T(x)=A(\widehat{x})\widehat{T}(\widehat{x})B(\widehat{x})
$$
微分$A$和$B$需要通过要变换的对象来具体确定。

4. 映射的导数
某些情况下还需要计算映射的导数，一阶导数就是上面的Jacobian矩阵。更高阶的导数类似，比如Jacobian矩阵的导数，还有Jacobian二阶导数等，具体形式参见帮助文档。



# deal.II的程序结构
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjkliv2w9j30gm0atjru.jpg)
deal.II采用面向对象编程，其中包含了很多的Modules，各自实现不同的功能，并有机地结合起来。如上图所示。具体为：
1. Triangulation
Triangulations是单元及其更低维度的边界的集合。triangulation存储了网格的几何和拓扑性质：单元怎样接触，它们的顶点在哪里。triangulation不知道将要在它上面使用的有限元的任何信息，甚至不知道它自己的单元的形状，它只知道在二维情形下有4条线段和4个顶点，三维下有6个面、12条线段和8个顶点。不过其他所有信息都定义在映射类mapping中，由该类将参考单元的坐标映射到真实单元的坐标上，通常采用线性映射。
当需要访问triangulation的性质和数据时，通过使用iterators迭代器对单元进行循环。

2. Finite Element
Finite element类用来描述定义在参考单元上的有限元空间的性质，比如在单元的顶点、线段或内部各有多少自由度，此外还给出节点上的形函数的值及其梯度。

3. Quadrature
跟Finite element相同，Quadrature也定义在单元上，用来描述参考单元上积分点的位置及其权重。

4. DoFHandler
DoFHandler对象是triangulations和finite elements的汇合点：finite element类描述了triangulation单元的点、线或内部需要多少自由度，而DoFHandler分配了这种空间，从而使得点、线或内部都有正确的数目，同时也给这些自由度统一编号。
也可以这样理解：triangulation和finite element描述了有限元空间的具体性质(有限元空间就是用来得到离散解的空间)，DoFHandler则枚举了该空间的基本框架，使得我们可以用一系列有序的系数$U\_j$来表示离散解$u\_h(x)=\sum\_jU\_j\phi\_i(x)$。
正如triangulation对象，DoFHandler也可以通过iterators迭代器对单元进行循环，从而得到具体的信息，比如单元的几何和拓扑信息(这些也可以由triangulation迭代获得，其实两个类是派生关系)，以及当前单元上的自由度数目和数值。需要注意的是，跟triangulation一样，DoFHandler也不知道从参考单元到它上面的真实单元的映射，也不知道对应于它所管理的自由度的形函数。

5. Mapping
Mapping类就是建立从参考单元到triangulation的单元的映射，包括形函数、积分点和积分权重，同时提供了这种派生的梯度和Jacobian行列式。

6. FEValues
这一步就是真正地取出某个单元，计算它上面在积分点上的形函数值及其梯度。注意，有限元空间不是连续的，积分都是在特定积分点上计算。

7. Linear Systems
一旦知道了怎样使用FEValues在单个单元上计算形函数值及其梯度，同时知道怎样使用DoFHandler获得自由度的全局标识，那么就可以使用矩阵类和向量类来存储和管理矩阵和向量的元素，从而形成线性系统。

8. Linear Solvers
构建好线性系统后，就可以使用求解器来求解该系统。

9. Output
求解完毕后，就可以输出结果到可视化软件中进行后处理。



# 源码解析
## 头文件
```cpp
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
```
这两个头文件分别处理triangulation和自由度。
```cpp
#include <deal.II/grid/grid_generator.h>
```
该文件用于生成网格。
```cpp
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
```
这三个文件用来对单元进行循环并获得单元上的信息。
```cpp
#include <deal.II/fe/fe_q.h>
```
该文件包含拉格朗日插值的描述。
```cpp
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
```
以上头文件还需仔细理解。

## step3类
跟之前两个例子不同，这次将信息都封装在了一个类里面。
```cpp
class Step3
{
    public:
        Step3 ();
        void run ();
```
这是类的public部分，包含一个构造函数和一个run函数，用来说明执行顺序。
```cpp
    private:
void make_grid ();
void setup_system ();
void assemble_system ();
void solve ();
void output_results () const;
```
这是类的private部分的成员函数，函数名说明了其要实现的功能。
```cpp
Triangulation<2> triangulation;
FE_Q<2> fe;
DoFHandler<2> dof_handler;
SparsityPattern sparsity_pattern;
SparseMatrix<double> system_matrix;
Vector<double> solution;
Vector<double> system_rhs;
};
```
还有成员变量，用来存储各种信息。
以下是step3类的各个成员函数的详解。
```cpp
Step3::Step3 ()
    :
    fe (1),
    dof_handler (triangulation)
{}
```
这是step3类的构造函数，它没有执行具体操作，只是调用了成员初始器对fe和dof_handler进行了初始化。fe是一个finite element对象，它接收1，1是多项式的次数，表明使用的是双线性插值的形函数。
之前的triangulation也被传递给了dof_handler。注意此时triangulation还没有具体建立网格，但dof_handler不介意，它只有在具体分配自由度时才关心网格。
下一步必须做的就是对计算域进行剖分，然后对每个顶点分配自由度。
```cpp
void Step3::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (5);
    std::cout << "Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl;
}
```
这个函数做的是第一步，即对计算域进行剖分，建立网格。计算域是$[-1,1]x[-1,1]$。
因为初次建立时只有一个单元，所以细化5次，形成1024个单元，这里用n_active_cells()验证一下个数。注意这里用的不是n_cells()函数，因为其还包涵父单元的概念。
下一步就是分配自由度：
```cpp
void Step3::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    std::cout << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
```
这里使用distribute_dofs()函数，接收的是fe，因为fe是线性插值，所以自由度是每个顶点上有一个。用n_dofs()输出一下，显示是1089。这是因为我们有32x32个网格，那么对应是33x33个节点。
然后创建稀疏矩阵
```cpp
DynamicSparsityPattern dsp(dof_handler.n_dofs());
DoFTools::make_sparsity_pattern (dof_handler, dsp);
sparsity_pattern.copy_from(dsp);
system_matrix.reinit (sparsity_pattern);
```
注意，SparsityPattern和SparseMatrix不同，前者只存储元素的位置，后者则存储具体的数值。
然后建立右端项向量和解向量：
```cpp
solution.reinit (dof_handler.n_dofs());
system_rhs.reinit (dof_handler.n_dofs());
}
```
下一步就是计算线性系统中的矩阵中的元素以及右端项的元素，这是每个有限元程序的核心部分！
组装矩阵和向量通常的方法是对所有单元进行循环，然后在每个单元上进行积分运算，得到该单元对整体的贡献。需要注意的是此时需要知道在每个真实单元上形函数在积分点位置上的值，但是！！形函数和积分点都是仅仅定义在参考单元上的，因此这些东西基本没用。那么关键问题来了，就是怎样将数据从参考单元上映射到真实单元上。这个活是由Mapping类来完成的，更加智能的是通常不需要人为指定它怎么做，它自动按标准双线性映射来操作。
现在我们需要处理三类东西：有限元finite element、积分quadrature和映射mapping。这些概念太多了，这里有一个类FEValues来将三者有机地整合起来。给它传进去这三个东西，它就能告诉你在真实单元上的形函数的值和梯度。
那么现在就开干吧：
```cpp
void Step3::assemble_system ()
{
    QGauss<2> quadrature_formula(2);
```
先确定在单元上的一套积分规则，这里使用的是Gauss数值积分方法，每个方向上选两个积分点。这套积分规则满足现在的问题。然后就可以生成FEValues的对象了：
```cpp
FEValues<2> fe_values (fe, quadrature_formula,
        update_values | update_gradients | update_JxW_values);
```
第一个参数告诉它参考单元是谁，第二个参数告诉它积分点及其权重(实际是一个Qudrature对象)，还有默认使用了双线性映射。最后告诉它需要在每个单元上计算什么，包括在积分点上的形函数值(为了计算右端项$\phi_i(x^K_q) f(x^K_q)$)、在积分点上的形函数梯度(为了计算矩阵元素$\nabla\phi_i(x^K_q) \cdot \nabla \phi_j(x^K_q)$)、Jacobian行列式。注意：这里都是积分点上的值。因为需要对单元进行循环，所有这些值需要更新，所以加上前缀update。这样显著地跟程序说明需要计算什么，就能加速运算，因为有的软件所有东西不管有用没用都一块计算，比如二阶导数、法向量等。注意最后用了按位或这一运算符，这在c语言中很常见，这里的意思就是我要计算谁“和”谁。
```cpp
const unsigned int dofs_per_cell = fe.dofs_per_cell;
const unsigned int n_q_points = quadrature_formula.size();
```
然后定义两个“快捷名”来称呼常用的两个变量：每个单元上的自由度个数和积分点个数。
现在终于开始一个单元一个单元地组装整体刚度矩阵和向量了。一种方法是直接将结果写入整体矩阵中，但这样对于大型矩阵的运算通常是很慢的。所以，这里是先在当前单元上计算该单元的单元刚度矩阵，然后将它的贡献叠加到整体上。计算右端项向量时也是这样的。
首先先创建以上两种单元矩阵和单元向量：
```cpp
FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
Vector<double> cell_rhs (dofs_per_cell);
```
当计算每个单元的贡献时，是对该单元上的自由度的局部标识进行循环，即从0到dofs_per_cell-1。
然而，当把结果传递给整体时，需要知道这些自由度的全局标识，这时需要一个临时的量来存储：
```cpp
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
```
来，现在真的要对单元进行循环：
```cpp
DoFHandler<2>::active_cell_iterator
cell = dof_handler.begin_active(),
     endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
```
现在，我们整个人站在单元上，我们想要知道在积分点上的形函数的值及其梯度以及参考单元和真实单元之间变换的Jacobian行列式。因为这些值与每个单元的几何信息有关，所以必须在每个单元上都需要让FEValues重算一下这些东西：
```cpp
fe_values.reinit (cell);
```
初始化单元的贡献为0，以便后面的赋值：
```cpp
cell_matrix = 0;
cell_rhs = 0;
```
然后开始积分，对积分点进行循环：
```cpp
for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
{
```
首先是对单元矩阵的计算：
```cpp
for (unsigned int i=0; i<dofs_per_cell; ++i)
    for (unsigned int j=0; j<dofs_per_cell; ++j)
        cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                fe_values.shape_grad (j, q_index) *
                fe_values.JxW (q_index));
```
对于拉普拉斯方程，每个单元上的矩阵是形函数i和j的梯度的积分，程序中用quadrature代替integral，所以变成两个梯度的乘积乘以Jacobian行列式(这是为了从参考单元到真实单元)和权重。这里的计算过程实际是这样的：
$fe_values.shape_grad(i,q_index)$是一个dim维(这里是2维)的向量，实际是一个Tensor<1,dim>的对象，后面的dim是维度，前面的1代表dim维的一阶向量，如果是0则代表标量，如果是2则代表dim乘以dim维的二阶张量。星号运算符则将前面的dim维向量跟后面的dim维向量正确地相乘，即两者做点积，结果是个标量，再跟后面的权重这一标量相乘。

然后对单元右端项向量的计算：
```cpp
for (unsigned int i=0; i<dofs_per_cell; ++i)
    cell_rhs(i) += (fe_values.shape_value (i, q_index) *
            1 *
            fe_values.JxW (q_index));
}
```
这里的积分是形函数的值乘以f乘以JxW。
Attention!!:实际上上面两步计算可以将两个循环合并起来，加速计算，见后面的step4。
现在有了单元的矩阵，下一步就是把它组装到整体上。我们先得问问这个单元，它的某个局部标号的自由度的全局标识是多少：
```cpp
cell->get_dof_indices (local_dof_indices);
```
后面就可以通过local_dof_indices[i]来获取全局标识。
然后再作循环叠加：
```cpp
for (unsigned int i=0; i<dofs_per_cell; ++i)
for (unsigned int j=0; j<dofs_per_cell; ++j)
system_matrix.add (local_dof_indices[i],
        local_dof_indices[j],
        cell_matrix(i,j));
for (unsigned int i=0; i<dofs_per_cell; ++i)
system_rhs(local_dof_indices[i]) += cell_rhs(i);
}
```
同理Attention！，这里也可以这样将两个循环合并。这样线性系统基本全做完了。But！还有一个重要的东西漏了：边界条件。事实上，如果该拉普拉斯方程不加上一个Dirichlet边界条件，它的解就不是惟一的，因为你可以在解上加一个任意的量。显然得解决这个问题：
```cpp
std::map<types::global_dof_index,double> boundary_values;
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        ZeroFunction<2>(),
        boundary_values);
```
这里使用的interpolate_boundary_values函数就是将函数值插入到特定位置的边界上，它需要的参数有：DoFHandler对象来获得边界上的自由度的全局标识;边界的一部分;边界上的值本身;输出对象。
”边界的一部分“意思是有时你只想在边界的一部分上赋予边界值，比如流体力学的入流和出流边界、变形体的固定部分等。这时可以对边界进行一部分一部分地标号，不同的标号施加不同的条件。默认条件下所有的边界标号都是0，这里也使用的是0，表明是对整个边界作用。描述边界值的函数这里用的是ZeroFunction，返回的是0，刚好适用现在的零边界条件。最后的输出对象boundary_values是一个列表，里面是成对的边界自由度全局标识及其对应的边界值。
知道上述信息后，再实际施加边界条件：
```cpp
MatrixTools::apply_boundary_values (boundary_values,
        system_matrix,
        solution,
        system_rhs);
}
```
完全建立好线性系统了，终于该求解了。这个问题的变量个数是1089，实际是很小的量，通常的问题一般是10万个变量这个量级。传统的求解线性代数的算法，如Gauss消去或LU分解，对于大型系统不适用，这里用的是共轭梯度算法，即CG算法。
```cpp
void Step3::solve ()
{
    SolverControl solver_control (1000, 1e-12);
```
首先告诉CG算法何时停止计算，这里是创建一个SolverControl对象来控制：最多迭代1000步，精度为1e-12。通常这个精度值是迭代停止的判据。
```cpp
SolverCG<> solver (solver_control);
```
然后创建一个CG的求解器。
```cpp
solver.solve (system_matrix, solution, system_rhs,
        PreconditionIdentity());
}
```
上面就是开始求解了。第四个参数是一个预条件子。求解完毕后，solution中就存储了解向量的离散值。
然后就是输出结果了：
```cpp
void Step3::output_results () const
{
    DataOut<2> data_out;
```
首先让它知道从哪里提取数据：
```cpp
data_out.attach_dof_handler (dof_handler);
data_out.add_data_vector (solution, "solution");
```
知道了目标数据后，离后面的可输出数据还隔了一个“中间数据”，因为deal.II是前后端分离的，这个中间数据像是个缓冲层，得到它的语句为：
```cpp
data_out.build_patches ();
```
然后再输出成最终的可被可视化软件读取的数据：
```cpp
std::ofstream output ("solution.gpl");
data_out.write_gnuplot (output);
}
```
上面一系列的成员函数通过run函数来按次序执行：
```cpp
void Step3::run ()
{
    make_grid ();
    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
}
```
该程序的main函数为：
```cpp
int main ()
{
    deallog.depth_console (2);
    Step3 laplace_problem;
    laplace_problem.run ();
    return 0;
}
```
第一个语句是打印log信息到屏幕上，后面就是创建step3对象，然后执行。

# 计算结果
输出到屏幕上的信息有：
```cpp
Number of active cells: 1024
Number of degrees of freedom: 1089
DEAL:cg::Starting value 0.121094
DEAL:cg::Convergence step 48 value 5.33692e-13
```
即，说明了单元个数1024、自由度个数1089，CG算法的起始残差是0.12，经过47步迭代后满足精度要求。
图形结果为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjklvv10zj30hs0dc77g.jpg)
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjkm5ntoaj30hs0dcjub.jpg)

