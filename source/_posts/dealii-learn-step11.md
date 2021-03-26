---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 11
tags: [deal.II]
categories: simulation
date: 2016-9-16
---

# 引子
这个例子的亮点：介绍了最简单的Laplace方程的组装矩阵和右端项的一键生成函数。
这个例子来求解仅考虑Neumann边界条件的Laplace问题：
$$
\begin{equation}
\begin{split}
-\Delta u &=f \qquad \mathrm{in}\ \Omega, \\\ 
\partial\_n u &=g \qquad \mathrm{on}\ \partial\Omega.
\end{split}
\end{equation}
$$
如果有解的话，解必须满足协调条件：
$$
\int\_\Omega f dx + \int\_{\partial\Omega} g ds = 0.
$$
这里考虑计算域是以原点为圆心，半径为1的圆，$f=-2,g=1$是满足协调条件的。
虽然协调条件允许有解，但解仍然是不定的：在方程中仅仅解的导数固定，解可以是任意常数。所以需要施加另外一个条件来固定这个常数。可以有以下可能的方法：
- 固定离散后的某个节点值为0或其他固定值。这意味着一个额外的条件$u\_h(x\_0)=0$。尽管这是通常的做法，但不是一个好方法，因为我们知道Laplace方程的解是在H1空间，它不允许定义某个点的值，因为这样不是连续函数的子集。因此，尽管固定某点的值在离散时是允许的，但这样不是连续函数，结果经常是数值上的一个尖峰。
- 固定计算域上的平均值是0或其他固定值。这满足了连续性。
- 固定计算域边界上的平均值是0或者其他固定值。这也满足了连续性。

这里选择最后一种，因为还想展示另一项技术。
具体到在程序中怎样解方程，除了额外的平均值限制，其他技术都已经涉及过了，这里需要把Dirichlet边界条件删掉，至于怎样映射，在绝大多数情况下，完全可以把它当成黑盒，程序已自动处理。唯一一点就是平均值限制。幸运的是，库中有一个类知道怎样处理这种限制。如果假定边界节点沿边界均匀分布，那么平均值限制：
$$
\int\_{\partial \Omega} u(x) ds = 0
$$
就可以写为：
$$
\sum\_{i\in\partial\Omega\_h} u\_i = 0,
$$
这个求和应该遍历所有在边界上的自由度。设$i\_0$是边界上指标最小的自由度，那么：
$$
u\_{i\_0} = \sum\_{i\in\partial\Omega\_h\backslash i\_0} -u\_i.
$$
这是ConstraintMatrix类所想要的形式。注意我们之前已经多次用到过这个类，比如悬点限制：中间节点上的值等于相邻节点的平均值。通常来说，ConstraintMatrix来处理均匀限制：
$$
CU=0
$$
其中C是矩阵，U是节点值的向量。

# 程序解析
```cpp
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
```
之前用过的头文件。
```cpp
#include <deal.II/lac/dynamic_sparsity_pattern.h>
```
这个是新的。
```cpp
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
```
C++的标准头文件。
```cpp
namespace Step11
{
    using namespace dealii;
    template <int dim>
        class LaplaceProblem
        {
            public:
                LaplaceProblem (const unsigned int mapping_degree);
                void run ();
            private:
                void setup_system ();
                void assemble_and_solve ();
                void solve ();
                Triangulation<dim> triangulation;
                FE_Q<dim> fe;
                DoFHandler<dim> dof_handler;
                MappingQ<dim> mapping;
                SparsityPattern sparsity_pattern;
                SparseMatrix<double> system_matrix;
                ConstraintMatrix mean_value_constraints;
                Vector<double> solution;
                Vector<double> system_rhs;
                TableHandler output_table;
        };
```
这个声明跟Step5差不多，仅有少许不同。它的构造函数中接收一个映射次数的参数。
```cpp
template <int dim>
LaplaceProblem<dim>::LaplaceProblem (const unsigned int mapping_degree) :
    fe (1),
    dof_handler (triangulation),
    mapping (mapping_degree)
    {
        std::cout << "Using mapping with degree " << mapping_degree << ":"
            << std::endl
            << "============================"
            << std::endl;
    }
```
构造函数，使用1阶有限单元，即fe的参数是1，映射次数是后续手动传入的。
```cpp
template <int dim>
void LaplaceProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
```
然后是建立系统：首先建立一个DoFHandler对象，初始化解和右端项。
```cpp
std::vector<bool> boundary_dofs (dof_handler.n_dofs(), false);
DoFTools::extract_boundary_dofs (dof_handler,
        ComponentMask(),
        boundary_dofs);
```
然后就是建立边界上自由度上的平均值为0的限制对象。这里首先得提取出边界上的自由度。DoFTools有一个函数能返回一个布尔值的列表，其中true代表节点在边界上。第二个参数是一个蒙版，选择矢量函数的某个分量。这里处理的是一个标量有限元，所以只有一个分量。
```cpp
const unsigned int first_boundary_dof
= std::distance (boundary_dofs.begin(),
        std::find (boundary_dofs.begin(),
            boundary_dofs.end(),
            true));
```
然后就是具体的限制。正如引子中所说，我们限制边界上某一个节点的值等于边界上其他所有自由度的值总和。所以，先找到第一个自由度，这里是查找第一个true值，然后计算它离第一个元素的距离从而确定它的指标。
```cpp
mean_value_constraints.clear ();
mean_value_constraints.add_line (first_boundary_dof);
for (unsigned int i=first_boundary_dof+1; i<dof_handler.n_dofs(); ++i)
    if (boundary_dofs[i] == true)
        mean_value_constraints.add_entry (first_boundary_dof,
                i, -1);
mean_value_constraints.close ();
```
然后创建一个限制对象。首先清除所有的之前的内容，然后加上这么一行限制，在其他所有自由度的总和前加上-1的权重。
```cpp
DynamicSparsityPattern dsp (dof_handler.n_dofs(),
        dof_handler.n_dofs());
DoFTools::make_sparsity_pattern (dof_handler, dsp);
mean_value_constraints.condense (dsp);
sparsity_pattern.copy_from (dsp);
system_matrix.reinit (sparsity_pattern);
}
```
然后就是创建稀疏矩阵。这里也是先使用动态稀疏模式来大致估计长度，然后再创建稀疏矩阵。
接下来是组装和求解：
```cpp
template <int dim>
void LaplaceProblem<dim>::assemble_and_solve ()
{
    const unsigned int gauss_degree
        = std::max (static_cast<unsigned int>(std::ceil(1.*(mapping.get_degree()+1)/2)),
                2U);
    MatrixTools::create_laplace_matrix (mapping, dof_handler,
            QGauss<dim>(gauss_degree),
            system_matrix);
    VectorTools::create_right_hand_side (mapping, dof_handler,
            QGauss<dim>(gauss_degree),
            ConstantFunction<dim>(-2),
            system_rhs);
```
首先必须组装矩阵和右端项。因为Laplace方程比较简单，库函数直接提供了工具，将单元循环什么的都集成了起来，正如上面所写。
```cpp
Vector<double> tmp (system_rhs.size());
VectorTools::create_boundary_right_hand_side (mapping, dof_handler,
        QGauss<dim-1>(gauss_degree),
        ConstantFunction<dim>(1),
        tmp);
```
边界力的计算也是直接调用。
```cpp
system_rhs += tmp;
```
然后将边界上的贡献叠加到整体右端项中。注意这个地方需要显式地叠加，因为这两个create函数都是先清除输出变量，所以不能直接叠加。
```cpp
mean_value_constraints.condense (system_matrix);
mean_value_constraints.condense (system_rhs);
solve ();
mean_value_constraints.distribute (solution);
```
然后施加限制条件，再求解。
```cpp
Vector<float> norm_per_cell (triangulation.n_active_cells());
VectorTools::integrate_difference (mapping, dof_handler,
        solution,
        ZeroFunction<dim>(),
        norm_per_cell,
        QGauss<dim>(gauss_degree+1),
        VectorTools::H1_seminorm);
const double norm = norm_per_cell.l2_norm();
output_table.add_value ("cells", triangulation.n_active_cells());
output_table.add_value ("|u|_1", norm);
output_table.add_value ("error", std::fabs(norm-std::sqrt(3.14159265358/2)));
}
```
这里是计算了范数并输出。
```cpp
template <int dim>
void LaplaceProblem<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> cg (solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    cg.solve (system_matrix, solution, system_rhs,
            preconditioner);
}
```
然后就是求解，跟Step5相同。
run函数：
```cpp
template <int dim>
void LaplaceProblem<dim>::run ()
{
    GridGenerator::hyper_ball (triangulation);
    static const SphericalManifold<dim> boundary;
    triangulation.set_all_manifold_ids_on_boundary(0);
    triangulation.set_manifold (0, boundary);
    for (unsigned int cycle=0; cycle<6; ++cycle, triangulation.refine_global(1))
    {
        setup_system ();
        assemble_and_solve ();
    };
    output_table.set_precision("|u|_1", 6);
    output_table.set_precision("error", 6);
    output_table.write_text (std::cout);
    std::cout << std::endl;
}
}
```
最后是main函数：
```cpp
int main ()
{
    try
    {
        std::cout.precision(5);
        for (unsigned int mapping_degree=1; mapping_degree<=3; ++mapping_degree)
            Step11::LaplaceProblem<2>(mapping_degree).run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    };
    return 0;
}
```

# 计算结果
```cpp
Using mapping with degree 1:
============================
cells |u|_1 error
5 0.680402 0.572912
20 1.085518 0.167796
80 1.208981 0.044334
320 1.242041 0.011273
1280 1.250482 0.002832
5120 1.252605 0.000709
Using mapping with degree 2:
============================
cells |u|_1 error
5 1.050963 0.202351
20 1.199642 0.053672
80 1.239913 0.013401
320 1.249987 0.003327
1280 1.252486 0.000828
5120 1.253108 0.000206
Using mapping with degree 3:
============================
cells |u|_1 error
5 1.086161 0.167153
20 1.204349 0.048965
80 1.240502 0.012812
320 1.250059 0.003255
1280 1.252495 0.000819
5120 1.253109 0.000205
```
其中一个有意思的地方是：使用线性映射的误差要比使用更高阶映射的三倍要大。因此在本例中使用更高阶映射，不是因为它提高了收敛阶数，而是它直接计算出了常数。另一方面，使用三次映射并没有显著提高精度。
