---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 5
tags: [deal.II]
categories: computational material science
date: 2016-8-28
---

# 引子
此例没有介绍革命性的功能，但有很多对前面例子的“微创新”，包括：
- 在不断细化的网格上的计算。数值计算通常要在不同的网格上进行，这样才能感受到精度。而且deal.II支持自适应网格，虽然这个例子中没有用到，但基础在这
- 读入非规则网格数据
- 计算优化
- debug模式，使用Assert宏
- 变系数Possion方程，使用预条件迭代求解器

这里要求解的方程是：
$$
\begin{equation}
\begin{split}
-\nabla\cdot a(x)\nabla u(x) &=1  \qquad in \Omega, \\\
u &=0 \qquad on \partial\Omega
\end{split}
\end{equation}
$$
如果$a(x)$是常系数，那么就成了Possion方程，如果它是空间相关的变系数，方程就复杂一些了。
还是得先写出方程的弱形式：
$$
(a \nabla \phi, \nabla u) = (\phi, 1) \qquad \forall \phi
$$

# 程序解析
以下是头文件们：
```cpp
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace dealii;
```
新增的是grid_in.h，是为了从硬盘中读入一个网格文件。manifold_lib.h是为了描述环形边界上的对象。

step5类模板：
```cpp
template <int dim>
class Step5
{
    public:
        Step5 ();
        void run ();
    private:
        void setup_system ();
        void assemble_system ();
        void solve ();
        void output_results (const unsigned int cycle) const;
        Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
};
```
因为这里处理的是变系数椭圆问题，使用的对象类型还是Function，不过这里没用value，而是用的value_list，它不再接收单个点，而是接收一系列的点，然后返回这些点上的函数值：
```cpp
template <int dim>
class Coefficient : public Function<dim>
{
    public:
        Coefficient () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                std::vector<double> &values,
                const unsigned int component = 0) const;
};
```
下面是单个点上的函数值：
```cpp
template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
        const unsigned int / *component* /) const
{
    if (p.square() < 0.5*0.5)
        return 20;
    else
        return 1;
}
```
那么下面就是一下计算很多点的函数值：
```cpp
template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
        std::vector<double> &values,
        const unsigned int component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

```
这个函数接收三个参数：一个是坐标点的列表，一个是存储这些点上的函数值的列表，一个是矢量component，这里应该是0,因为此处是标量函数。
很明显，输出列表values的大小跟输入列表points应该相同，但事实上90%的编程错误都是输入了无效参数，因此应该保证输入参数是valid的。此处，Assert宏是个好方法，因为它保证它的第一个参数，即条件，是有效的，如果无效，就抛出一个exception，即它的第二个参数，通常是终止程序。这将极快地定位错误，方便调试。另一方面，这些检查也不会明显地拖慢程序，而且，Assert宏可仅存在于debug模式，在优化模式中可以将其完全去掉。
事实上，如果将deal.II中的所有check都关掉，可以提速4倍，但同时有引入大量调试错误的问题。
所以，最好是程序稳定后，再将debug关闭。
上面代码就是Assert一下两者的尺寸是否相同，第一个参数是是否相同的条件，第二个参数是调用内置的一个函数来输出两者维度不匹配的信息。该算例的最后就是给出了一个触发这种不匹配的情形，可以发现很快就能定位错误，同时，如果程序是在一个调试器中运行，可以通过调用堆栈直接跳转到出错位置。
```cpp
Assert (component == 0,
        ExcIndexRange (component, 0, 1));
```
这里还检查了是不是标量函数，因为标量可视为只有一个分量的矢量，所以Assert一下是否component=0,如果越界了，就调用ExcIndexRange函数。
下面就是具体赋值代码：
```cpp
const unsigned int n_points = points.size();
for (unsigned int i=0; i<n_points; ++i)
{
    if (points[i].square() < 0.5*0.5)
        values[i] = 20;
    else
        values[i] = 1;
}
}
```
构造函数：
```cpp
template <int dim>
Step5<dim>::Step5 () :
    fe (1),
    dof_handler (triangulation)
{}
```
建立系统，跟之前不同的是没有生成网格这一步：
```cpp
template <int dim>
void Step5<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    std::cout << " Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}
```
组装系统：
```cpp
template <int dim>
void Step5<dim>::assemble_system ()
{
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values (fe, quadrature_formula,
            update_values | update_gradients |
            update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
```
这一步的前面部分跟之前的相同，不解释。
下面不同之处在于这里用的是变系数，所以先根据之前的系数模板创建这么一个对象：
```cpp
const Coefficient<dim> coefficient;
std::vector<double> coefficient_values (n_q_points);
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
     endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit (cell);
    coefficient.value_list (fe_values.get_quadrature_points(),
            coefficient_values);
    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (coefficient_values[q_index] *
                        fe_values.shape_grad(i,q_index) *
                        fe_values.shape_grad(j,q_index) *
                        fe_values.JxW(q_index));
            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                    1.0 *
                    fe_values.JxW(q_index));
        }
    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                    local_dof_indices[j],
                    cell_matrix(i,j));
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
}
std::map<types::global_dof_index,double> boundary_values;
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        ZeroFunction<dim>(),
        boundary_values);
MatrixTools::apply_boundary_values (boundary_values,
        system_matrix,
        solution,
        system_rhs);
}
```
比起step4做的优化是：在计算系数函数的值时，是在每个单位上一下计算了所有积分点上的值。因为从step4可以看出，在那里计算右端项时，计算了$dofs_per_cell*n_q_points times$次，如果还是这样计算，对应到这里计算系数时，就需要计算$dofs_per_cell*dofs_per_cell*n_q_points$次，但实际上相同积分点对应的这些函数值相同，没必要在自由度的循环中重复计算，同时这里还涉及虚函数调用，开销很大。综上，一次性计算完毕，优化了计算效率。
求解步如下：
```cpp
template <int dim>
void Step5<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> solver (solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve (system_matrix, solution, system_rhs,
            preconditioner);
    std::cout << " " << solver_control.last_step()
        << " CG iterations needed to obtain convergence."
        << std::endl;
}
```
这里使用了对称超松弛迭代算法作为预条件子。
输出部分如下：
```cpp
template <int dim>
void Step5<dim>::output_results (const unsigned int cycle) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    DataOutBase::EpsFlags eps_flags;
    eps_flags.z_scaling = 4;
    eps_flags.azimut_angle = 40;
    eps_flags.turn_angle = 10;
    data_out.set_flags (eps_flags);
    std::ostringstream filename;
    filename << "solution-"
        << cycle
        << ".eps";
    std::ofstream output (filename.str().c_str());
    data_out.write_eps (output);
}
```
这里将输出格式改成了EPS。因为EPS是一个打印格式，它不像其他格式的数据那样能输入到图像工具进行编辑，所以需要事先定义好，源码中就定义了多种flag，不详述。
```cpp
template <int dim>
void Step5<dim>::run ()
{
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);
    std::ifstream input_file("circle-grid.inp");
```
run函数中直接读入网格文件，这里是个inp后缀的。因为该网格是二维的，所以Assert一下如果不是二维的，就抛出一个异常：
```cpp
Assert (dim==2, ExcInternalError());
```
这是一个非规则网格文件UCD：
```cpp
grid_in.read_ucd (input_file);
```
```cpp
static const SphericalManifold<dim> boundary;
triangulation.set_all_manifold_ids_on_boundary(0);
triangulation.set_manifold (0, boundary);
for (unsigned int cycle=0; cycle<6; ++cycle)
{
    std::cout << "Cycle " << cycle << ':' << std::endl;
    if (cycle != 0)
        triangulation.refine_global (1);
    std::cout << " Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << " Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
    setup_system ();
    assemble_system ();
    solve ();
    output_results (cycle);
}
}
```
然后创建一个流形，来告诉triangulation在网格细化后如何在边界上加点。
然后是main函数：
```cpp
int main ()
{
    Step5<2> laplace_problem_2d;
    laplace_problem_2d.run ();
    /*
       Coefficient<2> coefficient;
       std::vector<Point<2> > points (2);
       std::vector<double> coefficient_values (1);
       coefficient.value_list (points, coefficient_values);
       */
    return 0;
}
```
注释起来的代码是为了得到Assert的异常信息。

# 计算结果
每次细化结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-5.solution-0.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-5.solution-1.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-5.solution-2.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-5.solution-3.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-5.solution-4.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-5.solution-5.png)
