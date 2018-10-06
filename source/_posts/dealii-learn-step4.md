---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 4
tags: [deal.II]
categories: computational material science
date: 2016-8-27
---

# 引子
deal.II有一个特性，叫作”维度无关的编程“。前面的例子中，很多类都有一个尖括号括起的数字的后缀。
这意味着该类能作用在不同的维度上，而不同维度的计算代码基本相同，这能显著地减少重复代码。这正是C++的模板template的拿手好戏。
在Step4中，将显示程序怎样维度无关的编程，具体是将step3中的Laplace问题同时在二维和三维上求解，以及右端项不再是常量、边界值不再为0。

# 程序解析
```cpp
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
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
#include <deal.II/base/logstream.h>
using namespace dealii;
```
最后一个头文件logstream是为了压缩输出信息。
下面就是step4类，它的形式跟step3相同，只不过将之前的2维改成这里的dim，相应地改用了类模板的形式。
```cpp
template <int dim>
class Step4
{
    public:
        Step4 ();
        void run ();
    private:
        void make_grid ();
        void setup_system();
        void assemble_system ();
        void solve ();
        void output_results () const;
        Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
};
```
下面的右端项和边界条件也都是类模板的形式：
```cpp
template <int dim>
class RightHandSide : public Function<dim>
{
    public:
        RightHandSide () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
};
template <int dim>
class BoundaryValues : public Function<dim>
{
    public:
        BoundaryValues () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
};
```
可以看出，两者都是继承自Function类，其中的value函数是一个虚函数，需要自定义一下：
```cpp
template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
        const unsigned int / *component* /) const
{
    double return_value = 0.0;
    for (unsigned int i=0; i<dim; ++i)
        return_value += 4.0 * std::pow(p(i), 4.0);
    return return_value;
}
```
其中，Point<dim>代表n维的点，可以用圆括号来访问它的分量。
右端项同理：
```cpp
template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
        const unsigned int / *component* /) const
{
    return p.square();
}
```
只是这里取的右端项正好是点的平方，所以直接调用平方函数即可。
下面是step4的具体应用，它的每个成员函数的具体实现前面也要加template。
```cpp
template <int dim>
Step4<dim>::Step4 ()
    :
        fe (1),
        dof_handler (triangulation)
{}
template <int dim>
void Step4<dim>::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (4);
    std::cout << " Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << " Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
}
template <int dim>
void Step4<dim>::setup_system ()
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
以上是其构造函数、make_grid和setup_system成员函数，跟step3中基本相同，差别就是加入了dim。
以下就是跟之前不同了，这里使用的是非常量的右端项和非零边界条件，与前面稍有不同，体现在代码上也是稍有不同：
```cpp
template <int dim>
void Step4<dim>::assemble_system ()
{
    QGauss<dim> quadrature_formula(2);
```
对于非常量的rhs，需要创建它的一个对象：
```cpp
const RightHandSide<dim> right_hand_side;
```
为了对每个单元都计算右端项，还得需要单元上的积分点信息，所以得在FEValues说明一下需要更新积分点信息：
```cpp
FEValues<dim> fe_values (fe, quadrature_formula,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);

```
然后就是对单元上的矩阵和右端项的计算：
```cpp
const unsigned int dofs_per_cell = fe.dofs_per_cell;
const unsigned int n_q_points = quadrature_formula.size();
FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
Vector<double> cell_rhs (dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
         endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
    fe_values.reinit (cell);
    cell_matrix = 0;
    cell_rhs = 0;
    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                        fe_values.shape_grad (j, q_index) *
                        fe_values.JxW (q_index));
            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                    right_hand_side.value (fe_values.quadrature_point (q_index)) *
                    fe_values.JxW (q_index));
        }
```
这里就将rhs和matrix同时在一个循环中计算。另外，在cell_rhs的计算时，乘以的不再是1，而是函数在积分点上的值。
然后再组装整体：
```cpp
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
```
可以看出，这里已将两个循环合并。
然后就是将不为0的边界条件加入，就是将step3中的ZeroFunction换成上面的BoundaryValues类的对象：
```cpp
std::map<types::global_dof_index,double> boundary_values;
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        BoundaryValues<dim>(),
        boundary_values);
MatrixTools::apply_boundary_values (boundary_values,
        system_matrix,
        solution,
        system_rhs);
}
```
下面是求解了：
```cpp
template <int dim>
void Step4<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> solver (solver_control);
    solver.solve (system_matrix, solution, system_rhs,
            PreconditionIdentity());
    std::cout << " " << solver_control.last_step()
        << " CG iterations needed to obtain convergence."
        << std::endl;
}
```
跟之前差不多，只是这里手动输出迭代步数。
```cpp
template <int dim>
void Step4<dim>::output_results () const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output (dim == 2 ?
            "solution-2d.vtk" :
            "solution-3d.vtk");
    data_out.write_vtk (output);
}
```
输出文件的格式也由gnuplot改成了VTK格式。也将2维和3维的数据用不同的文件名区分。
run函数无须多说：
```cpp
template <int dim>
void Step4<dim>::run ()
{
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    make_grid();
    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
}
```
main函数如下：
```cpp
int main ()
{
    deallog.depth_console (0);
    {
        Step4<2> laplace_problem_2d;
        laplace_problem_2d.run ();
    }
    {
        Step4<3> laplace_problem_3d;
        laplace_problem_3d.run ();
    }
    return 0;
}
```
2d和3d的切换是如此从容，注意：这里用花括号将两者分开，是为了及时销毁变量和释放内存。

# 计算结果
```cpp
Solving problem in 2 space dimensions.
Number of active cells: 256
Total number of cells: 341
Number of degrees of freedom: 289
26 CG iterations needed to obtain convergence.
Solving problem in 3 space dimensions.
Number of active cells: 4096
Total number of cells: 4681
Number of degrees of freedom: 4913
30 CG iterations needed to obtain convergence.
```
可以看出3维时的自由度数要大很多，计算量也相应增大。
2维计算结果为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjkk1jqg9j30qx0k5abx.jpg)
3维计算结果为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjkkbrbhcj30qx0k542b.jpg)

