---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 6
tags: [deal.II]
categories: computational material science
date: 2016-8-30
---

# 引子
本例主要着眼于处理局部细化的网格。如果临近单元细化多次以后，单元界面上的格点可能在另一边就不平衡，称为“悬点”。为了保证全局解在这些格点上也是连续的，必须对这些节点上的解的值施加一些限制，相应的核心的类是ConstraintMatrix。

# 程序解析
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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/fe/fe_q.h>
```
以上头文件之前用过，不解释。
```cpp
#include <deal.II/grid/grid_out.h>
```
本例中不从文件中读取网格，而是使用一个库函数来直接生成。但是需要输出网格(仅输出网格，不包括解，就像step1一样)，所以需要上述头文件。
```cpp
#include <deal.II/lac/constraint_matrix.h>
```
当局部细化网格时，就会产生悬点，但是，标准的有限元方法假定离散解空间是连续的，所以我们也得在悬点上的自由度加一些限制，使得全局解是连续的。这些限制条件就由ConstraintMatrix实现。
```cpp
#include <deal.II/grid/grid_refinement.h>
```
这个头文件提供函数来确定哪个单元来细化或粗化。
```cpp
#include <deal.II/numerics/error_estimator.h>
```
该头文件提供细化指示子的计算，其根据一些误差估计方法。通常来说自适应是跟所研究的问题密切相关的，不过该文件中的误差指示子对很多问题都具有很好的适用性。
```cpp
using namespace dealii;
```
最后是使用dealii命名空间。

step6的类模板：
```cpp
template <int dim>
class Step6
{
    public:
        Step6 ();
        ~Step6 ();
        void run ();
    private:
        void setup_system ();
        void assemble_system ();
        void solve ();
        void refine_grid ();
        void output_results (const unsigned int cycle) const;
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FE_Q<dim> fe;
        ConstraintMatrix constraints;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
};
```
基本跟之前的类模板相同，但多了几个新东西：refine_grid函数用来自适应地细化网格，这跟之前的全局细化不同;constraints对象来存储限制条件;还有一个析构函数。
变系数的引入是完全复制step5的：
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
template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
        const unsigned int) const
{
    if (p.square() < 0.5*0.5)
        return 20;
    else
        return 1;
}
template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
        std::vector<double> &values,
        const unsigned int component) const
{
    const unsigned int n_points = points.size();
    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));
    Assert (component == 0,
            ExcIndexRange (component, 0, 1));
    for (unsigned int i=0; i<n_points; ++i)
    {
        if (points[i].square() < 0.5*0.5)
            values[i] = 20;
        else
            values[i] = 1;
    }
}
```
构造函数如下：
```cpp
template <int dim>
Step6<dim>::Step6 ()
    :
        dof_handler (triangulation),
        fe (2)
{}
```
注意：这里因为要使用的单元是二次的，所以fe的参数是2。还要注意初始化器列表中的两个的位置顺序，这里是dof_handler在fe的前面，之前都是在fe的后面。这里的顺序变动将会产生一个很坏的副作用：
当我们使用dof_handler.distribute_dofs()分配自由度时，dof_handler也存储了一个指向正在使用的有限单元的指针，因为此指针一直在使用，直到使用另一个fe重新分配自由度或dof_handler被销毁掉。这样如果允许在dof_handler之前删除fe的话就会产生很大隐患。为了防止这个误操作，dof_handler在fe对象内部为一个计数器增值，这个计数器统计有多少对象使用这个有限元。如果该计数器大于0，那么这个fe对象就拒绝被销毁，因为有其他对象依赖于它。如果试图销毁它，一个异常就会被抛出，程序就会停止。
现在构造函数的初始化器列表这样写的话，如果不写析构函数，那么就会发生如上错误，因为销毁顺序是与之前的建立顺序相反的。在析构函数中应当做的就是告诉dof_handler释放它对fe的锁，当然这必须在它确实不再需要fe之后才行，即当所有的有限元相关的数据都被删除。DoFHandler有一个函数clear能够删除所有的自由度，同时释放对fe的锁。clear以后fe内部的计数器就变为0，然后就可以安全地删除fe了。
```cpp
template <int dim>
Step6<dim>::~Step6 ()
{
    dof_handler.clear ();
}
```
下一步就是要建立线性有限元系统的有关变量，如：自由度、矩阵、向量等。
```cpp
template <int dim>
void Step6<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
```
这一部分代码跟step5相同，下面就不同了：
```cpp
constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler,
        constraints);
```
上述就是对悬点的限制条件。clear就是清除其中可能残留的信息。
```cpp
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        ZeroFunction<dim>(),
        constraints);
```
这里将0值边界条件施加进去，将结果保存进constraints。注意：之前施加边界条件都是在组装完毕之后，这里却是在其之前。消除系统方程的边界节点，应该发生在消除悬点之后，这个是很重要的(感谢adadobe的指正)。所以在建立悬点限制后就将边界条件放进去。
下一步就是关闭该对象：
```cpp
constraints.close ();
```
接着就是创建稀疏矩阵：
```cpp
DynamicSparsityPattern dsp(dof_handler.n_dofs());
DoFTools::make_sparsity_pattern(dof_handler,
        dsp,
        constraints,
        / *keep_constrained_dofs = * / false);
```
注意，这里没有立即把pattern复制到最终的里面，而是又做了一步，至于是啥意思，没看明白。
然后就是真正创建矩阵：
```cpp
sparsity_pattern.copy_from(dsp);
system_matrix.reinit (sparsity_pattern);
}
```
再然后就是组装系统。跟之前的step5有两点不同：一是因为有限元形函数的多项式次数变大了，相应的积分公式的次数也要提高，这点很好实现，QGauss类接收每个方向上积分点的个数作为参数，之前双线性单元是2个积分点，对双二次单元则是3个。二是在组装总刚时，不再使用手写的循环，而是用ConstraintMatrix::distribute_local_to_global实现，它内置那个循环，并删除所有的限制。
```cpp
template <int dim>
void Step6<dim>::assemble_system ()
{
    const QGauss<dim> quadrature_formula(3);
    FEValues<dim> fe_values (fe, quadrature_formula,
            update_values | update_gradients |
            update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
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
        constraints.distribute_local_to_global (cell_matrix,
                cell_rhs,
                local_dof_indices,
                system_matrix,
                system_rhs);
    }
}
```
总套路跟之前差不多，但实际上隐藏的变化还是挺多的，只不过对用户不可见。
求解步如下：
```cpp
template <int dim>
void Step6<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> solver (solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve (system_matrix, solution, system_rhs,
            preconditioner);
    constraints.distribute (solution);
}
```
这里在最后也加入了限制。
下面是重头戏：局部细化。这里使用的是KellyErrorEstimator，顾名思义，该方法是由Kelly等人提出的。尽管该方法起初是用于拉普拉斯方程，但证明其对很多问题都能快速地产生局部细化的网格。原理上来讲，它着眼于单元之间的解的梯度阶跃(即是二阶导数的计算)，然后与单元尺寸比例一下，物理上来讲，它代表解在该单元的局部光滑度。
```cpp
template <int dim>
void Step6<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
            QGauss<dim-1>(3),
            typename FunctionMap<dim>::type(),
            solution,
            estimated_error_per_cell);
```
接收的参数分别为：一个DofHandler对象，描述自由度和每个自由度上的值，边界上的积分公式QGauss<dim-1>，Neumann边界对应的边界指示子FunctionMap<dim>::type()，这里因为没有施加Neumann边界条件，所以为空，解solution，还有返回的每个单元上的误差值estimated_error_per_cell。细化过程则为：对有最大误差值的30%的单元细化，而对有最小误差值的3%的单元粗化：
```cpp
GridRefinement::refine_and_coarsen_fixed_number (triangulation,
        estimated_error_per_cell,
        0.3, 0.03);
triangulation.execute_coarsening_and_refinement ();
}
```
仅输出网格：
```cpp
template <int dim>
void Step6<dim>::output_results (const unsigned int cycle) const
{
    Assert (cycle < 10, ExcNotImplemented());
    std::string filename = "grid-";
    filename += ('0' + cycle);
    filename += ".eps";
    std::ofstream output (filename.c_str());
    GridOut grid_out;
    grid_out.write_eps (triangulation, output);
}
```
run函数如下：
```cpp
template <int dim>
void Step6<dim>::run ()
{
    for (unsigned int cycle=0; cycle<8; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            GridGenerator::hyper_ball (triangulation);
            static const SphericalManifold<dim> boundary;
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold (0, boundary);
            triangulation.refine_global (1);
        }
        else
            refine_grid ();
        std::cout << " Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
        setup_system ();
        std::cout << " Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
        assemble_system ();
        solve ();
        output_results (cycle);
    }
DataOutBase::EpsFlags eps_flags;
eps_flags.z_scaling = 4;
DataOut<dim> data_out;
data_out.set_flags (eps_flags);
data_out.attach_dof_handler (dof_handler);
data_out.add_data_vector (solution, "solution");
data_out.build_patches ();
std::ofstream output ("final-solution.eps");
data_out.write_eps (output);
}
```
注意，每次循环都输出一下细化后的网格，同时在最后也输出一下解。
最后是main函数，它比之前的都高级，因为加入了异常捕获机制。有时候，程序只有在运行时才会出问题，比如没有足够的硬盘空间用于输出文件，没有足够的内存来分配矢量或矩阵，或者没有权限读写文件等。这里的代码具有一定的通用性，可以用于其他大型程序中：
```cpp
int main ()
{
    try
    {
        Step6<2> laplace_problem_2d;
        laplace_problem_2d.run ();
    }
```
首先try运行一下我们的程序，如果失败了，就要尽可能多地收集信息。
```cpp
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
```
如果抛出的异常属于C++标准类exception，那么就可以调用what这一成员函数来显示具体出错信息。
```cpp
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
}
```
如果不是一个来自标准exception类的异常，那么就无力作什么事情，只能打印一些提示字符。
```cpp
return 0;
}
```
如果没有异常，就顺序退出。

# 计算结果：
最终输出的解为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.solution.png)
每步循环得到的网格为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-0.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-1.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-2.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-3.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-4.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-5.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-6.png)
![](http://7xrm8i.com1.z0.glb.clouddn.com/step-6.grid-7.png)

# 可扩展性
## 求解器和预条件子
deal.II中提供多种求解器和预条件子来求解问题。
该例中的线性系统是对称且正定的，所以CG算法挺适合。这里可以更改预条件子来看看，如使用Jacobi(需要包含lac/sparse_ilu.h头文件)：
```cpp
PreconditionJacobi<> preconditioner;
preconditioner.initialize(system_matrix);
```
或LU分解：
```cpp
SparseILU<double> preconditioner;
preconditioner.initialize(system_matrix);
```
预条件子的选择需要根据具体问题来，不同类型的问题以及不同的有限单元可能有不一样的结论。

## 更好的网格
之前生成的网格可以看出来不是很能显示出圆形这一特征，Triangulation类仅能着眼于粗网格的集合，但是不知道它们组合起来应该怎么样。这可以通过更复杂的程序来调节，使之better represent the desired geometry。具体不说了，看帮助文档。
