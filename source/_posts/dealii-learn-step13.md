---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 13
tags: [deal.II]
categories: computational material science
date: 2016-9-20
---

本文来自亓欣波的博客[qixinbo.info](qixinbo.github.io)。
转载请保留上面信息，谢谢！

# 引子

本例亮点：
- 精确寻找某个节点并通过要求提供该节点上的自由度指标来取出该点上的值。
- 使用线程并行，否则只能串行等待上一步完成。线程是计算的最小单元。

在本例中，将不太关心描述使用deal.II的新方法，而是着眼于书写模块化及可扩展的有限元程序的方法。
这主要是为了考虑先进研究软件的大小和灵活性：使用了先进误差估计概念和自适应解的应用通常相当庞大。而大家的共识是这样的：庞大的程序，如果没有分成更小更独立的组件，将会很快消亡，因为甚至作者也会很快忘记程序内部各个部分之间的依赖关系。数据封装，比如面向对象编程和定义小巧且固定的界面的模块化编程，将会有效组织数据流以及理清相互依赖关系。这对于一个程序有多个开发者而言更是必不可少的。
本例仍然是一个Laplace求解，但跟之前不同的是：
- 负责数值求解的类不再是“求解—误差估计—细化—再求解”这样的过程，我们将它委派给一个外部函数。这样首先能把它当成一个构件来呼应更大的问题，比如Laplace求解仅是这个问题的一小部分的情形。再者也能基于该类建立求解其他问题的框架。
- 把分析所求解的过程分成很多类，原因是因为人们可能不关心方程的解，而是关系它的其他方面。比如在弹性问题中想要知道某个边界上的拉力，或者在给定位置的接收器上的地震波信号。这样分析的过程通常不会影响求解过程，将它分成若干类，构建不同的过滤器来用于不同的用途。
- 将网格细化的类从计算类中分离出来
- 将测试算例的描述从程序中剥离出来
- 用WorkStream将组装线性系统并行化

该程序所做的事不新，但是它的组织方式值得借鉴。不同的人有不同的软件设计方式，要勤加思考。
看这个程序时可以发现，它甚至是有些复杂的，但它代表了一种良好的能复用的设计方式，尤其是对大型程序。在编程开始时设计好程序是很重要的，否则可能在后期需要重构。
本例肯定也会有一些缺陷，但它却提示人们，尽量少编写紧耦合的代码！

本例最重要的是它的结构，但也得明白它干了啥：求解给定右端项的Laplace方程，使得解是$u(x,t)=\exp(x+\sin(10y+5x^2))$。计算目的是得到解在点$x\_0=(0.5,0.5)$处的值，同时对比两种细化准则下的精度。

# 程序解析
```cpp
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <iostream>
#include <fstream>
#include <list>
#include <sstream>
```
这次的头文件顺序有点讲究：base – lac – grid – dofs – fe – numerics。基本后面是构建于前面基础上。
然后开始创建本例的命名空间：
```cpp
namespace Step13
{
    using namespace dealii;
```

## 解的分析
这里也是封装成一个命名空间：
```cpp
namespace Evaluation
{
template <int dim>
    class EvaluationBase
    {
        public:
            virtual ~EvaluationBase ();
            void set_refinement_cycle (const unsigned int refinement_cycle);
            virtual void operator () (const DoFHandler<dim> &dof_handler,
                    const Vector<double> &solution) const = 0;
        protected:
            unsigned int refinement_cycle;
    };
```
这个定义了一个基类，主要是创建一个虚函数operator，接收DoFHandler和solution。这个“分析”基类可以用于多种分析用途，通过传入solution，然后定义不同的虚函数实现，从而达到目的。
然后是它的具体实现：
```cpp
template <int dim>
EvaluationBase<dim>::~EvaluationBase ()
{}
template <int dim>
void
EvaluationBase<dim>::set_refinement_cycle (const unsigned int step)
{
    refinement_cycle = step;
}
```
下面就是真正地用这个类——用作提取某点的值：
```cpp
template <int dim>
class PointValueEvaluation : public EvaluationBase<dim>
{
    public:
        PointValueEvaluation (const Point<dim> &evaluation_point,
                TableHandler &results_table);
        virtual void operator () (const DoFHandler<dim> &dof_handler,
                const Vector<double> &solution) const;
        DeclException1 (ExcEvaluationPointNotFound,
                Point<dim>,
                << "The evaluation point " << arg1
                << " was not found among the vertices of the present grid.");
    private:
        const Point<dim> evaluation_point;
        TableHandler &results_table;
};
```
点的位置是在构造函数中传入，输出在构造函数的第二个参数中。
如果不依赖已知所用的有限单元形式，来找到有限元场中任意一点的值是相当困难的，因为我们不能在节点之间内插。这里简单起见，会确保这个点确实是节点，如果在对所有节点循环过程中没有发现该点，就会抛出一个异常，而不是简单的忽略。这里使用的是跟Step9一样的DeclExceptionN宏。
虽然这里明确写出析构函数，编译器会产生一个默认的析构函数，它也是虚函数。
```cpp
template <int dim>
PointValueEvaluation<dim>::
PointValueEvaluation (const Point<dim> &evaluation_point,
        TableHandler &results_table)
    :
        evaluation_point (evaluation_point),
        results_table (results_table)
{}
```
这是构造函数，就是接收数据并存储下来。
下面就是最重要的部分，计算某点的值：
```cpp
template <int dim>
void
PointValueEvaluation<dim>::
operator () (const DoFHandler<dim> &dof_handler,
        const Vector<double> &solution) const
{
    double point_value = 1e20;
```
首先分配一个变量来存储该点上的数值。用一个明显是错误的值来初始化它，这样如果没有正确地赋值，就能迅速地发现错误。
```cpp
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
     endc = dof_handler.end();
bool evaluation_point_found = false;
for (; (cell!=endc) && !evaluation_point_found; ++cell)
    for (unsigned int vertex=0;
            vertex<GeometryInfo<dim>::vertices_per_cell;
            ++vertex)
        if (cell->vertex(vertex) == evaluation_point)
        {
```
然后对单元和单元上的节点进行循环，检查该节点是否是要找的点。如果找到了，同时要设置一个旗标，说明找到了，跳出循环。
```cpp
point_value = solution(cell->vertex_dof_index(vertex,0));
evaluation_point_found = true;
break;
        };
```
上面就是取出全局解在该点上的值，如果解是个矢量，就取出第一个分量。但这里需要说明的是这里使用的有限元的自由度与节点是相关的，但这不是完全通用的，比如对间断有限元就不适用。理想情况下，需要在前面加一个Assert语句来捕获异常。这里省略了，因为如果我们用语句vertex_dof_index来要求返回该点的自由度指标，但是其实没有时会导致错误。
```cpp
AssertThrow (evaluation_point_found,
        ExcEvaluationPointNotFound(evaluation_point));
```
然后再次判定一下确实找到了那个点。考虑到该点可能在细化和粗化过程中消失，所以判定是必要的。
```cpp
results_table.add_value ("DoFs", dof_handler.n_dofs());
results_table.add_value ("u(x_0)", point_value);
}
```
将结果存储下来。
一个额外的功能可以顺便实现了，就是将结果输出成图片形式，因为我们有了DoFHandler和解，万事俱备，可以顺手解决这个问题。
```cpp
template <int dim>
class SolutionOutput : public EvaluationBase<dim>
{
    public:
        SolutionOutput (const std::string &output_name_base,
                const DataOutBase::OutputFormat output_format);
        virtual void operator () (const DoFHandler<dim> &dof_handler,
                const Vector<double> &solution) const;
    private:
        const std::string output_name_base;
        const DataOutBase::OutputFormat output_format;
};
template <int dim>
SolutionOutput<dim>::
SolutionOutput (const std::string &output_name_base,
        const DataOutBase::OutputFormat output_format)
    :
        output_name_base (output_name_base),
        output_format (output_format)
{}
```
构造函数的接收参数是输出名字的主部分和输出格式。
然后具体操作过程是：
```cpp
template <int dim>
void
SolutionOutput<dim>::operator () (const DoFHandler<dim> &dof_handler,
        const Vector<double> &solution) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ostringstream filename;
    filename << output_name_base << "-"
        << this->refinement_cycle
        << data_out.default_suffix (output_format)
        << std::ends;
    std::ofstream out (filename.str().c_str());
    data_out.write (out, output_format);
}
}

## Laplace求解器
也是先设置一个命名空间：
```cpp
namespace LaplaceSolver
{
    template <int dim>
        class Base
        {
            public:
                Base (Triangulation<dim> &coarse_grid);
                virtual ~Base ();
                virtual void solve_problem () = 0;
                virtual void postprocess (const Evaluation::EvaluationBase<dim> &postprocessor) const = 0;
                virtual void refine_grid () = 0;
                virtual unsigned int n_dofs () const = 0;
            protected:
                const SmartPointer<Triangulation<dim> > triangulation;
        };
template <int dim>
    Base<dim>::Base (Triangulation<dim> &coarse_grid)
    :
        triangulation (&coarse_grid)
    {}
template <int dim>
    Base<dim>::~Base ()
    {}
```
先声明一个抽象基类，它不实现具体功能，只存储一个后来有用的指向triangulation的指针。这里使用的是一个智能指针，保证它一直存在。
这个基类是通用型的，可以用于其他的静态问题，它提供了这些函数声明：求解、后处理和细化网格等。
下面是一个具体的通用求解类：
```cpp
template <int dim>
class Solver : public virtual Base<dim>
{
    public:
        Solver (Triangulation<dim> &triangulation,
                const FiniteElement<dim> &fe,
                const Quadrature<dim> &quadrature,
                const Function<dim> &boundary_values);
        virtual
            ~Solver ();
        virtual
            void
            solve_problem ();
        virtual
            void
            postprocess (const Evaluation::EvaluationBase<dim> &postprocessor) const;
        virtual
            unsigned int
            n_dofs () const;
    protected:
        const SmartPointer<const FiniteElement<dim> > fe;
        const SmartPointer<const Quadrature<dim> > quadrature;
        DoFHandler<dim> dof_handler;
        Vector<double> solution;
        const SmartPointer<const Function<dim> > boundary_values;
        virtual void assemble_rhs (Vector<double> &rhs) const = 0;
    private:
        struct LinearSystem
        {
            LinearSystem (const DoFHandler<dim> &dof_handler);
            void solve (Vector<double> &solution) const;
            ConstraintMatrix hanging_node_constraints;
            SparsityPattern sparsity_pattern;
            SparseMatrix<double> matrix;
            Vector<double> rhs;
        };
        struct AssemblyScratchData
        {
            AssemblyScratchData (const FiniteElement<dim> &fe,
                    const Quadrature<dim> &quadrature);
            AssemblyScratchData (const AssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
        };
        struct AssemblyCopyData
        {
            FullMatrix<double> cell_matrix;
            std::vector<types::global_dof_index> local_dof_indices;
        };
        void
            assemble_linear_system (LinearSystem &linear_system);
        void
            local_assemble_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                    AssemblyScratchData &scratch_data,
                    AssemblyCopyData &copy_data) const;
        void
            copy_local_to_global(const AssemblyCopyData &copy_data,
                    LinearSystem &linear_system) const;
};
```
它除了继承基类中的求解和后处理函数外，还加了一个新的虚函数assemble_rhs。这是因为我们会用两种不同的方法来构建右端项。在protect和private中还有一些额外的数据成员，从它们的名字可以推知它们的功能。
```cpp
template <int dim>
Solver<dim>::Solver (Triangulation<dim> &triangulation,
        const FiniteElement<dim> &fe,
        const Quadrature<dim> &quadrature,
        const Function<dim> &boundary_values)
    :
        Base<dim> (triangulation),
        fe (&fe),
        quadrature (&quadrature),
        dof_handler (triangulation),
        boundary_values (&boundary_values)
{}
template <int dim>
Solver<dim>::~Solver ()
{
    dof_handler.clear ();
}
```
构造函数和析构函数如上。
下面是程序求解的主要架构：
```cpp
template <int dim>
void
Solver<dim>::solve_problem ()
{
    dof_handler.distribute_dofs (*fe);
    solution.reinit (dof_handler.n_dofs());
    LinearSystem linear_system (dof_handler);
    assemble_linear_system (linear_system);
    linear_system.solve (solution);
}
```
它用传入的有限单元构建了DoFHandler对象，然后根据它创建了整个线性系统，包括矩阵、右端项和解。
```cpp
template <int dim>
void
Solver<dim>::
postprocess (const Evaluation::EvaluationBase<dim> &postprocessor) const
{
    postprocessor (dof_handler, solution);
}
```
然后是后处理。
自由度的个数是自解释的：
```cpp
template <int dim>
unsigned int
Solver<dim>::n_dofs () const
{
    return dof_handler.n_dofs();
}
```
下面就是在每一步求解时组装矩阵和右端项：
```cpp
template <int dim>
void
Solver<dim>::assemble_linear_system (LinearSystem &linear_system)
{
    Threads::Task<> rhs_task = Threads::new_task (&Solver<dim>::assemble_rhs,
            *this,
            linear_system.rhs);
    WorkStream::run(dof_handler.begin_active(),
            dof_handler.end(),
            std_cxx11::bind(&Solver<dim>::local_assemble_matrix,
                this,
                std_cxx11::_1,
                std_cxx11::_2,
                std_cxx11::_3),
            std_cxx11::bind(&Solver<dim>::copy_local_to_global,
                this,
                std_cxx11::_1,
                std_cxx11::ref(linear_system)),
            AssemblyScratchData(*fe, *quadrature),
            AssemblyCopyData());
    linear_system.hanging_node_constraints.condense (linear_system.matrix);
```
这里使用了并行，并且是在多个层次上。在计算右端项，新开一个task，意思就是CPU这里有任务要做，当有可用的CPU核心时就分配给它，然后继续干其他的，当需要那个task的计算结果时等着它完成。在组装矩阵时，则是使用WorkStream。当然第一种比起第二种要好写得多。上面使用了C++11的语法，详情见帮助文档。
在等待右端项完成的同时，还可以构建边界条件：
```cpp
std::map<types::global_dof_index,double> boundary_value_map;
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        *boundary_values,
        boundary_value_map);
rhs_task.join ();
linear_system.hanging_node_constraints.condense (linear_system.rhs);
```
然后再将结果join起来。
最后是将边界条件加入到矩阵和右端项中：
```cpp
MatrixTools::apply_boundary_values (boundary_value_map,
        linear_system.matrix,
        solution,
        linear_system.rhs);
}
```
具体工作就是：
```cpp
template <int dim>
Solver<dim>::AssemblyScratchData::
AssemblyScratchData (const FiniteElement<dim> &fe,
        const Quadrature<dim> &quadrature)
    :
        fe_values (fe,
                quadrature,
                update_gradients | update_JxW_values)
{}
template <int dim>
Solver<dim>::AssemblyScratchData::
AssemblyScratchData (const AssemblyScratchData &scratch_data)
    :
        fe_values (scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                update_gradients | update_JxW_values)
{}
template <int dim>
void
Solver<dim>::local_assemble_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
        AssemblyScratchData &scratch_data,
        AssemblyCopyData &copy_data) const
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q_points = quadrature->size();
    copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);
    scratch_data.fe_values.reinit (cell);
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                copy_data.cell_matrix(i,j) += (scratch_data.fe_values.shape_grad(i,q_point) *
                        scratch_data.fe_values.shape_grad(j,q_point) *
                        scratch_data.fe_values.JxW(q_point));
    cell->get_dof_indices (copy_data.local_dof_indices);
}
template <int dim>
void
Solver<dim>::copy_local_to_global(const AssemblyCopyData &copy_data,
        LinearSystem &linear_system) const
{
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
        for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
            linear_system.matrix.add (copy_data.local_dof_indices[i],
                    copy_data.local_dof_indices[j],
                    copy_data.cell_matrix(i,j));
}
```
下面的也用到了并行：
```cpp
template <int dim>
Solver<dim>::LinearSystem::
LinearSystem (const DoFHandler<dim> &dof_handler)
{
    hanging_node_constraints.clear ();
    void (*mhnc_p) (const DoFHandler<dim> &,
            ConstraintMatrix &)
        = &DoFTools::make_hanging_node_constraints;
    Threads::Task<> side_task
        = Threads::new_task (mhnc_p,
                dof_handler,
                hanging_node_constraints);
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    side_task.join();
    hanging_node_constraints.close ();
    hanging_node_constraints.condense (dsp);
    sparsity_pattern.copy_from(dsp);
    matrix.reinit (sparsity_pattern);
    rhs.reinit (dof_handler.n_dofs());
}
```
然后就是求解算法：
```cpp
template <int dim>
void
Solver<dim>::LinearSystem::solve (Vector<double> &solution) const
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> cg (solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(matrix, 1.2);
    cg.solve (matrix, solution, rhs, preconditioner);
    hanging_node_constraints.distribute (solution);
}
```
下面提供了一个能自己组装右端项的类：
```cpp
template <int dim>
class PrimalSolver : public Solver<dim>
{
    public:
        PrimalSolver (Triangulation<dim> &triangulation,
                const FiniteElement<dim> &fe,
                const Quadrature<dim> &quadrature,
                const Function<dim> &rhs_function,
                const Function<dim> &boundary_values);
    protected:
        const SmartPointer<const Function<dim> > rhs_function;
        virtual void assemble_rhs (Vector<double> &rhs) const;
};
template <int dim>
PrimalSolver<dim>::
PrimalSolver (Triangulation<dim> &triangulation,
        const FiniteElement<dim> &fe,
        const Quadrature<dim> &quadrature,
        const Function<dim> &rhs_function,
        const Function<dim> &boundary_values)
    :
        Base<dim> (triangulation),
        Solver<dim> (triangulation, fe,
                quadrature, boundary_values),
        rhs_function (&rhs_function)
{}
template <int dim>
void
PrimalSolver<dim>::
assemble_rhs (Vector<double> &rhs) const
{
    FEValues<dim> fe_values (*this->fe, *this->quadrature,
            update_values | update_quadrature_points |
            update_JxW_values);
    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
    const unsigned int n_q_points = this->quadrature->size();
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<double> rhs_values (n_q_points);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell = this->dof_handler.begin_active(),
             endc = this->dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        cell_rhs = 0;
        fe_values.reinit (cell);
        rhs_function->value_list (fe_values.get_quadrature_points(),
                rhs_values);
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                cell_rhs(i) += (fe_values.shape_value(i,q_point) *
                        rhs_values[q_point] *
                        fe_values.JxW(q_point));
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            rhs(local_dof_indices[i]) += cell_rhs(i);
    };
}
```
至此，除了网格细化这一步，其他的函数都已经实现了。这里使用了两种方式来细化：全局细化和局部细化。
全局细化为：
```cpp
template <int dim>
class RefinementGlobal : public PrimalSolver<dim>
{
    public:
        RefinementGlobal (Triangulation<dim> &coarse_grid,
                const FiniteElement<dim> &fe,
                const Quadrature<dim> &quadrature,
                const Function<dim> &rhs_function,
                const Function<dim> &boundary_values);
        virtual void refine_grid ();
};
template <int dim>
RefinementGlobal<dim>::
RefinementGlobal (Triangulation<dim> &coarse_grid,
        const FiniteElement<dim> &fe,
        const Quadrature<dim> &quadrature,
        const Function<dim> &rhs_function,
        const Function<dim> &boundary_values)
    :
        Base<dim> (coarse_grid),
        PrimalSolver<dim> (coarse_grid, fe, quadrature,
                rhs_function, boundary_values)
{}
template <int dim>
void
RefinementGlobal<dim>::refine_grid ()
{
    this->triangulation->refine_global (1);
}
```
局部细化为：
```cpp
template <int dim>
class RefinementKelly : public PrimalSolver<dim>
{
    public:
        RefinementKelly (Triangulation<dim> &coarse_grid,
                const FiniteElement<dim> &fe,
                const Quadrature<dim> &quadrature,
                const Function<dim> &rhs_function,
                const Function<dim> &boundary_values);
        virtual void refine_grid ();
};
template <int dim>
RefinementKelly<dim>::
RefinementKelly (Triangulation<dim> &coarse_grid,
        const FiniteElement<dim> &fe,
        const Quadrature<dim> &quadrature,
        const Function<dim> &rhs_function,
        const Function<dim> &boundary_values)
    :
        Base<dim> (coarse_grid),
        PrimalSolver<dim> (coarse_grid, fe, quadrature,
                rhs_function, boundary_values)
{}
template <int dim>
void
RefinementKelly<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (this->triangulation->n_active_cells());
    KellyErrorEstimator<dim>::estimate (this->dof_handler,
            QGauss<dim-1>(3),
            typename FunctionMap<dim>::type(),
            this->solution,
            estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number (*this->triangulation,
            estimated_error_per_cell,
            0.3, 0.03);
    this->triangulation->execute_coarsening_and_refinement ();
}
}
```
基于学术目的，还想看看数值解跟解析解之间的差别，所以这里定义一个类来表现精确解：
```cpp
template <int dim>
class Solution : public Function<dim>
{
    public:
        Solution () : Function<dim> () {}
        virtual double value (const Point<dim> &p,
                const unsigned int component) const;
};
template <int dim>
double
Solution<dim>::value (const Point<dim> &p,
        const unsigned int / *component* /) const
{
    double q = p(0);
    for (unsigned int i=1; i<dim; ++i)
        q += std::sin(10*p(i)+5*p(0)*p(0));
    const double exponential = std::exp(q);
    return exponential;
}
template <int dim>
class RightHandSide : public Function<dim>
{
    public:
        RightHandSide () : Function<dim> () {}
        virtual double value (const Point<dim> &p,
                const unsigned int component) const;
};
template <int dim>
double
RightHandSide<dim>::value (const Point<dim> &p,
        const unsigned int / *component* /) const
{
    double q = p(0);
    for (unsigned int i=1; i<dim; ++i)
        q += std::sin(10*p(i)+5*p(0)*p(0));
    const double u = std::exp(q);
    double t1 = 1,
           t2 = 0,
           t3 = 0;
    for (unsigned int i=1; i<dim; ++i)
    {
        t1 += std::cos(10*p(i)+5*p(0)*p(0)) * 10 * p(0);
        t2 += 10*std::cos(10*p(i)+5*p(0)*p(0)) -
            100*std::sin(10*p(i)+5*p(0)*p(0)) * p(0)*p(0);
        t3 += 100*std::cos(10*p(i)+5*p(0)*p(0))*std::cos(10*p(i)+5*p(0)*p(0)) -
            100*std::sin(10*p(i)+5*p(0)*p(0));
    };
    t1 = t1*t1;
    return -u*(t1+t2+t3);
}
```
然后就是建立整个程序运行的过程：
```cpp
template <int dim>
void
run_simulation (LaplaceSolver::Base<dim> &solver,
        const std::list<Evaluation::EvaluationBase<dim> *> &postprocessor_list)
{
    std::cout << "Refinement cycle: ";
    for (unsigned int step=0; true; ++step)
    {
        std::cout << step << " " << std::flush;
        solver.solve_problem ();
        for (typename std::list<Evaluation::EvaluationBase<dim> *>::const_iterator
                i = postprocessor_list.begin();
                i != postprocessor_list.end(); ++i)
        {
            (*i)->set_refinement_cycle (step);
            solver.postprocess (**i);
        };
        if (solver.n_dofs() < 20000)
            solver.refine_grid ();
        else
            break;
    };
    std::cout << std::endl;
}
template <int dim>
void solve_problem (const std::string &solver_name)
{
    const std::string header = "Running tests with \"" + solver_name +
        "\" refinement criterion:";
    std::cout << header << std::endl
        << std::string (header.size(), '-') << std::endl;
    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (2);
    const FE_Q<dim> fe(1);
    const QGauss<dim> quadrature(4);
    const RightHandSide<dim> rhs_function;
    const Solution<dim> boundary_values;
    LaplaceSolver::Base<dim> *solver = 0;
    if (solver_name == "global")
        solver = new LaplaceSolver::RefinementGlobal<dim> (triangulation, fe,
                quadrature,
                rhs_function,
                boundary_values);
    else if (solver_name == "kelly")
        solver = new LaplaceSolver::RefinementKelly<dim> (triangulation, fe,
                quadrature,
                rhs_function,
                boundary_values);
    else
        AssertThrow (false, ExcNotImplemented());
    TableHandler results_table;
    Evaluation::PointValueEvaluation<dim>
        postprocessor1 (Point<dim>(0.5,0.5), results_table);
    Evaluation::SolutionOutput<dim>
        postprocessor2 (std::string("solution-")+solver_name,
                DataOutBase::gnuplot);
    std::list<Evaluation::EvaluationBase<dim> *> postprocessor_list;
    postprocessor_list.push_back (&postprocessor1);
    postprocessor_list.push_back (&postprocessor2);
    run_simulation (*solver, postprocessor_list);
    results_table.write_text (std::cout);
    delete solver;
    std::cout << std::endl;
}
}
```
然后是main函数：
```cpp
int main ()
{
    try
    {
        Step13::solve_problem<2> ("global");
        Step13::solve_problem<2> ("kelly");
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
某点的值：
```cpp
Running tests with "global" refinement criterion:
-------------------------------------------------
Refinement cycle: 0 1 2 3 4 5 6
DoFs u(x_0)
25 1.2868
81 1.6945
289 1.4658
1089 1.5679
4225 1.5882
16641 1.5932
66049 1.5945
Running tests with "kelly" refinement criterion:
------------------------------------------------
Refinement cycle: 0 1 2 3 4 5 6 7 8 9 10 11
DoFs u(x_0)
25 1.2868
47 0.8775
89 1.5365
165 1.2974
316 1.6442
589 1.5221
1093 1.5724
2042 1.5627
3766 1.5916
7124 1.5876
13111 1.5942
24838 1.5932
```
局部细化9次以后的网格：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjl4qyzqsj30o910eh05.jpg)
从上面看网格是这样的：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjl575tdjj30at0a9glf.jpg)
本例中还有一个问题：局部细化的效果要差于全局细化！全局细化能提供一个与网格尺寸相关的稳定收敛性，而局部细化的收敛性不规则，虽然也是网格越细越收敛，但收敛阶忽高忽低。
原教程中有答案使得局部细化的效果变好。
