---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 7
tags: [deal.II]
categories: simulation
date: 2016-9-2
---

# 引子
在本例中，将会着眼于以下两方面：
1. 验证程序的正确性，生成收敛性统计表格;
2. 对于Helmholtz方程施加非齐次Neumann边界条件。

另外还有一些小的优化点。

## 验证程序正确性
也许从来不会有任何一个有限元程序一开始就是正确的，所以找到方法来验证计算的解是否正确就很有必要。通常选择已知精确解析解，并且比较精确解析解和计算离散解两者之间差别来求证。如果随着误差次数提高，两者之间差别逐渐趋于0，就说明程序的正确性。deal.II中就提供了这样一个函数：VectorTools::integrate_difference()，它提供了很多种范数的计算：
$$
\begin{equation}
\begin{split}
{|| u-u\_h ||}\_{L\_1(K)} &=\int\_K |u-u\_h| dx, \\\
{|| u-u\_h ||}\_{L\_2(K)} &=\left( \int\_K |u-u\_h|^2 dx \right)^{1/2}, \\\
{|| u-u\_h ||}\_{L\_\infty(K)} &=\max\_{x \in K} |u(x) - u\_h(x)|, \\\
{| u-u\_h |}\_{H^1(K)} &=\left( \int\_K |\nabla(u-u\_h)|^2 dx \right)^{1/2}, \\\
{|| u-u\_h ||}\_{H^1(K)} &=\left( {|| u-u\_h ||}^2\_{L\_2(K)} +{| u-u\_h |}^2\_{H^1(K)} \right)^{1/2}
\end{split}
\end{equation}
$$
这些公式也适用于矢量函数。就像其他的积分一样，我们也需要用数值积分公式来计算这些范数，那么合适的积分公式对这些误差的计算就很重要，特别是对$L\_\infty$范数，因为需要在积分点上计算数值解和精确解的最大差别。该函数计算每个单元上的范数，然后返回一个vector存储每个单元上的这些值，从局部的范数，可以得到全局范数，如全局$L\_2$范数为：
$$
E=||e||=(\sum\_i e\_i^2)^{1/2}
$$
在本例中，将会展示怎样计算和使用这些量，同时监控随着网格细化其怎样变化。同时还将展示从得到的数据生成漂亮的表格，来自动计算收敛速率，而且将比较不同策略的网格细化。

## 非齐次Neumann边界条件
非齐次边界条件，即包括边界值及其梯度的条件，它们存在于边界积分中，然后计算时需要被组装进右端项中。具体到本例来说，要求解的方程是Holmholtz方程：
$$
-\Delta u+u=f
$$
计算域是$[-1,1]^2$。边界条件分两部分，在整体边界$\Gamma$的$\Gamma\_1$部分：
$$
u=g\_1
$$
在剩下的$\Gamma\_2$部分：
$$
{\mathbf n}\cdot \nabla u = g\_2
$$
具体边界划分为：$\Gamma\_1=\Gamma \cap ((x=1) \cup (y=1))$。
根据Method of Manufactured Solutions，得到本例的精确解为：
$$
\bar u(x) = \sum\_{i=1}^3 \exp\left(-\frac{|x-x\_i|^2}{\sigma^2}\right)
$$
其中:$ x\_1=(-\frac{1}{2},\frac{1}{2}),x\_2=(-\frac{1}{2},-\frac{1}{2}),x\_3=(\frac{1}{2},-\frac{1}{2}),\sigma=\frac{1}{8}$
弱形式为：
$$
(\nabla u, \nabla v)\_\Omega + (u,v)\_\Omega = (f,v)\_\Omega + (g\_2,v)\_{\Gamma\_2}
$$
其中，边界积分项$(g\_2,v)\_{\Gamma\_2}$已经考虑了在$\Gamma\_1$上$v=0$。
离散后单元上的矩阵和向量的形式为：
$$
\begin{equation}
\begin{split}
A\_{ij}^K &=\left(\nabla \varphi\_i, \nabla \varphi\_j\right)\_K +\left(\varphi\_i, \varphi\_j\right)\_K, \\\
f\_i^K &=\left(f,\varphi\_i\right)\_K +\left(g\_2, \varphi\_i\right)\_{\partial K\cap \Gamma\_2}
\end{split}
\end{equation}
$$
对于区域积分，之前已经有了很多介绍，就是用FEValues类来给出单元上的形函数的值及其梯度，以及Jacobian行列式及积分点等。而相对应地，对于边界上曲线积分，用FEFaceValues来做以上工作，只不过它的维度比domain要小1。

## 一个良好的编程习惯
一个良好的编程习惯就是使用命名空间，这样可以有效地预防命名冲突。格式为：
```cpp
... #includes
namespace Step7
{
    using namespace dealii;
    ...everything to do with the program...
}
int main ()
{
    ...do whatever main() does...
}
```

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
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
```
以上头文件不解释。
```cpp
#include <deal.II/dofs/dof_renumbering.h>
```
这里使用Cuthill-McKee算法对自由度重新排号。
```cpp
#include <deal.II/base/smartpointer.h>
```
以上头文件保证对象在被使用时不被删除。
```cpp
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
```
第一个头文件包含了VectorTools::integrate_difference()函数，第二个则是生成表格用。
```cpp
#include <deal.II/fe/fe_values.h>
```
还要使用FEValues类。
```cpp
#include <typeinfo>
#include <fstream>
#include <iostream>
```
下面开启step7命名空间，同时引入dealii空间：
```cpp
namespace Step7
{
    using namespace dealii;
```
首先创建一个类来存储精确解，这里把它作成一个基类，是为了以后跟右端项分享一些相同的特征(因为此例中右端项就是解的组合)：
```cpp
template <int dim>
class SolutionBase
{
    protected:
        static const unsigned int n_source_centers = 3;
        static const Point<dim> source_centers[n_source_centers];
        static const double width;
};
```
其特征包括三项：指数项的个数及其中心及其半宽度。此类与维度无关，先看它怎样对一维实现：
```cpp
template <>
const Point<1>
SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers]
= { Point<1>(-1.0 / 3.0),
    Point<1>(0.0),
    Point<1>(+1.0 / 3.0)
};
```
这里涉及模板显式特化语法。
二维是：
```cpp
template <>
const Point<2>
SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers]
= { Point<2>(-0.5, +0.5),
    Point<2>(-0.5, -0.5),
    Point<2>(+0.5, -0.5)
};
```
然后设定半宽度：
```cpp
template <int dim>
const double SolutionBase<dim>::width = 1./8.;
```
在声明和定义了右端项和解的特征以后，就需要真正声明这两个类了。它们都代表了连续函数，因此继承自Function<dim>基类，同时也继承上面的SolutionBase类。
```cpp
template <int dim>
class Solution : public Function<dim>,
    protected SolutionBase<dim>
{
    public:
        Solution () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
        virtual Tensor<1,dim> gradient (const Point<dim> &p,
                const unsigned int component = 0) const;
};
```
注意：为了计算离散解和连续解的误差，就必须提供精确解的值和梯度。Function类提供了关于值和梯度的虚函数，所以要做的就是对相应的虚函数进行重载。再次注意：在dim维空间的函数，它的梯度是具有dim维的一阶张量，如上所示。
值和梯度的计算如下：
```cpp
template <int dim>
double Solution<dim>::value (const Point<dim> &p,
        const unsigned int) const
{
    double return_value = 0;
    for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
        const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
        return_value += std::exp(-x_minus_xi.norm_square() /
                (this->width * this->width));
    }
    return return_value;
}
template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p,
        const unsigned int) const
{
    Tensor<1,dim> return_value;
    for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
        const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

        return_value += (-2 / (this->width * this->width) *
                std::exp(-x_minus_xi.norm_square() /
                    (this->width * this->width)) *
                x_minus_xi);
    }
    return return_value;
}
```
除了精确解，还需要一个右端项函数来组装离散方程的线性系统：
```cpp
template <int dim>
class RightHandSide : public Function<dim>,
    protected SolutionBase<dim>
{
    public:
        RightHandSide () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
};
template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
        const unsigned int) const
{
    double return_value = 0;
    for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
        const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
        return_value += ((2*dim - 4*x_minus_xi.norm_square()/
                    (this->width * this->width)) /
                (this->width * this->width) *
                std::exp(-x_minus_xi.norm_square() /
                    (this->width * this->width)));
        return_value += std::exp(-x_minus_xi.norm_square() /
                (this->width * this->width));
    }
    return return_value;
}
```
这里只用到它的值，用不着计算梯度。具体计算时解是由两部分构成：解的负laplace项和解本身。
然后就是求解这个问题的类了。它的界面跟之前的例子大体相同，但是有以下几点不同：
(1)用于不同的有限单元;(2)既可以自适应细化，也可以全局细化，具体怎样细化是在构造函数中判断。同时还有分析各种误差的
函数。
```cpp
template <int dim>
class HelmholtzProblem
{
    public:
        enum RefinementMode
        {
            global_refinement, adaptive_refinement
        };
        HelmholtzProblem (const FiniteElement<dim> &fe,
                const RefinementMode refinement_mode);
        ~HelmholtzProblem ();
        void run ();
    private:
        void setup_system ();
        void assemble_system ();
        void solve ();
        void refine_grid ();
        void process_solution (const unsigned int cycle);
```
下面是类的成员变量：
```cpp
Triangulation<dim> triangulation;
DoFHandler<dim> dof_handler;
SmartPointer<const FiniteElement<dim> > fe;
ConstraintMatrix hanging_node_constraints;
SparsityPattern sparsity_pattern;
SparseMatrix<double> system_matrix;
Vector<double> solution;
Vector<double> system_rhs;
```
其中比较特殊的是有限单元对象fe。从上面的类的构造函数可以看出，fe是传给它作为参数的。
考虑在所有程序中都会出现的情况：我们有一个triangulation对象，也有一个fe对象，当然也有一个同时使用它俩的DoFHandler对象。明显这三个对象的寿命要比其他对象要长。但问题是：我们能保证triangulation和fe的寿命足够长来供DoFHandler使用吗？这意味着DoFHandler要对这两者施加某些锁，只有在它已经清除了所有对这两者的使用后才能释放这些锁。正如step6所示，如果违反，则有异常抛出。
我们将要展示库是怎样找到是否还有对对象的使用的。过程大体是这样的：所有可能置于这些有潜在危险指针之下的对象都派生自Subscriptor类，比如Triangulation类、DoFHandler类、FiniteElement类。这个类不提供很多功能，但它有一个内置的计数器。一旦我们初始化一个指向该对象的指针，该计数器就加1，当移除指针或不再需要它时，就减1，这样就能检查还有多少对象仍然使用该对象。
另一方面，如果一个派生自Subscriptor类的类的对象销毁了，它也必须调用Subscriptor的析构函数。在这个析构函数中，也将检查那个计数器是否为0，如果是，那么就没有对该对象的引用，那么我们就可以安全地销毁它，否则，就会产生危险的指针，库就抛出一个异常来提醒程序员检查代码。
上面一切听起来都挺美好，但在使用上有一些问题：万一我忘了对计数器加1怎么办？万一我又忘了减1呢？这在调试程序时会很麻烦。解决这个问题的方法是使用C++的一个特性：SmartPointer智能指针。我们创建的类的对象让它就像一个指针一样。正如上面程序中fe的定义一样。
还有一个变量是存储细化方式，是一个枚举常量：
```cpp
const RefinementMode refinement_mode;
```
另一个变量是收敛性表格：
```cpp
ConvergenceTable convergence_table;
};
```
类的构造函数：
```cpp
template <int dim>
HelmholtzProblem<dim>::HelmholtzProblem (const FiniteElement<dim> &fe,
        const RefinementMode refinement_mode) :
    dof_handler (triangulation),
    fe (&fe),
    refinement_mode (refinement_mode)
{}
```
类的析构函数：
```cpp
template <int dim>
HelmholtzProblem<dim>::~HelmholtzProblem ()
{
    dof_handler.clear ();
}
```
建立系统：
```cpp
template <int dim>
void HelmholtzProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (*fe);
    DoFRenumbering::Cuthill_McKee (dof_handler);
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
            hanging_node_constraints);
    hanging_node_constraints.close ();
    DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    hanging_node_constraints.condense (dsp);
    sparsity_pattern.copy_from (dsp);
    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}
```
这里使用了算法对自由度序号重排，同时又有悬点问题，所以注意上面代码的顺序。
组装系统：
```cpp
template <int dim>
void HelmholtzProblem<dim>::assemble_system ()
{
    QGauss<dim> quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
```
跟之前不同的是，因为需要计算边界积分，所以需要声明边界积分公式。
```cpp
FEValues<dim> fe_values (*fe, quadrature_formula,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);
FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
        update_values | update_quadrature_points |
        update_normal_vectors | update_JxW_values);
```
然后是计算积分点上形函数的值、梯度，这些量需要在单元内部和边界上都得计算，两者有一个很大的差别，即单元内部的积分的权重需要测量单元，而边界积分需要在更低维度的流形上测量边界，无论如何，两者的界面是差不多的。注意：内部积分用的是FEValues类，这里需要计算积分点上的值、梯度、权重等，而边界积分用的是FEFaceValues，计算的是积分点上的形函数的值、权重，因为还要计算Neumann边值，所以还要计算法向量。
```cpp
const RightHandSide<dim> right_hand_side;
std::vector<double> rhs_values (n_q_points);
const Solution<dim> exact_solution;
```
然后是存储右端项和精确解的对象。
```cpp
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
     endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit (cell);
    right_hand_side.value_list (fe_values.get_quadrature_points(),
            rhs_values);
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += ((fe_values.shape_grad(i,q_point) *
                            fe_values.shape_grad(j,q_point)
                            +
                            fe_values.shape_value(i,q_point) *
                            fe_values.shape_value(j,q_point)) *
                        fe_values.JxW(q_point));
            cell_rhs(i) += (fe_values.shape_value(i,q_point) *
                    rhs_values [q_point] *
                    fe_values.JxW(q_point));
        }
```
以上是对每个单元的循环，单元刚度矩阵中已经根据Holmholtz方程进行了调整。同时上面的右端项的计算仅仅只包含了一项，下面是右端项的第二部分边界积分：
```cpp
for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
    if (cell->face(face_number)->at_boundary()
            &&
            (cell->face(face_number)->boundary_id() == 1))
    {
```
首先得找到$\Gamma\_2$的边界：
```cpp
for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
    if (cell->face(face_number)->at_boundary()
            &&
            (cell->face(face_number)->boundary_id() == 1))
    {
```
这里是判断单元的边界的标识是不是1，我们知道边界默认标识是0，而在后面的run函数中将$\Gamma\_2$人为指定成1。如果确认是它，那么就计算形函数的值，这通过reinit才实现，跟FEValues一样：
```cpp
fe_face_values.reinit (cell, face_number);
```
然后就是对积分点的循环：
```cpp
for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
{
    const double neumann_value
        = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                fe_face_values.normal_vector(q_point));
    for (unsigned int i=0; i<dofs_per_cell; ++i)
        cell_rhs(i) += (neumann_value *
                fe_face_values.shape_value(i,q_point) *
                fe_face_values.JxW(q_point));
}
}
```
其中，法向导数的值是根据精确解的梯度和法向量的乘积计算得到。
然后就是组装系统：
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
施加边界条件：
```cpp
hanging_node_constraints.condense (system_matrix);
hanging_node_constraints.condense (system_rhs);
std::map<types::global_dof_index,double> boundary_values;
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        Solution<dim>(),
        boundary_values);
MatrixTools::apply_boundary_values (boundary_values,
        system_matrix,
        solution,
        system_rhs);
}
```
注意：上面的边界中只包含了$\Gamma\_1$，这正是我们想要的。
求解：
```cpp
template <int dim>
void HelmholtzProblem<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> cg (solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    cg.solve (system_matrix, solution, system_rhs,
            preconditioner);
    hanging_node_constraints.distribute (solution);
}
```
然后就是细化网格。根据传递给构造函数的参数决定是自适应细化还是全局细化。
```cpp
template <int dim>
void HelmholtzProblem<dim>::refine_grid ()
{
    switch (refinement_mode)
    {
        case global_refinement:
            {
                triangulation.refine_global (1);
                break;
            }
        case adaptive_refinement:
            {
                Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
                KellyErrorEstimator<dim>::estimate (dof_handler,
                        QGauss<dim-1>(3),
                        typename FunctionMap<dim>::type(),
                        solution,
                        estimated_error_per_cell);
                GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                        estimated_error_per_cell,
                        0.3, 0.03);
                triangulation.execute_coarsening_and_refinement ();
                break;
            }
        default:
            {
                Assert (false, ExcNotImplemented());
            }
    }
}
```
细化方案跟之前相同，不多说，注意最后的缺省情形不要忘加。
下一步就是对解的处理：
```cpp
template <int dim>
void HelmholtzProblem<dim>::process_solution (const unsigned int cycle)
{
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
            solution,
            Solution<dim>(),
            difference_per_cell,
            QGauss<dim>(3),
            VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
```
首先是计算误差范数。创建一个Vector来存放每个单元上的误差值。然后计算L2范数，接收的参数是DoFHandler对象、数值解的节点值、精确解、存放每个单元上的误差值的量、计算该范数的积分公式、范数类型。
然后计算H1范数：
```cpp
ectorTools::integrate_difference (dof_handler,
        solution,
        Solution<dim>(),
        difference_per_cell,
        QGauss<dim>(3),
        VectorTools::H1_seminorm);
const double H1_error = difference_per_cell.l2_norm();
```
然后计算最大范数，当然也是在积分点上的最大范数，不可能是全局最大范数：
```cpp
const QTrapez<1> q_trapez;
const QIterated<dim> q_iterated (q_trapez, 5);
VectorTools::integrate_difference (dof_handler,
        solution,
        Solution<dim>(),
        difference_per_cell,
        q_iterated,
        VectorTools::Linfty_norm);
const double Linfty_error = difference_per_cell.linfty_norm();
```
然后将所有结果输出到表格中：
```cpp
const unsigned int n_active_cells=triangulation.n_active_cells();
const unsigned int n_dofs=dof_handler.n_dofs();
std::cout << "Cycle " << cycle << ':'
    << std::endl
    << " Number of active cells: "
    << n_active_cells
    << std::endl
    << " Number of degrees of freedom: "
    << n_dofs
    << std::endl;
convergence_table.add_value("cycle", cycle);
convergence_table.add_value("cells", n_active_cells);
convergence_table.add_value("dofs", n_dofs);
convergence_table.add_value("L2", L2_error);
convergence_table.add_value("H1", H1_error);
convergence_table.add_value("Linfty", Linfty_error);
}
```
接下来是run函数，控制程序的运行过程。与之前不同的是，需要先设定好边界标识，这里是根据坐标值来确定。而且是对所有单元循环，不仅仅是活动单元，这是因为细化时子网格会继承父网格的边界标识，如果仅细化活动单元，之前定义的边界就继承不下来。当然也可以在细化之前对最初的粗网格进行标识，然后再细化。
```cpp
template <int dim>
void HelmholtzProblem<dim>::run ()
{
    const unsigned int n_cycles = (refinement_mode==global_refinement)?5:9;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (3);
            typename Triangulation<dim>::cell_iterator
                cell = triangulation.begin (),
                     endc = triangulation.end();
            for (; cell!=endc; ++cell)
                for (unsigned int face_number=0;
                        face_number<GeometryInfo<dim>::faces_per_cell;
                        ++face_number)
                    if ((std::fabs(cell->face(face_number)->center()(0) - (-1)) < 1e-12)
                            ||
                            (std::fabs(cell->face(face_number)->center()(1) - (-1)) < 1e-12))
                        cell->face(face_number)->set_boundary_id (1);
        }
        else
            refine_grid ();
        setup_system ();
        assemble_system ();
        solve ();

        process_solution (cycle);
    }
```
在最后一步迭代后，输出最细网格上的解。输出文件根据细化方式、单元类型来命名：
```cpp
std::string vtk_filename;
switch (refinement_mode)
{
    case global_refinement:
        vtk_filename = "solution-global";
        break;
    case adaptive_refinement:
        vtk_filename = "solution-adaptive";
        break;
    default:
        Assert (false, ExcNotImplemented());
}
switch (fe->degree)
{
    case 1:
        vtk_filename += "-q1";
        break;
    case 2:
        vtk_filename += "-q2";
        break;
    default:
        Assert (false, ExcNotImplemented());
}
vtk_filename += ".vtk";
std::ofstream output (vtk_filename.c_str());
DataOut<dim> data_out;
data_out.attach_dof_handler (dof_handler);
data_out.add_data_vector (solution, "solution");
```
下面就是建立中间格式的数据。跟以前不同的是，我们这里有时会使用双二次单元。但大多数的输出格式仅支持双线性数据，如果强行转换就会丢失部分数据。当然我们不能改变图像处理程序的输入文件的格式，但可以变着花样写出来。比如把每个单元分成有双线性数据的四个单元。
```cpp
data_out.build_patches (fe->degree);
data_out.write_vtk (output);
```
build_patches接收一个参数，表明每个单元的单个方向上应该划分成几个子单元。
然后是输出误差表格：
```cpp
convergence_table.set_precision("L2", 3);
convergence_table.set_precision("H1", 3);
convergence_table.set_precision("Linfty", 3);
convergence_table.set_scientific("L2", true);
convergence_table.set_scientific("H1", true);
convergence_table.set_scientific("Linfty", true);
convergence_table.set_tex_caption("cells", "\\# cells");
convergence_table.set_tex_caption("dofs", "\\# dofs");
convergence_table.set_tex_caption("L2", "L^2-error");
convergence_table.set_tex_caption("H1", "H^1-error");
convergence_table.set_tex_caption("Linfty", "L^\\infty-error");
convergence_table.set_tex_format("cells", "r");
convergence_table.set_tex_format("dofs", "r");
std::cout << std::endl;
convergence_table.write_text(std::cout);
std::string error_filename = "error";
switch (refinement_mode)
{
    case global_refinement:
        error_filename += "-global";
        break;
    case adaptive_refinement:
        error_filename += "-adaptive";
        break;
    default:
        Assert (false, ExcNotImplemented());
}
switch (fe->degree)
{
    case 1:
        error_filename += "-q1";
        break;
    case 2:
        error_filename += "-q2";
        break;
    default:
        Assert (false, ExcNotImplemented());
}
error_filename += ".tex";
std::ofstream error_table_file(error_filename.c_str());
convergence_table.write_tex(error_table_file);
```
这里面包含了输出成TeX的格式。
对于全局细化的话，还可以输出收敛速率：
```cpp
if (refinement_mode==global_refinement)
{
    convergence_table.add_column_to_supercolumn("cycle", "n cells");
    convergence_table.add_column_to_supercolumn("cells", "n cells");
    std::vector<std::string> new_order;
    new_order.push_back("n cells");
    new_order.push_back("H1");
    new_order.push_back("L2");
    convergence_table.set_column_order (new_order);
    convergence_table
        .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
    convergence_table
        .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
    convergence_table
        .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
    convergence_table
        .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
    std::string conv_filename = "convergence";
    switch (refinement_mode)
    {
        case global_refinement:
            conv_filename += "-global";
            break;
        case adaptive_refinement:
            conv_filename += "-adaptive";
            break;
        default:
            Assert (false, ExcNotImplemented());
    }
    switch (fe->degree)
    {
        case 1:
            conv_filename += "-q1";
            break;
        case 2:
            conv_filename += "-q2";
            break;
        default:
            Assert (false, ExcNotImplemented());
    }
    conv_filename += ".tex";
    std::ofstream table_file(conv_filename.c_str());
    convergence_table.write_tex(table_file);
}
}
}
```
然后就是main函数：
```cpp
int main ()
{
    const unsigned int dim = 2;
    try
    {
        using namespace dealii;
        using namespace Step7;
        {
            std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl
                << "=============================================" << std::endl
                << std::endl;
            FE_Q<dim> fe(1);
            HelmholtzProblem<dim>
                helmholtz_problem_2d (fe, HelmholtzProblem<dim>::adaptive_refinement);
            helmholtz_problem_2d.run ();
            std::cout << std::endl;
        }
        {
            std::cout << "Solving with Q1 elements, global refinement" << std::endl
                << "===========================================" << std::endl
                << std::endl;
            FE_Q<dim> fe(1);
            HelmholtzProblem<dim>
                helmholtz_problem_2d (fe, HelmholtzProblem<dim>::global_refinement);
            helmholtz_problem_2d.run ();
            std::cout << std::endl;
        }
        {
            std::cout << "Solving with Q2 elements, global refinement" << std::endl
                << "===========================================" << std::endl
                << std::endl;
            FE_Q<dim> fe(2);
            HelmholtzProblem<dim>
                helmholtz_problem_2d (fe, HelmholtzProblem<dim>::global_refinement);
            helmholtz_problem_2d.run ();
            std::cout << std::endl;
        }
        {
            std::cout << "Solving with Q2 elements, adaptive refinement" << std::endl
                << "===========================================" << std::endl
                << std::endl;
            FE_Q<dim> fe(2);
            HelmholtzProblem<dim>
                helmholtz_problem_2d (fe, HelmholtzProblem<dim>::adaptive_refinement);
            helmholtz_problem_2d.run ();
            std::cout << std::endl;
        }
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
    }
    return 0;
}
```
# 计算结果

以下是使用双二次单元的自适应计算的结果：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk8u4xgkj30qx0k50tt.jpg)
收敛性结果如下：
```cpp
Solving with Q1 elements, adaptive refinement
=============================================
Cycle 0:
Number of active cells: 64
Number of degrees of freedom: 81
Cycle 1:
Number of active cells: 124
Number of degrees of freedom: 157
Cycle 2:
Number of active cells: 280
Number of degrees of freedom: 341
Cycle 3:
Number of active cells: 577
Number of degrees of freedom: 690
Cycle 4:
Number of active cells: 1099
Number of degrees of freedom: 1264
Cycle 5:
Number of active cells: 2191
Number of degrees of freedom: 2452
Cycle 6:
Number of active cells: 4165
Number of degrees of freedom: 4510
Cycle 7:
Number of active cells: 7915
Number of degrees of freedom: 8440
Cycle 8:
Number of active cells: 15196
Number of degrees of freedom: 15912
cycle cells dofs L2 H1 Linfty
0 64 81 1.576e-01 1.418e+00 2.707e-01
1 124 157 4.285e-02 1.285e+00 1.469e-01
2 280 341 1.593e-02 7.909e-01 8.034e-02
3 577 690 9.359e-03 5.096e-01 2.784e-02
4 1099 1264 2.865e-03 3.038e-01 9.822e-03
5 2191 2452 1.480e-03 2.106e-01 5.679e-03
6 4165 4510 6.907e-04 1.462e-01 2.338e-03
7 7915 8440 4.743e-04 1.055e-01 1.442e-03
8 15196 15912 1.920e-04 7.468e-02 7.259e-04
Solving with Q1 elements, global refinement
===========================================
Cycle 0:
Number of active cells: 64
Number of degrees of freedom: 81
Cycle 1:
Number of active cells: 256
Number of degrees of freedom: 289
Cycle 2:
Number of active cells: 1024
Number of degrees of freedom: 1089
Cycle 3:
Number of active cells: 4096
Number of degrees of freedom: 4225
Cycle 4:
Number of active cells: 16384
Number of degrees of freedom: 16641
cycle cells dofs L2 H1 Linfty
0 64 81 1.576e-01 1.418e+00 2.707e-01
1 256 289 4.280e-02 1.285e+00 1.444e-01
2 1024 1089 1.352e-02 7.556e-01 7.772e-02
3 4096 4225 3.423e-03 3.822e-01 2.332e-02
4 16384 16641 8.586e-04 1.917e-01 6.097e-03
n cells H1 L2
0 64 1.418e+00 - - 1.576e-01 - -
1 256 1.285e+00 1.10 0.14 4.280e-02 3.68 1.88
2 1024 7.556e-01 1.70 0.77 1.352e-02 3.17 1.66
3 4096 3.822e-01 1.98 0.98 3.423e-03 3.95 1.98
4 16384 1.917e-01 1.99 1.00 8.586e-04 3.99 2.00
Solving with Q2 elements, global refinement
===========================================
Cycle 0:
Number of active cells: 64
Number of degrees of freedom: 289
Cycle 1:
Number of active cells: 256
Number of degrees of freedom: 1089
Cycle 2:
Number of active cells: 1024
Number of degrees of freedom: 4225
Cycle 3:
Number of active cells: 4096
Number of degrees of freedom: 16641
Cycle 4:
Number of active cells: 16384
Number of degrees of freedom: 66049
cycle cells dofs L2 H1 Linfty
0 64 289 1.606e-01 1.278e+00 3.029e-01
1 256 1089 7.638e-03 5.248e-01 4.816e-02
2 1024 4225 8.601e-04 1.086e-01 4.827e-03
3 4096 16641 1.107e-04 2.756e-02 7.802e-04
4 16384 66049 1.393e-05 6.915e-03 9.971e-05
n cells H1 L2
0 64 1.278e+00 - - 1.606e-01 - -
1 256 5.248e-01 2.43 1.28 7.638e-03 21.03 4.39
2 1024 1.086e-01 4.83 2.27 8.601e-04 8.88 3.15
3 4096 2.756e-02 3.94 1.98 1.107e-04 7.77 2.96
4 16384 6.915e-03 3.99 1.99 1.393e-05 7.94 2.99
Solving with Q2 elements, adaptive refinement
===========================================
Cycle 0:
Number of active cells: 64
Number of degrees of freedom: 289
Cycle 1:
Number of active cells: 124
Number of degrees of freedom: 577
Cycle 2:
Number of active cells: 289
Number of degrees of freedom: 1353
Cycle 3:
Number of active cells: 547
Number of degrees of freedom: 2531
Cycle 4:
Number of active cells: 1057
Number of degrees of freedom: 4919
Cycle 5:
Number of active cells: 2059
Number of degrees of freedom: 9223
Cycle 6:
Number of active cells: 3913
Number of degrees of freedom: 17887
Cycle 7:
Number of active cells: 7441
Number of degrees of freedom: 33807
Cycle 8:
Number of active cells: 14212
Number of degrees of freedom: 64731
cycle cells dofs L2 H1 Linfty
0 64 289 1.606e-01 1.278e+00 3.029e-01
1 124 577 7.891e-03 5.256e-01 4.852e-02
2 289 1353 1.070e-03 1.155e-01 4.868e-03
3 547 2531 5.962e-04 5.101e-02 1.876e-03
4 1057 4919 1.977e-04 3.094e-02 7.923e-04
5 2059 9223 7.738e-05 1.974e-02 7.270e-04
6 3913 17887 2.925e-05 8.772e-03 1.463e-04
7 7441 33807 1.024e-05 4.121e-03 8.567e-05
8 14212 64731 3.761e-06 2.108e-03 2.167e-05 
```

# 进一步扩展
## 更高阶的单元
如果使用更高阶的单元，如Q3、Q4，可能就会触发一些异常，比如文件保存阶段。即使把这些错误修正了，也不能产生理论预测的正确的收敛结果，这是因为积分公式的次数不够，而这是在程序中硬编码的。那么如何动态地选择这个次数呢？
## 收敛性对比
Q1单元和Q2哪个更好？自适应细化和全局细化哪个更好？
注意：峰的半宽影响自适应或全局细化哪个更好。如果解足够光滑，那么局部细化比全局细化没有优势。

