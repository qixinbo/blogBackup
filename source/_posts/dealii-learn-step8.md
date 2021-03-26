---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 8
tags: [deal.II]
categories: simulation
date: 2016-9-6
---

# 引子
真实生活中，大多数的偏微分方程都是一组方程，相应地，解也通常是矢量场。跟之前单方程的求解以及解是标量场相比，这种问题只是在组装矩阵和右端项时有些复杂，其他都一样。
本例中求解的是弹性问题：
$$
-\partial\_j (c\_{ijkl} \partial\_k u\_l) = f\_i, \qquad i=1 \ldots d
$$
其中，$c\_{ijkl}$是刚度系数，通常与空间坐标相关。
弹性方程是Laplace方程的一种扩散形式，其解$u\_l$是矢量场，表示一个刚体在力的作用下的位移。力$f\_i$也是矢量场。
$c\_{ijkl}$是一个四阶张量，共有$3^4=81$个分量，但其实只有21个分量是独立的，这是因为：
首先因为$\sigma\_{ij}$和$\epsilon\_{ij}$都是对称张量，导致$c\_{ijkl}=c\_{jikl}$和$c\_{ijkl}=c\_{ijlk}$，这样就将独立的弹性常数减少到$(3\*2\*1)^2=36$个。
然后因为应变能密度函数与应力应变的关系，导致$c\_{ijkl}=c\_{klij}$，这样代表沿对角线对称，这样将弹性常数减少到21个。
这样三维情形下应力应变关系矩阵形式为：
$$
\begin{bmatrix}
\sigma\_{11} \\\
\sigma\_{22} \\\
\sigma\_{33} \\\
\sigma\_{12} \\\
\sigma\_{23} \\\
\sigma\_{31} 
\end{bmatrix}
=
\begin{bmatrix}
c\_{1111} & c\_{1122} & c\_{1133} & c\_{1112} & c\_{1123} & c\_{1131} \\\
          & c\_{2222} & c\_{2233} & c\_{2212} & c\_{2223} & c\_{2231} \\\
          &           & c\_{3333} & c\_{3312} & c\_{3323} & c\_{3331} \\\
          &           &           & c\_{1212} & c\_{1223} & c\_{1231} \\\
          &           &           &           & c\_{2323} & c\_{2331} \\\
          &           &           &           &           & c\_{3131} \\\
\end{bmatrix}
\begin{bmatrix}
\epsilon\_{11} \\\
\epsilon\_{22} \\\
\epsilon\_{33} \\\
2\epsilon\_{12} \\\
2\epsilon\_{23} \\\
2\epsilon\_{31} \\\
\end{bmatrix}
$$

在各向同性材料中，通过引入两个系数，系数张量变成(可以通过理论证明，各向同性的均匀弹性体的弹性常数只有两个)：
$$
c\_{ijkl} = \lambda \delta\_{ij} \delta\_{kl} + \mu (\delta\_{ik} \delta\_{jl} + \delta\_{il} \delta\_{jk}).
$$
比如，当$i=j=k=l=1$时：
$$
c\_{1111} = \lambda + 2\mu
$$
当$i=j=k=1,l=2$时，
$$
c\_{1112} = 0 
$$
当$i=k=1,j=l=2$时，
$$
c\_{1212} = \mu
$$
当$i=j=1,k=l=2$时，
$$
c\_{1122} = \lambda
$$
这就是各向同性材料的应力应变关系矩阵：
$$
\begin{bmatrix}
\lambda+2\mu & \lambda      & \lambda      & 0   & 0   & 0   \\\
             & \lambda+2\mu & \lambda      & 0   & 0   & 0   \\\
             &              & \lambda+2\mu & 0   & 0   & 0   \\\
             &              &              & \mu & 0   & 0   \\\
             &              &              &     & \mu & 0   \\\
             &              &              &     &     & \mu 
\end{bmatrix}
$$
回到力与位移的系数矩阵，按各个字母的序号循环可得：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk3ow7k0j32eo37k4qq.jpg)
它的系数矩阵是一个$2\*2$的矩阵，其中各个元素可由上图得到。
对于各向同性材料，代入上面的取值后，有：
$$
-\nabla\lambda(\nabla\cdot{\mathbf u})-(\nabla\cdot\mu\nabla){\mathbf u}-\nabla\cdot\mu (\nabla{\mathbf u})^T = {\mathbf f},
$$
其展开就是如图：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk60wib7j32eo37khdu.jpg)
注意各种标量、矢量和张量的梯度和散度运算。
其弱形式为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk70cfz7j32eo37ke82.jpg)

下面就是怎样组装这个线性系统。第一件事情就是需要知道在矢量值的有限元中形函数是怎样工作的。大体过程这样：设$n$为要建立的矢量单元的分量，即标量单元，的形函数的个数，比如之前用的双线性单元，二维情形下$n=4$。设$N$是矢量单元的形函数的个数，二维情形下，$N=2n$，那么矢量单元的第i个形函数的形式为：
$$
\Phi\_i({\mathbf x})=\varphi\_{\text{base}(i)}({\mathbf x}){\mathbf e}\_{\text{comp}(i)},
$$
其中，$\text{comp}(i)$是告诉我们$\Phi\_i$的哪个分量非0(对每一个矢量形函数，只有一个分量非0，其余分量都为0)，比如$\text{comp}(1)=0$表示第1个矢量形函数的第0个分量非0。$\varphi\_{\text{base}(i)}(x)$描述形函数与坐标的关系，就是标量单元的第$\text{base}(i)$个形函数，比如具体形式是这样的：
$$
\begin{equation}
\begin{split}
\Phi\_0({\mathbf x}) &= 
\begin{bmatrix}
\varphi\_0({\mathbf x}) \\\
0 
\end{bmatrix}, \\\
\Phi\_1({\mathbf x}) &=
\begin{bmatrix}
0 \\\ 
\varphi\_0({\mathbf x}) 
\end{bmatrix}, \\\
\Phi\_2({\mathbf x}) &= 
\begin{bmatrix}
\varphi\_1({\mathbf x}) \\\
0 
\end{bmatrix}, \\\
\Phi\_3({\mathbf x}) &=
\begin{bmatrix}
0 \\\ 
\varphi\_1({\mathbf x}) 
\end{bmatrix}, ...
\end{split}
\end{equation}
$$
其中：
$$
\text{comp}(0)=0, \quad \text{comp}(1)=1, \quad \text{comp}(2)=0, \quad \text{comp}(3)=1, \quad \ldots \\\
\text{base}(0)=0, \quad \text{base}(1)=0, \quad \text{base}(2)=1, \quad \text{base}(3)=1, \quad \ldots
$$
在绝大多数情况下，不需要知道哪个$\varphi\_{\text{base}(i)}$属于$\Phi\_i$，于是定义：
$$
\phi\_i = \varphi\_{\text{base}(i)}
$$
所以，矢量形函数表示为：
$$
\Phi\_i({\mathbf x}) = \phi\_{i}({\mathbf x}){\mathbf e}\_{\text{comp}(i)}.
$$
使用上述矢量形函数，构造出离散有限元解：
$$
{\mathbf u}\_h({\mathbf x}) = \sum\_i \Phi\_i({\mathbf x})\ U\_i
$$
其中，$U\_i$是系数，是标量。定义一个类似的函数${\mathbf v}\_h$作为试探函数，那么问题就变为：找到系数$U\_i$，使得：
$$
a({\mathbf u}\_h, {\mathbf v}\_h) = ({\mathbf f}, {\mathbf v}\_h) \qquad \forall {\mathbf v}\_h.
$$
将双线性的具体形式代入，可得：
$$
\sum\_{i,j} U\_i V\_j \sum\_{k,l}[ \left( \lambda \partial\_l (\Phi\_i)\_l, \partial\_k (\Phi\_j)\_k \right)\_\Omega + \left( \mu \partial\_l (\Phi\_i)\_k, \partial\_l (\Phi\_j)\_k \right)\_\Omega+\left( \mu \partial\_l (\Phi\_i)\_k, \partial\_k (\Phi\_j)\_l \right)\_\Omega ] \\\
= \sum\_j V\_j \sum\_l \left( f\_l, (\Phi\_j)\_l \right)\_\Omega. 
$$
注意到：下标k和l是对所有空间方向进行循环，$0\le k,l < d$，而下标i和j是对所有自由度进行循环。
那么，单元K上的单元刚度矩阵就是：
$$
A\_{ij}^K= \sum\_{k,l}[ \left( \lambda \partial\_l (\Phi\_i)\_l, \partial\_k (\Phi\_j)\_k \right)\_K + \left( \mu \partial\_l (\Phi\_i)\_k, \partial\_l (\Phi\_j)\_k \right)\_K + \left( \mu \partial\_l (\Phi\_i)\_k, \partial\_k (\Phi\_j)\_l \right)\_K]
$$
这里，i和j是局部自由度，有$0\le i,j < N$。
在这些公式中，我们通常取矢量形函数的部分分量，根据定义，有：
$$
(\Phi\_i)\_l = \phi\_i \delta\_{l,\text{comp}(i)},
$$
那么，进一步简化得到：
$$
\begin{equation}
\begin{split}
A^K\_{ij} &=\sum\_{k,l}[ \left(\lambda\partial\_l \phi\_i\delta\_{l,\text{comp}(i)}, \partial\_k \phi\_j\delta\_{k,\text{comp}(j)} \right)\_K
+\left(\mu\partial\_l \phi\_i\delta\_{k,\text{comp}(i)}, \partial\_l \phi\_j\delta\_{k,\text{comp}(j)} \right)\_K + \left(\mu\partial\_l \phi\_i\delta\_{k,\text{comp}(i)},\partial\_k \phi\_j\delta\_{l,\text{comp}(j)} \right)\_K ] \\\
&= \left(\lambda\partial\_{\text{comp}(i)} \phi\_i, \partial\_{\text{comp}(j)}\phi\_j \right)\_K + \sum\_l \left(\mu\partial\_l \phi\_i, \partial\_l \phi\_j \right)\_K \delta\_{\text{comp}(i),\text{comp}(j)} + \left(\mu\partial\_{\text{comp}(j)} \phi\_i, \partial\_{\text{comp}(i)} \phi\_j \right)\_K \\\
&= \left( \lambda \partial\_{\text{comp}(i)} \phi\_i, \partial\_{\text{comp}(j)} \phi\_j \right)\_K + \left( \mu \nabla \phi\_i, \nabla \phi\_j \right)\_K \delta\_{\text{comp}(i),\text{comp}(j)} + \left( \mu \partial\_{\text{comp}(j)} \phi\_i, \partial\_{\text{comp}(i)} \phi\_j \right)\_K.
\end{split}
\end{equation}
$$
同样地，单元对右端项的贡献为：
$$
\begin{equation}
\begin{split}
f^K\_j &=\sum\_l \left( f\_l, (\Phi\_j)\_l \right)\_K \\\
&= \sum\_l \left( f\_l, \phi\_j \delta\_{l,\text{comp}(j)} \right)\_K \\\
&= \left( f\_{\text{comp}(j)}, \phi\_j \right)\_K.
\end{split}
\end{equation}
$$


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
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
```
以上是以前用过的头文件。
```cpp
#include <deal.II/fe/fe_system.h>
```
该头文件提供对矢量值的有限元的支持。
```cpp
#include <deal.II/fe/fe_q.h>
```
从常规的Q1单元组合得到矢量值的有限元，Q1单元在以上头文件中。
```cpp
#include <fstream>
#include <iostream>
```
C++的标准库。
建立step8的命名空间：
```cpp
namespace Step8
{
    using namespace dealii;
template <int dim>
    class ElasticProblem
    {
        public:
            ElasticProblem ();
            ~ElasticProblem ();
            void run ();
        private:
            void setup_system ();
            void assemble_system ();
            void solve ();
            void refine_grid ();
            void output_results (const unsigned int cycle) const;
            Triangulation<dim> triangulation;
            DoFHandler<dim> dof_handler;
            FESystem<dim> fe;
            ConstraintMatrix hanging_node_constraints;
            SparsityPattern sparsity_pattern;
            SparseMatrix<double> system_matrix;
            Vector<double> solution;
            Vector<double> system_rhs;
    };
```
step8的类跟step6差不多，唯一一个变化是fe的类型，这里使用的是FESystem，不再是FE_Q。实际上FESystem本身不是一个有限元类型，不提供形函数。它就是将多个单元集合起来形成一个矢量的有限单元。
然后建立右端项：
```cpp
template <int dim>
class RightHandSide : public Function<dim>
{
    public:
        RightHandSide ();
        virtual void vector_value (const Point<dim> &p,
                Vector<double> &values) const;
        virtual void vector_value_list (const std::vector<Point<dim> > &points,
                std::vector<Vector<double> > &value_list) const;
};
```
vector_value是取得某个位置的矢量值，vector_value_list是一下取得很多。
```cpp
template <int dim>
RightHandSide<dim>::RightHandSide ()
    :
        Function<dim> (dim)
{}
```
析构函数中给Function传递了参数，代表分量的个数，这里就是dim。
```cpp
template <int dim>
inline
void RightHandSide<dim>::vector_value (const Point<dim> &p,
        Vector<double> &values) const
{
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    Assert (dim >= 2, ExcNotImplemented());
```
这里加入了几个提前判断，用来保证维数和矢量大小正确。
```cpp
Point<dim> point_1, point_2;
point_1(0) = 0.5;
point_2(0) = -0.5;
if (((p-point_1).norm_square() < 0.2*0.2) ||
        ((p-point_2).norm_square() < 0.2*0.2))
    values(0) = 1;
else
    values(0) = 0;

```
如果在离两个圆心一定范围内，就把x方向的力设为1，否则设为0。
```cpp
if (p.norm_square() < 0.2*0.2)
    values(1) = 1;
else
    values(1) = 0;
}
```
如果离原点一定范围内，就把y方向的力设为1，否则设为0。
```cpp
template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
        std::vector<Vector<double> > &value_list) const
{
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));
    const unsigned int n_points = points.size();
    for (unsigned int p=0; p<n_points; ++p)
        RightHandSide<dim>::vector_value (points[p],
                value_list[p]);
}
```
然后就是一下取得好多点上的值。
```cpp
template <int dim>
ElasticProblem<dim>::ElasticProblem ()
    :
        dof_handler (triangulation),
        fe (FE_Q<dim>(1), dim)
{}
template <int dim>
ElasticProblem<dim>::~ElasticProblem ()
{
    dof_handler.clear ();
}
```
这是求解类的构造和析构函数。在构造函数中，给fe传递两个参数：一个是构造矢量有限元所基于的标量有限元，另一个就是多少个标量堆起来等于1个矢量，这里就是dim。知道这些信息后，FESystem就知道该怎么合成矢量有限元了。
```cpp
template <int dim>
void ElasticProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
            hanging_node_constraints);
    hanging_node_constraints.close ();
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
            dsp,
            hanging_node_constraints,
            / *keep_constrained_dofs = * / true);
    sparsity_pattern.copy_from (dsp);
    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}
```
建立系统。过程跟step6相同，这里的这些类都能处理矢量的单元，实际无论矢量还是标量单元，这些类都一视同仁。
重头戏就是组装系统了。
```cpp
template <int dim>
void ElasticProblem<dim>::assemble_system ()
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
第一部分跟之前的相同：设置合适的积分公式、初始化FEValues对象、声明一些附加数组等。其中，获取每个单元上的自由度数，是从合成过的有限单元上取得，而不是那个标量Q1对象。这里，自由度数等于dim乘以Q1单元的每个单元上的自由度数。
```cpp
std::vector<double> lambda_values (n_q_points);
std::vector<double> mu_values (n_q_points);
```
然后存储所有积分点上两个系数的值。
```cpp
ConstantFunction<dim> lambda(1.), mu(1.);
```
这里将两个系数都设为定值。
```cpp
RightHandSide<dim> right_hand_side;
std::vector<Vector<double> > rhs_values (n_q_points,
        Vector<double>(dim));
```
建立右端项的对象。因为是矢量，所以rhs_values的类型也变化了。
然后开始单元循环：
```cpp
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
         endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit (cell);
    lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
    mu.value_list (fe_values.get_quadrature_points(), mu_values);
    right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
            rhs_values);
```
上面计算系数和右端项在积分点上的值。
然后就是计算单元刚度矩阵和右端项：
```cpp
for (unsigned int i=0; i<dofs_per_cell; ++i)
{
    const unsigned int
        component_i = fe.system_to_component_index(i).first;
    for (unsigned int j=0; j<dofs_per_cell; ++j)
    {
        const unsigned int
            component_j = fe.system_to_component_index(j).first;
        for (unsigned int q_point=0; q_point<n_q_points;
                ++q_point)
        {
            cell_matrix(i,j)
                +=
```
整个计算过程完全对应前面引子中的推导。component_i就是非0分量的指标，它通过fe.system_to_component_index(i).first这个函数取得，实际上first取得矢量形函数的非0分量的指标，而second取得具体这个形函数的值，即引子中的base。
```cpp
(
 (fe_values.shape_grad(i,q_point)[component_i] *
  fe_values.shape_grad(j,q_point)[component_j] *
  lambda_values[q_point])
 +
 (fe_values.shape_grad(i,q_point)[component_j] *
  fe_values.shape_grad(j,q_point)[component_i] *
  mu_values[q_point])
 +
```
这一项计算的是(lambda d_i u_i, d_j v_j) + (mu d_i u_j, d_j v_i)。
```cpp
((component_i == component_j) ?
 (fe_values.shape_grad(i,q_point) *
  fe_values.shape_grad(j,q_point) *
  mu_values[q_point]) :
 0)
)
*
fe_values.JxW(q_point);
        }
    }
}
```
这一项计算的是(mu nabla u_i, nabla v_j)。注意这里的grad没有加后面的方括号，用了重载的乘号。
且使用了条件表达式，判断两个下标是否相同。
单元右端项的计算：
```cpp
for (unsigned int i=0; i<dofs_per_cell; ++i)
{
    const unsigned int
        component_i = fe.system_to_component_index(i).first;
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        cell_rhs(i) += fe_values.shape_value(i,q_point) *
            rhs_values[q_point](component_i) *
            fe_values.JxW(q_point);
}
```
组装：
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
hanging_node_constraints.condense (system_matrix);
hanging_node_constraints.condense (system_rhs);
```
施加边界条件：
```cpp
std::map<types::global_dof_index,double> boundary_values;
VectorTools::interpolate_boundary_values (dof_handler,
        0,
        ZeroFunction<dim>(dim),
        boundary_values);
MatrixTools::apply_boundary_values (boundary_values,
        system_matrix,
        solution,
        system_rhs);
}
```
这里做了一些小修改，因为解是矢量的，所以边界条件施加的也应该是矢量的。而ZeroFunction可以接收参数来形成不同类型的量，这里传递的是dim。
求解器：
```cpp
template <int dim>
void ElasticProblem<dim>::solve ()
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
求解过程不变，求解器不管具体问题是什么，只要线性系统是正定且对称的，CG算法就能用。
细化网格：
```cpp
template <int dim>
void ElasticProblem<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
            QGauss<dim-1>(2),
            typename FunctionMap<dim>::type(),
            solution,
            estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
            estimated_error_per_cell,
            0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
}
```
指示子是用的所有方向上的位移具有相同的权重。
结果输出：
```cpp
template <int dim>
void ElasticProblem<dim>::output_results (const unsigned int cycle) const
{
    std::string filename = "solution-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());
    filename += ".vtk";
    std::ofstream output (filename.c_str());
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    std::vector<std::string> solution_names;
    switch (dim)
    {
        case 1:
            solution_names.push_back ("displacement");
            break;
        case 2:
            solution_names.push_back ("x_displacement");
            solution_names.push_back ("y_displacement");
            break;
        case 3:
            solution_names.push_back ("x_displacement");
            solution_names.push_back ("y_displacement");
            solution_names.push_back ("z_displacement");
            break;
        default:
            Assert (false, ExcNotImplemented());
    }
    data_out.add_data_vector (solution, solution_names);
    data_out.build_patches ();
    data_out.write_vtk (output);
}
```
因为结果是矢量，所以给每个分量都有一个名字。
运行函数：
```cpp
template <int dim>
void ElasticProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle<8; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (2);
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
}
}
```
Attention!!
这里有个小问题，刚开始产生网格后，就全局细化了两次。这是因为这里选择的右端项相当局域化，如果只细化一次，网格的积分点很稀疏，它捕捉的右端项的值全是0，这样就计算错误了。所以，要考虑到网格细化对值的正确捕捉。
Attention完毕！！
main函数如下：
```cpp
int main ()
{
    try
    {
        Step8::ElasticProblem<2> elastic_problem_2d;
        elastic_problem_2d.run ();
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
x分量为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk7bplf8j30qx0k5dhl.jpg)
y分量为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk7ml9eij30qx0k5dhh.jpg)
注意，虽然这两个分量组合起来是位移，即它们两个不是完全孤立的，比如说是压力和浓度的关系，而是一个量的两个分量，但现在的output方式没法将两者组合起来，即这里还是将两者看成两个孤立的量，真正组合起来显示矢量的例子是step22，到时再说。
