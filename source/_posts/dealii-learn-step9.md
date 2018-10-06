---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 9
tags: [deal.II]
categories: computational material science
date: 2016-9-15
---

# 引子
本例将要完成以下目标：
- 求解对流方程$\beta \cdot \nabla u = f$
- 使用多线程求解
- 设计一个简单的细化准则



## 方程离散
对流方程中$\beta$是一个描述对流方向和速度的矢量场，可能与空间位置相关，$f$是源项，$u$是解。这个方程求解的物理过程可以是浓度u在给定流场下的传输，没有扩散过程，但有源。
明显地，在入流侧，上述方程需要在边界上获得补充：
$$
u = g \qquad\qquad \mathrm{on}\ \partial\Omega\_-,
$$
其中，$\Omega\_-$代表边界的入流部分，其定义为：
$$
\partial\Omega_-=[{\mathbf x}\in\partial\Omega:\beta\cdot{\mathbf n}({\mathbf x}) < 0],
$$
其中，${\mathbf n}({\mathbf x})$是点$x$上的外法线。这种定义方式是非常直观的：因为如果外法线是朝外的，那么在入流边界上流动方向就肯定是朝内的。而且，数学理论要求不能在外流边界上施加任何边界条件。在下面的弱形式中也可以看出，确实是没有对外流边界作任何处理。这里跟Fluent软件中的自由外流出口边界还不同，Fluent中默认其在出流界面上法线通量为0，即出流速度不影响前方的速度，其是由完全外推得到。但这里因为没有应用分部积分，所以也不需要处理外流边界上的速度值，只是在方程两侧显性地加上了入流条件。
注意：这个传输方程使用标准的有限元方法是不能稳定求解的。这里使用“流线扩散稳定方法”，测试函数的形式变成$v+\delta\beta\cdot v$，其中$\delta$是一个与局部网格间距$h$有关的参数，通常$\delta=0.1 h$。当网格尺寸为0时，测试函数就回归之前的形式。具体原理可以参见相关书籍和论文。
方程的弱形式为：
$$
(\beta \cdot \nabla u\_h, v\_h + \delta \beta\cdot\nabla v\_h)\_\Omega - (\beta\cdot {\mathbf n} u\_h, v\_h)\_{\partial\Omega\_-} = (f, v\_h + \delta \beta\cdot\nabla v\_h)\_\Omega - (\beta\cdot {\mathbf n} g, v\_h)\_{\partial\Omega\_-}.
$$
Attention！！上式已经加入了边界条件，即右端第二项。而为了满足等式，在左侧也加入了对应的积分项，所以，左侧边界积分项中是未知解，而右端项边界积分中是对应的边界值。
那么，其中的刚度矩阵就是：
$$
a\_{ij} = (\beta \cdot \nabla \varphi\_i, \varphi\_j + \delta \beta\cdot\nabla \varphi\_j)\_\Omega - (\beta\cdot {\mathbf n} \varphi\_i, \varphi\_j)\_{\partial\Omega\_-},
$$
但是这样的话就需要求解这样的方程：
$$
{\mathbf u}^T A = {\mathbf f}^T,
$$
实际求解时需要转置一下：
$$
A^T{\mathbf u} = {\mathbf f},
$$
此时如果$A^T=A$还好说，即A是对称矩阵，但此时的问题中的矩阵非对称，所以具体组装矩阵时就要注意元素的位置，为了防止出现需要计算转置这种问题，养成这样一种习惯：每次都把试探函数左乘，而不是右乘，则有：
$$
(v\_h + \delta \beta\cdot\nabla v\_h, \beta \cdot \nabla u\_h)\_\Omega - (\beta\cdot {\mathbf n} v\_h, u\_h)\_{\partial\Omega\_-} = (v\_h + \delta \beta\cdot\nabla v\_h, f)\_\Omega - (\beta\cdot {\mathbf n} v\_h, g)\_{\partial\Omega\_-}
$$
那么矩阵就变成：
$$
a\_{ij} = (\varphi\_i + \delta \beta \cdot \nabla \varphi\_i, \beta\cdot\nabla \varphi\_j)\_\Omega - (\beta\cdot {\mathbf n} \varphi\_i, \varphi\_j)\_{\partial\Omega\_-},
$$
对于方程的解，因为矩阵不再是对称正定的，CG算法不再适用，这里采用双共轭梯度稳定算法。
方程中的具体数值为：
$$
\begin{equation}
\begin{split}
\Omega &=[-1,1]^d \\\ 
\beta({\mathbf x}) &=
\begin{bmatrix}
2 \\\ 
1+\frac{4}{5}\sin(8\pi x),
\end{bmatrix} \\\ 
f({\mathbf x}) &=
\begin{bmatrix}
\frac {1}{10 s^d} & \mathrm{for}\ |{\mathbf x}-{\mathbf x}\_0|<s, \\\
0                 & \mathrm{else},
\end{bmatrix}
\qquad\qquad {\mathbf x}\_0 = 
\begin{bmatrix}
-\frac{3}{4} \\\
-\frac{3}{4}
\end{bmatrix} \\\
g &=e^{5(1-|{\mathbf x}|^2)} \sin(16\pi|{\mathbf x}|^2)
\end{split}
\end{equation}
$$

## 基于任务task并行
传统的在共享内存机器上的并行是将程序分成多个线程，但这里先讲基于任务的并行。
“任务”是程序的独立部分，它们之间或者互相依赖，或者互相不依赖。如：
```cpp
1 dof_handler.distribute_dofs (fe);
2 DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);
3 DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
4 hanging_node_constraints.condense (sparsity_pattern);
```
上述4个操作每个都需要较大的计算量。但是注意并不是全部都互相依赖：明显地，操作1必须首先运行，操作4也必须最后运行，但2和3没有先后顺序。
Attention：这里如果用线程来表现2和3的独立性，就需要开两个线程，然后在每个线程上分别运行2和3。然后等待两个线程完成后，再将两者join起来：
```cpp
dof_handler.distribute_dofs (fe);
Threads::Thread<void>
thread_1 = Threads::new_thread (&DoFTools::make_hanging_node_constraints,
        dof_handler, hanging_node_constraints);
Threads::Thread<void>
thread_2 = Threads::new_thread (&DoFTools::make_sparsity_pattern,
        dof_handler, sparsity_pattern);
thread_1.join();
thread_2.join();
hanging_node_constraints.condense (sparsity_pattern);
```
那么问题来了：如果只有一个处理器核心，或者有两个，但早就有程序的另外部分在用着？这些情况下，上面的代码仍然会开两个线程，但因为没有额外的计算资源，所以程序运行得不会更快，其实反而更慢，因为线程的创建和销毁需要时间，同时系统还需要调度线程给过载的计算资源。
一个更好的方法是：识别独立的任务，然后创建一个任务与计算资源相对应的调度程序scheduler，根据它来切换任务。这样，程序将会在每个核心上创建一个线程，然后开始计算，任务将会一直运行到结束，而不是都同时运行，防止突然打断的线程使计算切换到另一个线程上。在此例中，如果有两个核心，2和3将会同时运行，如果只有一个，调度程序将会先计算完2，然后再切换到3上，或者反过来。这种方法在有大量任务时会很有效率，见下面的WorkStream。
deal.II自己不建立调度程序，而是使用TBB库函数。
如果使用task来并行，则代码为：
```cpp
Threads::Task<void>
thread
= Threads::new_task (&DoFTools::make_hanging_node_constraints,
        dof_handler,
        hanging_node_constraints);
```
这里使用new_task来创建一个任务，跟之前的new_thread类似。
如果有很多tasks，得等它们都停下以后才能往后运行，这时每个都写join会很麻烦，可以把它们放进一个group中，一次等待所有的任务：
```cpp
dof_handler.distribute_dofs (fe);
Threads::TaskGroup<void> task_group;
task_group += Threads::new_task (&DoFTools::make_hanging_node_constraints,
        dof_handler, hanging_node_constraints);
task_group += Threads::new_task (&DoFTools::make_sparsity_pattern,
        dof_handler, sparsity_pattern);
task_group.join_all ();
hanging_node_constraints.condense (sparsity_pattern);
```
至于如何调度任务，是在TBB库内部实现的。TBB的文档给出了如何将任务调度给线程，让其执行的详细过程，但没有明确说创建了多少线程。然而，合理猜测是TBB根据核数创建了相同数量的线程，这样就能充分使用整个系统，也没有过多的线程导致频繁的打断。TBB调度程序分配了任务，然后让线程运行它们，线程会充分运行，直到任务结束，而不会中途打断。这意味着缓存永远是时刻使用着的。
但是基于任务的并行也有缺点，CPU只在以下两者情况都满足的情况下才满负荷运转：一是有足够数目的任务；二是这些任务确实在做事。比如如果任务数目不够或数目够了但一些任务在闲着，比如等待写入硬盘或等待读入数据，都使CPU没有充分利用。其他情形还有任务在等待外部事件，比如与其他任务的同步等。上述情形下可以创建基于线程的并行，来充分利用CPU。

### 工作流 Work Streams
在上面例子中，任务的数目大于或等于CPU的核数，以使它们满负荷运行，但也没有太多量，比如只有4个核却有百万个任务。然而也会有百万量级任务的情形，比如将单元上的贡献叠加到整体刚度矩阵中;估计每个单元上的误差指示子等。这些情形可以使用一种称为“WorkStream”的软件设计模式。
详情见相应的帮助文档。

## 基于线程的并行
有时候基于任务的并行不能发挥CPU的百分百性能，这时可以使用基于线程的并行。比如下面的这个例子：
```cpp
template <int dim>
void MyClass<dim>::output_and_estimate_error () const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output ("solution.vtk");
    Threads::Thread<void>
        thread = Threads::new_thread (&DataOut<dim>::write_vtk, data_out, output);
    Vector<float> error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
            QGauss<dim-1>(3),
            typename FunctionMap<dim>::type(),
            solution,
            estimated_error_per_cell);
    thread.join ();
```
new_thread新开了一个线程用来输出图片，它跟后面的误差估计并行。这里用线程，是因为往硬盘中写数据不会使得CPU很忙，所以没必要把它做成一个任务。
正如上面所说，deal.II使用的是TBB来调度线程，TBB不建议明确地给定线程数目。但是有时候也需要明确指定线程数目，比如在有一个资源管理器的机器上或在获取某些计算资源很费劲的机器上，这时需要限制数目，防止TBB调用过多CPU。还有一种情况是有多个MPI并行任务时，每个任务也只能调用一个子集的计算资源。

## 细化准则
之前在有自适应网格的例子中，都是使用了Kelly等人提出的误差指示子，其实际使用的是类似解的二阶导数作为判据。所以应用这个指示子的前提是解有二阶导数。但在有些情况下，比如本例中，解甚至在某些地方没有一阶导数，所以Kelly指示子不再适用。这里提出一种判断梯度的公式。注意到给定两个单元和连结两个单元中心的向量，那么解的梯度可以近似表示为：
$$
\frac{\mathbf y\_{KK'}^T}{|\mathbf y\_{KK'}|} \nabla u \approx \frac{u(K') - u(K)}{|\mathbf y\_{KK'}|},
$$
那么：
$$
\underbrace{ \left(\sum\_{K'} \frac{\mathbf y\_{KK'} \mathbf y\_{KK'}^T} {|\mathbf y\_{KK'}|^2}\right)}\_{=:Y} \nabla u \approx \sum\_{K'} \frac{\mathbf y\_{KK'}}{|\mathbf y\_{KK'}|} \frac{u(K') - u(K)}{|\mathbf y\_{KK'}|}.
$$
再推导有：
$$
\nabla u \approx Y^{-1} \left( \sum\_{K'} \frac{\mathbf y\_{KK'}}{|\mathbf y\_{KK'}|} \frac{u(K') - u(K)}{|\mathbf y\_{KK'}|} \right).
$$
右端用$\nabla\_h u(K)$来表示，细化准则则为：
$$
\eta\_K = h^{1+d/2} |\nabla\_h u\_h(K)|,
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
#include <deal.II/lac/solver_bicgstab.h>
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
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
```
以上是之前提过的一些头文件。
```cpp
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
```
第一个头文件是WorkStream命名空间，包含并行计算时所用的函数和类。第二个头文件是获得电脑的处理器核数，从而确定并行时开多少线程。
```cpp
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/error_estimator.h>
#include <fstream>
#include <iostream>
```
还有一个新头文件是TensorFunction，不同于之前的Function类，它返回的是个张量值。
```cpp
namespace Step9
{
    using namespace dealii;
```
然后就开始具体问题的Step9命名空间定义。
```cpp
template <int dim>
class AdvectionProblem
{
    public:
        AdvectionProblem ();
        ~AdvectionProblem ();
        void run ();
    private:
        void setup_system ();
```
上面开始声明问题类。
然后开始组装矩阵：
```cpp
struct AssemblyScratchData
{
    AssemblyScratchData (const FiniteElement<dim> &fe);
    AssemblyScratchData (const AssemblyScratchData &scratch_data);
    FEValues<dim> fe_values;
    FEFaceValues<dim> fe_face_values;
};
struct AssemblyCopyData
{
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
};
void assemble_system ();
void local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
        AssemblyScratchData &scratch,
        AssemblyCopyData &copy_data);
void copy_local_to_global (const AssemblyCopyData &copy_data);
```
组装矩阵时可以很好地应用并行，因为单元矩阵运算可以自己进行，与其他单元无关，只需要在组装时同步就行。其中的ScratchData是临时数据。
然后就是之前的一些声明：
```cpp
void solve ();
void refine_grid ();
void output_results (const unsigned int cycle) const;
Triangulation<dim> triangulation;
DoFHandler<dim> dof_handler;
FE_Q<dim> fe;
ConstraintMatrix hanging_node_constraints;
SparsityPattern sparsity_pattern;
SparseMatrix<double> system_matrix;
Vector<double> solution;
Vector<double> system_rhs;
};
```
下面就是声明一个对流场的类：
```cpp
template <int dim>
class AdvectionField : public TensorFunction<1,dim>
{
    public:
        AdvectionField () : TensorFunction<1,dim> () {}
        virtual Tensor<1,dim> value (const Point<dim> &p) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                std::vector<Tensor<1,dim> > &values) const;
        DeclException2 (ExcDimensionMismatch,
                unsigned int, unsigned int,
                << "The vector has size " << arg1 << " but should have "
                << arg2 << " elements.");
};
```
这个类明显是个矢量，因为有对应于空间维度数目的分量。可以像以前一样使用Function类，不过这里使用的是描述张量函数的TensorFunction类。这里面也设置了捕捉异常的函数，用来判断两个矢量的大小是否相同。
```cpp
template <int dim>
Tensor<1,dim>
AdvectionField<dim>::value (const Point<dim> &p) const
{
    Point<dim> value;
    value[0] = 2;
    for (unsigned int i=1; i<dim; ++i)
        value[i] = 1+0.8*std::sin(8*numbers::PI*p[0]);
    return value;
}
template <int dim>
void
AdvectionField<dim>::value_list (const std::vector<Point<dim> > &points,
        std::vector<Tensor<1,dim> > &values) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = AdvectionField<dim>::value (points[i]);
}
```
以上是value和value_list的具体实现。第二个函数中有矢量的尺寸大小判断，加上这么一个Assert语句能有效防止错误。
下面就是描述源项：
```cpp
template <int dim>
class RightHandSide : public Function<dim>
{
    public:
        RightHandSide () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                std::vector<double> &values,
                const unsigned int component = 0) const;
    private:
        static const Point<dim> center_point;
};
template <>
const Point<1> RightHandSide<1>::center_point = Point<1> (-0.75);
template <>
const Point<2> RightHandSide<2>::center_point = Point<2> (-0.75, -0.75);
template <>
const Point<3> RightHandSide<3>::center_point = Point<3> (-0.75, -0.75, -0.75);
```
从方程中可以看出，源项是围绕在某个源点附近，注意不是原点。这里定义了在不同维度下的源点。
```cpp
template <int dim>
double
RightHandSide<dim>::value (const Point<dim> &p,
        const unsigned int component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));
    const double diameter = 0.1;
    return ( (p-center_point).norm_square() < diameter*diameter ?
            .1/std::pow(diameter,dim) :
            0);
}
template <int dim>
void
RightHandSide<dim>::value_list (const std::vector<Point<dim> > &points,
        std::vector<double> &values,
        const unsigned int component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = RightHandSide<dim>::value (points[i], component);
}
```
以上是具体实现。这里也加入了尺寸判断。
然后就是边界值：
```cpp
template <int dim>
class BoundaryValues : public Function<dim>
{
    public:
        BoundaryValues () : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                const unsigned int component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                std::vector<double> &values,
                const unsigned int component = 0) const;
};
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim> &p,
        const unsigned int component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));
    const double sine_term = std::sin(16*numbers::PI*std::sqrt(p.norm_square()));
    const double weight = std::exp(-5*p.norm_square()) / std::exp(-5.);
    return sine_term * weight;
}
template <int dim>
void
BoundaryValues<dim>::value_list (const std::vector<Point<dim> > &points,
        std::vector<double> &values,
        const unsigned int component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = BoundaryValues<dim>::value (points[i], component);
}
```
## 梯度估计
下面开始计算单元上的梯度以及网格尺寸的乘方作为权重，就像引子中所示。这个类是DerivativeApproximation的简化版本，DerivativeApproximatio还能计算高阶的导数。
```cpp
class GradientEstimation
{
    public:
        template <int dim>
            static void estimate (const DoFHandler<dim> &dof,
                    const Vector<double> &solution,
                    Vector<float> &error_per_cell);
        DeclException2 (ExcInvalidVectorLength,
                int, int,
                << "Vector has length " << arg1 << ", but should have "
                << arg2);
        DeclException0 (ExcInsufficientDirections);
    private:
        template <int dim>
            struct EstimateScratchData
            {
                EstimateScratchData (const FiniteElement<dim> &fe,
                        const Vector<double> &solution);
                EstimateScratchData (const EstimateScratchData &data);
                FEValues<dim> fe_midpoint_value;
                Vector<double> solution;
            };
        struct EstimateCopyData
        {};
        template <int dim>
            static
            void estimate_cell (const SynchronousIterators<std_cxx11::tuple<typename DoFHandler<dim>::active_cell_iterator,
                    Vector<float>::iterator> > &cell,
                    EstimateScratchData<dim> &scratch_data,
                    const EstimateCopyData &copy_data);
};
```
这个类有一个静态Public函数estimate来计算误差指示子，还有几个Private函数来做具体的工作。还有两个异常捕获函数，用来捕捉没有相邻单元的单元以及尺寸不对的参数。
还有两个事情需要注意：一是该类没有非静态的成员函数或变量，所以它不是一个真正的类，而是命名空间的功能。这里选择类而不是命名空间是因为想要声明私有函数。这也可以同样使用命名空间来实现，但这涉及多文件，这里只有一个文件，所以在此例中不能隐藏函数。二是维度参数没有放在类中，而是放在了函数上，这样不用手动加上维度这个参数，而是编译器能够从DoF等上自动推导。
还要讲一下并行策略：之前主类中已经说了，并行时需要定义：一是定义临时和复制对象；二是在单元上进行局部计算；三是将局部对象复制到整体对象中的函数。但是这里做一点改变。具体改变见帮助文档。

## 主类的实现
```cpp
template <int dim>
AdvectionProblem<dim>::AdvectionProblem ()
    :
        dof_handler (triangulation),
        fe(1)
{}
template <int dim>
AdvectionProblem<dim>::~AdvectionProblem ()
{
    dof_handler.clear ();
}
template <int dim>
void AdvectionProblem<dim>::setup_system ()
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
这是主类的构造函数、析构函数和建立系统，不详述。
下面是组装系统的并行策略：
```cpp
template <int dim>
void AdvectionProblem<dim>::assemble_system ()
{
    WorkStream::run(dof_handler.begin_active(),
            dof_handler.end(),
            *this,
            &AdvectionProblem::local_assemble_system,
            &AdvectionProblem::copy_local_to_global,
            AssemblyScratchData(fe),
            AssemblyCopyData());
```
使用WorkStream来统一调度，能充分利用电脑的线程。
```cpp
hanging_node_constraints.condense (system_matrix);
hanging_node_constraints.condense (system_rhs);
}
```
同样也需要消除悬点限制。因为不能分开在单个线程上进行，所以需要统一进行。
Attention！！跟之前不同，这里没有施加边界条件，当然这是因为已经把边界条件加在了弱形式中。！
```cpp
template <int dim>
AdvectionProblem<dim>::AssemblyScratchData::
AssemblyScratchData (const FiniteElement<dim> &fe)
    :
        fe_values (fe,
                QGauss<dim>(2),
                update_values | update_gradients |
                update_quadrature_points | update_JxW_values),
        fe_face_values (fe,
                QGauss<dim-1>(2),
                update_values | update_quadrature_points |
                update_JxW_values | update_normal_vectors)
{}
template <int dim>
AdvectionProblem<dim>::AssemblyScratchData::
AssemblyScratchData (const AssemblyScratchData &scratch_data)
    :
        fe_values (scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                update_values | update_gradients |
                update_quadrature_points | update_JxW_values),
        fe_face_values (scratch_data.fe_face_values.get_fe(),
                scratch_data.fe_face_values.get_quadrature(),
                update_values | update_quadrature_points |
                update_JxW_values | update_normal_vectors)
{}
```
这是并行时用到的临时对象。这些对象包含FEValues和FEFaceValues，其中的flag一定要传递全面。
下面就是具体要干的事了。
```cpp
template <int dim>
void
AdvectionProblem<dim>::
local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
        AssemblyScratchData &scratch_data,
        AssemblyCopyData &copy_data)
{
```
帮助文档中关于这里有一大段关于效率的问题，没看明白，留坑待填。
首先创建描述边界值、右端项和对流场的对象，这里声明为const，以便让编译器进行优化：
```cpp
const AdvectionField<dim> advection_field;
const RightHandSide<dim> right_hand_side;
const BoundaryValues<dim> boundary_values;
```
然后定义一些缩写，防止冗长的描述：
```cpp
const unsigned int dofs_per_cell = fe.dofs_per_cell;
const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();
const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();
```
初始化单元刚度矩阵和单元右端项和单元自由度的全局标识：
```cpp
copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
copy_data.cell_rhs.reinit (dofs_per_cell);
copy_data.local_dof_indices.resize(dofs_per_cell);
```
还有创建对象来存储单元和边界积分中要用到的右端项、对流方向和边界的值。
```cpp
std::vector<double> rhs_values (n_q_points);
std::vector<Tensor<1,dim> > advection_directions (n_q_points);
std::vector<double> face_boundary_values (n_face_q_points);
std::vector<Tensor<1,dim> > face_advection_directions (n_face_q_points);
```
初始化FEValues对象：
```cpp
scratch_data.fe_values.reinit (cell);
```
得到积分点上右端项和对流方向的值：
```cpp
advection_field.value_list (scratch_data.fe_values.get_quadrature_points(),
        advection_directions);
right_hand_side.value_list (scratch_data.fe_values.get_quadrature_points(),
        rhs_values);
```
设置流线扩散法中参数的值，参见引子：
```cpp
const double delta = 0.1 * cell->diameter ();
```
然后计算单元刚度矩阵和右端项：
```cpp
for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
            copy_data.cell_matrix(i,j) += ((advection_directions[q_point] *
                        scratch_data.fe_values.shape_grad(j,q_point) *
                        (scratch_data.fe_values.shape_value(i,q_point) +
                         delta *
                         (advection_directions[q_point] *
                          scratch_data.fe_values.shape_grad(i,q_point)))) *
                    scratch_data.fe_values.JxW(q_point));
        copy_data.cell_rhs(i) += ((scratch_data.fe_values.shape_value(i,q_point) +
                    delta *
                    (advection_directions[q_point] *
                     scratch_data.fe_values.shape_grad(i,q_point)) ) *
                rhs_values[q_point] *
                scratch_data.fe_values.JxW (q_point));
    }
```
Attention！！上面项中没有加入边界项。所以还要检查这个单元的边是否在入流边界上，如果是，那么必须把它的贡献也得叠加上去。注意仅仅是入流边界，所以还得知道积分点的绝对位置，及该点上的流动方向：
```cpp
for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
    if (cell->face(face)->at_boundary())
    {
```
如果进入了上面的if，说明该单元的边是在边界上了，然而别忘了初始化FEFaceValues对象，然后获取入流函数的值和流动方向：
```cpp
scratch_data.fe_face_values.reinit (cell, face);
boundary_values.value_list (scratch_data.fe_face_values.get_quadrature_points(),
        face_boundary_values);
advection_field.value_list (scratch_data.fe_face_values.get_quadrature_points(),
        face_advection_directions);
```
现在遍历所有积分点，判断是否入流边界：
```cpp
for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
    if (scratch_data.fe_face_values.normal_vector(q_point) *
            face_advection_directions[q_point]
            < 0)
```
如果是入流边界，那么就计算这个边的贡献：
```cpp
for (unsigned int i=0; i<dofs_per_cell; ++i)
{
    for (unsigned int j=0; j<dofs_per_cell; ++j)
        copy_data.cell_matrix(i,j) -= (face_advection_directions[q_point] *
                scratch_data.fe_face_values.normal_vector(q_point) *
                scratch_data.fe_face_values.shape_value(i,q_point) *
                scratch_data.fe_face_values.shape_value(j,q_point) *
                scratch_data.fe_face_values.JxW(q_point));
    copy_data.cell_rhs(i) -= (face_advection_directions[q_point] *
            scratch_data.fe_face_values.normal_vector(q_point) *
            face_boundary_values[q_point] *
            scratch_data.fe_face_values.shape_value(i,q_point) *
            scratch_data.fe_face_values.JxW(q_point));
}
}
```
终于到组装了：
```cpp
cell->get_dof_indices (copy_data.local_dof_indices);
}
template <int dim>
void
AdvectionProblem<dim>::copy_local_to_global (const AssemblyCopyData &copy_data)
{
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
    {
        for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
            system_matrix.add (copy_data.local_dof_indices[i],
                    copy_data.local_dof_indices[j],
                    copy_data.cell_matrix(i,j));
        system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
    }
}
```
跟之前的差不多。
然后就是求解：
```cpp
template <int dim>
void AdvectionProblem<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverBicgstab<> bicgstab (solver_control);
    PreconditionJacobi<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    bicgstab.solve (system_matrix, solution, system_rhs,
            preconditioner);
    hanging_node_constraints.distribute (solution);
}
```
因为方程不再对称，所以使用了BICGStab算法。
然后就是细化网格：
template <int dim>
void AdvectionProblem<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    GradientEstimation::estimate (dof_handler,
            solution,
            estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
            estimated_error_per_cell,
            0.5, 0.03);
    triangulation.execute_coarsening_and_refinement ();
}
```
然后输出：
```cpp
template <int dim>
void AdvectionProblem<dim>::output_results (const unsigned int cycle) const
{
    std::string filename = "grid-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());
    filename += ".eps";
    std::ofstream output (filename.c_str());
    GridOut grid_out;
    grid_out.write_eps (triangulation, output);
}
```
运行函数：
```cpp
template <int dim>
void AdvectionProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle<6; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (4);
        }
        else
        {
            refine_grid ();
        }
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
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output ("final-solution.vtk");
    data_out.write_vtk (output);
}
```

## 梯度估计类的实现
先是构造函数：
```cpp
template <int dim>
GradientEstimation::EstimateScratchData<dim>::
EstimateScratchData (const FiniteElement<dim> &fe,
        const Vector<double> &solution)
    :
        fe_midpoint_value(fe,
                QMidpoint<dim> (),
                update_values | update_quadrature_points),
        solution(solution)
{}
template <int dim>
GradientEstimation::EstimateScratchData<dim>::
EstimateScratchData(const EstimateScratchData &scratch_data)
    :
        fe_midpoint_value(scratch_data.fe_midpoint_value.get_fe(),
                scratch_data.fe_midpoint_value.get_quadrature(),
                update_values | update_quadrature_points),
        solution(scratch_data.solution)
{}
```
然后就是estimate成员函数：
```cpp
template <int dim>
void
GradientEstimation::estimate (const DoFHandler<dim> &dof_handler,
        const Vector<double> &solution,
        Vector<float> &error_per_cell)
{
    Assert (error_per_cell.size() == dof_handler.get_triangulation().n_active_cells(),
            ExcInvalidVectorLength (error_per_cell.size(),
                dof_handler.get_triangulation().n_active_cells()));
    typedef std_cxx11::tuple<typename DoFHandler<dim>::active_cell_iterator,Vector<float>::iterator>
        IteratorTuple;
    SynchronousIterators<IteratorTuple>
        begin_sync_it (IteratorTuple (dof_handler.begin_active(),
                    error_per_cell.begin())),
                      end_sync_it (IteratorTuple (dof_handler.end(),
                                  error_per_cell.end()));
    WorkStream::run (begin_sync_it,
            end_sync_it,
            &GradientEstimation::template estimate_cell<dim>,
            std_cxx11::function<void (const EstimateCopyData &)> (),
            EstimateScratchData<dim> (dof_handler.get_fe(),
                solution),
            EstimateCopyData ());
}
```
这里也检查了矢量的尺寸是否正确，避免不易发现的错误。然后还设置了一些迭代器，没看明白。。
Attention！！然后计算梯度的有限差分近似：
```cpp
std_cxx11::function<void (const SynchronousIterators<IteratorTuple> &,
        EstimateScratchData<dim> &,
        EstimateCopyData &)>
(std_cxx11::bind (&GradientEstimation::template estimate_cell<dim>,
                  std_cxx11::_1,
                  std_cxx11::_2))
```
整体思路是：首先计算当前单元的活跃邻居的列表，然后计算这些邻居单元上的量，参见引子。注意这里是活跃邻居单元，保证是在同一层级或更粗一级上。如果不活跃，还得看它的child。
具体为：
```cpp
template <int dim>
void
GradientEstimation::estimate (const DoFHandler<dim> &dof_handler,
        const Vector<double> &solution,
        Vector<float> &error_per_cell)
{
    Assert (error_per_cell.size() == dof_handler.get_triangulation().n_active_cells(),
            ExcInvalidVectorLength (error_per_cell.size(),
                dof_handler.get_triangulation().n_active_cells()));
    typedef std_cxx11::tuple<typename DoFHandler<dim>::active_cell_iterator,Vector<float>::iterator>
        IteratorTuple;
    SynchronousIterators<IteratorTuple>
        begin_sync_it (IteratorTuple (dof_handler.begin_active(),
                    error_per_cell.begin())),
                      end_sync_it (IteratorTuple (dof_handler.end(),
                                  error_per_cell.end()));
    WorkStream::run (begin_sync_it,
            end_sync_it,
            &GradientEstimation::template estimate_cell<dim>,
            std_cxx11::function<void (const EstimateCopyData &)> (),
            EstimateScratchData<dim> (dof_handler.get_fe(),
                solution),
            EstimateCopyData ());
}
template <int dim>
void
GradientEstimation::estimate_cell (const SynchronousIterators<std_cxx11::tuple<typename DoFHandler<dim>::active_cell_iterator,
        Vector<float>::iterator> > &cell,
        EstimateScratchData<dim> &scratch_data,
        const EstimateCopyData &)
{
    Tensor<2,dim> Y;
    std::vector<typename DoFHandler<dim>::active_cell_iterator> active_neighbors;
    active_neighbors.reserve (GeometryInfo<dim>::faces_per_cell *
            GeometryInfo<dim>::max_children_per_face);
    typename DoFHandler<dim>::active_cell_iterator cell_it(std_cxx11::get<0>(cell.iterators));
    scratch_data.fe_midpoint_value.reinit (cell_it);
    Tensor<1,dim> projected_gradient;
    active_neighbors.clear ();
    for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
        if (! std_cxx11::get<0>(cell.iterators)->at_boundary(face_no))
        {
            const typename DoFHandler<dim>::face_iterator
                face = std_cxx11::get<0>(cell.iterators)->face(face_no);
            const typename DoFHandler<dim>::cell_iterator
                neighbor = std_cxx11::get<0>(cell.iterators)->neighbor(face_no);
            if (neighbor->active())
                active_neighbors.push_back (neighbor);
            else
            {
                if (dim == 1)
                {
                    typename DoFHandler<dim>::cell_iterator
                        neighbor_child = neighbor;
                    while (neighbor_child->has_children())
                        neighbor_child = neighbor_child->child (face_no==0 ? 1 : 0);
                    Assert (neighbor_child->neighbor(face_no==0 ? 1 : 0)
                            ==std_cxx11::get<0>(cell.iterators),ExcInternalError());
                    active_neighbors.push_back (neighbor_child);
                }
                else
                    for (unsigned int subface_no=0; subface_no<face->n_children(); ++subface_no)
                        active_neighbors.push_back (
                                std_cxx11::get<0>(cell.iterators)->neighbor_child_on_subface(face_no,subface_no));
            }
        }
    const Point<dim> this_center = scratch_data.fe_midpoint_value.quadrature_point(0);
    std::vector<double> this_midpoint_value(1);
    scratch_data.fe_midpoint_value.get_function_values (scratch_data.solution, this_midpoint_value);
    std::vector<double> neighbor_midpoint_value(1);
    typename std::vector<typename DoFHandler<dim>::active_cell_iterator>::const_iterator
        neighbor_ptr = active_neighbors.begin();
    for (; neighbor_ptr!=active_neighbors.end(); ++neighbor_ptr)
    {
        const typename DoFHandler<dim>::active_cell_iterator
            neighbor = *neighbor_ptr;
        scratch_data.fe_midpoint_value.reinit (neighbor);
        const Point<dim> neighbor_center = scratch_data.fe_midpoint_value.quadrature_point(0);
        scratch_data.fe_midpoint_value.get_function_values (scratch_data.solution,
                neighbor_midpoint_value);
        Tensor<1,dim> y = neighbor_center - this_center;
        const double distance = y.norm();
        y /= distance;
        for (unsigned int i=0; i<dim; ++i)
            for (unsigned int j=0; j<dim; ++j)
                Y[i][j] += y[i] * y[j];
        projected_gradient += (neighbor_midpoint_value[0] -
                this_midpoint_value[0]) /
            distance *
            y;
    }
    AssertThrow (determinant(Y) != 0,
            ExcInsufficientDirections());
    const Tensor<2,dim> Y_inverse = invert(Y);
    Tensor<1,dim> gradient = Y_inverse * projected_gradient;
    *(std_cxx11::get<1>(cell.iterators)) = (std::pow(std_cxx11::get<0>(cell.iterators)->diameter(),
                1+1.0*dim/2) *
            std::sqrt(gradient.norm_square()));
}
```
细节见帮助文档。简要说一下其中的几个特点：一是能判断单元是否活跃；二是能得到单元的子单元，甚至是它的左边子单元还是右边；三是能得到单元的中点上的函数值。
然后是main函数：
```cpp
int main ()
{
    try
    {
        ::MultithreadInfo::set_thread_limit();
        Step9::AdvectionProblem<2> advection_problem_2d;
        advection_problem_2d.run ();
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
跟之前差不多，区别是使用MultithreadInfo来限制线程最大数目。

# 计算结果
自适应网格是：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk098ejhj30mb0m0aii.jpg)
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjk0jwpe3j30qx0k5wkl.jpg)
