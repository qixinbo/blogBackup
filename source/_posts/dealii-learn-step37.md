---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 37
tags: [deal.II]
categories: computational material science
date: 2016-12-26
---

2016-12-26更新：
将方程求解过程更加细致地表述。

# 引子
本例展示了一种无矩阵(matrix-free)方法的使用，即不明确存储矩阵元素来求解二阶变系数Possion方程。同时求解时使用了多重网格算法。
应用无矩阵方法的原因是：当前科学计算的一个瓶颈是对于内存而不是高速缓存中的数据的读取：比如对于一个矩阵和一个向量的乘法运算，现在的CPU能很快地计算浮点数的乘法和加法，但通常需要很长时间等待数据从内存中传入。因此，如果我们不再从内存中寻找矩阵元素，而是重新计算它们，那么可能整体时间就能减少。
这个无矩阵方法中还涉及向量化/矢量化编程：把for循环的操作，用矩阵操作的形式代替。在向量化编程中，程序设计以向量为基本操作单位,采用向量运算代替循环操作以提高运行效率，这里的“向量”不同于一般数学中的概念, 它指的是数组或矩阵。该类具体进行向量化时，将一些单元合并成一个宏单元，这样用一条指令就可以同时对多个单元进行操作。

## 算例
这里要求解的是变系数Possion方程：
\begin{equation}
\begin{split}
-\nabla \cdot a (\mathbf x) \nabla u &= 1, \\\
u &= 0 \quad \text{on}\ \partial \Omega
\end{split}
\end{equation}
计算域是$\Omega=[0,1]^3$，变系数是$a(\mathbf x)=\frac{1}{0.05+2\|\mathbf x \|^2}$。系数虽然是以原点对称，但计算域不是，所以得到的结果也是不对称的。

方程的弱形式为：
\begin{equation}
\begin{split}
a (\mathbf x) \nabla u \nabla w &= 1.0*w, \\\
u &= 0 \quad \text{on}\ \partial \Omega
\end{split}
\end{equation}

所以，各项对应到程序中就是：
左端项：
```cpp
phi.submit_gradient (coefficient(cell,q) *
phi.get_gradient(q), q);
```
右端项：
```cpp
rhs_val += (fe_values.shape_value(i,q) * 1.0 *
fe_values.JxW(q));
```
注意，上述左端项中仅有试探函数的梯度，而没有试探函数的值，所以，程序中是：
```cpp
phi.evaluate (false,true,false);
phi.integrate (false,true);
```
如果仅用到试探函数的值，那么就得变成：
```cpp
phi.evaluate (true,false,false);
phi.submit_value(...);
phi.integrate (true,false);
```

## 矩阵与向量的乘法
求解一个方程，形如：
$$
Au=b
$$
时，如果按照传统思路，需要求一个稀疏矩阵A与一个向量u的乘积。

先来看看有限元矩阵A是怎样组装的：
$$
A=\sum\_{\mathrm{cell}=1}^{\mathrm{n\_{cells}}} P\_{\mathrm{cell,{loc-glob}}}^T A\_{\mathrm{cell}} P\_{\mathrm{cell,{loc-glob}}}
$$
上式中，长方矩阵$P\_{\mathrm {cell,{loc-glob}}}$定义当前单元从局部自由度到全局自由度的指标映射。
如果想要对上述矩阵A乘以一个向量u，即：
$$
\begin{split} 
y &= A\cdot u = (\sum\_{\text{cell}=1}^{\mathrm{n\_cells}} P\_\mathrm{cell,{loc-glob}}^T A\_\mathrm{cell} P\_\mathrm{cell,{loc-glob}}) \cdot u \\\ 
&= \sum\_{\mathrm{cell}=1}^{\mathrm{n\_cells}} P\_\mathrm{cell,{loc-glob}}^T A\_\mathrm{cell} u\_\mathrm{cell} \\\ 
&= \sum\_{\mathrm{cell}=1}^{\mathrm{n\_cells}} P\_\mathrm{cell,{loc-glob}}^T v\_\mathrm{cell} 
\end{split}
$$
编程思路是这样的：先建立单元矩阵，即$A\_{cell}$，然后用局部自由度和全局自由度之间的映射将真实空间中的u转换成单元上的量$u\_{cell}$。再用vmult函数实现单元矩阵与单元向量的乘法，即：
$$
A\_{cell}u\_{cell}=v\_{cell}
$$，得到目标向量后，再用局部到全局的映射将该向量转换到真实空间中。

这个思路很正确，但是很慢。对于每个单元，都要用三个嵌套循环来构建单元矩阵，然后再用两个嵌套循环作乘法。一个改进的方法是意识到单元矩阵可在概念上视为三个矩阵的乘积：
$$
A\_\mathrm{cell} = B\_\mathrm{cell}^T D\_\mathrm{cell} B\_\mathrm{cell}
$$
这个形式跟力学中的单元刚度矩阵就是完全相同的了，矩阵B是形函数梯度矩阵，D是弹性矩阵。
当这个矩阵跟一个向量相乘时：
$$
A\_\mathrm{cell}\cdot u\_\mathrm{cell} = B\_\mathrm{cell}^T D\_\mathrm{cell} B\_\mathrm{cell}\cdot u\_\mathrm{cell}
$$
这样，就是从右往左做三次矩阵-向量乘法。这避免了在构建单元矩阵时的三次嵌套循环。
上述代码中一个瓶颈在于对每个单元都做FEValues::reinit，这个操作所用的时间可能跟其他所有操作加起来相同(至少对非规则网格是这样的，规则网格上的梯度通常不变)。这明显不理想，所以最好是优化一下。reinit所做的工作是：通过雅各比矩阵将参考单元上的梯度进行变换，从而计算真实空间上的梯度。这在每个单元上的每个积分点上对每个形函数都要操作。通常雅各比矩阵不依赖于形函数，但它在不同的积分点上不同。在之前的算例中，矩阵只构建一次，那么就没有必要对这个reinit函数做什么，因此在计算局部矩阵元素时必须得做这个雅各比变换。
然而，在应用无矩阵运算时，我们对只使用矩阵一次没有兴趣，而是想要对矩阵多次使用。所以，就想能不能缓存什么东西用于加速计算，但也不能缓存太多数据，否则就会陷入之前获取内存中数据缓慢的瓶颈。
这里用的方法就是识别出那个雅各比变换，然后仅在一个参考单元上应用一次。
即：
$$
v\_\mathrm{cell} = B\_\mathrm{ref\_cell}^T J\_\mathrm{cell}^T D J\_\mathrm{cell} B\_\mathrm{ref\_cell} u\_\mathrm{cell}, \quad v = \sum\_{\mathrm{cell}=1}^{\mathrm{n\_cells}} P\_\mathrm{cell,{loc-glob}}^T v\_\mathrm{cell}.
$$

使用一个无矩阵、基于单元的有限元算子，需要使用跟之前代码不同的设计思路，deal.II中存储这种数据结构的类是MatrixFree类，然后FEEvaluation类来具体使用它。

## 无矩阵对象的使用
本例中，除了问题类的定义，还有一个类，LaplaceOperator，用来表示差分算子。为了通用性，将它设计成一个矩阵，即你可以获得它的大小，也可以将它用在一个向量上。跟真实的矩阵不同点在于：它不存储矩阵的元素，仅仅知道当它用于一个向量时应该怎么做。在这个类中用来存储数据的类是MatrixFree，它包含了局部自由度与整体自由度之间的映射关系，即雅各比矩阵，也能在并行计算时遍历所有单元，它能保证只对不共享自由度的单元进行操作，这就保证了线程安全性，它比之前提到的WorkStream类更加先进。
然后就是这个类中对MatrixFree类型数据的使用，包括以下功能：返回数据的维度、多种形式的矩阵-向量乘法、初始化数据等。这个类需要三个参数：维度(所以能处理不同维度的问题)、有限元的度(这是为了后续FEEvaluation的快速计算)、潜在的标量类型(我们想要对于最终矩阵使用double类型，而对于多重网格上的矩阵使用float类型)。
这个类的数据成员包括：真正使用的MatrixFree对象、在所有积分点上计算的变系数(这样在矩阵-向量乘法中就不用重复计算)、存储矩阵对角元素的容器(用于多重网格平滑)。

初始化数据时，最重要的是创建了MatrixFree的对象实例，同时计算了积分点上的系数：
```cpp
additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
update_quadrature_points);
data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
additional_data);
evaluate_coefficient(Coefficient<dim>());
```
注意，上面在MatrixFree的AdditionalData中需要设定要更新的flag。

然后就是这个类的主要功能所在：计算矩阵-向量乘积。注意：该类中的单元范围通常不等于triangulation的单元数。事实上，这里的“单元”可能是个错误的概念，因为它实际上是多个单元上的积分点的集合，MatrixFree对象将多个单元的积分点分组，变成一个块，形成一个新的向量化高度。这些“单元”的个数可以通过MatrixFree类的MatrixFree::n_macro_cells()获得。与之前的单元迭代器相比，MatrixFree类的所有单元都在同一层级的数组中，这样就可以直接通过整数指标来索引它们。
LaplaceOperator算子的使用步骤如下：
先创建一个FEEvaluation对象，用来进行后续对MatrixFree对象的计算。这个对象接收五个参数，分别是：维度、有限单元的degree、每个方向上积分点的个数(默认是fe.degree+1)、分量的个数(可以是向量，但这里是标量)、数据类型(因为想对多重网格预条件子只设置float类型)。然后就对给定的“单元”范围进行循环，在每个“macro cell”上具体所做的事情如下：
```cpp
phi.reinit (cell);
phi.read_dof_values(src);
phi.evaluate (false,true,false);
for (unsigned int q=0; q<phi.n_q_points; ++q)
phi.submit_gradient (coefficient(cell,q) *
phi.get_gradient(q), q);
phi.integrate (false,true);
phi.distribute_local_to_global (dst);
```
(1)告诉那个FEEvaluation对象它要作用的单元,
(2)读入源向量的值，即上面分析中的$u\_{cell}$,
(3)计算参考单元的梯度。因为FEEvaluation既能计算函数值，也能计算梯度，所以它提供了一个统一的界面来计算从0阶到2阶的梯度(0阶梯度即值本身)。因为这里只需计算梯度，所以就在第二个参数位置设置为true，在第一个和第三个参数位置设置为false。这一步的复杂度比传统的用FEValues来计算的复杂度要降低很多。
(4)然后就是雅各比矩阵变换、与变系数和积分权重的相乘。FEEvaluation有个函数getGradient能应用雅各比变换，同时获得真实空间中的梯度，那么就用它乘以变系数。再用submitGradient施加第二个雅各比矩阵和积分权重。注意：这里不要对积分点重复，否则会造成在此积分点上多次运算。
(5)然后就是积分。使用函数integrate，它的两个参数分别是说明是否对值、梯度进行积分。因为这里仅需对梯度积分，所以将第一个设置为false，第二个设置为true。
(6)最后就是将单元贡献叠加到整体上去。

# 程序解析
## 头文件
头文件需要增加两个：
```cpp
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
```
## 驱动程序
```cpp
using namespace Step37;
LaplaceProblem<dimension> laplace_problem;
laplace_problem.run ();
```
创建问题类，注意这个类中不再使用稀疏矩阵，而是使用无矩阵方式：
```cpp
typedef LaplaceOperator<dim,degree_finite_element,double> SystemMatrixType;
typedef LaplaceOperator<dim,degree_finite_element,float> LevelMatrixType;
SystemMatrixType system_matrix;
MGLevelObject<LevelMatrixType> mg_matrices;
```
其中无矩阵对象是LaplaceOperator类中的一个数据成员：
```cpp
MatrixFree<dim,number> data;
```
问题类中还提供一个用于输出每段程序进行多长时间的输出流：
```cpp
ConditionalOStream time_details;
```
在程序中其默认是关闭的，需要在构造函数初始化时更改一下让其起作用：
```cpp
template <int dim>
LaplaceProblem<dim>::LaplaceProblem ()
:
fe (degree_finite_element),
dof_handler (triangulation),
time_details (std::cout, true)
{}
```
然后执行run函数。

## 运行函数
run函数控制程序运行的整个流程，包括创建并加密网格、创建系统、组装系统、组装多重网格、求解、输出结果：
```cpp
if (cycle == 0)
{
GridGenerator::hyper_cube (triangulation, 0., 1.);
triangulation.refine_global (3-dim);
}
triangulation.refine_global (1);
setup_system ();
assemble_system ();
assemble_multigrid ();
solve ();
output_results (cycle);
std::cout << std::endl;
```
## 创建系统
注意是做一些初始化工作，初始化无矩阵对象、解向量、右端项等。
比如，初始化无矩阵对象：
```cpp
system_matrix.reinit (dof_handler, constraints);
```
此处只有一句话，实际做的东西很多：
```cpp
template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::reinit (const DoFHandler<dim> &dof_handler,
const ConstraintMatrix &constraints,
const unsigned int level)
{
typename MatrixFree<dim,number>::AdditionalData additional_data;
additional_data.tasks_parallel_scheme =
MatrixFree<dim,number>::AdditionalData::partition_color;
additional_data.level_mg_handler = level;
additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
update_quadrature_points);
data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
additional_data);
evaluate_coefficient(Coefficient<dim>());
}
```
从上可以看出，主要做的就是对无矩阵对象的data初始化，同时最后一句话还将算子类中的系数进行了赋值，具体来说就是先用一个FEEvaluation对象将data中的信息提取出来，用其取得积分点位置，这样就能通过系数函数确定积分点上的系数值，然后赋值：
```cpp
template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::
evaluate_coefficient (const Coefficient<dim> &coefficient_function)
{
const unsigned int n_cells = data.n_macro_cells();
FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
coefficient.reinit (n_cells, phi.n_q_points);
for (unsigned int cell=0; cell<n_cells; ++cell)
{
phi.reinit (cell);
for (unsigned int q=0; q<phi.n_q_points; ++q)
coefficient(cell,q) =
coefficient_function.value(phi.quadrature_point(q));
}
}
```
## 组装系统
这一步不需要组装矩阵，只需要组装右端项，做法跟传统的组装右端项相同：
```cpp
template <int dim>
void LaplaceProblem<dim>::assemble_system ()
{
Timer time;
QGauss<dim> quadrature_formula(fe.degree+1);
FEValues<dim> fe_values (fe, quadrature_formula,
update_values | update_JxW_values);
const unsigned int dofs_per_cell = fe.dofs_per_cell;
const unsigned int n_q_points = quadrature_formula.size();
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
cell->get_dof_indices (local_dof_indices);
fe_values.reinit (cell);
for (unsigned int i=0; i<dofs_per_cell; ++i)
{
double rhs_val = 0;
for (unsigned int q=0; q<n_q_points; ++q)
rhs_val += (fe_values.shape_value(i,q) * 1.0 *
fe_values.JxW(q));
system_rhs(local_dof_indices[i]) += rhs_val;
}
}
constraints.condense(system_rhs);
setup_time += time.wall_time();
time_details << "Assemble right hand side (CPU/wall) "
<< time() << "s/" << time.wall_time() << "s" << std::endl;
}
```
## 组装多重网格
## 求解
用共轭梯度法求解：
```cpp
cg.solve (system_matrix, solution, system_rhs,
preconditioner);
```
可以看出，求解形式跟之前的有矩阵的形式是相同的，这个地方是个坑，看似相同，但奥秘隐藏在算子类的vmult函数中。实际，共轭梯度法在计算时会调用这个函数。
```cpp
template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::vmult (Vector<double> &dst,
const Vector<double> &src) const
{
dst = 0;
vmult_add (dst, src);
}
```
然后，
```cpp
template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::vmult_add (Vector<double> &dst,
const Vector<double> &src) const
{
data.cell_loop (&LaplaceOperator::local_apply, this, dst, src);
const std::vector<unsigned int> &
constrained_dofs = data.get_constrained_dofs();
for (unsigned int i=0; i<constrained_dofs.size(); ++i)
dst(constrained_dofs[i]) += src(constrained_dofs[i]);
}
```
然后，
```cpp
template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::
local_apply (const MatrixFree<dim,number> &data,
Vector<double> &dst,
const Vector<double> &src,
const std::pair<unsigned int,unsigned int> &cell_range) const
{
FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
{
phi.reinit (cell);
phi.read_dof_values(src);
phi.evaluate (false,true,false);
for (unsigned int q=0; q<phi.n_q_points; ++q)
phi.submit_gradient (coefficient(cell,q) *
phi.get_gradient(q), q);
phi.integrate (false,true);
phi.distribute_local_to_global (dst);
}
}
```
所以，这个地方是个连环调用。

# 扩展
要深刻理解方程中各项在程序中是怎样实现的，这样才能扩展。
本例的求解过程实际是prisms-pf中椭圆型方程的求解思路，仔细领会之。


