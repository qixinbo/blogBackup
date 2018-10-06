---
title: 求解偏微分方程开源有限元软件deal.II学习--Step 10
tags: [deal.II]
categories: computational material science
date: 2016-9-16
---

# 引子
这个小例子主要演示更高阶的映射。映射的意思是参考单元和真实单元之间的变换，之前的例子中默认使用线性或双线性映射。这里的线性映射不是形函数用线性近似，同时也不是Cn单元的概念，Cn单元代表解的最高阶导数的概念。如果边界是弯曲的话，用线性近似就可能不充分。如果使用分段二次抛物线来近似的话，就说这是二次或Q2近似。如果使用分段三次多项式近似的话，就称为Q3近似。依次类推。
这里是用逼近圆周率来说明映射问题。用两种方法：一种是对半径为1的圆进行三角剖分，然后积分，如果完全是圆，那么它的面积精确为圆周率，但因为是多项式近似，所以肯定不精确，这里展示不同映射下随着网格细化的收敛性。第二种不是计算面积，而是计算周长，那么圆周率就大约是周长的一半。

# 程序解析
```cpp
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
```
这是之前用过的头文件。
```cpp
#include <deal.II/fe/mapping_q.h>
```
这是新的头文件。用来声明MappingQ类，进行任意阶数的多项式映射。
```cpp
#include <iostream>
#include <fstream>
#include <cmath>
```
C++的标准头文件。
然后开始定义这个问题的命名空间。
```cpp
namespace Step10
{
    using namespace dealii;
```
先定义一种非常精确的圆周率值用于后面比较：
```cpp
const long double pi = 3.141592653589793238462643;
```
下面是输出。这里因为是个小程序，没有用类，而是用函数模板，参数是空间维度。
```cpp
template <int dim>
void gnuplot_output()
{
    std::cout << "Output of grids into gnuplot files:" << std::endl
        << "===================================" << std::endl;
    Triangulation<dim> triangulation;
    GridGenerator::hyper_ball (triangulation);
    static const SphericalManifold<dim> boundary;
    triangulation.set_all_manifold_ids_on_boundary(0);
    triangulation.set_manifold (0, boundary);
```
首先产生一个粗的三角剖分，施加一个合适的边界描述。
```cpp
for (unsigned int refinement=0; refinement<2;
        ++refinement, triangulation.refine_global(1))
{
    std::cout << "Refinement level: " << refinement << std::endl;
    std::string filename_base = "ball";
    filename_base += '0'+refinement;
    for (unsigned int degree=1; degree<4; ++degree)
    {
        std::cout << "Degree = " << degree << std::endl;
        const MappingQ<dim> mapping (degree);
```
然后使用不同的映射，分别是Q1、Q2和Q3。当分段线性映射，即Q1时，可以直接使用MappingQ1这个类。Attention！！MappingQ1也是很多函数和类默认使用的映射方式，如果不明确指定映射方式的话。
```cpp
GridOut grid_out;
GridOutFlags::Gnuplot gnuplot_flags(false, 30);
grid_out.set_flags(gnuplot_flags);
std::string filename = filename_base+"_mapping_q";
filename += ('0'+degree);
filename += ".dat";
std::ofstream gnuplot_file (filename.c_str());
grid_out.write_gnuplot (triangulation, gnuplot_file, &mapping);
    }
    std::cout << std::endl;
}
}
```
然后就是使用Gnuplot进行输出。
下面就进入该算例的主要部分：圆周率的计算。
因为这里的圆的半径是1，所以圆的面积的计算就是对常数1在整个计算域上积分：
$$
\int\_K 1 dx=\int\_{\hat K} 1 \ \textrm{det}\ J(\hat x) d\hat x \approx \sum\_i \textrm{det} \ J(\hat x\_i)w(\hat x\_i)
$$
计算开始：
```cpp
template <int dim>
void compute_pi_by_area ()
{
    std::cout << "Computation of Pi by the area:" << std::endl
        << "==============================" << std::endl;
    const QGauss<dim> quadrature(4);
    for (unsigned int degree=1; degree<5; ++degree)
    {
        std::cout << "Degree = " << degree << std::endl;
        Triangulation<dim> triangulation;
        GridGenerator::hyper_ball (triangulation);
        static const SphericalManifold<dim> boundary;
        triangulation.set_all_manifold_ids_on_boundary (0);
        triangulation.set_manifold(0, boundary);
        const MappingQ<dim> mapping (degree);
```
这里选择的积分精度足够大，保证在这里任意映射下都能正确求解。
```cpp
const FE_Q<dim> dummy_fe (1);
DoFHandler<dim> dof_handler (triangulation);
```
这里创建了虚假的有限单元和自由度句柄，因为这里不需要知道在积分点上有限元上的值，只是为了知道那个积分权重。注意，这里有限单元的形函数次数是1，再次说明线性形函数跟线性映射不是一个概念。
```cpp
FEValues<dim> fe_values (mapping, dummy_fe, quadrature,
        update_JxW_values);
```
这里传递给FEValues的第一个参数是mapping对象，之前的例子中这个参数是省略的，默认使用MappingQ1这样的对象。
```cpp
ConvergenceTable table;
```
这里再创建一个收敛性表格，来看收敛速率。
然后开始对细化步数循环：
```cpp
for (unsigned int refinement=0; refinement<6;
        ++refinement, triangulation.refine_global (1))
{
    table.add_value("cells", triangulation.n_active_cells());
    dof_handler.distribute_dofs (dummy_fe);
    long double area = 0;
```
然后对所有单元进行循环：
```cpp
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
     endc = dof_handler.end();
for (; cell!=endc; ++cell)
{
    fe_values.reinit (cell);
    for (unsigned int i=0; i<fe_values.n_quadrature_points; ++i)
        area += fe_values.JxW (i);
}
table.add_value("eval.pi", static_cast<double> (area));
table.add_value("error", static_cast<double> (std::fabs(area-pi)));
}
```
并且存储进table中。
然后开始计算收敛性：
```cpp
table.omit_column_from_convergence_rate_evaluation("cells");
table.omit_column_from_convergence_rate_evaluation("eval.pi");
table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
table.set_precision("eval.pi", 16);
table.set_scientific("error", true);
table.write_text(std::cout);
std::cout << std::endl;
}
}

下面的计算方法不是计算面积，而是计算周长：
```cpp
template <int dim>
void compute_pi_by_perimeter ()
{
    std::cout << "Computation of Pi by the perimeter:" << std::endl
        << "===================================" << std::endl;
    const QGauss<dim-1> quadrature(4);
```
注意，上面的QGauss的维度是dim-1，这是因为是对边积分，而不是对体单元。
```cpp
for (unsigned int degree=1; degree<5; ++degree)
{
    std::cout << "Degree = " << degree << std::endl;
    Triangulation<dim> triangulation;
    GridGenerator::hyper_ball (triangulation);
    static const SphericalManifold<dim> boundary;
    triangulation.set_all_manifold_ids_on_boundary (0);
    triangulation.set_manifold (0, boundary);
    const MappingQ<dim> mapping (degree);
    const FE_Q<dim> fe (1);
    DoFHandler<dim> dof_handler (triangulation);
```
这些跟上面的一样。
```cpp
FEFaceValues<dim> fe_face_values (mapping, fe, quadrature,
        update_JxW_values);
ConvergenceTable table;
for (unsigned int refinement=0; refinement<6;
        ++refinement, triangulation.refine_global (1))
{
    table.add_value("cells", triangulation.n_active_cells());
    dof_handler.distribute_dofs (fe);
```
Attention！这里是创建的FEFaceValues而不是FEValues对象。
```cpp
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
     endc = dof_handler.end();
long double perimeter = 0;
for (; cell!=endc; ++cell)
    for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
        if (cell->face(face_no)->at_boundary())
        {
            fe_face_values.reinit (cell, face_no);
            for (unsigned int i=0; i<fe_face_values.n_quadrature_points; ++i)
                perimeter += fe_face_values.JxW (i);
        }
table.add_value("eval.pi", static_cast<double> (perimeter/2.));
table.add_value("error", static_cast<double> (std::fabs(perimeter/2.-pi)));
}
table.omit_column_from_convergence_rate_evaluation("cells");
table.omit_column_from_convergence_rate_evaluation("eval.pi");
table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
table.set_precision("eval.pi", 16);
table.set_scientific("error", true);
table.write_text(std::cout);
std::cout << std::endl;
}
}
}
```
Attention！！这里积分时，只有边界上的才能被叠加进去，所以需要有一个判断是否边界的语句。
然后是main函数：
```cpp
int main ()
{
    try
    {
        std::cout.precision (16);
        Step10::gnuplot_output<2>();
        Step10::compute_pi_by_area<2> ();
        Step10::compute_pi_by_perimeter<2> ();
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
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjl2m8t1wj308c08c74g.jpg)
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjl33o61aj308c08c74i.jpg)
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjl3b1j24j308c081mxd.jpg)
上面展示的是粗网格上的结果。实线是网格，虚线是精确的圆。
可以看出，即使在粗网格上，二次和三次映射都取得了很好的吻合结果。
然后是计算精度：
```cpp
Computation of Pi by the area:
==============================
Degree = 1
cells eval.pi error
5 1.9999999999999993 1.1416e+00 -
20 2.8284271247461894 3.1317e-01 1.87
80 3.0614674589207178 8.0125e-02 1.97
320 3.1214451522580520 2.0148e-02 1.99
1280 3.1365484905459393 5.0442e-03 2.00
5120 3.1403311569547534 1.2615e-03 2.00
Degree = 2
cells eval.pi error
5 3.1045694996615865 3.7023e-02 -
20 3.1391475703122271 2.4451e-03 3.92
80 3.1414377167038303 1.5494e-04 3.98
320 3.1415829366419015 9.7169e-06 4.00
1280 3.1415920457576911 6.0783e-07 4.00
5120 3.1415926155921139 3.7998e-08 4.00
Degree = 3
cells eval.pi error
5 3.1410033851499310 5.8927e-04 -
20 3.1415830393583861 9.6142e-06 5.94
80 3.1415925017363837 1.5185e-07 5.98
320 3.1415926512106722 2.3791e-09 6.00
1280 3.1415926535525962 3.7197e-11 6.00
5120 3.1415926535892140 5.7923e-13 6.00
Degree = 4
cells eval.pi error
5 3.1415871927401127 5.4608e-06 -
20 3.1415926314742437 2.2116e-08 7.95
80 3.1415926535026228 8.7170e-11 7.99
320 3.1415926535894529 3.4036e-13 8.00
1280 3.1415926535897927 2.9187e-16 10.19
5120 3.1415926535897944 1.3509e-15 -2.21
Computation of Pi by the perimeter:
===================================
Degree = 1
cells eval.pi error
5 2.8284271247461898 3.1317e-01 -
20 3.0614674589207178 8.0125e-02 1.97
80 3.1214451522580520 2.0148e-02 1.99
320 3.1365484905459393 5.0442e-03 2.00
1280 3.1403311569547525 1.2615e-03 2.00
5120 3.1412772509327729 3.1540e-04 2.00
Degree = 2
cells eval.pi error
5 3.1248930668550594 1.6700e-02 -
20 3.1404050605605449 1.1876e-03 3.81
80 3.1415157631807014 7.6890e-05 3.95
320 3.1415878042798617 4.8493e-06 3.99
1280 3.1415923498174534 3.0377e-07 4.00
5120 3.1415926345932004 1.8997e-08 4.00
Degree = 3
cells eval.pi error
5 3.1414940401456057 9.8613e-05 -
20 3.1415913432549156 1.3103e-06 6.23
80 3.1415926341726914 1.9417e-08 6.08
320 3.1415926532906893 2.9910e-10 6.02
1280 3.1415926535851360 4.6571e-12 6.01
5120 3.1415926535897203 7.2845e-14 6.00
Degree = 4
cells eval.pi error
5 3.1415921029432576 5.5065e-07 -
20 3.1415926513737600 2.2160e-09 7.96
80 3.1415926535810712 8.7218e-12 7.99
320 3.1415926535897594 3.3668e-14 8.02
1280 3.1415926535897922 1.0617e-15 4.99
5120 3.1415926535897931 1.0061e-16 3.40
```
可以看出，随着映射次数提高，计算精度也在提高，而Q1映射在最细网格上的精度还不如Q3映射在最粗网格上的精度。最后一栏是收敛阶，可以看出阶数达到了$h^{2p}$。


