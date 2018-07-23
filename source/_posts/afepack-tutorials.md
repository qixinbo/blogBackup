---
title: 自适应有限元包AFEPack教学系列
tags: [AFEPack, linux]
categories: programming
date: 2016-3-11
---
写在前面：本文转自原AFEPack论坛afepack.org。

> AFEPack教学系列转载自北京大学数学学院李若教授之个人博客。
> 如果需要转载本教学系列，请保持文章完整性，并注明转载出处。


# 李老师的话
学校的服务器老是有问题，希望这个地方能够稳定一些。我把AFEPack的使用写一些系列的小例子，从最简单的开始，希望能够为我这个软件包的使用者提供一些方便。

我在大年三十的晚上更新了AFEPack 的主页上的一点内容，现在主要是照抄下来，给想用AFEPack的人一些方向性的指导。在过去的几年中间，我多少花了一些力气来推广这个包，希望能够为计算数学的发展做一点事情。由于一些令人失望的经验，我对于推广使用这个软件包并不抱太乐观的态度。由于Linux 操作系统和C++这两个东西对于我们大多数同行和同学来说是两个巨大的困难，真正学会了我的这个软件包的使用者少之又少。如果期望从不熟悉环境到能够使用这个包来进行研究工作，假设您确实有研究上的需求，因而有学习使用这个包的动力，那么至少需要在半年时间中，每天都花上两个小时以上学习它，才有可能有所小成。在当今这个日新月异的时代里，这个时间恐怕已经超出了大多数期望尽快发表SCI论文而能够付出的可接受时间尺度。这个包的用户对象是研究人员，是那些为了能够研究新的算法而需求而设计的，如果您期望使用鼠标完成您的工作，那么这个包绝非正确的选择。

说了这么多吓人的话，您如果还能坚持读下去，我也期望对能够坚持学习下去的用户，描述一下如果您能够真正的掌握这个包的开发方式的话，可能会见到的光明前景。一般来说，在这个包的帮助下，您写一个复杂的偏微分方程 组的有限元程序的工作，在一到两个小时内完成是并不困难的。而且您的程序天生会具有相当大的可重用性：解一个新的问题常常只需要在旧程序上进行少许修改就能完成，而很多并不是太高要求的问题都可以在库中附带的例子上进行不多的修改即可解决。

在过去的两年时间里，AFEPack的主页一直都没有更新。一个问题是，两年没有更新的网页为什么现在要进行更新呢？事实上，这次终于进行的更新也只是进行了少许的修改，更新主要的目的是期望为了在我基本上终止对这个包的进一步开发的时候，为它的发展最后加入一点动力。AFEPack在2006年的八月申请了一个计算机软件版权号，它现在对于其能力所能够覆盖的问题，基本上能够比较完好的解决了，但是对于其能力不能覆盖的问题，是不可能在其现在的基本数据结构下进行简单的修改和补充就能够解决的。我现在将精力放在了新一代的软件包的开发上，期望能够在整个的自适应算法上有比较全面的发展，这个新的包的结构虽然在我看来非常精致而美丽，但是也很难以理解。

AFEPack 是刘文斌教授和我开发的基础有限元和有限元自适应软件包，请在它的主页上了解详细的信息。这是一个给基础数值技术研究人员使用的C++库函数包，熟练的使用者能够通过这个包非常大程度的提高工作效率。这个源码包以后若非修正Bug，将不会再频繁更新。

# AFEPack安装的步骤
一、请先安装deal.II软件包，因为我使用了deal.II的线性代数包，所以您只需要编译deal.II的base和lac部分就可以了，具体操作如下：
1. 下载deal.II 的源码包；
2. 将这个软件包解压缩到，比如在/usr/local/deal.II 下面；
3. 进入/usr/local/deal.II，运行./configure；
4. 编译：make base lac；
5. 将deal.II 的头文件和库文件链接到系统目录下，在我的机器上为：
```cpp
[rli@circus /usr/local/include]$ ln -s ../deal.II/base/include/base .
[rli@circus /usr/local/include]$ ln -s ../deal.II/lac/include/lac .
[rli@circus /usr/local/lib]$ ln -s ../deal.II/base/lib/lib* .
[rli@circus /usr/local/lib]$ ln -s ../deal.II/lac/lib/lib* .
```
二、然后您就可以编译AFEPack了，先获得AFEPack的源码包，并进行解压，比如解压到 /usr/local/AFEPack下面；

三、请您在AFEPack的源码目录中，运行 ./configure；
如果包括 deal.II 的头文件或者库文件找不到，可以使用指定EXTRA_INCDIR和EXCTRA_LIBDIR这两个环境变量来指定deal.II 的头文件和库文件的路径:
```cpp
env EXTRA_INCDIR="-Ipath/to/dealii/head/file" EXTRA_LIBDIR="-Lpath/to/dealii/libs" ./configure
```
如果还有问题，可能是因为GNU开发工具包的版本的问题，您可以首先将过去运行automake和autoconf等的缓冲目录删除(名为autom4*)，然后使用命令序列：
```cpp
aclocal
automake
autoconf
```
重新产生出configure 脚本；
configure 脚本接收 --help 参数的时候可以给您帮助信息；

四、运行make进行编译；
编译过程也可以分开进行，您可以到library、template目录中分别运行make进行编译；
目录example下的内容编译的时候会有问题，您可以在做完下一步以后再在exmaple目录下运行make进行编译；
在example/tools下会编译出来很多很有用的可执行程序，帮助您做很多数据文件的格式转换，很值得试一下；

五、将头文件和编译得到的库文件链接到系统目录下：
```cpp
[rli@circus /usr/local/include]$ ln -s ../AFEPack/library/include AFEPack
[rli@circus /usr/local/lib]$ ln -s ../AFEPack/library/lib/lib* .
```
您使用make install也可以做这件事。

六、现在您就能够使用AFEPack提供的功能进行编程了，具体情况请参阅文档。

您现在可以到example目录中编译和研究我提供的一些使用AFEPack进行开发的小例子，这些例子对于帮助您学习使用AFEPack进行编程，是非常有用处的。很多问题直接通过对这些小例子中的某一个进行简单的修改就可以解决。

这个软件的开发在我的博士后研究期间得到EPSRC的支持，但是开发的动机是出于研究的兴趣，没有任何商业上的企图，现在已经没有经费的支持，所以文档更新的不是太及时，我也不能对于其稳定性和正确性提供任何保证，但如果有什么问题请给我发email，我会尽力给您帮助的。如果您有兴趣参与这个软件的开发，我也非常欢迎。

我现在使用AFEPack进行研究工作中的开发的程序包括：
- 间断Galerkin方法+移动网格方法求解守恒律；
- 最优控制问题的自适应方法，包括椭圆型和抛物型的分布式控制问题，参数估计问题；
- p-Laplacian方程的快速算法；
- 求解一些Fokker-Planck方程的求解算法；

如果您也从事相应的研究工作，我可以提供大部分的源程序以供参考。如果您有使用AFEPack 进行开发的实例，希望您告诉我。如果比较成功，让我分享您的喜悦；如果还有问题，或许我可以提供一些帮助。

AFEPack 支持的相关软件下载：

- easymesh : 一个二维的网格产生程序；
[这里](http://dsec.pku.edu.cn/~rli/WiKi/Easymesh.html)有我翻译的 easymesh 的文档；
- gmsh : 一个能进行三维造型和产生网格的程序，rpm 包；
[这里](http://dsec.pku.edu.cn/~rli/WiKi/GmshTutorial.html)有我翻译的 gmsh 的文档；
- bamg : 一个二维的网格产生程序；
- IBM Open Data Explorer : 非常专业的画图软件，我最喜欢的！AFEPack支持它的数据格式，推荐使用。

AFEPack 还支持将计算结果输出成为Tecplot使用的数据格式，不过这个软件不是免费的，请您自己安排购买该软件。

# AFEPack教学系列之二: 第一个例子－－求解泊松方程

AFEPack 的基本使用方法，我们通过下面的求解泊松方程的程序来具体讲解。假设我们现在需要在区域$\Omega=[0, 1]\times[0, 1]$上来求解一个狄氏边值问题的泊松方程
$$
-\Delta u = f, \qquad u |\_{\partial \Omega} = u\_b.
$$
作为有限元程序，第一个问题就是要对区域进行剖分，得到网格的数据。我们这里使用一个简单的二维三角形网格剖分软件EasyMesh来去区域进行剖分。EasyMesh有一个一页纸的说明书，请读者自己去阅读它。EasyMesh 需要用户手工写一个对区域进行描述的文件作为输入文件，我们使用的文件名为D.d，其内容如下：
```cpp
4 # 区域的顶点的个数 #
0: 0.0 0.0 0.05 1
1: 1.0 0.0 0.05 1
2: 1.0 1.0 0.05 1
3: 0.0 1.0 0.05 1

4 # 区域的边界上边的条数 #
0: 0 1 1
1: 1 2 1
2: 2 3 1
3: 3 4 1
```
其中前面一个部分描述区域中的顶点，共有4个，然后每一行描述一个顶点的信息，其意义为

顶点的序号:    x坐标    y坐标    剖分密度h    材料标识

后面一个部分则描述区域的边界上的边的条数，共有4条，然后每一行描述一条边的信息，其意义为

边的序号:    起始顶点序号    结束顶点序号    材料标识

以这个文件的主文件名D作为输入，运行easymesh D，即可获得网格数据，存储在三个不同的文件中，名为D.n，D.s，D.e，存储的分别是节点、边和单元的信息。EasyMesh的说明中有对这些文件格式的详细说明，而AFEPack也提供了对这种数据格式的文件读入的接口。

在得到了网格数据以后，我们即可以开始对程序进行说明。我们的整个程序的原文和详细的说明如下：
```cpp
/**
 * 下面这些包含的头文件都是属于AFEPack的。
 */
#include <AFEPack/AMGSolver.h>        /// 代数多重网格求解器
#include <AFEPack/TemplateElement.h>  /// 参考单元
#include <AFEPack/FEMSpace.h>         /// 有限元空间
#include <AFEPack/Operator.h>         /// 算子
#include <AFEPack/BilinearOperator.h> /// 双线性算子
#include <AFEPack/Functional.h>       /// 泛函
#include <AFEPack/EasyMesh.h>         /// EasyMesh 接口

#define DIM 2                /**< 区域的维数 */
#define PI (4.0*atan(1.0))   /**< 常数 \pi   */

/**
 * 精确解 u 的表达式
 *
 * @param p 区域上的点的坐标: x = p[0], y = p[1]
 *
 * @return u(x, y)
 */
double _u_(const double * p)
{
  return sin(PI*p[0]) * sin(2*PI*p[1]);
}

/**
 *  右端项函数 f 的表达式
 *
 * @param p 区域上的点的坐标: x = p[0], y = p[1]
 *
 * @return f(x, y)
 */
double _f_(const double * p)
{
  return 5*PI*PI*u(p);
}

/**
 * 主函数：程序在运行的时候，我们要求有一个命令行参数，即为网格数据文
 *         件名 D。
 *
 * @param argc 命令行参数个数
 * @param argv 命令行参数列表
 *
 * @return 返回 0
 */
int main(int argc, char * argv[])
{
  /**
   * AFEPack 中的 EasyMesh 类能够读入 EasyMesh 产生的数据文件的接口，
   * 它本身是由网格类 Mesh 派生出来的，可以当成一个网格使用
   */
  EasyMesh mesh;
  mesh.readData(argv[1]); /// 以第一个命令行参数为文件名，读入网格数据

  /**
   * 下面这个段落是为有限元空间准备参考单元的数据。AFEPack 将参考单元
   * 的数据分成为四个不同的信息，包括：
   *
   *   - 参考单元的几何信息
   *   - 参考单元到网格中的单元的坐标变换
   *   - 单元上的自由度分布
   *   - 单元上的基函数
   *
   * 使用这几个信息反复组合，可以得到很多种不同的参考单元。这四个信息
   * 的管理使用了四个不同的类，名为 TemplateGeometry, CoordTransform,
   * TemplateDOF 和 BasisFunctionAdmin。这四个类都可以从数据文件中读入
   * 信息来构造其自身的结构，而对于很多常用的单元，这些数据文件都已经
   * 准备在了 AFEPack 的 template 子目录下。这些数据文件的数据格式在
   * AFEPack 的文档之中有比较详细的说明。为了能够使得 AFEPack 顺利的找
   * 到这些数据文件，我们需要设置环境变量 AFEPACK_TEMPLATE_PATH。比如
   * 对于我们这个问题的计算， 我们需要设置(假设 bash)
   *
   * <pre>
   *   $ export AFEPACK_TEMPLATE_PATH=$AFEPACK_PATH/template/triangle
   * </pre>
   *
   * 其中 AFEPACK_PATH 假设为 AFEPack 的安装路径。
   */

  /**
   * 参考单元的几何信息：其中模板参数 DIM 为参考单元的维数，对于我们现 
   * 在的三角形来说就是 2。 参考单元的几何信息中包括参考单元的几何结构， 
   * 以及进行数值积分的时候的一系列积分公式的信息。
   *
   */
  TemplateGeometry<DIM> triangle_template_geometry;
  triangle_template_geometry.readData("triangle.tmp_geo");

  /**
   * 从参考单元到网格中的单元的坐标变换：这个类有两个模板参数，分别为
   * 参考单元的维数和网格中单元的维数。这两个数可能是不一样的，比如从
   * 标准三角形到球面三角形的坐标变换。这个类中提供了将参考单元中的点
   * 和实际网格单元中的点相互进行变换以及变换的雅可比行列式的计算方法。
   */
  CoordTransform<DIM,DIM> triangle_coord_transform;
  triangle_coord_transform.readData("triangle.crd_trs");

  /**
   * 自由度分布指定了在单元的每个构件几何体上分布的自由度的个数，因此
   * 它需要在知道了参考单元的几何构型的情况下进行初始化。
   */
  TemplateDOF<DIM> triangle_template_dof(triangle_template_geometry);
  triangle_template_dof.readData("triangle.1.tmp_dof");

  /**
   * 类 BasisFunctionAdmin 事实上管理着一组基函数，每个基函数为指定自
   * 己依赖的几何体，在参考单元上的插值点，一组称为"基函数识别协议"的
   * 信息，以及每个基函数的函数值和函数梯度的计算方法。
   */
  BasisFunctionAdmin<double,DIM,DIM> triangle_basis_function(triangle_template_dof);
  triangle_basis_function.readData("triangle.1.bas_fun");

  /**
   * 将上面的这四个信息组合到类 TemplateElement 中，就得到了一个完整的
   * 参考单元。而建立有限元空间需要一个网格和一组参考单元，现在我们只
   * 使用一个参考单元。
   */
  std::vector<TemplateElement<double,DIM,DIM> > template_element(1);
  template_element[0].reinit(triangle_template_geometry,
                             triangle_template_dof,
                             triangle_coord_transform,
                             triangle_basis_function);

  /**
   * 使用我们读入的 EasyMesh 得到的网格，和刚刚创建的参考单元组来初始
   * 化有限元空间。
   */
  FEMSpace<double,DIM> fem_space(mesh, template_element);
        
  /**
   * 我们先取出网格中的单元的个数，然后据此为有限元空间中的单元保留网
   * 格中的几何体单元同样数量的空间。此后，我们使得有限元空间中的每个
   * 单元是根据网格中的相应序号的几何体单元映射上第零号参考单元得到的。
   */
  int n_element = mesh.n_geometry(DIM);
  fem_space.element().resize(n_element);
  for (int i = 0;i < n_element;i ++) {
    fem_space.element(i).reinit(fem_space,i,0);
  }

  /**
   * 下面三行分别建立有限元空间中的单元的内部数据结构，然后在整个网格
   * 上分布自由度以及为所有的自由度确定材料标识。自此，整个有限元空间
   * 的所有数据就都准备好了。
   */
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();

  /**
   * 在上面建立的这个有限元空间上，我们计算一个刚度矩阵，就是由元素
   *
   * \f[
   *    a_{ij} = \int_\Omega \nabla \phi^i \cdot \nabla \phi^j dx
   * \f]
   *
   * 所构成的矩阵。由于这是常用矩阵，因此库中间有准备这个矩阵。建立这
   * 个矩阵，我们先使用上面的有限元空间构造这个矩阵的对象，然后设置计
   * 算矩阵时候使用的数值积分公式的代数精度的阶数。最后，调用函数
   * build 即可将矩阵计算出来。
   */
  StiffMatrix<DIM,double> stiff_matrix(fem_space);
  stiff_matrix.algebricAccuracy() = 3;
  stiff_matrix.build();

  /**
   * 有限元函数 u_h 用来逼近解函数 u
   */
  FEMFunction<double,DIM> u_h(fem_space);

  /**
   * 向量 f_h 用来计算右端项 f 在离散过后得到的向量，其元素为
   *
   * \f[
   *    f_i = \int_\Omega f \phi^i dx
   * \f]
   *
   * 函数 Operator::L2Discretize 帮助我们完成这个向量的计算。
   */
  Vector<double> f_h;
  Operator::L2Discretize(&_f_, fem_space, f_h, 3);

  /**
   * 下面这段是使用上狄氏边值条件：我们为材料标识为 1 的自由度使用函数
   * 表达式计算了其函数值，从而应用上了想要的边值条件。我们通过直接修
   * 改获得的线性系统中的稀疏矩阵和右端项来达成这样的结果。
   */
  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET,
                                        1, &_u_);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(stiff_matrix, u_h, f_h);

  /**
   * 调用 AFEPack 中的代数多重网格求解器，求解线性系统
   */
  AMGSolver solver(stiff_matrix);
  solver.solve(u_h, f_h);      

  /**
   * 将解输出到一个数据文件中。我们这里使用了 Open Data Explorer 的数
   * 据文件格式，输出的数据可以使用该软件来做可视化。
   */
  u_h.writeOpenDXData("u.dx");

  /**
   * 由于这个问题有精确解，我们计算一下数值解和精确解的 L^2 误差。
   */
  double error = Functional::L2Error(u_h, FunctionFunction<double>(&_u_), 3);
  std::cerr << "\nL2 error = " << error << std::endl;

  return 0;
}
```
这个程序是使用AFEPack进行程序设计的基本套路，虽然其中基本上没有用户输入的关于计算方面的内容，但是很多问题都可以通过对这个程序进行不多的修改就能完成计算。

下面是上面程序的一个算例的图形:
1. 网格
http://rli.bloghome.cn/photos/p_original/7fa80b4fff007a6c9db4ce72559c9129.jpg
2. 解的等高线
http://rli.bloghome.cn/photos/p_original/775be126588a4ce7377c04f8165e67bf.jpg

# AFEPack教学系列之三: 变二次系数的椭圆型方程
我们将上文中求解的泊松方程进行一些修改，变成一个变二次项系数的问题。从而，我们现在的问题为
$$
-\nabla \cdot \left( A \nabla u \right) = f, \qquad u|\_{\partial\Omega} = u\_b.
$$
其中$A$是一个$2\times2$的矩阵，依赖于坐标，在我们的算例中，我们将其取为一个标量函数$a(x)$乘以单位矩阵。其与前一节中的泊松方程唯一的不同之处在于现在我们需要计算二次型
$$
a(u,v) = \int\_\Omega \nabla u \cdot A \cdot \nabla v dx
$$
的刚度矩阵。

在求解这个问题的过程中，我们先要把程序的结构进行整理，将变二次项系数的因素加进去，并使得其更加有利于复杂一些的问题的求解。为此，我们使用一个类来管理这个求解的问题，先写一个对这个类进行声明的头文件EllipticEquation.h。请看下面该头文件的源码和详细注释
```cpp
/**
 * @file   EllipticEquation.h
 * @author Robert Lie
 * @date   Wed Feb 21 11:45:03 2007
 *
 * @brief  类 EllipticEquation 的声明，附带的我们还声明了刚度矩阵的类
 *         型 Matrix。
 *
 */

#ifndef __EllipticEquation_h__
#define __EllipticEquation_h__

#include <AFEPack/AMGSolver.h>        /// 代数多重网格求解器
#include <AFEPack/TemplateElement.h>  /// 参考单元
#include <AFEPack/FEMSpace.h>         /// 有限元空间
#include <AFEPack/Operator.h>         /// 算子
#include <AFEPack/BilinearOperator.h> /// 双线性算子
#include <AFEPack/Functional.h>       /// 泛函
#include <AFEPack/EasyMesh.h>         /// EasyMesh 接口

#define DIM 2

/**
 * 类EllipticEquation 的声明，可以看到，在这个类的声明中，我们基本上
 * 只是把前一节的程序中的很多变量，搬到了这里作为这个类的成员变量。注
 * 意我们使得本类是从 EasyMesh 上派生出来的，从而本类的对象本身可以作
 * 为一个网格来使用，同时这是为了下面扩展到移动网格方法的方便。
 */
class EllipticEquation : public EasyMesh
{
 private:
  /**
   * 下面的几个变量用来构造参考单元，已经在上一节进行过详细解释。
   */
  TemplateGeometry<DIM> triangle_template_geometry;
  CoordTransform<DIM,DIM> triangle_coord_transform;
  TemplateDOF<DIM> triangle_template_dof;
  BasisFunctionAdmin<double,DIM,DIM> triangle_basis_function;

  std::vector<TemplateElement<double,DIM,DIM> > template_element;

  FEMSpace<double,DIM> fem_space; /// 有限元空间
  FEMFunction<double,DIM> u_h;    /// 有限元函数

 public:
  /**
   * 这个函数进行初始化的工作，包括读入网格数据文件，以及构造参考单元，
   * 为建立有限元空间做准备。
   */
  void initialize();
  /**
   * 这个函数将构造有限元空间。
   */
  void buildSpace();
  /**
   * 在构造的有限元空间上计算刚度矩阵并求解获得的线性系统，最终完成求
   * 解近似解。
   */
  void solve();
};

/**
 * 这个类是用来计算现在的变系数二次算子的刚度矩阵用的。它从库中的刚度
 * 矩阵类 StiffMatrix 派生出来，唯一需要进行重载的函数就是计算单元刚度
 * 矩阵的函数 getElementMatrix。
 */
class Matrix : public StiffMatrix<DIM,double>
{
 public:
  Matrix(FEMSpace<double,DIM>& sp) :
    StiffMatrix<DIM,double>(sp) {};
  virtual ~Matrix() {};
 public:
  virtual void
    getElementMatrix(const Element<double,2>& ele0,
                     const Element<double,2>& ele1,
                     const ActiveElementPairIterator<DIM>::State state);
};

#endif // __EllipticEquation_h__

/**
 * end of file
 *
 */
\\end{verbatim}



针对这个头文件，其中提到的函数的实现文件 的源码和注释如下：

\\begin{verbatim}
/**
 * @file   EllipticEquation.cpp
 * @author Robert Lie
 * @date   Wed Feb 21 15:26:48 2007
 *
 * @brief  类 EllipticEquation 和 Matrix 的实现文件。
 *
 */

#include "EllipticEquation.h"

/**
 * 计算二次项系数的函数，表达式为
 *                4, x^2 > y^2;
 *    a(x, y) = {
 *                1, x^2 < y^2.
 *
 * @param p 点的坐标： x = p[0], y = p[1]
 *
 * @return 返回函数 a(x, y) 的值
 *
 */
double _a_(const double * p)

{
  if (p[0]*p[0] > p[1]*p[1])
    return 4;
  else
    return 1;
}

/**
 * 计算矩阵 Matrix 的单元刚度矩阵。这个函数本来是为了计算双线性型 b(u,
 * v) 而设计的，其中 u 和 v 可以在不同的有限元空间上。因此，这个函数接
 * 受两个单元的参数，分别是 u 和 v 所在的有限元空间的单元，在我们现在
 * 的情况下，这两个参数是同一个单元。第三个参数是为了实现库中的所谓多
 * 套网格方法而预留的，在这里我们可以先忽略这个参数的作用。
 *
 * @param ele0 有限元空间的单元
 * @param ele1 这里是和 ele0 相同的单元
 *
 * 下面的这段程序是比较标准的进行单元上的数值积分的程序的套路，弄清楚
 * 其作用对于掌握 AFEPack 的程序设计是至关重要的。计算一个函数 f(x) 在
 * 一个单元上的数值积分可以使用通用的公式
 *
 *    sum |J| w_l f(x_l) |e|
 *
 * 其中 |e| 是参考单元的体积，w_l 是积分公式中在第 l 个积分点上的权重，
 * 而 |J| 是从参考单元到实际单元进行变换的雅可比行列式，x_l 是第 l 个
 * 积分点。在 AFEPack 中，积分点是首先给在参考单元上的，然后通过坐标变
 * 换映射到实际单元上，成为实际单元上的积分点。下面的程序基本上就是在
 * 实现这样的一个数值积分公式。
 *
 */
void
Matrix::getElementMatrix(const Element<double,DIM>& ele0,
                         const Element<double,DIM>& ele1,
                         const ActiveElementPairIterator<DIM>::State state)
{
  /// 计算参考单元的体积
  double volume = ele0.templateElement().volume();

  /// 取出参考单元上的数值积分公式
  const QuadratureInfo<DIM>& quad_info =
    ele0.findQuadratureInfo(algebricAccuracy());

  /// 计算在积分点上的坐标变换的雅可比矩阵
  std::vector<double> jacobian =
    ele0.local_to_global_jacobian(quad_info.quadraturePoint());

  /// 这是积分点的个数
  int n_quadrature_point = quad_info.n_quadraturePoint();

  /// 将参考单元上的积分点变换到网格中的实际单元上去
  std::vector<Point<DIM> > q_point =
    ele0.local_to_global(quad_info.quadraturePoint());

  /// 计算单元上的基函数的梯度值
  std::vector<std::vector<std::vector<double> > > basis_grad =
    ele0.basis_function_gradient(q_point);

  int n_ele_dof = eleDof0().size(); /// 单元上的自由度的个数

  /// 对积分点做求和
  for (int l = 0;l < n_quadrature_point;l ++) {
    /// Jxw 就是积分公式中的项 |e| |J| w_l
    double Jxw = quad_info.weight(l)*jacobian[l]*volume;

    /// 计算函数 a(x, y) 在积分点的值
    double a_val = _a_(q_point[l]);

    for (int j = 0;j < n_ele_dof;j ++) { /// 对自由度做循环
      for (int k = 0;k < n_ele_dof;k ++) { /// 对自由度做循环

        /// 将积分点上的值往单元刚度矩阵上进行累加
        elementMatrix(j,k) += Jxw*a_val*innerProduct(basis_grad[j][l],
                                                     basis_grad[k][l]);
      }
    }
  }
}

/**
 * 初始化函数中准备了参考单元以及读入了网格数据文件。
 */
void EllipticEquation::initialize()
{
  /// 读入参考单元的几何信息及坐标变换
  triangle_template_geometry.readData("triangle.tmp_geo");
  triangle_coord_transform.readData("triangle.crd_trs");

  /// 在参考单元的几何信息更新以后，模板单元上的自由度分布需要重新初始
  /// 化一下，对于下面的基函数，也是相似的情况。
  triangle_template_dof.reinit(triangle_template_geometry);
  triangle_template_dof.readData("triangle.1.tmp_dof");
  triangle_basis_function.reinit(triangle_template_dof);
  triangle_basis_function.readData("triangle.1.bas_fun");

  /// 用上面读入的信息合成参考单元
  template_element.resize(1);
  template_element[0].reinit(triangle_template_geometry,
                             triangle_coord_transform,
                             triangle_template_dof,
                             triangle_basis_function);

  /// 读入网格数据文件
  this->readData("D");
}

/**
 * 构造有限元空间，这里的代码是完全从前一节中搬过来的。
 */
void EllipticEquation::buildSpace()
{
  /// 对有限元空间的对象重新初始化一下
  fem_space,reinit(*this, template_element);
        
  int n_element = mesh.n_geometry(DIM);
  fem_space.element().resize(n_element);
  for (int i = 0;i < n_element;i ++) {
    fem_space.element(i).reinit(fem_space,i,0);
  }

  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();
}

/**
 * 构造线性系统，并求解，得到需要的有限元函数。这一段的程序也基本上是
 * 由前一节搬过来的。
 */
void EllipticEquation::solve()
{
  /// 注意这里我们使用 Matrix 来当我们的刚度矩阵
  Matrix stiff_matrix(fem_space);
  stiff_matrix.algebricAccuracy() = 3;
  stiff_matrix.build();

  /// 有限元函数 u_h 初始化
  u_h.reinit(fem_space);

  /**
   * 向量 f_h 用来计算右端项 f 在离散过后得到的向量。
   *
   * 注意：我们需要使用前一节中的函数表达式 _f_ 和 _u_，为了节省篇幅，
   * 我们这个文件中没有重复写这两个函数的表达式。
   */
  Vector<double> f_h;
  Operator::L2Discretize(&_f_, fem_space, f_h, 3);

  /// 使用狄氏边值条件。
  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET,
                                        1, &_u_);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(stiff_matrix, u_h, f_h);

  /// 调用 AFEPack 中的代数多重网格求解器，求解线性系统。
  AMGSolver solver(stiff_matrix);
  solver.solve(u_h, f_h);      

  /// 将解输出。
  u_h.writeOpenDXData("u.dx");
}

/**
 * end of file
 *
 */
```
除去了这两个文件，我们另外写一个主文件，用来调用EllipticEquation类中的几个公有函数完成计算，全文如下：
```cpp
/**
 * @file   main.cpp
 * @author Robert Lie
 * @date   Wed Feb 21 16:44:21 2007
 *
 * @brief  主文件，我们在这里开始运行，其中的这几行程序的意义是显而易
 *         见的，就不多解释了。
 *
 */

#include "EllipticEquation.h"

int main(int argc, char * argv[])
{
  try {
    EllipticEquation the_app;
    the_app.initialize();
    the_app.buildSpace();
    the_app.solve();
  }
  catch(std::exception& e) {
    std::cerr << "Exception caughted:" << std::endl
              << e.what ()
              << std::endl;
  }
  catch(...) {
    std::cerr << "Exception caughted:" << std::endl
              << "unknown exception caughted."
              << std::endl;
  }

  return 0;
}

/**
 * end of file
 *
 */
```
下面是这个结果的图形，可以看到间断系数带来的解上的褶皱：
1. 等高线
http://rli.bloghome.cn/photos/p_original/16251db7005cc5179486c6f28e19ea30.jpg
2. 曲面
http://rli.bloghome.cn/photos/p_original/9c9154be5fa9a3789c245e06606e4eb1.jpg

# AFEPack教学系列之四: 简单的抛物型方程的例子
```cpp
/**
 * @file   test.cpp
 * @author Robert Lie
 * @date   Wed Feb 28 11:45:08 2007
 *
 * @brief  这是个抛物型方程的例子，非常简单，中间就没有解释太多了
 *
 */

#include <AFEPack/EasyMesh.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>

#define DIM 2

/// 初值和边值的表达式
double _u_(const double * p)
{
  return p[0]*exp(p[1]);
}

/// 右端项
double _f_(const double * p)
{
  return p[0]*p[1] + sin(p[1]);
}


/// 1/dt - \Delta 离散出来的矩阵
class Matrix : public L2InnerProduct<DIM,double>
{
private:
  double _dt;
public:
  Matrix(FEMSpace<double,DIM>& sp, double dt) :
    L2InnerProduct<DIM,double>(sp, sp), _dt(dt) {}
  virtual void getElementMatrix(const Element<double,DIM>& e0,
                                const Element<double,DIM>& e1,
                                const ActiveElementPairIterator< DIM >::State s)
  {
    double vol = e0.templateElement().volume();
    u_int acc = algebricAccuracy();
    const QuadratureInfo<DIM>& qi = e0.findQuadratureInfo(acc);
    u_int n_q_pnt = qi.n_quadraturePoint();
    std::vector<double> jac = e0.local_to_global_jacobian(qi.quadraturePoint());
    std::vector<Point<DIM> > q_pnt = e0.local_to_global(qi.quadraturePoint());
    std::vector<std::vector<double> > bas_val = e0.basis_function_value(q_pnt);
    std::vector<std::vector<std::vector<double> > > bas_grad = e0.basis_function_gradient(q_pnt);
    u_int n_ele_dof = e0.dof().size();
    for (u_int l = 0;l < n_q_pnt;++ l) {
      double Jxw = vol*qi.weight(l)*jac[l];
      for (u_int i = 0;i < n_ele_dof;++ i) {
        for (u_int j = 0;j < n_ele_dof;++ j) {
          elementMatrix(i,j) += Jxw*(bas_val[i][l]*bas_val[j][l]/_dt +
                                     innerProduct(bas_grad[i][l], bas_grad[j][l]));
        }
      }
    }
  }
};

int main(int argc, char * argv[])
{
  /// 准备网格
  EasyMesh mesh;
  mesh.readData(argv[1]);

  /// 准备参考单元
  TemplateGeometry<DIM> tmp_geo;
  tmp_geo.readData("triangle.tmp_geo");
  CoordTransform<DIM,DIM> crd_trs;
  crd_trs.readData("triangle.crd_trs");
  TemplateDOF<DIM> tmp_dof(tmp_geo);
  tmp_dof.readData("triangle.1.tmp_dof");
  BasisFunctionAdmin<double,DIM,DIM> bas_fun(tmp_dof);
  bas_fun.readData("triangle.1.bas_fun");

  std::vector<TemplateElement<double,DIM> > tmp_ele(1);
  tmp_ele[0].reinit(tmp_geo, tmp_dof, crd_trs, bas_fun);

  /// 定制有限元空间
  FEMSpace<double,DIM> fem_space(mesh, tmp_ele);
  u_int n_ele = mesh.n_geometry(DIM);
  fem_space.element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    fem_space.element(i).reinit(fem_space, i, 0);
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();

  /// 准备初值
  FEMFunction<double,DIM> u_h(fem_space);
  Operator::L2Interpolate(&_u_, u_h);

  /// 准备边界条件
  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET,
                                        1,
                                        &_u_);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);

  double t;//

  do {
    double dt = 0.01; /// 简单起见，随手取个时间步长算了

    /// 准备线性系统的矩阵
    Matrix mat(fem_space, dt);
    mat.algebricAccuracy() = 3;
    mat.build();

    /// 准备右端项
    Vector<double> rhs(fem_space.n_dof());
    FEMSpace<double,DIM>::ElementIterator the_ele = fem_space.beginElement();
    FEMSpace<double,DIM>::ElementIterator end_ele = fem_space.endElement();
    for (;the_ele != end_ele;++ the_ele) {
      double vol = the_ele->templateElement().volume();
      const QuadratureInfo<DIM>& qi = the_ele->findQuadratureInfo(3);
      u_int n_q_pnt = qi.n_quadraturePoint();
      std::vector<double> jac = the_ele->local_to_global_jacobian(qi.quadraturePoint());
      std::vector<Point<DIM> > q_pnt = the_ele->local_to_global(qi.quadraturePoint());
      std::vector<std::vector<double> > bas_val = the_ele->basis_function_value(q_pnt);

      /// 当基函数的值已知情况下，可以使用下面的函数来加速
      std::vector<double> u_h_val = u_h.value(bas_val, *the_ele);
      std::vector<std::vector<double> > u_h_grad = u_h.gradient(q_pnt, *the_ele);
      const std::vector<int>& ele_dof = the_ele->dof();
      u_int n_ele_dof = ele_dof.size();
      for (u_int l = 0;l < n_q_pnt;++ l) {
        double Jxw = vol*qi.weight(l)*jac[l];
        double f_val = _f_(q_pnt[l]);
        for (u_int i = 0;i < n_ele_dof;++ i) {
          rhs(ele_dof[i]) += Jxw*bas_val[i][l]*(u_h_val[l]/dt + f_val);
        }
      }
    }

    /// 应用边界条件
    boundary_admin.apply(mat, u_h, rhs);

    /// 求解线性系统
    AMGSolver solver;
    solver.lazyReinit(mat);
    solver.solve(u_h, rhs, 1.0e-08, 50);

    /// 输出数据画图
    u_h.writeOpenDXData("u_h.dx");

    t += dt; /// 更新时间
    
    std::cout << "\n\tt = " <<  t << std::endl;
  } while (1);
 
  return 0;
}

/**
 * end of file
 *
 */
```
# AFEPack教学系列之五: 一个没有注释的程序，考考你！

下面是我们讨论班上心血来潮随手写的例子，一句注释都没有写，考考你看是什么意思，如果看明白了告诉我啊。
```cpp
/**
 * @file   test.cpp
 * @author Robert Lie
 * @date   Wed Feb 28 11:45:08 2007
 *
 * @brief  
 *
 *
 */

#include <AFEPack/EasyMesh.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>

#define DIM 2

double _u_(const double * p)
{
  return p[0]*exp(p[1]);
}

double _f_(const double * p)
{
  return p[0]*p[1] + sin(p[1]);
}

class Matrix : public L2InnerProduct<DIM,double>
{
private:
  FEMFunction<double,DIM> * p_u_h;
public:
  Matrix(FEMSpace<double,DIM>& sp, FEMFunction<double,DIM>& u_h) :
    L2InnerProduct<DIM,double>(sp, sp), p_u_h(&u_h) {}
  virtual void getElementMatrix(const Element<double,DIM>& e0,
                                const Element<double,DIM>& e1,
                                const ActiveElementPairIterator< DIM >::State s)
  {
    double vol = e0.templateElement().volume();
    u_int acc = algebricAccuracy();
    const QuadratureInfo<DIM>& qi = e0.findQuadratureInfo(acc);
    u_int n_q_pnt = qi.n_quadraturePoint();
    std::vector<double> jac = e0.local_to_global_jacobian(qi.quadraturePoint());
    std::vector<Point<DIM> > q_pnt = e0.local_to_global(qi.quadraturePoint());
    std::vector<std::vector<double> > bas_val = e0.basis_function_value(q_pnt);
    std::vector<std::vector<std::vector<double> > >
      bas_grad = e0.basis_function_gradient(q_pnt);
    std::vector<double> u_h_val = p_u_h->value(q_pnt, e0);
    u_int n_ele_dof = e0.dof().size();
    for (u_int l = 0;l < n_q_pnt;++ l) {
      double Jxw = vol*qi.weight(l)*jac[l];
      for (u_int i = 0;i < n_ele_dof;++ i) {
        for (u_int j = 0;j < n_ele_dof;++ j) {
          elementMatrix(i,j) += Jxw*(u_h_val[l]*u_h_val[l]*
                                     innerProduct(bas_grad[i][l],
                                                  bas_grad[j][l]));
        }
      }
    }
  }
};

int main(int argc, char * argv[])
{
  EasyMesh mesh;
  mesh.readData(argv[1]);

  TemplateGeometry<DIM> tmp_geo;
  tmp_geo.readData("triangle.tmp_geo");
  CoordTransform<DIM,DIM> crd_trs;
  crd_trs.readData("triangle.crd_trs");
  TemplateDOF<DIM> tmp_dof(tmp_geo);
  tmp_dof.readData("triangle.1.tmp_dof");
  BasisFunctionAdmin<double,DIM,DIM> bas_fun(tmp_dof);
  bas_fun.readData("triangle.1.bas_fun");

  std::vector<TemplateElement<double,DIM> > tmp_ele(1);
  tmp_ele[0].reinit(tmp_geo, tmp_dof, crd_trs, bas_fun);

  FEMSpace<double,DIM> fem_space(mesh, tmp_ele);
  u_int n_ele = mesh.n_geometry(DIM);
  fem_space.element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    fem_space.element(i).reinit(fem_space, i, 0);
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();

  FEMFunction<double,DIM> u_h(fem_space);
  Operator::L2Interpolate(&_u_, u_h);

  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET,
                                        1,
                                        &_u_);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);

  do {
    Matrix mat(fem_space, u_h);
    mat.algebricAccuracy() = 3;
    mat.build();

    Vector<double> rhs(fem_space.n_dof());
    FEMSpace<double,DIM>::ElementIterator the_ele = fem_space.beginElement();
    FEMSpace<double,DIM>::ElementIterator end_ele = fem_space.endElement();
    for (;the_ele != end_ele;++ the_ele) {
      double vol = the_ele->templateElement().volume();
      const QuadratureInfo<DIM>& qi = the_ele->findQuadratureInfo(3);
      u_int n_q_pnt = qi.n_quadraturePoint();
      std::vector<double> jac = the_ele->local_to_global_jacobian(qi.quadraturePoint());
      std::vector<Point<DIM> > q_pnt = the_ele->local_to_global(qi.quadraturePoint());
      std::vector<std::vector<double> > bas_val = the_ele->basis_function_value(q_pnt);
      std::vector<double> u_h_val = u_h.value(q_pnt, *the_ele);
      std::vector<std::vector<double> > u_h_grad = u_h.gradient(q_pnt, *the_ele);
      const std::vector<int>& ele_dof = the_ele->dof();
      u_int n_ele_dof = ele_dof.size();
      for (u_int l = 0;l < n_q_pnt;++ l) {
        double Jxw = vol*qi.weight(l)*jac[l];
        double f_val = _f_(q_pnt[l]);
        for (u_int i = 0;i < n_ele_dof;++ i) {
          rhs(ele_dof[i]) += Jxw*bas_val[i][l]*
            (f_val - u_h_val[l]*(u_h_grad[l][0] + u_h_grad[l][1]));
        }
      }
    }
    
    boundary_admin.apply(mat, u_h, rhs);

    AMGSolver solver;
    solver.lazyReinit(mat);
    solver.solve(u_h, rhs, 1.0e-08, 50);

    u_h.writeOpenDXData("u_h.dx");

    Vector<double> res(fem_space.n_dof());
    the_ele = fem_space.beginElement();
    for (;the_ele != end_ele;++ the_ele) {
      double vol = the_ele->templateElement().volume();
      const QuadratureInfo<DIM>& qi = the_ele->findQuadratureInfo(3);
      u_int n_q_pnt = qi.n_quadraturePoint();
      std::vector<double> jac = the_ele->local_to_global_jacobian(qi.quadraturePoint());
      std::vector<Point<DIM> > q_pnt = the_ele->local_to_global(qi.quadraturePoint());
      std::vector<std::vector<double> > bas_val = the_ele->basis_function_value(q_pnt);
      std::vector<double> u_h_val = u_h.value(q_pnt, *the_ele);
      std::vector<std::vector<std::vector<double> > > bas_grad = the_ele->basis_function_gradient(q_pnt);
      std::vector<std::vector<double> > u_h_grad = u_h.gradient(q_pnt, *the_ele);
      const std::vector<int>& ele_dof = the_ele->dof();
      u_int n_ele_dof = ele_dof.size();
      for (u_int l = 0;l < n_q_pnt;++ l) {
        double Jxw = vol*qi.weight(l)*jac[l];
        double f_val = _f_(q_pnt[l]);
        for (u_int i = 0;i < n_ele_dof;++ i) {
      if (fem_space.dofBoundaryMark(ele_dof[i]) != 0) continue;
          res(ele_dof[i]) += Jxw*
            (bas_val[i][l]*(f_val - u_h_val[l]*(u_h_grad[l][0] + u_h_grad[l][1])) -
             u_h_val[l]*u_h_val[l]*innerProduct(bas_grad[i][l], u_h_grad[l]));
        }
      }
    }

    
    std::cout << "res = " << res.l2_norm()
              << ", press Enter to continue ..." << std::flush;
    getchar();
  } while (1);
 
  return 0;
}

/**
 * end of file
 *
 */
```
# AFEPack教学系列之六: 加入移动网格方法的椭圆型方程

基于前面的变系数的椭圆型方程的例子，我们来加入移动网格的模块，使得网格能够更加集中于数值解中的弱间断的位置。我们的移动网格模块的算法，整个网格的边界和内部将会耦合移动，因此，在区域的不同边界上，需要使用不同的材料标识才能使得程序对每个不同的边界上的信息加以区分。我们先将给EasyMesh使用的输入文件D.d更新为下面的样子：
```cpp
4 # 区域的顶点的个数 #
0: 0.0 0.0 0.05 1
1: 1.0 0.0 0.05 2
2: 1.0 1.0 0.05 3
3: 0.0 1.0 0.05 4

4 # 区域的边界上边的条数 #
0: 0 1 1
1: 1 2 2
2: 2 3 3
3: 3 4 4
```
正方形的每个边上现在各有不同的边界条件。现在请以这个文件作为输入，使用easymesh产生出新的网格数据文件。另外，我们还需要一个文件来对问题的逻辑区域进行描述，这个文件命名为``文件名'' + ``.log''，现在我们的主文件名为D，因此我们提供名为D.log 的文件如下：
```cpp
0.0 0.0
1.0 0.0
1.0 1.0
0.0 1.0
```
这个文件的每一行是逻辑区域的一个顶点的坐标，和物理区域的各个顶点相对应。由于理论上的考虑，请使得逻辑区域是一个凸区域。为了加入移动网格的模块，需要从头文件 EllipticEquation.h 改起：
```cpp
#ifndef __EllipticEquation_h__
#define __EllipticEquation_h__

#include <AFEPack/AMGSolver.h>        /// 代数多重网格求解器
#include <AFEPack/TemplateElement.h>  /// 参考单元
#include <AFEPack/FEMSpace.h>         /// 有限元空间
#include <AFEPack/Operator.h>         /// 算子
#include <AFEPack/BilinearOperator.h> /// 双线性算子
#include <AFEPack/Functional.h>       /// 泛函
#include <AFEPack/MovingMesh.h>       /// 移动网格模块

#define DIM 2

/**
 * 现在，本类是从 MovingMesh 这个类派生出来的。在库的结构中，MovingMesh
 * 是从 EasyMesh 这个类派生出来的。因此，现在本类中只是多了在 MovingMesh
 * 中实现的一些算法的接口。
 */
class EllipticEquation : public MovingMesh
{
 private:
  /**
   * 构造参考单元的变量。
   */
  TemplateGeometry<DIM> triangle_template_geometry;
  CoordTransform<DIM,DIM> triangle_coord_transform;
  TemplateDOF<DIM> triangle_template_dof;
  BasisFunctionAdmin<double,DIM,DIM> triangle_basis_function;

  std::vector<TemplateElement<double,DIM,DIM> > template_element;

  FEMSpace<double,DIM> fem_space; /// 有限元空间
  FEMFunction<double,DIM> u_h;    /// 有限元函数

 public:
  /// 下面这三个函数还是完成它们在上一个小节的功能
  void initialize();
  void buildSpace();
  void solve();

  /**
   * 现在由于要进行自适应，意味着我们要反复求解上面的椭圆型方程，因此
   * 我们在上面的椭圆型方程的求解上再套上一重循环来完成整个工作。在
   * run 这个函数中，就是一个这样的循环，在每个循环步中，整个程序求解
   * 椭圆型方程，然后调整网格，然后走向下一个循环步。
   */
  void run();
  /**
   * getMonitor是在 MovingMesh 中的一个虚函数，缺省情况下，它将
   * MovingMesh 类中用作控制函数的数组填充为一个值全为 1 的数组。这个
   * 数组的长度和网格中的单元个数是一支的。现在这里重载这个函数，用来
   * 控制网格的移动。
   */
  virtual void getMonitor();
  /**
   * 在网格移动了以后，一般来说我们需要将旧网格上的解通过某种方式更新
   * 到新网格上来，因此，MovingMesh 类中也定义了这个虚函数。
   */
  virtual void updateSolution();
  /**
   * 这个函数本来是调试的时候用的，在基类设计的时候没有计划得很好，做
   * 成了纯虚的，所以这里也写一个它的实现。
   */
  virtual void outputSolution();
};

/**
 * 计算刚度矩阵的程序的声明和实现都不做改变。
 */
class Matrix : public StiffMatrix<DIM,double>
{
 public:
  Matrix(FEMSpace<double,DIM>& sp) :
    StiffMatrix<DIM,double>(sp) {};
  virtual ~Matrix() {};
 public:
  virtual void
    getElementMatrix(const Element<double,DIM>& ele0,
                     const Element<double,DIM>& ele1,
                     const ActiveElementPairIterator<DIM>::State state);
};

#endif // __EllipticEquation_h__
```


下面是类的实现文件的源码：
```cpp
#include "EllipticEquation.h"

/**
 * 计算二次项系数的函数的表达式。
 */
double _a_(const double * p)

{
  if (p[0]*p[0] > p[1]*p[1])
    return 4;
  else
    return 1;
}

/**
 * 计算矩阵 Matrix 的单元刚度矩阵，未做改变。
 */
void
Matrix::getElementMatrix(const Element<double,DIM>& ele0,
                         const Element<double,DIM>& ele1,
                         const ActiveElementPairIterator<DIM>::State state)
{
  double volume = ele0.templateElement().volume();
  const QuadratureInfo<DIM>& quad_info =
    ele0.findQuadratureInfo(algebricAccuracy());
  std::vector<double> jacobian =
    ele0.local_to_global_jacobian(quad_info.quadraturePoint());
  int n_quadrature_point = quad_info.n_quadraturePoint();
  std::vector<Point<DIM> > q_point =
    ele0.local_to_global(quad_info.quadraturePoint());
  std::vector<std::vector<std::vector<double> > > basis_grad =
    ele0.basis_function_gradient(q_point);
  int n_ele_dof = eleDof0().size();
  for (int l = 0;l < n_quadrature_point;l ++) {
    double Jxw = quad_info.weight(l)*jacobian[l]*volume;
    double a_val = _a_(q_point[l]);
    for (int j = 0;j < n_ele_dof;j ++) {
      for (int k = 0;k < n_ele_dof;k ++) {
        elementMatrix(j,k) += Jxw*a_val*innerProduct(basis_grad[j][l],
                                                     basis_grad[k][l]);
      }
    }
  }
}

/**
 * 初始化函数也基本没有改变，但是在读入网格数据文件的时候，我们现在需
 * 要使用 MovingMesh 类提供的 readDomain 函数。这个函数会不但读入网格
 * 剖分文件，并且会对区域描述文件 D.d 和 D.log 进行分析。注意：D.d 中
 * 的节点附近的剖分尺度在这个部分是不会用到的。
 */
void EllipticEquation::initialize()
{
  triangle_template_geometry.readData("triangle.tmp_geo");
  triangle_coord_transform.readData("triangle.crd_trs");
  triangle_template_dof.reinit(triangle_template_geometry);
  triangle_template_dof.readData("triangle.1.tmp_dof");
  triangle_basis_function.reinit(triangle_template_dof);
  triangle_basis_function.readData("triangle.1.bas_fun");

  template_element.resize(1);
  template_element[0].reinit(triangle_template_geometry,
                             triangle_coord_transform,
                             triangle_template_dof,
                             triangle_basis_function);

  /// 读入网格数据文件和区域描述文件
  this->readDomain("D");
}

/**
 * 构造有限元空间。我们只需要做一次这个操作，即使是网格移动了以后，有
 * 限元空间也不必更新。事实上，网格移动了以后，有限元空间中每个自由度
 * 的插值点会不对。但是由于我们下面的计算并不使用到插值操作，因此就不
 * 必更新自由度的插值点了。
 */
void EllipticEquation::buildSpace()
{
  fem_space,reinit(*this, template_element);
        
  int n_element = mesh.n_geometry(DIM);
  fem_space.element().resize(n_element);
  for (int i = 0;i < n_element;i ++) {
    fem_space.element(i).reinit(fem_space,i,0);
  }

  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();
  
}

/**
 * 构造线性系统，并求解，得到需要的有限元函数。在这个函数中，我们仅仅
 * 对边界条件的部分进行了更新，因为我们为了能够应用移动网格模块，更新
 * 了区域描述文件中的边界材料标识。
 */
void EllipticEquation::solve()
{
  Matrix stiff_matrix(fem_space);
  stiff_matrix.algebricAccuracy() = 3;
  stiff_matrix.build();

  u_h.reinit(fem_space);

  Vector<double> f_h;
  Operator::L2Discretize(&_f_, fem_space, f_h, 3);

  /// 使用狄氏边值条件。现在我们需要对四个边分别添加边值条件。
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET,
                                         1, &_u_);
  BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET,
                                         2, &_u_);
  BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET,
                                         3, &_u_);
  BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET,
                                         4, &_u_);
  boundary_admin.add(boundary1);
  boundary_admin.add(boundary2);
  boundary_admin.add(boundary3);
  boundary_admin.add(boundary4);
  boundary_admin.apply(stiff_matrix, u_h, f_h);

  AMGSolver solver(stiff_matrix);
  solver.solve(u_h, f_h);      
}

/**
 * run 是整个程序运行的主要驱动引擎。这里是最外层的循环。
 */
void EllipticEquation::run()
{
  initialize(); /// 完成准备工作
  buildSpace(); /// 建立有限元空间
  /**
   * 这里要加一个对u_h的初始化，否则在调用getMonitor的时候会出错: 
   * 在buildSpace()之后是moveMesh()，而在moveMesh()里最先调用     
   * getMoveDirection()，该函数又首先调用getMonitor()。在
   * getMonitor()里会用到u_h来计算梯度，所以在此之前必须初始化u_h。
   * 对于时间依赖的问题，因为要用到初始值，所以在移动之前，u_h已经
   * 有了初值了，但是本算例是静态问题，所以必须先解一次，给u_h一个
   * 合理的初值，然后才可以移动。这里我们首先求解一次，然后调用
   * moveMesh()（by hghdd）
   */
  solve();
  /**
   * 调用 MovingMesh 的 moveMesh 函数进行网格移动，主循环事实上隐藏在
   * 这个函数中。
   */
  moveMesh();
}

/**
 * 这个函数计算控制网格移动的控制函数。我们使用解的梯度的阶跃来作为控
 * 制量。由于我们现在使用最简单的线性逼近的方式，我们可以手工在计算这
 * 个量，避免要进行面单元上的数值积分带来的麻烦。请仔细理解下面这段代
 * 码。
 */
void EllipticEquation::getMonitor()
{
  u_int n_ele = n_geometry(DIM); /// 网格中所有的单元的个数
  u_int n_side = n_geometry(1); /// 网格中的所有边的条数。
  std::vector<bool> sflag(n_side, false); /// 加在每个边上的一个标志
  std::vector<double> sjump(n_side); /// 用来边上的阶跃
  std::vector<double> earea(n_ele); /// 用来存储每个单元的面积

  /// 对单元做循环
  FEMSpace<double,DIM>::ElementIterator the_ele = fem_space.beginElement();
  FEMSpace<double,DIM>::ElementIterator end_ele = fem_space.endElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    /// 取出单元的几何体
    const GeometryBM& geo = the_ele->geometry();

    /// 下面是单元的三个顶点的坐标
    const Point<DIM>& p0 = point(geometry(0, geo.vertex(0)).vertex(0));
    const Point<DIM>& p1 = point(geometry(0, geo.vertex(1)).vertex(0));
    const Point<DIM>& p2 = point(geometry(0, geo.vertex(2)).vertex(0));

    /// 手工计算此单元的面积
    earea[i] = 0.5*((p1[0] - p0[0])*(p2[1] - p0[1]) -
                    (p1[1] - p0[1])*(p2[0] - p0[0]));

    /**
     * 计算单元上解函数的梯度。由于使用分片线性逼近，解函数的梯度是一
     * 个常数，因此在下面的函数中，我们将计算的坐标点设为 p0 是没有关
     * 系的。如果不是这样的逼近方式，下面的代码是不对的。
     */
    std::vector<double> u_h_grad = u_h.gradient(p0, *the_ele);

    /**
     * 下面对三角形的三条边进行循环，分别计算每条边上的梯度法向分量。
     * 并加到相应的边上去。
     */
    for (u_int j = 0;j < 3;++ j) {
      u_int k = geo.boundary(j); /// 这是边的序号
      const GeometryBM& sid = geometry(1, k); /// 取出边的几何体

      /// 下面是边的两个端点
      const Point<DIM>& ep0 = point(geometry(0, sid.vertex(0)).vertex(0));
      const Point<DIM>& ep1 = point(geometry(0, sid.vertex(1)).vertex(0));

      /// 计算梯度的法向分量乘以边的长度
      double norm_grad = (u_h_grad[0]*(ep0[1] - ep1[1]) +
                          u_h_grad[1]*(ep1[0] - ep0[0]));
      /**
       * 如果 sflag[k] 是 false，说明对面的单元上还没有计算过， 因此我 
       * 们将 norm_grad 直接设为该边上的值； 否则说明对面的单元上已经计 
       * 算过了，我们将本单元的 norm_grad 减去，就得到了边上的梯度阶跃 
       * 了。
       */
      if (sflag[k] == false) {
        sjump[k] = norm_grad;
        sflag[k] = true; /// 将标志改为 true，通知对面单元
      } else {
        sjump[k] -= norm_grad;
        sflag[k] = false; /// 将标志恢复为 false，表面这个边不在边界上
      }
    }
  }

  /// 重新开始对单元做循环
  the_ele = fem_space.beginElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    const GeometryBM& geo = the_ele->geometry(); /// 单元的几何体
    monitor(i) = 0.0;
    /// 将三个边上的阶跃都算在这个单元头上
    for (u_int j = 0;j < 3;++ j) {
      u_int k = geo.boundary(j);
      const GeometryBM& sid = geometry(1, k);
      if (sflag[k] == true) continue; /// 如果在区域边界上，则跳过去
      monitor(i) += sjump[k]*sjump[k];
    }
    monitor(i) /= earea[i]; /// 除以面积，得到分布函数
  }
  smoothMonitor(2); /// 对控制函数进行磨光操作

  /**
   * beta 是用来调节对比度的重要参数，当这个数越大的时候，就使得网格的
   * 尺寸对比越大。我们目前还不知道如何选择这个参数，只能靠试验和经验
   * 了。
   */
  double beta = 1.0e+02;
  for (u_int i = 0;i < n_ele;++ i) { /// 将控制函数取调和形式
    monitor(i) = 1.0/sqrt(1.0 + beta*monitor(i));
  }
}

/**
 * 由于是静态问题，我们在这里先直接通过在新的网格上直接求解来获得新的
 * 解。
 */
void EllipticEquation::updateSolution()
{
  solve();
}

/// 输出解函数做可视化用
void EllipticEquation::outputSolution()
{
  u_h.writeOpenDXData("u_h.dx");
}
```


最后，我们将main.cpp改成下面的样子，使得它在里面直接调用run这个函数：
```cpp
#include "EllipticEquation.h"

int main(int argc, char * argv[])
{
  try {
    EllipticEquation the_app;
    the_app.run();
  }
  catch(std::exception& e) {
    std::cerr << "Exception caughted:" << std::endl
              << e.what ()
              << std::endl;
  }
  catch(...) {
    std::cerr << "Exception caughted:" << std::endl
              << "unknown exception caughted."
              << std::endl;
  }

  return 0;
}
```
由于我们在MovingMesh中的边界控制问题的求解算法上进行了比较本质的改进，现在类MovingMesh将不会继续维护，而代之以一个新的类MovingMesh2D。因此在此谨提醒用户都移到这个新类上来，在一段时间以后，MovingMesh这个类会被废弃的。这个新的类在接口上基本上维持了MovingMesh的样子，原先的程序直接把MovingMesh改为MovingMesh2D就都能使用了。

# AFEPack教学系列之七: 加一行，改成Burgers方程的程序
```cpp
/**
 * @file   bug.cpp
 * @author Robert Lie
 * @date   Thu Mar  8 14:55:37 2007
 *
 * @brief  将系列之四的抛物型方程的例子稍微改一下，就成为了黏性Burgers方程
 *         的程序了。其它的部分我就都删除了，只留下了修改过的那一点点。
 *
 */

    /// 准备右端项
    Vector<double> rhs(fem_space.n_dof());
    FEMSpace<double,DIM>::ElementIterator the_ele = fem_space.beginElement();
    FEMSpace<double,DIM>::ElementIterator end_ele = fem_space.endElement();
    for (;the_ele != end_ele;++ the_ele) {
      double vol = the_ele->templateElement().volume();
      const QuadratureInfo<DIM>& qi = the_ele->findQuadratureInfo(3);
      u_int n_q_pnt = qi.n_quadraturePoint();
      std::vector<double> jac = the_ele->local_to_global_jacobian(qi.quadraturePoint());
      std::vector<Point<DIM> > q_pnt = the_ele->local_to_global(qi.quadraturePoint());
      std::vector<std::vector<double> > bas_val = the_ele->basis_function_value(q_pnt);

      /// 当基函数的值已知情况下，可以使用下面的函数来加速
      std::vector<double> u_h_val = u_h.value(bas_val, *the_ele);
      std::vector<std::vector<double> > u_h_grad = u_h.gradient(q_pnt, *the_ele);
      const std::vector<int>& ele_dof = the_ele->dof();
      u_int n_ele_dof = ele_dof.size();
      for (u_int l = 0;l < n_q_pnt;++ l) {
        double Jxw = vol*qi.weight(l)*jac[l];
        double f_val = _f_(q_pnt[l]);
        for (u_int i = 0;i < n_ele_dof;++ i) {
          rhs(ele_dof[i]) += Jxw*bas_val[i][l]*(u_h_val[l]*/dt + f_val
                                                u_h_val[l]*(u_h_grad[l][0] + u_h_grad[l][1]));
          /// 就加了上面这一行
        }
      }
    }

/**
 * end of file
 *
 */
```
# AFEPack教学系列之八: 网格和几何体

应Wang Han的要求，写一些解释AFEPack的基本数据结构的内容。首先，我们说一下关于网格和几何体的数据结构。

几何体的类就是Geometry，如果需要材料标识，则叫做GeometryBM，这个名字比较难看，是因为开始的时候没有设计好造成的。单独一个几何体中存储的数据是没有意义的，几何体这个类的对象必须在网格类中间存储，才能够有意义。

Mesh这个类存储着各维几何体的数组。其中，最高维的几何体就是网格中间的单元，最低维的几何体则是网格中的节点。因为使用了数组，就可以使用随机读取方式来取其中的元素，因此，在AFEPack内部的设计语言中，每个几何体的表述方法，都是按照“第n维的第m个几何体”这样的方式来叙述的。

在Geometry中，于是就使用存储序号的方式进行相互引用。Geometry中有两个数组，一个存储它的顶点的序号，一个存储它的拓扑边界几何体的序号。

几何体的顶点是网格中的节点，我们称为“第0维几何体”。而对于一个“第m维几何体” 来说，它的拓扑边界几何体是“第m-1维的几何体”。每个几何体总是往低维进行索引，从而获得关于其自身结构的所有信息。

于是，对于一个给定的网格类的对象a_mesh，为了得到某一个维数(比如第k维)的几何体的个数，我们可以使用
```cpp
    u_int n = a_mesh.n_geometry(k);
```
得到。而为了得到第k维的第l个几何体，则可以使用
```cpp
    GeometryBM& geo = a_mesh.geometry(k, l);
```
这样，我们获得了一个几何体的对象，然后就可以知道它的顶点和拓扑边界的序号，就可以继续往低维进行索引了，比如我们取出上面得到的这个几何geo自己的边界，可以使用如下的代码：
```cpp
    u_int n_bnd = geo.n_boundary(); /// geo 的拓扑边界几何体的个数
    for (u_int i = 0;i < n_bnd;i ++) { /// 进行遍历
      u_int bnd_idx = geo.boundary(i); /// 第i个边界几何体的序号
      GeometryBM& bnd_geo = a_mesh.geometry(k-1, bnd_idx); /// 取出来
      ... ...
    }
```
网格类有两个模板参数，一个是DIM，一个是DOW。其中DIM是网格所在的区域，作为一个流形的维数。比如平面上的一个多边形，是二维的，球面也是二维的。这个数决定了Mesh类中间，用来存储各维几何体的数组的数组的大小，事实上这是一个DIM+1个元素的数组，分别存储第0维一直到第DIM维的几何体。

另外一个模板参数DOW是Dimension Of World的缩写，标识的是网格所在的流形所在的欧式空间的维数。比如平面上的一个多边形，DOW=2，而对于球面，则DOW=3。

不知你注意到没有，我们事实上到现在为止还没有说网格中的点的坐标呢！DOW事实上给定的就是网格中的点的坐标的维数。在Mesh类中，存储着一个坐标的数组，其中每个元素是一个Point<DOW>型的对象，事实上就是一个坐标而已。

对于一个“0维几何体”，它的顶点序号具有不一样的意义，事实上是说自己作为网格中的一个节点，坐标是坐标数组中的哪一个。而“0维几何体界”的拓扑边界是没有意义的，所以请不要使用其中的信息：那个数AFEPack不会负责维护的！

“0维几何体” 的这种特殊性，给我们的程序常常带来了一些麻烦。比如，我们想取一个几何体geo的第k个顶点的坐标，需要使用下面的方式：
```cpp
    u_int vtx_idx = geo.vertex(k); /// 节点几何体的序号
    GeometryBM& vtx_geo = a_mesh.geometry(0, vtx_idx); /// 取出该节点几何体
    u_int pnt_idx = vtx_geo.vertex(0); /// 取出坐标的序号
    Point<DOW>& pnt = a_mesh.point(pnt_idx); /// 取出坐标
```
有时候，你自己可以保证vtx_idx和pnt_idx是一样的，这样可以省下来两行，但是库的内部并不能保证这一点。
AFEPack的内部设计语言将节点称为“0维几何体”，把线段称为“1维几何体”等等的目标并不是想标新立异，而是为了统一，因为计算机更善于理解更加统一的语言。而且，在这样的语言下，我们事实上可以描述任意维数的网格，对网格中的几何体的形状也没有什么既定的要求，是一种非常灵活的描述网格的数据格式。

一个Mesh类中的数据，如果被写到文件中，就是完全按照其在内存中的方式进行输出的。首先输出的是坐标，然后是0维几何体，1维几何体，...，直到DIM维几何体。

# AFEPack 教学系列之九: 局部加密功能的展示

网格的局部加密从原理上来说是很简单的事情，但是具体的实现相当复杂。AFEPack支持在二维和三维的单纯形网格上进行局部加密和稀疏化的操作。为了实现加密和稀疏化，我们事实上要处理不同的网格之间的关系。AFEPack通过一个所谓几何遗传树的数据结构来实现。几何遗传树是一个由不断加密的几何体构成的树型数据结构，而每个网格则是这棵树的一个所谓截面上的所有叶子节点构成的。AFEPack支持的网格上的操作可以是非常大幅度的操作，它通过几个分解了的动作简单地完成了需要构造复杂的算法来完成的功能。涉及到的概念包括非正则网格、半正则网格和正则网格。在一个网格被进行了自适应操作后，获得的是一个所谓非正则网格。这个非正则网格进行一下半正则化，就得到一个半正则网格。然后再进行一下正则化，就得到了正则网格。在正则网格上就可以按照正常的程序建造有限元空间了。这中间的过程相当复杂，不是一两句话就可以说清楚的。如果有兴趣，我们可以在讨论班上慢慢介绍。下面就是一个简单的例子：
```cpp
/**
 * @file   refine.cpp
 * @author Robert Lie
 * @date   Wed Mar  7 11:43:40 2007
 *
 * @brief 
 *
 *
 */

#include <AFEPack/HGeometry.h>

#define DIM 2

int main(int argc, char * argv[])
{
  /// 声明几何遗传树，并从 Easymesh 格式的文件中读入数据
  HGeometryTree<DIM> h_tree;
  h_tree.readEasyMesh(argv[1]);

  /// 在背景网格上建立第一个非正则网格，并均匀加密三次
  IrregularMesh<DIM> irregular_mesh(h_tree);
  irregular_mesh.globalRefine(3);

  do {
    /// 对非正则网格做半正则化和正则化
    irregular_mesh.semiregularize();
    irregular_mesh.regularize(false);

    /// 这就是通过正则化得到的网格
    RegularMesh<DIM>& regular_mesh = irregular_mesh.regularMesh();

    /// 将这个网格输出到数据文件中
    regular_mesh.writeOpenDXData("D.dx");
    std::cout << "Press ENTER to continue or CTRL+C to stop ..." << std::flush;
    getchar();

    /**
     * 下面一段计算用于加密的指示子。我们在以 c0 和 c1 两个点为中心，以半
     * 径处于 0.004 到 0.005 直接的环状区域中设置指示子的值为单元的面积，
     * 从而会将这两个环状区域中的地带均匀加密掉。其他部分的指示子是零。
     * 所有这些计算都是手工进行的，您需要深入了解网格中的数据放置方式。
     *
     */
    Indicator<DIM> indicator(regular_mesh);
    Point<DIM> c0(0.495, 0.5);
    Point<DIM> c1(0.505, 0.5);
    for (int i = 0;i < regular_mesh.n_geometry(2);i ++) {
      /// 这是三角形的三个顶点。对于三角形和双生三角形都是这样的。
      Point<DIM>& p0 = regular_mesh.point(regular_mesh.geometry(2,i).vertex(0));
      Point<DIM>& p1 = regular_mesh.point(regular_mesh.geometry(2,i).vertex(1));
      Point<DIM>& p2 = regular_mesh.point(regular_mesh.geometry(2,i).vertex(2));

      /// 这是三角形的重心
      Point<DIM> p((p0[0] + p1[0] + p2[0])/3., (p0[1] + p1[1] + p2[1])/3.);

      /// 手工计算三角形的面积
      double area = ((p1[0] - p0[0])*(p2[1] - p0[1]) -
                     (p2[0] - p0[0])*(p1[1] - p0[1]));

      /// 计算三角形的重心到 c0 和 c1 的距离。如果三角形过大，这个计算会抓
      /// 不住正确的指示子，所以一开始我们把网格均匀加密了三倍。
      double d0 = (p - c0).length();
      double d1 = (p - c1).length();
      if ((d0 > 0.004 && d0 < 0.005) ||
          (d1 > 0.004 && d1 < 0.005)) { /// 在环状区域中设置指示子
        indicator[i] = area;
      }
    }

    /// 下面的几行调用进行自适应的函数，都是套话。
    MeshAdaptor<DIM> mesh_adaptor(irregular_mesh);
    mesh_adaptor.convergenceOrder() = 0.; /// 设置收敛阶为0
    mesh_adaptor.refineStep() = 1; /// 最多允许加密一步
    mesh_adaptor.setIndicator(indicator);
    mesh_adaptor.tolerence() = 2.5e-10; /// 自适应的忍量
    mesh_adaptor.adapt(); /// 完成自适应
  } while (1);   

  return 0;
}

/**
 * end of file
 *
 */
```
下面是一个算例网格的样子：
http://rli.bloghome.cn/photos/p_small/8d365cc574e6f262ae930c9491549b0e.gif

# AFEPack教学系列之十: 处理边界条件(I)

我们学偏微分方程的数值解的课程的时候，首先就是解泊松方程。而在会用有限元程序求解泊松方程了以后，马上就会考虑泊松方程的一些变形。当然，如果不使用一些软件包，白手起家来做一个有限元程序求解泊松方程的话，确实需要相当的勇气和时间的。

在考虑泊松方程的变形的时候，或许第一件想做的事情就是期望解一下Neuman边值问题了。但是事实上的情况是，求解变二次系数的方程的实现，比实现Neumann边值的程序要简单得多。

前面，我们已经看到了变二次系数的方程的求解。这里，让我们来讨论Neumann边值问题的实现方法。应用边值条件这个问题之所以麻烦，就麻烦在边值条件其实是一个约束。我们实现它的方法，却绝对不能将它作为一个约束加上去，否则就会面临要解一个非正定矩阵的尴尬局面，是无论如何不能接受的。因此，我们必须将这个约束手工从线性系统中消去。

要想手工地修正线性系统，我们必须深入了解线性系统中实现矩阵和向量的这些类。现在AFEPack使用的是来自deal.II的线性代数包，所以，我们在这里先介绍一下deal.II中提供的稀疏矩阵和向量的类的结构和使用方法。

deal.II 的作者Wolfgang曾经因为AFEPack中实现的多线程类的界面中，使用了deal.II的代码中实现技术而发信质问我。但是我其实在那里已经给了deal.II包citation的，只是因为我现在为了方便使用中文的用户，改用中文写注释了。而Wolfgang是因为看不懂中文找不到给他的引用在什么地方而已。

我在这里提到这个，主要是希望提醒一些同僚，以后在这些敏感的位置，还是细致一些比较恰当。Wolfgang本人是一个广受称赞的绝对nice guy，但是对于credit这样的事情也是毫不含糊的哦，;-)

鉴于这个原因，我在这下面写一句英文的说明，主要是为了给deal.II的作者们看的。

In the following, let us discussed the usage of the sparse matrix and vector implemented 
in deal.II(url: http://www.dealii.org). Please refer the original document obtained from 
its homepage for details.

AFEPack使用了deal.II中的线性代数包，看中的是deal.II中的实现结构很漂亮，给使用者的接口通过细致的面对对象的分析，相当合理和有扩展性。作为设计算法的研究人员，我并不在意其中的求解器是否高效－－－反正我们是要自己写求解器的。另外，现在新版的deal.II开始支持使用petsc的求解器，这或许对很多不想自己写求解器的人来说，是个大好事啊。

deal.II中的稀疏矩阵，使用的是所谓的行压缩存储结构。这个行压缩存储结构，事实上就是用三个数组来存储一个稀疏矩阵的非零元素。如果您不知道啥叫做稀疏矩阵的话，我这里就不补课了。
deal.II中稀疏矩阵的数据结构:
考虑一个M x N的稀疏矩阵，数据结构中的三个数组分别叫做 ：

1. 行开始指标数组(rowstarts)
这是一个长度为M + 1的整数数组，其中第i个位置的这个整数的值，是我们从矩阵的第0行开始，按行来数矩阵中的非零元素的个数，一直数到第i行的第一个非零元素的时候，数出来的这个数。由于是用C++，这里我们的记数都是从0开始的。从而我们知道，这个数组的第0个元素总是0的，第M个元素则是该矩阵中所有非零元的个数，记为nnz。

2. 列指标数组(colnums)
这也是一个整数数组，长度为nnz。其中的每个元素事实上就是每一行的非零元素是哪一列的列指标。比如，对于第i行，我们知道这一行总共的非零元的个数是

rowstarts[i + 1] - rowstarts[i]

而这一行的哪几个列是非零的，这些列的列指标就是colnums数组中的

colnums[rowstarts[i]],
colnums[rowstarts[i] + 1],
... ...,
colnums[rowstarts[i + 1] - 1]

这些个数。

其实，有了上面的这两个数组，我们就已经获得了这个稀疏矩阵的非零元分布的样子了。所以，这两个数组在deal.II中先首先包装成为了所谓的稀疏模板类SparsityPattern。这样的包装是非常必要的，因为我们常常会使用相同的稀疏模板构造很多不同的稀疏矩阵。

3. 矩阵元素数组(val)
这是长度为nnz的实数数组，现在deal.II对float, double两种数据类型进行有特别的实现。这个数组中的第k个元素的意义为：
如果rowstarts[i] <= k < rowstarts[i + 1]，而j = colnums[k]，则矩阵中的元素
a(i, j) = val[k]

deal.II的稀疏矩阵类SparseMatrix中包含一个对SparsityPattern的引用以及一个val数组。

所以，您需要学会SparsityPattern和SparseMatrix这两个类的使用方法。

写太长了，只能在下一篇中说边界条件的问题 了... ...

# AFEPack教学系列之十: 处理边界条件(II)
继续侃，希望您能够搞明白，:-)

现在在库函数中，自动处理了狄氏边值的问题。处理的方式为：首先由用户提供在每种材料标识的情况下，解函数在自由度的插值点上的函数值的表达式；然后，库通过判断每个自由度的材料标识、以及使用自由度的插值点的坐标，获得自由度本身的。从而，在需要求解的线性系统中，我们就已经知道了一部分解变量的值，于是，现在的问题就可以如下来说了：我们有一个线性方程组

                  A x = b

现在，我们可以将其分块为

                [ A_11 A_12 ] [ x_1 ]   [ b_1 ]
                [           ] [     ] = [     ]
                [ A_21 A_22 ] [ x_2 ]   [ b_2 ]


现在呢，我们已经知道了x_2的值了。于是，我们的手段就是用下面的线性系统来代替上面的：

                [ A_11   0 ] [ x_1 ]   [ b_1 - A_12 x_2 ]
                [          ] [     ] = [                ]
                [  0     I ] [ x_2 ]   [       x_2      ]

其中I是单位对角块，0表示零块。可以看到，修改过的系统和原系统相比，具有完全相同的解，而且，整个矩阵的结构可以保持不必修改。在实际实现的时候，可以用A_22的对角线来代替I这块，效果更好一些。

上面就是库中间处理狄氏边值的原理。但是，如果基函数不是正交的拉格朗日插值函数，库中的程序会给出错误的结果的。在事实上遇到这样的情况的时候，解决的方案也是有的，之所以能够解决的基本原理在于，狄氏边值是强制加上的，所以边值的处理可以和区域内部的求解解耦合，我们就不讨论这个话题了。

下面我们来看看加上Neumann边值的方法。首先，对于Neumann边值问题，我们知道边值将会贡献一个边界积分到右端项上去，因此，我们需要能够做边界积分的功能；另外，Neumann边值问题的解具有一个自由度(假设单连通区域)，在解上加上一个常数的话还是解，因此，我们需要在线性系统中将这个因素带来的矩阵退化消掉，这个得我们手工做一下。

我们先来看边界积分的问题。为了能够做边界积分，我们需要使用所谓的DGFEMSpace类。这个类是后来加上去的，使得空间中不但有体单元，而且有面单元，只是面单元上不分配自由度而已。请看下面的代码示例：
```cpp
#define DIM 2 /// 考虑二维问题

Mesh<DIM> mesh; /// 网格对象
mesh.readData(data_file) /// 读入网格数据

/**
 * 三角形单元上的一阶参考单元
 *
 */
TemplateGeometry<DIM> triangle_template_geometry;
CoordTransform<DIM,DIM> triangle_coord_transform;
TemplateDOF<DIM> triangle_1_template_dof;
BasisFunctionAdmin<double,DIM,DIM> triangle_1_basis_function;
UnitOutNormal<DIM> triangle_unit_out_normal; /// 注意这一行

triangle_template_geometry.readData("triangle.tmp_geo");
triangle_coord_transform.readData("triangle.crd_trs");
triangle_unit_out_normal.readData("triangle.out_nrm");

triangle_1_template_dof.reinit(triangle_template_geometry);
triangle_1_template_dof.readData("triangle.1.tmp_dof");
triangle_1_basis_function.reinit(triangle_1_template_dof);
triangle_1_basis_function.readData("triangle.1.bas_fun");

/// 做体单元的参考单元
std::vector<TemplateElement<double,DIM,DIM> > template_element(1);
template_element[0].reinit(triangle_template_geometry,
                           triangle_1_template_dof,
                           triangle_coord_transform,
                           triangle_1_basis_function,
                           triangle_unit_out_normal);
/**
 * 将一维的线变换到二维的线的模板几何信息
 *
 */
TemplateGeometry<DIM-1> interval_template_geometry;
CoordTransform<DIM-1,DIM> interval_to2d_coord_transform;

interval_template_geometry.readData("interval.tmp_geo");
interval_to2d_coord_transform.readData("interval.to2d.crd_trs");

/// 做面单元的模板单元
std::vector<TemplateDGElement<DIM-1,DIM> > dg_template_element(1);
dg_template_element[0].reinit(interval_template_geometry,
                              interval_to2d_coord_transform);

/// 声明有限元空间
DGFEMSpace<double,DIM> fem_space(mesh,
                                 template_element,
                                 dg_template_element);

/// 建立有限元空间的体单元
u_int n_ele = mesh.n_geometry(DIM);
fem_space.element().resize(n_ele);
for (u_int i = 0;i < n_ele;++ i) {
  fem_space.element(i).reinit(fem_space,i,0);
}
fem_space.buildElement();
fem_space.buildDof();
fem_space.buildDofBoundaryMark();

u_int n_side = mesh.n_geometry(DIM-1);
u_int n_dg_ele = 0; /// 我们来统计一下边界上的边的个数
for (u_int i = 0;i < n_side;++ i) {
  if (mesh.geometry(DIM-1,i).boundaryMark() != 0) {
    n_dg_ele += 1;
  }
}
/// 建立有限元空间的面单元
fem_space.dgElement().resize(n_dg_ele);
for (u_int i = 0, j = 0;i < n_side;++ i) {
  if (mesh.geometry(DIM-1,i).boundaryMark() != 0) {
    fem_space.dgElement(j).reinit(fem_space, i, 0);
    j += 1;
  }
}
fem_space.buildDGElement();

/// 哈哈，现在已经可以在面单元上积分了
DGFEMSpace<double,DIM>::DGElementIterator
  the_dgele = fem_space.beginDGElement(),
  end_dgele = fem_space.endDGElement();
for (;the_dgele != end_dgele;++ the_dgele) {

  /// 参考单元的体积
  double vol = the_dgele->templateElement().volume();

  /// 积分公式，注意它的积分点的维数少一啊
  const QuadratureInfo<DIM-1>& qi = the_dgele->findQuadratureInfo(3);
  u_int n_q_pnt = qi.n_quadraturePoint();

  /// 变换的雅可比行列式
  std::vector<double> jac = the_dgele->local_to_global_jacobian(qi.quadraturePoint());

  /// 积分点变换到网格中去
  std::vector<Point<DIM> > q_pnt = the_dgele->local_to_global(qi.quadraturePoint());

  /// 和它邻接的两个体单元的指针
  Element<double,DIM> * p_neigh0 = the_dgele->p_neighbourElement(0);
  Element<double,DIM> * p_neigh1 = the_dgele->p_neighbourElement(1);

  /// 法向单位向量，对 0 号邻居来说是外法向
  std::vector<std::vector<double> > un = unitOutNormal(q_pnt, *p_neigh0, *the_dgele);

  /// 我想下面的数值积分程序你应该会写了吧，不明白你还需要什么更多的信
  /// 息了，;-)
}
```
太长了，太长了，下回分解吧... ...

# AFEPack教学系列之十: 处理边界条件(III)

现在，我们已经会了在边界上进行的积分，从而能够将非齐次Neumann边界条件对线性系统的右端项上的贡献算出来了。那么，现在的矩阵仍然是一个奇异矩阵，而且我们精确地知道，只要我们固定住解中间的一个变量，就能够得到唯一解。下面，我们就来对矩阵进行手工地修改，实现这个效果。事实上，我们做的事情，相当于对一个变量，使用了狄氏边界条件而已。我假设您已经了解了deal.II中的SparsityPattern、SparseMatrix和Vector这几个类的用法了，我们的线性系统已经准备好了为A x = b，请看示例的代码：
```cpp
  SparseMatrix<double> stiff_matrix; /// 这是矩阵 A
  FEMFunction<double> u_h; /// 这是解变量 x
  Vector<double> f_h; /// 这是右端项 b

  { /// 先修改矩阵的第一行和第一列
    const unsigned int * row_start
      = stiff_matrix.get_sparsity_pattern().get_rowstart_indices();
    const unsigned int * column
      = stiff_matrix.get_sparsity_pattern().get_column_numbers();
    for (int j = row_start[0] + 1;j < row_start[1];j ++) {
      stiff_matrix.global_entry(j) = 0.0;
      int k = column[j];
      const unsigned int * p = std::find(&column[row_start[k] + 1],
                                         &column[row_start[k + 1]], 0);
      if (p != &column[row_start[k + 1]]) {
        stiff_matrix.global_entry(p - &column[row_start[0]]) = 0.0;
      }
    }

    /// 这是修改解变量和右端项的第一个元素
    u_h(0) = 0.0;
    f_h(0) = 0.0;
  }
```

这样，矩阵就成为了非奇异的矩阵，可以调用正定矩阵的求解器来求解了。只是这个矩阵尽管和狄氏边值问题的矩阵的规模差不太多，但是它的条件数比狄氏边值问题的条件数要坏很多，求解所花的迭代步数要多不少，怎么加快求解的技巧值得深入地挖一挖。

# AFEPack教学系列之十一：解二次方程+局部加密
```cpp
/**
 * @file   refine.cpp
 * @author Robert Lie
 * @date   Wed Mar  7 11:43:40 2007
 *
 * @brief  下面的这个程序，是将求解偏微分方程和局部网格加密简单组合在
 *         了一起。我们用的方程是一个含有间断二次系数的椭圆型方程，在
 *         系数间断的位置，解会有一个弱间断。我们瞎算了一个长得象误差
 *         估计的量来做自适应指示子，您可以试试它的效果，;-)
 *
 */

#include <AFEPack/AMGSolver.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/EasyMesh.h>
#include <AFEPack/HGeometry.h>

#define DIM 2

/// 边界条件
double _u_b_(const double * p)
{
  return sin(p[0] + p[1]);
}

/// 右端项
double _f_(const double * p)
{
  return sin(p[0]) + exp(p[1]);
}

/// 二次系数
double A(const Point<DIM>& p)
{
  if (p[0] > p[1]*p[1]) {
    return 4.0;
  } else if (p[0] > sin(p[1])) {
    return 2.0;
  } else {
    return 1.0;
  }
}

/// 二次问题的刚度矩阵，这一段程序我们已经在前面的文章中解释过了。
class Matrix : public StiffMatrix<DIM,double>
{
public:
  Matrix(FEMSpace<double,DIM>& sp) : StiffMatrix<DIM,double>(sp) {};
  virtual void getElementMatrix(const Element<double,DIM> & ele0,
                const Element<double,DIM> & ele1,
                const ActiveElementPairIterator<DIM>::State
                state=ActiveElementPairIterator<DIM>::EQUAL) {
    int n_ele_dof = elementDof0().size();
    double volume = ele0.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = ele0.findQuadratureInfo(algebricAccuracy());
    std::vector<double> jacobian = ele0.local_to_global_jacobian(quad_info.quadraturePoint());
    int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<Point<DIM> > q_point = ele0.local_to_global(quad_info.quadraturePoint());
    std::vector<std::vector<std::vector<double> > > basis_gradient = ele0.basis_function_gradient(q_point);
    std::vector<std::vector<double> > basis_value = ele0.basis_function_value(q_point);
    for (int l = 0;l < n_quadrature_point;l ++) {
      double Jxw = quad_info.weight(l)*jacobian[l]*volume;
      double a_val = A(q_point[l]);
      for (int j = 0;j < n_ele_dof;j ++) {
        for (int k = 0;k < n_ele_dof;k ++) {
          elementMatrix(j,k) += Jxw*a_val*
            innerProduct(basis_gradient[j][l], basis_gradient[k][l]);
        }
      }
    }
  }
};

/// 在当前网格上求解二次方程，然后计算自适应指示子。关于求解方程的部分
/// 和前面的其他程序是一模一样的，但是计算自适应指示子的部分则是完全手
/// 工做的。我想您自己读懂这个部分会比听我讲清楚这个部分要获益更多，我
/// 就特意把注释都抹去了，;-)
void get_indicator(Indicator<DIM>& ind,
                   RegularMesh<DIM>& mesh)
{
  TemplateGeometry<DIM> triangle_template_geometry;
  triangle_template_geometry.readData("triangle.tmp_geo");
  CoordTransform<DIM,DIM> triangle_coord_transform;
  triangle_coord_transform.readData("triangle.crd_trs");
  TemplateDOF<DIM> triangle_template_dof(triangle_template_geometry);
  triangle_template_dof.readData("triangle.1.tmp_dof");
  BasisFunctionAdmin<double,DIM,DIM> triangle_basis_function(triangle_template_dof);
  triangle_basis_function.readData("triangle.1.bas_fun");

  TemplateGeometry<DIM> twin_triangle_template_geometry;
  twin_triangle_template_geometry.readData("twin_triangle.tmp_geo");
  CoordTransform<DIM,DIM> twin_triangle_coord_transform;
  twin_triangle_coord_transform.readData("twin_triangle.crd_trs");
  TemplateDOF<DIM> twin_triangle_template_dof(twin_triangle_template_geometry);
  twin_triangle_template_dof.readData("twin_triangle.1.tmp_dof");
  BasisFunctionAdmin<double,DIM,DIM> twin_triangle_basis_function(twin_triangle_template_dof);
  twin_triangle_basis_function.readData("twin_triangle.1.bas_fun");

  std::vector<TemplateElement<double,DIM,DIM> > template_element(2);
  template_element[0].reinit(triangle_template_geometry,
                             triangle_template_dof,
                             triangle_coord_transform,
                             triangle_basis_function);
  template_element[1].reinit(twin_triangle_template_geometry,
                             twin_triangle_template_dof,
                             twin_triangle_coord_transform,
                             twin_triangle_basis_function);

  FEMSpace<double,DIM> fem_space(mesh, template_element);

  int n_element = mesh.n_geometry(DIM);
  fem_space.element().resize(n_element);
  for (int i = 0;i < n_element;i ++) {
    if (mesh.geometry(DIM,i).n_vertex() == 3) {
      fem_space.element(i).reinit(fem_space,i,0);
    } else {
      fem_space.element(i).reinit(fem_space,i,1);      
    }
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();

  Matrix stiff_matrix(fem_space);
  stiff_matrix.algebricAccuracy() = 2;
  stiff_matrix.build();

  FEMFunction<double,DIM> u_h(fem_space);
  Vector<double> f_h;
  Operator::L2Discretize(&_f_, fem_space, f_h, 3);

  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET,
                                        1, &_u_b_);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(stiff_matrix, u_h, f_h);

  AMGSolver solver(stiff_matrix);
  solver.solve(u_h, f_h);

  u_h.writeOpenDXData("u_h.dx");

  /// 从这里开始就是计算自适应指示子的部分了，;-)
  u_int n_side = mesh.n_geometry(1);
  std::vector<bool> sflag(n_side, false);
  std::vector<double> sjump(n_side);
  FEMSpace<double,DIM>::ElementIterator
    the_ele = fem_space.beginElement(),
    end_ele = fem_space.endElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    const GeometryBM& ele_geo = mesh.geometry(DIM,i);
    u_int n_bnd = ele_geo.n_boundary();
    for (u_int j = 0;j < n_bnd;++ j) {
      u_int sid_idx = ele_geo.boundary(j);
      const GeometryBM& sid_geo = mesh.geometry(1, sid_idx);
      const GeometryBM& vtx0 = mesh.geometry(0, sid_geo.vertex(0));
      const Point<DIM>& p0 = mesh.point(vtx0.vertex(0));
      const GeometryBM& vtx1 = mesh.geometry(0, sid_geo.vertex(1));
      const Point<DIM>& p1 = mesh.point(vtx1.vertex(0));
      Point<DIM> p(0.5*(p0[0] + p1[0]), 0.5*(p0[1] + p1[1]));
      std::vector<double> u_h_grad = u_h.gradient(p, *the_ele);

      if (sflag[sid_idx] == false) {
        sjump[sid_idx] = (u_h_grad[0]*(p0[1] - p1[1]) -
                          u_h_grad[1]*(p1[0] - p0[0]));
        sflag[sid_idx] = true;
      } else {
        sjump[sid_idx] -= (u_h_grad[0]*(p0[1] - p1[1]) -
                           u_h_grad[1]*(p1[0] - p0[0]));
        sflag[sid_idx] = false;
      }
    }
  }

  the_ele = fem_space.beginElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    const GeometryBM& ele_geo = mesh.geometry(DIM,i);
    u_int n_bnd = ele_geo.n_boundary();
    ind[i] = 0.0;
    for (u_int j = 0;j < n_bnd;++ j) {
      u_int sid_idx = ele_geo.boundary(j);
      if (sflag[sid_idx] == true) continue;
      ind[i] += sjump[sid_idx]*sjump[sid_idx];
    }
  }
}


int main(int argc, char * argv[])
{
  /// 声明几何遗传树，并从Easymesh格式的文件中读入数据
  HGeometryTree<DIM> h_tree;
  h_tree.readEasyMesh(argv[1]);

  /// 在背景网格上建立第一个非正则网格，并均匀加密三次
  IrregularMesh<DIM> irregular_mesh(h_tree);
  irregular_mesh.globalRefine(2);

  do {
    /// 对非正则网格做半正则化和正则化
    irregular_mesh.semiregularize();
    irregular_mesh.regularize(false);

    /// 这就是通过正则化得到的网格
    RegularMesh<DIM>& regular_mesh = irregular_mesh.regularMesh();

    /// 将这个网格输出到数据文件中
    regular_mesh.writeOpenDXData("D.dx");
    std::cout << "Press ENTER to continue or CTRL+C to stop ..." << std::flush;
    getchar();

    /// 下面一段计算用于加密的指示子。
    Indicator<DIM> indicator(regular_mesh);
    get_indicator(indicator, regular_mesh);

    /// 下面的几行调用进行自适应的函数，都是套话。
    MeshAdaptor<DIM> mesh_adaptor(irregular_mesh);
    mesh_adaptor.convergenceOrder() = 1.; /// 设置收敛阶为0
    mesh_adaptor.refineStep() = 2; /// 最多允许加密一步
    mesh_adaptor.setIndicator(indicator);
    mesh_adaptor.tolerence() = 1.0e-06; /// 自适应的忍量
    mesh_adaptor.adapt(); /// 完成自适应
  } while (1);    

  return 0;
}

/**
 * end of file
 *
 */
```
列几个问题在这里，看有没有人能回答：
- 现在我们没有利用到上一步的解来作为下一步的解的初值，您能够写出这样的程序么？
- 我们这里瞎写的这个自适应指示子特别地抹去了所有注释，您能够读懂这是个什么量么？
- 您读懂了上面的计算自适应指示子的程序后，能够自己把这个部分更新为正确的后验误差估计的形式么？ 

# AFEPack教学系列之十二: 在流形上做网格加密的例子

我们常常看见进行自适应加密和稀疏化的程序，但是如果问题不是在一个欧式空间中的问题，而是在一个上提出的，比如是在一个球面上，那么我们怎么能够完成这样的网格上的h-自适应呢？

这个事情看上去好像比较复杂，但是仔细考虑起来其实是比较简单的。这中间最关键的部分其实在于我们如何定义“直线”，如果你给直线一个不一样的定义，那我们就可以在一个流形上，按照欧式空间中的算法，来做加密和稀疏化了。为了实现这一点，AFEPack 提供一个接口让用户来定义直线的方案，而为了定义直线，我们最根本的事实上要定义中点 。如果对任何一个两个端点已知的线段，我们能够知道其中点的计算方法，那么完成流形上的网格的加密和稀疏化的工作就显而易见了。请看下面的例子：这是前面二维平面上的加密和稀疏化的程序上做一个简单修改得到的。

我们使用的初始网格事实上就是一个正方形， 数据如下：
```cpp
o.n
-------------------------------------------------
4 2 5 **(Nnd, Nee, Nsd)**
0 -1 -1 0 1
1  1 -1 0 1
2  1  1 0 1
3 -1  1 0 1
-------------------------------------------------

o.s
-------------------------------------------------
5
0 0 1 0 -1 1
1 1 2 0 -1 1
2 0 2 1 0 0
3 2 3 1 -1 1
4 0 3 -1 1 1
-------------------------------------------------

o.e
-------------------------------------------------
2 4 5   **(Nee, Nnd, Nsd)**
0 0 1 2 -1 1 -1 1 2 0
1 0 2 3 -1 -1 0 3 4 2
-------------------------------------------------
```
程序如下：
```cpp
#include <AFEPack/HGeometry.h>

#define DIM 2  /// 我们在一个 2 维流形上操作
#define DOW 3  /// 这个 2 维的流形上的点的坐标是在三维欧式空间中

/**
 * 流形上计算中点的方法：猜猜看这样计算中点得到一个什么样子的流形啊
 *
 * @param p0 线段的一个端点
 * @param p1 线段的另一个端点
 * @param bm 线段的材料标识
 * @param p  线段的中点
 */
void my_midpoint(const Point<DOW>& p0,
                 const Point<DOW>& p1,
                 const int& bm,
                 Point<DOW>& p)
{
  double R1 = sqrt(2), R0 = 0.3*R1;
  p = (p0 + p1);
  p /= 2.0;
  double r = sqrt(p[0]*p[0] + p[1]*p[1]);
  if (r < R0) {
    p[2] = fabs(p[2] - R1);
    r = p.length();
    p /= r/R0;
    p[2] += R1;
  }
  else {
    r = p.length();
    p /= r/R1;
  }
}

int main(int argc, char * argv[])
{
  /// 声明几何遗传树，并从Easymesh格式的文件中读入数据
  HGeometryTree<DIM,DOW> h_tree;
  h_tree.readEasyMesh(argv[1]);

  /// 秘密都藏在这一行中哦
  HGeometry<1,DOW>::mid_point = &(my_midpoint);

  /// 在背景网格上建立第一个非正则网格，并均匀加密三次
  IrregularMesh<DIM,DOW> irregular_mesh(h_tree);
  irregular_mesh.globalRefine(6);

  /// 对非正则网格做半正则化和正则化
  irregular_mesh.semiregularize();
  irregular_mesh.regularize(false);

  /// 这就是通过正则化得到的网格
  RegularMesh<DIM,DOW>& regular_mesh = irregular_mesh.regularMesh();

  /// 将这个网格输出到数据文件中
  regular_mesh.writeOpenDXData("D.dx");

  return 0;
}

/**
 * end of file
 *
 */
```
图：
http://rli.bloghome.cn/photos/p_original/13c87f2eaff69aeeb6fa3e42c4275bbe.gif

# AFEPack教学系列之十三: 试试特征线法解对流扩散方程

```cpp
/**
 * @file   test.cpp
 * @author Robert Lie
 * @date   Thu May 17 22:39:30 2007
 *
 * @brief  我们在一个三角形网格上使用了特征线法来求解一个对流扩散方程。
 *
 *
 */

#include <AFEPack/EasyMesh.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/DGFEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>

/*!

  为了求解对流占优的对流扩散方程

  \f[
    u_t + b \cdot \nabla u = \Delta u + f
  \]

  一个经常的考虑就是通过特征线法来进行离散，去除由于强对流带来的困难。
  在实现特征线法的时候，一个唯一的困难就是在计算上一个时间步的函数值的
  时候，需要为其提供正确的由特征线给出的坐标点所在的单元这个参数。我们
  假设速度场可以手工积分，下面写的函数 find_element 可以找到这个单元参
  数，从而使得上面的方程求解划归为一个抛物型方程求解的样子。所以，只需
  要理解了这个函数，世界就顿时有秩序了。

*/

#define DIM 2

/**
 * 假设网格为一个三角形网格，从一个单元出发，寻找点所位于的单元。要求
 * 单元是一个 DGFEMSpace 空间的单元。
 *
 * @param p 点的坐标
 * @param ele 出发单元的指针
 * @param sp 单元所在的有限元空间
 *
 * @return 最后找到的单元的指针，如果寻找到区域的边界之外，则所返回一
 *         个空指针。
 *
 */
inline Element<double,DIM> *
find_element(const Point<DIM>& p,
         Element<double,DIM> * ele,
         DGFEMSpace<double,DIM>& sp)
{
  Mesh<DIM>& mesh = sp.mesh();                    // 取得有限元空间的网格
  GeometryBM& geo = ele->geometry();              // 取得单元的几何体
  double * v[3] = {mesh.point(geo.vertex(0)),
                   mesh.point(geo.vertex(1)),
                   mesh.point(geo.vertex(2))}; // 几何体的三个顶点
  double ele_vol = (v[1][0] - v[0][0])*(v[2][1] - v[0][1])
                 - (v[1][1] - v[0][1])*(v[2][0] - v[0][0]);    // 三角形的面积
  for (u_int i = 0;i < 3;++ i) {                  // 对每个节点做循环
    u_int i1 = (i + 1)%3, i2 = (i + 2)%3;
    double lambda = (p[0] - v[i1][0])*(p[1] - v[i2][1])
                  - (p[1] - v[i1][1])*(p[0] - v[i2][0]);      // 第 i 个面积坐标
    if (lambda/ele_vol < -1.0e-08) {              // 如果为负，说明需要向这个方向搜索
      /// 取出这个边界上的面单元
      DGElement<double,DIM>& dgele = sp.dgElement(geo.boundary(i));
      if (dgele.p_neighbourElement(0) == ele) {  // 如果本单元是 0 号邻居
    if (dgele.p_neighbourElement(1) != NULL) { /**
                            * 如果对面单元不为空，
                            * 则跳到对面继续搜索
                            */
      return find_element(p, dgele.p_neighbourElement(1), sp);
    }
    else { // 否则就是已经到达区域的边界的情况，返回空指针
      return NULL;
    }
      }
      else { // 否则跳到对面单元上去搜索
    return find_element(p, dgele.p_neighbourElement(0), sp);
      }
    }
  }
  return ele;
}

/**
 * 计算当前(时间为 t)的点 p，在时间 t-dt 时候的位置 q，从而有
 *
 * \f[
 *    p = q + \int_{t - dt}^t b(.;t) dt
 * \f]
 *
 * @param p 当前(时间为 t)点坐标
 * @param t 当前时间
 * @param dt 时间步长
 * @param q t-dt时的点坐标
 */
void _b_(const double * p,
         double t,
         double dt,
         double * q)
{
  /// 在这里写速度场
}

/// 初值和边值的表达式
double _u_(const double * p)
{
  return p[0]*exp(p[1]);
}

/// 右端项
double _f_(const double * p)
{
  return p[0]*p[1] + sin(p[1]);
}


/// 1/dt - \Delta 离散出来的矩阵
class Matrix : public L2InnerProduct<DIM,double>
{
private:
  double _dt;
public:
  Matrix(FEMSpace<double,DIM>& sp, double dt) :
    L2InnerProduct<DIM,double>(sp, sp), _dt(dt) {}
  virtual void getElementMatrix(const Element<double,DIM>& e0,
                                const Element<double,DIM>& e1,
                                const ActiveElementPairIterator< DIM >::State s)
  {
    double vol = e0.templateElement().volume();
    u_int acc = algebricAccuracy();
    const QuadratureInfo<DIM>& qi = e0.findQuadratureInfo(acc);
    u_int n_q_pnt = qi.n_quadraturePoint();
    std::vector<double> jac = e0.local_to_global_jacobian(qi.quadraturePoint());
    std::vector<Point<DIM> > q_pnt = e0.local_to_global(qi.quadraturePoint());
    std::vector<std::vector<double> > bas_val = e0.basis_function_value(q_pnt);
    std::vector<std::vector<std::vector<double> > > bas_grad = e0.basis_function_gradient(q_pnt);
    u_int n_ele_dof = e0.dof().size();
    for (u_int l = 0;l < n_q_pnt;++ l) {
      double Jxw = vol*qi.weight(l)*jac[l];
      for (u_int i = 0;i < n_ele_dof;++ i) {
        for (u_int j = 0;j < n_ele_dof;++ j) {
          elementMatrix(i,j) += Jxw*(bas_val[i][l]*bas_val[j][l]/_dt +
                                     innerProduct(bas_grad[i][l], bas_grad[j][l]));
        }
      }
    }
  }
};

int main(int argc, char * argv[])
{
  /// 准备网格
  EasyMesh mesh;
  mesh.readData(argv[1]);

  /// 准备参考单元
  TemplateGeometry<DIM> tmp_geo;
  tmp_geo.readData("triangle.tmp_geo");
  CoordTransform<DIM,DIM> crd_trs;
  crd_trs.readData("triangle.crd_trs");
  TemplateDOF<DIM> tmp_dof(tmp_geo);
  tmp_dof.readData("triangle.1.tmp_dof");
  BasisFunctionAdmin<double,DIM,DIM> bas_fun(tmp_dof);
  bas_fun.readData("triangle.1.bas_fun");
  UnitOutNormal<DIM> unit_out_normal;
  unit_out_normal.readData("triangle.out_nrm");

  std::vector<TemplateElement<double,DIM> > tmp_ele(1);
  tmp_ele[0].reinit(tmp_geo, tmp_dof, crd_trs, bas_fun, unit_out_normal);

  TemplateGeometry<DIM-1> interval_tmp_geo;
  interval_tmp_geo.readData("interval.tmp_geo");
  CoordTransform<DIM-1,DIM> interval_crd_trs;
  interval_crd_trs.readData("interval.crd_trs");  

  std::vector<TemplateDGElement<DIM-1,DIM> > dg_tmp_ele(1);
  dg_tmp_ele[0].reinit(interval_tmp_geo, interval_crd_trs);

  /// 定制有限元空间
  DGFEMSpace<double,DIM> fem_space(mesh, tmp_ele, dg_tmp_ele);
  u_int n_ele = mesh.n_geometry(DIM);
  fem_space.element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    fem_space.element(i).reinit(fem_space, i, 0);
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();

  u_int n_edge = mesh.n_geometry(DIM-1); //网格边的条数
  fem_space.dgElement().resize(n_edge);
  for (u_int i = 0;i < n_edge;i ++) {
    fem_space.dgElement(i).reinit(fem_space, i, 0);
  }
  fem_space.buildDGElement();

  /// 准备初值
  FEMFunction<double,DIM> u_h(fem_space);
  Operator::L2Interpolate(&_u_, u_h);

  /// 准备边界条件
  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET,
                                        1,
                                        &_u_);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);

  double t = 0.0;
  Point<DIM> qq_pnt;
  do {
    double dt = 0.01; /// 简单起见，随手取个时间步长算了

    /// 准备线性系统的矩阵
    Matrix mat(fem_space, dt);
    mat.algebricAccuracy() = 3;
    mat.build();

    /// 准备右端项
    Vector<double> rhs(fem_space.n_dof());
    FEMSpace<double,DIM>::ElementIterator the_ele = fem_space.beginElement();
    FEMSpace<double,DIM>::ElementIterator end_ele = fem_space.endElement();
    for (;the_ele != end_ele;++ the_ele) {
      double vol = the_ele->templateElement().volume();
      const QuadratureInfo<DIM>& qi = the_ele->findQuadratureInfo(3);
      u_int n_q_pnt = qi.n_quadraturePoint();
      std::vector<double> jac = the_ele->local_to_global_jacobian(qi.quadraturePoint());
      std::vector<Point<DIM> > q_pnt = the_ele->local_to_global(qi.quadraturePoint());
      std::vector<std::vector<double> > bas_val = the_ele->basis_function_value(q_pnt);

      /// 当基函数的值已知情况下，可以使用下面的函数来加速
      std::vector<double> u_h_val = u_h.value(bas_val, *the_ele);
      const std::vector<int>& ele_dof = the_ele->dof();
      u_int n_ele_dof = ele_dof.size();
      for (u_int l = 0;l < n_q_pnt;++ l) {
        double Jxw = vol*qi.weight(l)*jac[l];
        _b_(q_pnt[l], t, dt, qq_pnt);
        Element<double,DIM> * ele = find_element(qq_pnt, &(*the_ele), fem_space);
        double u_h_val;
        if (ele != NULL) {
          u_h_val = u_h.value(qq_pnt, *ele);
        }
        else {
          u_h_val = _u_(qq_pnt); /// 此时坐标出了区域边界，需要使用边界条件来
                                 /// 给定值。
        }
        double f_val = _f_(q_pnt[l]);
        for (u_int i = 0;i < n_ele_dof;++ i) {
          rhs(ele_dof[i]) += Jxw*bas_val[i][l]*(u_h_val/dt + f_val);
        }
      }
    }

    /// 应用边界条件
    boundary_admin.apply(mat, u_h, rhs);

    /// 求解线性系统
    AMGSolver solver;
    solver.lazyReinit(mat);
    solver.solve(u_h, rhs, 1.0e-08, 50);

    t += dt; /// 更新时间
    
    std::cout << "\n\tt = " <<  t << std::endl;
  } while (1);
 
  return 0;
}

/**
 * end of file
 *
 */
```
# AFEPack教学系列之十四: 讨论班例子(I)

前些天，在我们的讨论班上讲了几次AFEPack的高级功能，把几个例子贴上来供大家参考。由于开始比较简单一些的文件没有保留下来，下面的第一个例子就是两个分线性变量耦合的方程组的移动网格方法的程序。比较懒，不写注释了，反正您要是想学会的话就得自己真正看懂的。如果那位愿意帮我把注释一段一段写上去就好了。这个例子有两个文件，一个是头文件，一个是实现文件，请看下面：
```cpp
/**
 * @file   prob.h
 * @author Robert Lie
 * @date   Sat Jul 14 09:06:42 2007
 * 
 * @brief  
 * 
 * 
 */

#ifndef __prob_h__
#define __prob_h__

#include <AFEPack/EasyMesh.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/AMGSolver.h>
#include <AFEPack/MovingMesh2D.h>

#define DIM 2

class Matrix : public StiffMatrix<DIM, double>
{
private:
  double _dt;
  FEMFunction<double,DIM> * p_u_h;
public:
  Matrix(FEMSpace<double,DIM>& sp,
         double dt,
         FEMFunction<double,DIM>& u_h) : 
    StiffMatrix<DIM,double>(sp), _dt(dt), p_u_h(&u_h) {};
  virtual 
  void getElementMatrix(const Element<double,DIM> & ele0, 
                        const Element<double,DIM> & ele1, 
                        const ActiveElementPairIterator<DIM>::State 
                        state=ActiveElementPairIterator<DIM>::EQUAL);
};

class problem : public MovingMesh2D {
 private:
  TemplateGeometry<DIM>       template_geometry;
  CoordTransform<DIM,DIM>     coord_transform;
  TemplateDOF<DIM>            template_dof;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function;

  std::vector<TemplateElement<double,DIM> > template_element;

  FEMSpace<double,DIM>        fem_space;
  FEMFunction<double,DIM>     u_h;
  FEMFunction<double,DIM>     v_h;
  double                      t;
  double                      dt;

 public:
  void initialize();
  void run();
  void stepForward();

  virtual void getMonitor();
  virtual void updateSolution();
  virtual void outputSolution() {}
};

#endif // __prob_h__

/**
 * end of file
 * 
 */
```
下面这个是实现文件：
```cpp
/**
 * @file   prob.cpp
 * @author Robert Lie
 * @date   Sat Jul 14 09:13:06 2007
 * 
 * @brief  
 * 
 * 
 */

#include "prob.h"

double f(const double * p)
{
  return p[0] + p[1]*p[1];
}

double u_b(const double * p)
{
  return 0.0;
}

double g(const double * p)
{
  return exp(p[0]) + sin(p[1]*p[1]);
}

double v_b(const double * p)
{
  return p[0]*p[1];
}


double a(const double * p)
{
  if (p[0] < 0.6)
    return 2;
  else
    return 1;
}

void Matrix::getElementMatrix(const Element<double,DIM> & ele0, 
			      const Element<double,DIM> & ele1, 
			      ActiveElementPairIterator<DIM>::State)
{
  double vol = ele0.templateElement().volume();
  const QuadratureInfo<DIM>& quad_info = 
    ele0.findQuadratureInfo(algebricAccuracy());
  int n_quadrature_point = quad_info.n_quadraturePoint();

  std::vector<double> jacobian = 
    ele0.local_to_global_jacobian(quad_info.quadraturePoint());

  std::vector<Point<DIM> > q_point = 
    ele0.local_to_global(quad_info.quadraturePoint());

  std::vector<std::vector<double> >
    basis_value = ele0.basis_function_value(q_point);

  std::vector<std::vector<std::vector<double> > > 
    basis_gradient = ele0.basis_function_gradient(q_point);

  const std::vector<int>& ele_dof = ele0.dof();
  u_int n_ele_dof = ele_dof.size();

  for (int l = 0;l < n_quadrature_point;++ l) {
    double Jxw = vol*jacobian[l]*quad_info.weight(l);
    double a_value = a(q_point[l]);
    
    double u_h_val = p_u_h->value(q_point[l], ele0);

    for (int i = 0;i < n_ele_dof;++ i) {
      for (int j = 0;j < n_ele_dof;++ j) {
        elementMatrix(i, j) += Jxw*
          ((1/_dt)*basis_value[i][l]*basis_value[j][l]
           + a_value*innerProduct(basis_gradient[i][l], 
                                  basis_gradient[j][l])
           + 10000*u_h_val*u_h_val*basis_value[i][l]*basis_value[j][l]);
      }
    }
    
  }
}


void problem::initialize()
{
  this->readDomain("M");

  template_geometry.readData("triangle.tmp_geo");
  coord_transform.readData("triangle.crd_trs");
  template_dof.reinit(template_geometry);
  template_dof.readData("triangle.1.tmp_dof");
  basis_function.reinit(template_dof);
  basis_function.readData("triangle.1.bas_fun");

  template_element.resize(1);
  template_element[0].reinit(template_geometry,
                             template_dof,
                             coord_transform,
                             basis_function);

  u_int n_ele = n_geometry(DIM);
  fem_space.reinit(*this, template_element);
  fem_space.element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    fem_space.element(i).reinit(fem_space, i, 0);
  }
  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark();

  u_h.reinit(fem_space);
  v_h.reinit(fem_space);

  t = 0, dt = 1.0e-02;
}

void problem::run()
{
  do {
    moveMesh();

    stepForward();

    t += dt;
    u_h.writeOpenDXData("u_h.dx");

    std::cout << "t = " << t << std::endl;
  } while (1);
}

void problem::stepForward()
{
  FEMFunction<double,DIM> last_u_h(u_h);
  FEMFunction<double,DIM> last_v_h(v_h);

  u_int step = 0;
  do {

    {
      Matrix stiff_matrix(fem_space, dt, u_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(g, fem_space, rhs, 3);

      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space.beginElement(),
        end_ele = fem_space.endElement();
      for (;the_ele != end_ele;++ the_ele) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele);
        std::vector<std::vector<double> > 
          v_h_grad = last_v_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                    u_h_val[l]*v_h_grad[l][0] -
                                    v_h_val[l]*v_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &v_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &v_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &v_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &v_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &v_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, v_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(v_h, rhs);
    }

    {
      FEMFunction<double,DIM> old_u_h(u_h);

      Matrix stiff_matrix(fem_space, dt, v_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(f, fem_space, rhs, 3);

      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space.beginElement(),
        end_ele = fem_space.endElement();
      for (;the_ele != end_ele;++ the_ele) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele);
        std::vector<std::vector<double> > 
          u_h_grad = last_u_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*u_h_val[l] - 
                                    u_h_val[l]*u_h_grad[l][0] -
                                    v_h_val[l]*u_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &u_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &u_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &u_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &u_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &u_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, u_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(u_h, rhs);

      old_u_h.add(-1.0, u_h);
      double error = Functional::L2Norm(old_u_h, 3);

      std::cout << "step " << step ++ << ", error = " << error << std::endl;

      if (error < 1.0e-08) break;
    }

  } while (1);
}


void problem::getMonitor()
{
  FEMSpace<double,DIM>::ElementIterator
    the_ele = fem_space.beginElement(),
    end_ele = fem_space.endElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    Point<DIM> p;
    std::vector<double> u_h_grad = u_h.gradient(p, *the_ele);
    monitor(i) = innerProduct(u_h_grad, u_h_grad);
  }
  smoothMonitor(2);
  double epsilon = 1.0e-02;
  for (u_int i = 0;i < fem_space.n_element();++ i) {
    monitor(i) = 1.0/sqrt(epsilon + monitor(i));
  }
}

void problem::updateSolution()
{
  double msl = moveStepLength();
  Vector<double> rhs_u(fem_space.n_dof());
  Vector<double> rhs_v(fem_space.n_dof());

  FEMSpace<double,DIM>::ElementIterator
    the_ele = fem_space.beginElement(),
    end_ele = fem_space.endElement();
  for (;the_ele != end_ele;++ the_ele) {
    double vol = the_ele->templateElement().volume();
 
    const QuadratureInfo<DIM>& quad_info = 
      the_ele->findQuadratureInfo(2);
    int n_quadrature_point = quad_info.n_quadraturePoint();

    std::vector<double> jacobian = 
      the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

    std::vector<Point<DIM> > q_point = 
      the_ele->local_to_global(quad_info.quadraturePoint());

    std::vector<std::vector<double> >
      basis_value = the_ele->basis_function_value(q_point);

    std::vector<double> u_h_value = u_h.value(q_point, *the_ele);
    std::vector<double> v_h_value = v_h.value(q_point, *the_ele);

    std::vector<std::vector<double> > 
      u_h_grad = u_h.gradient(q_point, *the_ele);
    std::vector<std::vector<double> > 
      v_h_grad = v_h.gradient(q_point, *the_ele);

    std::vector<std::vector<double> >
      move_vec = moveDirection(q_point, the_ele->index());

    const std::vector<int>& ele_dof = the_ele->dof();
    u_int n_ele_dof = ele_dof.size();

    for (int l = 0;l < n_quadrature_point;++ l) {
      double Jxw = vol*jacobian[l]*quad_info.weight(l);
      for (u_int j = 0;j < n_ele_dof;++ j) {
        rhs_u(ele_dof[j]) += Jxw*(u_h_value[l] +
                                  msl*innerProduct(u_h_grad[l], move_vec[l])
                                  )*basis_value[j][l];
        rhs_v(ele_dof[j]) += Jxw*(v_h_value[l] +
                                  msl*innerProduct(v_h_grad[l], move_vec[l])
                                  )*basis_value[j][l];
      }
    }
  }

  MassMatrix<DIM,double> mass_matrix(fem_space);
  mass_matrix.algebricAccuracy() = 2;
  mass_matrix.build();

  AMGSolver solver;
  solver.lazyReinit(mass_matrix);
  solver.solve(u_h, rhs_u);
  solver.solve(v_h, rhs_v);
}

int main(int argc, char * argv[])
{
  problem the_app;
  the_app.initialize();
  the_app.run();

  return 0;
}


/**
 * end of file
 * 
 */
```
我们用的计算区域是一个五边形，给easymesh 的源文件如下:
```cpp
5

0: 0 0 0.1 1
1: 1 0 0.2 1
2: 2 0.8 0.05 1
3: 0.7 2.2 0.2 1
4: 0 1 0.05 1

5

0: 0 1 1
1: 1 2 2
2: 2 3 3
3: 3 4 4
4: 4 0 5
```
# AFEPack教学系列之十四: 讨论班例子(II)

下面这个例子解的还是上面的方程，一个不同之处就在于现在我们是在使用局部加密和稀疏化方法做自适应，不过，为了能够循序渐近起见，现在两个变量的有限元空间总是一样的，请看代码：

头文件：
```cpp
/**
 * @file   prob.h
 * @author Robert Lie
 * @date   Sat Jul 14 09:06:42 2007
 * 
 * @brief  
 * 
 * 
 */

#ifndef __prob_h__
#define __prob_h__

#include <AFEPack/EasyMesh.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/AMGSolver.h>
#include <AFEPack/HGeometry.h>

#define DIM 2

class Matrix : public StiffMatrix<DIM, double>
{
private:
  double _dt;
  FEMFunction<double,DIM> * p_u_h;
public:
  Matrix(FEMSpace<double,DIM>& sp,
         double dt,
         FEMFunction<double,DIM>& u_h) : 
    StiffMatrix<DIM,double>(sp), _dt(dt), p_u_h(&u_h) {};
  virtual 
  void getElementMatrix(const Element<double,DIM> & ele0, 
                        const Element<double,DIM> & ele1, 
                        const ActiveElementPairIterator<DIM>::State 
                        state=ActiveElementPairIterator<DIM>::EQUAL);
};

class problem {
 private:
  TemplateGeometry<DIM>       template_geometry;
  CoordTransform<DIM,DIM>     coord_transform;
  TemplateDOF<DIM>            template_dof;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function;

  TemplateGeometry<DIM>       template_geometry1;
  CoordTransform<DIM,DIM>     coord_transform1;
  TemplateDOF<DIM>            template_dof1;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function1;

  std::vector<TemplateElement<double,DIM> > template_element;

  HGeometryTree<DIM>          h_tree;
  IrregularMesh<DIM>        * ir_mesh;

  FEMSpace<double,DIM>      * fem_space;
  FEMFunction<double,DIM>   * u_h;
  FEMFunction<double,DIM>   * v_h;
  double                      t;
  double                      dt;

 public:
  void initialize();
  void run();
  void stepForward();

  void buildFEMSpace();
  void getIndicator(Indicator<DIM>&);
  void adaptMesh();

};

#endif // __prob_h__

/**
 * end of file
 * 
 */
```
实现文件：
```cpp
/**
 * @file   prob.cpp
 * @author Robert Lie
 * @date   Sat Jul 14 09:13:06 2007
 * 
 * @brief  
 * 
 * 
 */

#include "prob.h"

double f(const double * p)
{
  return p[0] + p[1]*p[1];
}

double u_b(const double * p)
{
  return 0.0;
}

double g(const double * p)
{
  return exp(p[0]) + sin(p[1]*p[1]);
}

double v_b(const double * p)
{
  return p[0]*p[1];
}


double a(const double * p)
{
  if (p[0] < 0.6)
    return 2;
  else
    return 1;
}

void Matrix::getElementMatrix(const Element<double,DIM> & ele0, 
			      const Element<double,DIM> & ele1, 
			      ActiveElementPairIterator<DIM>::State)
{
  double vol = ele0.templateElement().volume();
  const QuadratureInfo<DIM>& quad_info = 
    ele0.findQuadratureInfo(algebricAccuracy());
  int n_quadrature_point = quad_info.n_quadraturePoint();

  std::vector<double> jacobian = 
    ele0.local_to_global_jacobian(quad_info.quadraturePoint());

  std::vector<Point<DIM> > q_point = 
    ele0.local_to_global(quad_info.quadraturePoint());

  std::vector<std::vector<double> >
    basis_value = ele0.basis_function_value(q_point);

  std::vector<std::vector<std::vector<double> > > 
    basis_gradient = ele0.basis_function_gradient(q_point);

  const std::vector<int>& ele_dof = ele0.dof();
  u_int n_ele_dof = ele_dof.size();

  for (int l = 0;l < n_quadrature_point;++ l) {
    double Jxw = vol*jacobian[l]*quad_info.weight(l);
    double a_value = a(q_point[l]);
    
    double u_h_val = p_u_h->value(q_point[l], ele0);

    for (int i = 0;i < n_ele_dof;++ i) {
      for (int j = 0;j < n_ele_dof;++ j) {
        elementMatrix(i, j) += Jxw*
          ((1/_dt)*basis_value[i][l]*basis_value[j][l]
           + a_value*innerProduct(basis_gradient[i][l], 
                                  basis_gradient[j][l])
           + 10000*u_h_val*u_h_val*basis_value[i][l]*basis_value[j][l]);
      }
    }
    
  }
}


void problem::initialize()
{
  h_tree.readEasyMesh("M");
  ir_mesh = new IrregularMesh<DIM>(h_tree);
  ir_mesh->globalRefine(3);
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);

  template_geometry.readData("triangle.tmp_geo");
  coord_transform.readData("triangle.crd_trs");
  template_dof.reinit(template_geometry);
  template_dof.readData("triangle.1.tmp_dof");
  basis_function.reinit(template_dof);
  basis_function.readData("triangle.1.bas_fun");

  template_geometry1.readData("twin_triangle.tmp_geo");
  coord_transform1.readData("twin_triangle.crd_trs");
  template_dof1.reinit(template_geometry1);
  template_dof1.readData("twin_triangle.1.tmp_dof");
  basis_function1.reinit(template_dof1);
  basis_function1.readData("twin_triangle.1.bas_fun");

  template_element.resize(2);
  template_element[0].reinit(template_geometry,
                             template_dof,
                             coord_transform,
                             basis_function);
  template_element[1].reinit(template_geometry1,
                             template_dof1,
                             coord_transform1,
                             basis_function1);

  buildFEMSpace();

  t = 0, dt = 1.0e-02;
}

void problem::buildFEMSpace()
{
  RegularMesh<DIM>& mesh = ir_mesh->regularMesh();

  u_int n_ele = mesh.n_geometry(DIM);
  fem_space = new FEMSpace<double,DIM>(mesh, template_element);
  fem_space->element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    u_int n_vtx = mesh.geometry(DIM, i).n_vertex();
    if (n_vtx == 3) {
      fem_space->element(i).reinit(*fem_space, i, 0);
    }
    else {
      fem_space->element(i).reinit(*fem_space, i, 1);
    }
  }
  fem_space->buildElement();
  fem_space->buildDof();
  fem_space->buildDofBoundaryMark();

  u_h = new FEMFunction<double,DIM>(*fem_space);
  v_h = new FEMFunction<double,DIM>(*fem_space);
}

void problem::adaptMesh()
{
  Indicator<DIM> ind(ir_mesh->regularMesh());
  getIndicator(ind);

  IrregularMesh<DIM> * old_ir_mesh = ir_mesh;
  ir_mesh = new IrregularMesh<DIM>(*old_ir_mesh);

  MeshAdaptor<DIM> mesh_adaptor(*old_ir_mesh, *ir_mesh);
  mesh_adaptor.setIndicator(ind);
  mesh_adaptor.tolerence() = 1.0e-04;
  mesh_adaptor.convergenceOrder() = 1;
  mesh_adaptor.adapt();

  ir_mesh->semiregularize();
  ir_mesh->regularize(false);

  FEMSpace<double,DIM> * old_fem_space = fem_space;
  FEMFunction<double,DIM> * old_u_h = u_h;
  FEMFunction<double,DIM> * old_v_h = v_h;

  buildFEMSpace();

  Operator::L2Interpolate(*old_u_h, *u_h);
  Operator::L2Interpolate(*old_v_h, *v_h);

  delete old_u_h;
  delete old_v_h;
  delete old_fem_space;
  delete old_ir_mesh;
}

void problem::run()
{
  do {
    stepForward();

    adaptMesh();

    t += dt;
    u_h->writeOpenDXData("u_h.dx");
    v_h->writeOpenDXData("v_h.dx");

    std::cout << "t = " << t << std::endl;
  } while (1);
}

void problem::stepForward()
{
  FEMFunction<double,DIM> last_u_h(*u_h);
  FEMFunction<double,DIM> last_v_h(*v_h);

  u_int step = 0;
  do {

    {
      Matrix stiff_matrix(*fem_space, dt, *u_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(g, *fem_space, rhs, 3);

      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space->beginElement(),
        end_ele = fem_space->endElement();
      for (;the_ele != end_ele;++ the_ele) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele);
        std::vector<std::vector<double> > 
          v_h_grad = last_v_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                    u_h_val[l]*v_h_grad[l][0] -
                                    v_h_val[l]*v_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &v_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &v_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &v_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &v_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &v_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(*fem_space);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, *v_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(*v_h, rhs);
    }

    {
      FEMFunction<double,DIM> old_u_h(*u_h);

      Matrix stiff_matrix(*fem_space, dt, *v_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(f, *fem_space, rhs, 3);

      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space->beginElement(),
        end_ele = fem_space->endElement();
      for (;the_ele != end_ele;++ the_ele) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele);
        std::vector<std::vector<double> > 
          u_h_grad = last_u_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*u_h_val[l] - 
                                    u_h_val[l]*u_h_grad[l][0] -
                                    v_h_val[l]*u_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &u_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &u_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &u_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &u_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &u_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(*fem_space);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, *u_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(*u_h, rhs);

      old_u_h.add(-1.0, *u_h);
      double error = Functional::L2Norm(old_u_h, 3);

      std::cout << "step " << step ++ << ", error = " << error << std::endl;

      if (error < 1.0e-08) break;
    }

  } while (1);
}


void problem::getIndicator(Indicator<DIM>& ind)
{
  RegularMesh<DIM>& mesh = ir_mesh->regularMesh();
  u_int n_face = mesh.n_geometry(DIM - 1);
  std::vector<bool> flag(n_face, false);
  std::vector<double> jump(n_face);
  FEMSpace<double,DIM>::ElementIterator
    the_ele = fem_space->beginElement(),
    end_ele = fem_space->endElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    Point<DIM> p;
    std::vector<double> u_h_grad = u_h->gradient(p, *the_ele);
    GeometryBM& geo = the_ele->geometry();
    u_int n_bnd = geo.n_boundary();
    for (u_int j = 0;j < n_bnd;++ j) {
      GeometryBM& bnd = mesh.geometry(DIM-1, geo.boundary(j));
      Point<DIM>& p0 = mesh.point(bnd.vertex(0));
      Point<DIM>& p1 = mesh.point(bnd.vertex(1));
      double a = (u_h_grad[0]*(p1[1] - p0[1]) -
                  u_h_grad[1]*(p1[0] - p0[0]));
      if (flag[bnd.index()] == false) {
        jump[bnd.index()] = a;
        flag[bnd.index()] = true;
      }
      else {
        jump[bnd.index()] -= a;
        flag[bnd.index()] = false;
      }
    }
  }

  the_ele = fem_space->beginElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    GeometryBM& geo = the_ele->geometry();
    u_int n_bnd = geo.n_boundary();
    ind[i] = 0.0;
    for (u_int j = 0;j < n_bnd;++ j) {
      GeometryBM& bnd = mesh.geometry(DIM-1, geo.boundary(j));
      if (flag[bnd.index()]) continue;
      ind[i] += jump[bnd.index()];
    }
  }
}

int main(int argc, char * argv[])
{
  problem the_app;
  the_app.initialize();
  the_app.run();

  return 0;
}


/**
 * end of file
 * 
 */
```
# AFEPack 教学系列之十四: 讨论班例子(III)

下面这个例子是在上面这个的基础上改成的，我们仅仅做了一点儿修改，那就是两个变量现在是在不同的有限元空间中了，但是，这两个有限元空间还是建立在同一个网格上的。请看代码：

头文件：
```cpp
/**
 * @file   prob.h
 * @author Robert Lie
 * @date   Sat Jul 14 09:06:42 2007
 * 
 * @brief  
 * 
 * 
 */

#ifndef __prob_h__
#define __prob_h__

#include <AFEPack/EasyMesh.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/AMGSolver.h>
#include <AFEPack/HGeometry.h>

#define DIM 2

class Matrix : public StiffMatrix<DIM, double>
{
private:
  double _dt;
  FEMSpace<double,DIM>::ElementIterator v_it;
  FEMFunction<double,DIM> * p_u_h;
public:
  Matrix(FEMSpace<double,DIM>& sp,
         double dt,
         FEMFunction<double,DIM>& u_h) : 
    StiffMatrix<DIM,double>(sp), _dt(dt), p_u_h(&u_h) {};
  virtual void buildSparseMatrix();
  virtual 
  void getElementMatrix(const Element<double,DIM> & ele0, 
                        const Element<double,DIM> & ele1, 
                        const ActiveElementPairIterator<DIM>::State 
                        state=ActiveElementPairIterator<DIM>::EQUAL);
};

class problem {
 private:
  TemplateGeometry<DIM>       template_geometry;
  CoordTransform<DIM,DIM>     coord_transform;
  TemplateDOF<DIM>            template_dof;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function;

  TemplateGeometry<DIM>       template_geometry1;
  CoordTransform<DIM,DIM>     coord_transform1;
  TemplateDOF<DIM>            template_dof1;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function1;

  TemplateDOF<DIM>            template_dof2;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function2;

  TemplateDOF<DIM>            template_dof3;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function3;

  std::vector<TemplateElement<double,DIM> > template_element;

  HGeometryTree<DIM>          h_tree;
  IrregularMesh<DIM>        * ir_mesh_u;
  IrregularMesh<DIM>        * ir_mesh_v;

  FEMSpace<double,DIM>      * fem_space_u;
  FEMSpace<double,DIM>      * fem_space_v;
  FEMFunction<double,DIM>   * u_h;
  FEMFunction<double,DIM>   * v_h;
  double                      t;
  double                      dt;

 public:
  void initialize();
  void run();
  void stepForward();

  void buildFEMSpace();
  void getIndicator(Indicator<DIM>&);
  void adaptMesh();

};

#endif // __prob_h__

/**
 * end of file
 * 
 */
```
实现文件：
```cpp
/**
 * @file   prob.cpp
 * @author Robert Lie
 * @date   Sat Jul 14 09:13:06 2007
 * 
 * @brief  
 * 
 * 
 */

#include "prob.h"

double f(const double * p)
{
  return p[0] + p[1]*p[1];
}

double u_b(const double * p)
{
  return 0.0;
}

double g(const double * p)
{
  return exp(p[0]) + sin(p[1]*p[1]);
}

double v_b(const double * p)
{
  return p[0]*p[1];
}


double a(const double * p)
{
  if (p[0] < 0.6)
    return 2;
  else
    return 1;
}

void Matrix::buildSparseMatrix()
{
  SparseMatrix<double>::reinit(getSparsityPattern());

  FEMSpace<double,DIM>::ElementIterator 
    the_ele = FEMSpace0()->beginElement(),
    end_ele = FEMSpace0()->endElement(),
    v_it = p_u_h->femSpace().beginElement();
  for (;the_ele != end_ele;the_ele ++, v_it ++) {
    getElementPattern(*the_ele, *the_ele);
    elementMatrix().reinit(elementDof0().size(), elementDof1().size());
    getElementMatrix(*the_ele, *the_ele);
    addElementMatrix();
  }
}

void Matrix::getElementMatrix(const Element<double,DIM> & ele0, 
			      const Element<double,DIM> & ele1, 
			      ActiveElementPairIterator<DIM>::State)
{
  double vol = ele0.templateElement().volume();
  const QuadratureInfo<DIM>& quad_info = 
    ele0.findQuadratureInfo(algebricAccuracy());
  int n_quadrature_point = quad_info.n_quadraturePoint();

  std::vector<double> jacobian = 
    ele0.local_to_global_jacobian(quad_info.quadraturePoint());

  std::vector<Point<DIM> > q_point = 
    ele0.local_to_global(quad_info.quadraturePoint());

  std::vector<std::vector<double> >
    basis_value = ele0.basis_function_value(q_point);

  std::vector<std::vector<std::vector<double> > > 
    basis_gradient = ele0.basis_function_gradient(q_point);

  const std::vector<int>& ele_dof = ele0.dof();
  u_int n_ele_dof = ele_dof.size();

  for (int l = 0;l < n_quadrature_point;++ l) {
    double Jxw = vol*jacobian[l]*quad_info.weight(l);
    double a_value = a(q_point[l]);
    
    double u_h_val = p_u_h->value(q_point[l], *v_it);

    for (int i = 0;i < n_ele_dof;++ i) {
      for (int j = 0;j < n_ele_dof;++ j) {
        elementMatrix(i, j) += Jxw*
          ((1/_dt)*basis_value[i][l]*basis_value[j][l]
           + a_value*innerProduct(basis_gradient[i][l], 
                                  basis_gradient[j][l])
           + 10000*u_h_val*u_h_val*basis_value[i][l]*basis_value[j][l]);
      }
    }
    
  }
}


void problem::initialize()
{
  h_tree.readEasyMesh("M");
  ir_mesh = new IrregularMesh<DIM>(h_tree);
  ir_mesh->globalRefine(3);
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);

  template_geometry.readData("triangle.tmp_geo");
  coord_transform.readData("triangle.crd_trs");
  template_dof.reinit(template_geometry);
  template_dof.readData("triangle.1.tmp_dof");
  basis_function.reinit(template_dof);
  basis_function.readData("triangle.1.bas_fun");

  template_dof2.reinit(template_geometry);
  template_dof2.readData("triangle.2.tmp_dof");
  basis_function2.reinit(template_dof2);
  basis_function2.readData("triangle.2.bas_fun");

  template_geometry1.readData("twin_triangle.tmp_geo");
  coord_transform1.readData("twin_triangle.crd_trs");
  template_dof1.reinit(template_geometry1);
  template_dof1.readData("twin_triangle.1.tmp_dof");
  basis_function1.reinit(template_dof1);
  basis_function1.readData("twin_triangle.1.bas_fun");

  template_dof3.reinit(template_geometry1);
  template_dof3.readData("twin_triangle.2.tmp_dof");
  basis_function3.reinit(template_dof3);
  basis_function3.readData("twin_triangle.2.bas_fun");

  template_element.resize(4);
  template_element[0].reinit(template_geometry,
                             template_dof,
                             coord_transform,
                             basis_function);
  template_element[1].reinit(template_geometry1,
                             template_dof1,
                             coord_transform1,
                             basis_function1);
  template_element[2].reinit(template_geometry,
                             template_dof2,
                             coord_transform,
                             basis_function2);
  template_element[3].reinit(template_geometry1,
                             template_dof3,
                             coord_transform1,
                             basis_function3);

  buildFEMSpace();

  t = 0, dt = 1.0e-02;
}

void problem::buildFEMSpace()
{
  RegularMesh<DIM>& mesh_u = ir_mesh_u->regularMesh();

  u_int n_ele = mesh_u.n_geometry(DIM);
  fem_space_u = new FEMSpace<double,DIM>(mesh_u, template_element);
  fem_space_u->element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    u_int n_vtx = mesh_u.geometry(DIM, i).n_vertex();
    if (n_vtx == 3) {
      fem_space_u->element(i).reinit(*fem_space_u, i, 0);
    }
    else {
      fem_space_u->element(i).reinit(*fem_space_u, i, 1);
    }
  }
  fem_space_u->buildElement();
  fem_space_u->buildDof();
  fem_space_u->buildDofBoundaryMark();

  RegularMesh<DIM>& mesh_v = ir_mesh_v->regularMesh();

  n_ele = mesh_v.n_geometry(DIM);
  fem_space_v = new FEMSpace<double,DIM>(mesh_v, template_element);
  fem_space_v->element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    u_int n_vtx = mesh_v.geometry(DIM, i).n_vertex();
    if (n_vtx == 3) {
      fem_space_v->element(i).reinit(*fem_space_v, i, 0);
    }
    else {
      fem_space_v->element(i).reinit(*fem_space_v, i, 1);
    }
  }
  fem_space_v->buildElement();
  fem_space_v->buildDof();
  fem_space_v->buildDofBoundaryMark();

  u_h = new FEMFunction<double,DIM>(*fem_space_u);
  v_h = new FEMFunction<double,DIM>(*fem_space_v);
}

void problem::adaptMesh()
{
  Indicator<DIM> ind_u(ir_mesh_u->regularMesh());
  getIndicator_u(ind_u);

  IrregularMesh<DIM> * old_ir_mesh_u = ir_mesh_u;
  ir_mesh_u = new IrregularMesh<DIM>(*old_ir_mesh_u);

  MeshAdaptor<DIM> mesh_adaptor(*old_ir_mesh_u, *ir_mesh_u);
  mesh_adaptor.setIndicator(ind_u);
  mesh_adaptor.tolerence() = 1.0e-04;
  mesh_adaptor.convergenceOrder() = 1;
  mesh_adaptor.adapt();

  ir_mesh_u->semiregularize();
  ir_mesh_u->regularize(false);

  FEMSpace<double,DIM> * old_fem_space_u = fem_space_u;
  FEMFunction<double,DIM> * old_u_h = u_h;

  ///////////////////////////////////////////

  Indicator<DIM> ind_v(ir_mesh_v->regularMesh());
  getIndicator_v(ind_v);

  IrregularMesh<DIM> * old_ir_mesh_v = ir_mesh_v;
  ir_mesh_v = new IrregularMesh<DIM>(*old_ir_mesh_v);

  mesh_adaptor.reinit(*old_ir_mesh_v, *ir_mesh_v);
  mesh_adaptor.setIndicator(ind_v);
  mesh_adaptor.tolerence() = 1.0e-04;
  mesh_adaptor.convergenceOrder() = 1;
  mesh_adaptor.adapt();

  ir_mesh_v->semiregularize();
  ir_mesh_v->regularize(false);

  FEMSpace<double,DIM> * old_fem_space_v = fem_space_v;
  FEMFunction<double,DIM> * old_v_h = v_h;

  buildFEMSpace();

  Operator::L2Interpolate(*old_u_h, *u_h);

  delete old_u_h;
  delete old_fem_space_u;
  delete old_ir_mesh_u;

  ////////////////////////////////////////

  Operator::L2Interpolate(*old_v_h, *v_h);

  delete old_v_h;
  delete old_fem_space_v;
  delete old_ir_mesh_v;
}

void problem::run()
{
  do {
    stepForward();

    adaptMesh();

    t += dt;
    u_h->writeOpenDXData("u_h.dx");
    v_h->writeOpenDXData("v_h.dx");

    std::cout << "t = " << t << std::endl;
  } while (1);
}

void problem::stepForward()
{
  FEMFunction<double,DIM> last_u_h(*u_h);
  FEMFunction<double,DIM> last_v_h(*v_h);

  u_int step = 0;
  do {

    {
      Matrix stiff_matrix(*fem_space_v, dt, *u_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(g, *fem_space_v, rhs, 3);

      IrregularMeshPair<DIM> mesh_pair(*ir_mesh_u, *ir_mesh_v);
      ActiveElementPairIterator<DIM> 
        the_pair = mesh_pair.beginActiveElementPair(),
        end_pair = mesh_pair.endActiveElementPair();
      for (;the_pair != end_pair;++ the_pair) {
        const HElement<DIM>& h_element0 = the_pair(0);
        const HElement<DIM>& h_element1 = the_pair(1);
        Element<double,DIM>& ele0 = fem_space_u->element(h_element0.index);
        Element<double,DIM>& ele1 = fem_space_v->element(h_element1.index);
        if (the_pair.State() == -1) { // GREAT_THAN
          double vol = ele1.templateElement().volume();
 
          const QuadratureInfo<DIM>& quad_info = 
            ele1.findQuadratureInfo(2);
          int n_quadrature_point = quad_info.n_quadraturePoint();

          std::vector<double> jacobian = 
            ele1.local_to_global_jacobian(quad_info.quadraturePoint());

          std::vector<Point<DIM> > q_point = 
            ele1.local_to_global(quad_info.quadraturePoint());
          
          std::vector<std::vector<double> >
            basis_value = ele1.basis_function_value(q_point);

          std::vector<double> u_h_val = last_u_h.value(q_point, ele0);
          std::vector<double> v_h_val = last_v_h.value(q_point, ele1);
          std::vector<std::vector<double> > 
            v_h_grad = last_v_h.gradient(q_point, ele1);

          const std::vector<int>& ele_dof = ele1.dof();
          u_int n_ele_dof = ele_dof.size();

          for (int l = 0;l < n_quadrature_point;++ l) {
            double Jxw = vol*jacobian[l]*quad_info.weight(l);
            for (u_int j = 0;j < n_ele_dof;++ j) {
              rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                      u_h_val[l]*v_h_grad[l][0] -
                                      v_h_val[l]*v_h_grad[l][1]
                                      )*basis_value[j][l];
            }
          }
        } else {
          double vol = ele0.templateElement().volume();
 
          const QuadratureInfo<DIM>& quad_info = 
            ele0.findQuadratureInfo(2);
          int n_quadrature_point = quad_info.n_quadraturePoint();

          std::vector<double> jacobian = 
            ele0.local_to_global_jacobian(quad_info.quadraturePoint());

          std::vector<Point<DIM> > q_point = 
            ele0.local_to_global(quad_info.quadraturePoint());
          
          std::vector<std::vector<double> >
            basis_value = ele1.basis_function_value(q_point);

          std::vector<double> u_h_val = last_u_h.value(q_point, ele0);
          std::vector<double> v_h_val = last_v_h.value(q_point, ele1);
          std::vector<std::vector<double> > 
            v_h_grad = last_v_h.gradient(q_point, ele1);

          const std::vector<int>& ele_dof = ele1.dof();
          u_int n_ele_dof = ele_dof.size();

          for (int l = 0;l < n_quadrature_point;++ l) {
            double Jxw = vol*jacobian[l]*quad_info.weight(l);
            for (u_int j = 0;j < n_ele_dof;++ j) {
              rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                      u_h_val[l]*v_h_grad[l][0] -
                                      v_h_val[l]*v_h_grad[l][1]
                                      )*basis_value[j][l];
            }
          }
        }
      }



      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space_v->beginElement(),
        end_ele = fem_space_v->endElement(),
        the_ele_u = fem_space_u->beginElement();
      for (;the_ele != end_ele;++ the_ele, ++ the_ele_u) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele_u);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele);
        std::vector<std::vector<double> > 
          v_h_grad = last_v_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                    u_h_val[l]*v_h_grad[l][0] -
                                    v_h_val[l]*v_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &v_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &v_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &v_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &v_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &v_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(*fem_space_v);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, *v_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(*v_h, rhs);
    }

    {
      FEMFunction<double,DIM> old_u_h(*u_h);

      Matrix stiff_matrix(*fem_space_u, dt, *v_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(f, *fem_space_u, rhs, 3);

      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space_u->beginElement(),
        end_ele = fem_space_u->endElement(),
        the_ele_v = fem_space_v->beginElement();
      for (;the_ele != end_ele;++ the_ele, ++ the_ele_v) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele_v);
        std::vector<std::vector<double> > 
          u_h_grad = last_u_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*u_h_val[l] - 
                                    u_h_val[l]*u_h_grad[l][0] -
                                    v_h_val[l]*u_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &u_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &u_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &u_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &u_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &u_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(*fem_space_u);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, *u_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(*u_h, rhs);

      old_u_h.add(-1.0, *u_h);
      double error = Functional::L2Norm(old_u_h, 3);

      std::cout << "step " << step ++ << ", error = " << error << std::endl;

      if (error < 1.0e-08) break;
    }

  } while (1);
}


void problem::getIndicator(Indicator<DIM>& ind)
{
  RegularMesh<DIM>& mesh = ir_mesh->regularMesh();
  u_int n_face = mesh.n_geometry(DIM - 1);
  std::vector<bool> flag(n_face, false);
  std::vector<double> jump(n_face);
  FEMSpace<double,DIM>::ElementIterator
    the_ele = fem_space_u->beginElement(),
    end_ele = fem_space_u->endElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    Point<DIM> p;
    std::vector<double> u_h_grad = u_h->gradient(p, *the_ele);
    GeometryBM& geo = the_ele->geometry();
    u_int n_bnd = geo.n_boundary();
    for (u_int j = 0;j < n_bnd;++ j) {
      GeometryBM& bnd = mesh.geometry(DIM-1, geo.boundary(j));
      Point<DIM>& p0 = mesh.point(bnd.vertex(0));
      Point<DIM>& p1 = mesh.point(bnd.vertex(1));
      double a = (u_h_grad[0]*(p1[1] - p0[1]) -
                  u_h_grad[1]*(p1[0] - p0[0]));
      if (flag[bnd.index()] == false) {
        jump[bnd.index()] = a;
        flag[bnd.index()] = true;
      }
      else {
        jump[bnd.index()] -= a;
        flag[bnd.index()] = false;
      }
    }
  }

  the_ele = fem_space_u->beginElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    GeometryBM& geo = the_ele->geometry();
    u_int n_bnd = geo.n_boundary();
    ind[i] = 0.0;
    for (u_int j = 0;j < n_bnd;++ j) {
      GeometryBM& bnd = mesh.geometry(DIM-1, geo.boundary(j));
      if (flag[bnd.index()]) continue;
      ind[i] += jump[bnd.index()];
    }
  }
}

int main(int argc, char * argv[])
{
  problem the_app;
  the_app.initialize();
  the_app.run();

  return 0;
}

/**
 * end of file
 * 
 */
```
# AFEPack教学系列之十四: 讨论班例子(IV)

下面这个例子算是我们这个系列的最终目标了 ：我们求解了
- 两个变量耦合的偏微分方程组；
- 方程组中包含对流、扩散、非线性；
- 我们对两个变量在两个不同的网格上逼近；
- 两个变量在自己的网格上分别使用了一次元空 间和二次元空间来逼近；
- 两个变量还分别在自己的网格上做h-自适应； 

这个代码中有些重复的部分我没有写出来，请您自己去补全吧。

头文件：
```cpp
/**
 * @file   prob.h
 * @author Robert Lie
 * @date   Sat Jul 14 09:06:42 2007
 * 
 * @brief  
 * 
 * 
 */

#ifndef __prob_h__
#define __prob_h__

#include <AFEPack/EasyMesh.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/AMGSolver.h>
#include <AFEPack/HGeometry.h>

#define DIM 2

class Matrix : public StiffMatrix<DIM, double>
{
private:
  double _dt;
  FEMSpace<double,DIM>::ElementIterator v_it;
  FEMFunction<double,DIM> * p_u_h;
public:
  Matrix(FEMSpace<double,DIM>& sp,
         double dt,
         FEMFunction<double,DIM>& u_h) : 
    StiffMatrix<DIM,double>(sp), _dt(dt), p_u_h(&u_h) {};
  virtual void buildSparseMatrix();
  virtual 
  void getElementMatrix(const Element<double,DIM> & ele0, 
                        const Element<double,DIM> & ele1, 
                        const ActiveElementPairIterator<DIM>::State 
                        state=ActiveElementPairIterator<DIM>::EQUAL);
};

class problem {
 private:
  TemplateGeometry<DIM>       template_geometry;
  CoordTransform<DIM,DIM>     coord_transform;
  TemplateDOF<DIM>            template_dof;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function;

  TemplateGeometry<DIM>       template_geometry1;
  CoordTransform<DIM,DIM>     coord_transform1;
  TemplateDOF<DIM>            template_dof1;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function1;

  TemplateDOF<DIM>            template_dof2;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function2;

  TemplateDOF<DIM>            template_dof3;
  BasisFunctionAdmin<double,DIM,DIM>  basis_function3;

  std::vector<TemplateElement<double,DIM> > template_element;

  HGeometryTree<DIM>          h_tree;
  IrregularMesh<DIM>        * ir_mesh_u;
  IrregularMesh<DIM>        * ir_mesh_v;

  FEMSpace<double,DIM>      * fem_space_u;
  FEMSpace<double,DIM>      * fem_space_v;
  FEMFunction<double,DIM>   * u_h;
  FEMFunction<double,DIM>   * v_h;
  double                      t;
  double                      dt;

 public:
  void initialize();
  void run();
  void stepForward();

  void buildFEMSpace();
  void getIndicator(FEMFunction<double,DIM>&,
                    Indicator<DIM>&);
  void adaptMesh();

};

#endif // __prob_h__

/**
 * end of file
 * 
 */
```
实现文件：
```cpp
/**
 * @file   prob.cpp
 * @author Robert Lie
 * @date   Sat Jul 14 09:13:06 2007
 * 
 * @brief  
 * 
 * 
 */

#include "prob.h"

double f(const double * p)
{
  return p[0] + p[1]*p[1];
}

double u_b(const double * p)
{
  return 0.0;
}

double g(const double * p)
{
  return exp(p[0]) + sin(p[1]*p[1]);
}

double v_b(const double * p)
{
  return p[0]*p[1];
}


double a(const double * p)
{
  if (p[0] < 0.6)
    return 2;
  else
    return 1;
}

void Matrix::buildSparseMatrix()
{
  SparseMatrix<double>::reinit(getSparsityPattern());

  FEMSpace<double,DIM>& fem_space0 = FEMSpace0();
  femSpace<double,DIM>& fem_space1 = p_u_h->FEMSpace();
  RegularMesh<DIM>& mesh0 = dynamic_cast<RegularMesh<DIM>&>(fem_space0.mesh());
  RegularMesh<DIM>& mesh1 = dynamic_cast<RegularMesh<DIM>&>(fem_space1.mesh());
  IrregularMesh<DIM>& ir_mesh0 = mesh0.irregularMesh();
  IrregularMesh<DIM>& ir_mesh1 = mesh1.irregularMesh();
  MeshPair<DIM> mesh_pair(ir_mesh0, ir_mesh1);
  ActiveElementPairIterator<DIM> 
    the_pair = mesh_pair.beginActiveElementPair(),
    end_pair = mesh_pair.endActiveElementPair();
  for (;the_pair != end_pair;++ the_pair) {
    const HElement<DIM>& h_element0 = the_pair(0);
    const HElement<DIM>& h_element1 = the_pair(1);
    Element<double,DIM>& ele0 = fem_space0.element(h_element0.index);
    Element<double,DIM>& ele1 = fem_space1.element(h_element1.index);
    getElementPattern(ele0, ele0);
    elementMatrix().reinit(elementDof0().size(), elementDof1().size());
    getElementMatrix(ele0, ele1, the_pair.state());
    addElementMatrix();
  }
}

void Matrix::getElementMatrix(const Element<double,DIM> & ele0, 
			      const Element<double,DIM> & ele1, 
			      ActiveElementPairIterator<DIM>::State state)
{
  if (state == ActiveElementPairIterator<DIM>::GREAT_THAN) {
    /// 此时单元 ele0 比较大，我们使用 ele1 上的积分公式
    double vol = ele1.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = 
      ele1.findQuadratureInfo(algebricAccuracy());
    int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<double> jacobian = 
      ele1.local_to_global_jacobian(quad_info.quadraturePoint());
    std::vector<Point<DIM> > q_point = 
      ele1.local_to_global(quad_info.quadraturePoint());

    std::vector<std::vector<double> >
      basis_value = ele0.basis_function_value(q_point);

    std::vector<std::vector<std::vector<double> > > 
      basis_gradient = ele0.basis_function_gradient(q_point);

    const std::vector<int>& ele_dof = ele0.dof();
    u_int n_ele_dof = ele_dof.size();

    for (int l = 0;l < n_quadrature_point;++ l) {
      double Jxw = vol*jacobian[l]*quad_info.weight(l);
      double a_value = a(q_point[l]);
    
      double u_h_val = p_u_h->value(q_point[l], ele1);

      for (int i = 0;i < n_ele_dof;++ i) {
        for (int j = 0;j < n_ele_dof;++ j) {
          elementMatrix(i, j) += Jxw*
            ((1/_dt)*basis_value[i][l]*basis_value[j][l]
             + a_value*innerProduct(basis_gradient[i][l], 
                                    basis_gradient[j][l])
             + 10000*u_h_val*u_h_val*basis_value[i][l]*basis_value[j][l]);
        }
      }
    }
  } else {
    /// 此时单元 ele1 比较大，我们使用 ele0 上的积分公式
    double vol = ele0.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = 
      ele0.findQuadratureInfo(algebricAccuracy());
    int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<double> jacobian = 
      ele0.local_to_global_jacobian(quad_info.quadraturePoint());
    std::vector<Point<DIM> > q_point = 
      ele0.local_to_global(quad_info.quadraturePoint());

    std::vector<std::vector<double> >
      basis_value = ele0.basis_function_value(q_point);

    std::vector<std::vector<std::vector<double> > > 
      basis_gradient = ele0.basis_function_gradient(q_point);

    const std::vector<int>& ele_dof = ele0.dof();
    u_int n_ele_dof = ele_dof.size();

    for (int l = 0;l < n_quadrature_point;++ l) {
      double Jxw = vol*jacobian[l]*quad_info.weight(l);
      double a_value = a(q_point[l]);
    
      double u_h_val = p_u_h->value(q_point[l], ele1);

      for (int i = 0;i < n_ele_dof;++ i) {
        for (int j = 0;j < n_ele_dof;++ j) {
          elementMatrix(i, j) += Jxw*
            ((1/_dt)*basis_value[i][l]*basis_value[j][l]
             + a_value*innerProduct(basis_gradient[i][l], 
                                    basis_gradient[j][l])
             + 10000*u_h_val*u_h_val*basis_value[i][l]*basis_value[j][l]);
        }
      }
    }
  }
}


void problem::initialize()
{
  h_tree.readEasyMesh("M");
  ir_mesh = new IrregularMesh<DIM>(h_tree);
  ir_mesh->globalRefine(3);
  ir_mesh->semiregularize();
  ir_mesh->regularize(false);

  template_geometry.readData("triangle.tmp_geo");
  coord_transform.readData("triangle.crd_trs");
  template_dof.reinit(template_geometry);
  template_dof.readData("triangle.1.tmp_dof");
  basis_function.reinit(template_dof);
  basis_function.readData("triangle.1.bas_fun");

  template_dof2.reinit(template_geometry);
  template_dof2.readData("triangle.2.tmp_dof");
  basis_function2.reinit(template_dof2);
  basis_function2.readData("triangle.2.bas_fun");

  template_geometry1.readData("twin_triangle.tmp_geo");
  coord_transform1.readData("twin_triangle.crd_trs");
  template_dof1.reinit(template_geometry1);
  template_dof1.readData("twin_triangle.1.tmp_dof");
  basis_function1.reinit(template_dof1);
  basis_function1.readData("twin_triangle.1.bas_fun");

  template_dof3.reinit(template_geometry1);
  template_dof3.readData("twin_triangle.2.tmp_dof");
  basis_function3.reinit(template_dof3);
  basis_function3.readData("twin_triangle.2.bas_fun");

  template_element.resize(4);
  template_element[0].reinit(template_geometry,
                             template_dof,
                             coord_transform,
                             basis_function);
  template_element[1].reinit(template_geometry1,
                             template_dof1,
                             coord_transform1,
                             basis_function1);
  template_element[2].reinit(template_geometry,
                             template_dof2,
                             coord_transform,
                             basis_function2);
  template_element[3].reinit(template_geometry1,
                             template_dof3,
                             coord_transform1,
                             basis_function3);

  buildFEMSpace();

  t = 0, dt = 1.0e-02;
}

void problem::buildFEMSpace()
{
  RegularMesh<DIM>& mesh_u = ir_mesh_u->regularMesh();

  u_int n_ele = mesh_u.n_geometry(DIM);
  fem_space_u = new FEMSpace<double,DIM>(mesh_u, template_element);
  fem_space_u->element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    u_int n_vtx = mesh_u.geometry(DIM, i).n_vertex();
    if (n_vtx == 3) {
      fem_space_u->element(i).reinit(*fem_space_u, i, 0);
    }
    else {
      fem_space_u->element(i).reinit(*fem_space_u, i, 1);
    }
  }
  fem_space_u->buildElement();
  fem_space_u->buildDof();
  fem_space_u->buildDofBoundaryMark();

  RegularMesh<DIM>& mesh_v = ir_mesh_v->regularMesh();

  n_ele = mesh_v.n_geometry(DIM);
  fem_space_v = new FEMSpace<double,DIM>(mesh_v, template_element);
  fem_space_v->element().resize(n_ele);
  for (u_int i = 0;i < n_ele;++ i) {
    u_int n_vtx = mesh_v.geometry(DIM, i).n_vertex();
    if (n_vtx == 3) {
      fem_space_v->element(i).reinit(*fem_space_v, i, 2);
    }
    else {
      fem_space_v->element(i).reinit(*fem_space_v, i, 3);
    }
  }
  fem_space_v->buildElement();
  fem_space_v->buildDof();
  fem_space_v->buildDofBoundaryMark();

  u_h = new FEMFunction<double,DIM>(*fem_space_u);
  v_h = new FEMFunction<double,DIM>(*fem_space_v);
}

void problem::adaptMesh()
{
  Indicator<DIM> ind_u(ir_mesh_u->regularMesh());
  getIndicator(*u_h, ind_u);

  IrregularMesh<DIM> * old_ir_mesh_u = ir_mesh_u;
  ir_mesh_u = new IrregularMesh<DIM>(*old_ir_mesh_u);

  MeshAdaptor<DIM> mesh_adaptor(*old_ir_mesh_u, *ir_mesh_u);
  mesh_adaptor.setIndicator(ind_u);
  mesh_adaptor.tolerence() = 1.0e-04;
  mesh_adaptor.convergenceOrder() = 1;
  mesh_adaptor.adapt();

  ir_mesh_u->semiregularize();
  ir_mesh_u->regularize(false);

  FEMSpace<double,DIM> * old_fem_space_u = fem_space_u;
  FEMFunction<double,DIM> * old_u_h = u_h;

  ///////////////////////////////////////////

  Indicator<DIM> ind_v(ir_mesh_v->regularMesh());
  getIndicator(*v_h, ind_v);

  IrregularMesh<DIM> * old_ir_mesh_v = ir_mesh_v;
  ir_mesh_v = new IrregularMesh<DIM>(*old_ir_mesh_v);

  mesh_adaptor.reinit(*old_ir_mesh_v, *ir_mesh_v);
  mesh_adaptor.setIndicator(ind_v);
  mesh_adaptor.tolerence() = 1.0e-04;
  mesh_adaptor.convergenceOrder() = 1;
  mesh_adaptor.adapt();

  ir_mesh_v->semiregularize();
  ir_mesh_v->regularize(false);

  FEMSpace<double,DIM> * old_fem_space_v = fem_space_v;
  FEMFunction<double,DIM> * old_v_h = v_h;

  buildFEMSpace();

  Operator::L2Interpolate(*old_u_h, *u_h);

  delete old_u_h;
  delete old_fem_space_u;
  delete old_ir_mesh_u;

  ////////////////////////////////////////

  Operator::L2Interpolate(*old_v_h, *v_h);

  delete old_v_h;
  delete old_fem_space_v;
  delete old_ir_mesh_v;
}

void problem::run()
{
  do {
    stepForward();

    adaptMesh();

    t += dt;
    u_h->writeOpenDXData("u_h.dx");
    v_h->writeOpenDXData("v_h.dx");

    std::cout << "t = " << t << std::endl;
  } while (1);
}

void problem::stepForward()
{
  FEMFunction<double,DIM> last_u_h(*u_h);
  FEMFunction<double,DIM> last_v_h(*v_h);

  u_int step = 0;
  do {

    {
      Matrix stiff_matrix(*fem_space_v, dt, *u_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(g, *fem_space_v, rhs, 3);

      IrregularMeshPair<DIM> mesh_pair(*ir_mesh_u, *ir_mesh_v);
      ActiveElementPairIterator<DIM> 
        the_pair = mesh_pair.beginActiveElementPair(),
        end_pair = mesh_pair.endActiveElementPair();
      for (;the_pair != end_pair;++ the_pair) {
        const HElement<DIM>& h_element0 = the_pair(0);
        const HElement<DIM>& h_element1 = the_pair(1);
        Element<double,DIM>& ele0 = fem_space_u->element(h_element0.index);
        Element<double,DIM>& ele1 = fem_space_v->element(h_element1.index);
        if (the_pair.State() == ActiveElementPairIterator<DIM>::GREAT_THAN) {
          double vol = ele1.templateElement().volume();
 
          const QuadratureInfo<DIM>& quad_info = 
            ele1.findQuadratureInfo(2);
          int n_quadrature_point = quad_info.n_quadraturePoint();

          std::vector<double> jacobian = 
            ele1.local_to_global_jacobian(quad_info.quadraturePoint());

          std::vector<Point<DIM> > q_point = 
            ele1.local_to_global(quad_info.quadraturePoint());
          
          std::vector<std::vector<double> >
            basis_value = ele1.basis_function_value(q_point);

          std::vector<double> u_h_val = last_u_h.value(q_point, ele0);
          std::vector<double> v_h_val = last_v_h.value(q_point, ele1);
          std::vector<std::vector<double> > 
            v_h_grad = last_v_h.gradient(q_point, ele1);

          const std::vector<int>& ele_dof = ele1.dof();
          u_int n_ele_dof = ele_dof.size();

          for (int l = 0;l < n_quadrature_point;++ l) {
            double Jxw = vol*jacobian[l]*quad_info.weight(l);
            for (u_int j = 0;j < n_ele_dof;++ j) {
              rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                      u_h_val[l]*v_h_grad[l][0] -
                                      v_h_val[l]*v_h_grad[l][1]
                                      )*basis_value[j][l];
            }
          }
        } else {
          double vol = ele0.templateElement().volume();
 
          const QuadratureInfo<DIM>& quad_info = 
            ele0.findQuadratureInfo(2);
          int n_quadrature_point = quad_info.n_quadraturePoint();

          std::vector<double> jacobian = 
            ele0.local_to_global_jacobian(quad_info.quadraturePoint());

          std::vector<Point<DIM> > q_point = 
            ele0.local_to_global(quad_info.quadraturePoint());
          
          std::vector<std::vector<double> >
            basis_value = ele1.basis_function_value(q_point);

          std::vector<double> u_h_val = last_u_h.value(q_point, ele0);
          std::vector<double> v_h_val = last_v_h.value(q_point, ele1);
          std::vector<std::vector<double> > 
            v_h_grad = last_v_h.gradient(q_point, ele1);

          const std::vector<int>& ele_dof = ele1.dof();
          u_int n_ele_dof = ele_dof.size();

          for (int l = 0;l < n_quadrature_point;++ l) {
            double Jxw = vol*jacobian[l]*quad_info.weight(l);
            for (u_int j = 0;j < n_ele_dof;++ j) {
              rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                      u_h_val[l]*v_h_grad[l][0] -
                                      v_h_val[l]*v_h_grad[l][1]
                                      )*basis_value[j][l];
            }
          }
        }
      }



      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space_v->beginElement(),
        end_ele = fem_space_v->endElement(),
        the_ele_u = fem_space_u->beginElement();
      for (;the_ele != end_ele;++ the_ele, ++ the_ele_u) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele_u);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele);
        std::vector<std::vector<double> > 
          v_h_grad = last_v_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*v_h_val[l] - 
                                    u_h_val[l]*v_h_grad[l][0] -
                                    v_h_val[l]*v_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &v_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &v_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &v_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &v_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &v_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(*fem_space_v);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, *v_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(*v_h, rhs);
    }

    {
      FEMFunction<double,DIM> old_u_h(*u_h);

      Matrix stiff_matrix(*fem_space_u, dt, *v_h);
      stiff_matrix.algebricAccuracy() = 3;
      stiff_matrix.build();

      Vector<double> rhs;
      Operator::L2Discretize(f, *fem_space_u, rhs, 3);

      FEMSpace<double,DIM>::ElementIterator 
        the_ele = fem_space_u->beginElement(),
        end_ele = fem_space_u->endElement(),
        the_ele_v = fem_space_v->beginElement();
      for (;the_ele != end_ele;++ the_ele, ++ the_ele_v) {
        double vol = the_ele->templateElement().volume();
 
        const QuadratureInfo<DIM>& quad_info = 
          the_ele->findQuadratureInfo(2);
        int n_quadrature_point = quad_info.n_quadraturePoint();

        std::vector<double> jacobian = 
          the_ele->local_to_global_jacobian(quad_info.quadraturePoint());

        std::vector<Point<DIM> > q_point = 
          the_ele->local_to_global(quad_info.quadraturePoint());

        std::vector<std::vector<double> >
          basis_value = the_ele->basis_function_value(q_point);

        std::vector<double> u_h_val = last_u_h.value(q_point, *the_ele);
        std::vector<double> v_h_val = last_v_h.value(q_point, *the_ele_v);
        std::vector<std::vector<double> > 
          u_h_grad = last_u_h.gradient(q_point, *the_ele);

        const std::vector<int>& ele_dof = the_ele->dof();
        u_int n_ele_dof = ele_dof.size();

        for (int l = 0;l < n_quadrature_point;++ l) {
          double Jxw = vol*jacobian[l]*quad_info.weight(l);
          for (u_int j = 0;j < n_ele_dof;++ j) {
            rhs(ele_dof[j]) += Jxw*((1/dt)*u_h_val[l] - 
                                    u_h_val[l]*u_h_grad[l][0] -
                                    v_h_val[l]*u_h_grad[l][1]
                                    )*basis_value[j][l];
          }
        }
      }

      BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 
                                             1, &u_b);
      BoundaryFunction<double,DIM> boundary2(BoundaryConditionInfo::DIRICHLET, 
                                             2, &u_b);
      BoundaryFunction<double,DIM> boundary3(BoundaryConditionInfo::DIRICHLET, 
                                             3, &u_b);
      BoundaryFunction<double,DIM> boundary4(BoundaryConditionInfo::DIRICHLET, 
                                             4, &u_b);
      BoundaryFunction<double,DIM> boundary5(BoundaryConditionInfo::DIRICHLET, 
                                             5, &u_b);
      BoundaryConditionAdmin<double,DIM> boundary_admin(*fem_space_u);
      boundary_admin.add(boundary1);
      boundary_admin.add(boundary2);
      boundary_admin.add(boundary3);
      boundary_admin.add(boundary4);
      boundary_admin.add(boundary5);
      boundary_admin.apply(stiff_matrix, *u_h, rhs);

      AMGSolver solver;
      solver.lazyReinit(stiff_matrix);
      solver.solve(*u_h, rhs);

      old_u_h.add(-1.0, *u_h);
      double error = Functional::L2Norm(old_u_h, 3);

      std::cout << "step " << step ++ << ", error = " << error << std::endl;

      if (error < 1.0e-08) break;
    }

  } while (1);
}


void problem::getIndicator(FEMFunction<double,DIM>& w_h,
                           Indicator<DIM>& ind)
{
  FEMSpace<double,DIM>& fem_space = w_h.FEMSpace();
  Mesh<DIM>& mesh = fem_space.mesh();
  u_int n_face = mesh.n_geometry(DIM - 1);
  std::vector<bool> flag(n_face, false);
  std::vector<double> jump(n_face);
  FEMSpace<double,DIM>::ElementIterator
    the_ele = fem_space.beginElement(),
    end_ele = fem_space.endElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    Point<DIM> p;
    std::vector<double> w_h_grad = w_h.gradient(p, *the_ele);
    GeometryBM& geo = the_ele->geometry();
    u_int n_bnd = geo.n_boundary();
    for (u_int j = 0;j < n_bnd;++ j) {
      GeometryBM& bnd = mesh.geometry(DIM-1, geo.boundary(j));
      Point<DIM>& p0 = mesh.point(bnd.vertex(0));
      Point<DIM>& p1 = mesh.point(bnd.vertex(1));
      double a = (w_h_grad[0]*(p1[1] - p0[1]) -
                  w_h_grad[1]*(p1[0] - p0[0]));
      if (flag[bnd.index()] == false) {
        jump[bnd.index()] = a;
        flag[bnd.index()] = true;
      }
      else {
        jump[bnd.index()] -= a;
        flag[bnd.index()] = false;
      }
    }
  }

  the_ele = fem_space_u->beginElement();
  for (u_int i = 0;the_ele != end_ele;++ the_ele, ++ i) {
    GeometryBM& geo = the_ele->geometry();
    u_int n_bnd = geo.n_boundary();
    ind[i] = 0.0;
    for (u_int j = 0;j < n_bnd;++ j) {
      GeometryBM& bnd = mesh.geometry(DIM-1, geo.boundary(j));
      if (flag[bnd.index()]) continue;
      ind[i] += jump[bnd.index()];
    }
  }
}

int main(int argc, char * argv[])
{
  problem the_app;
  the_app.initialize();
  the_app.run();

  return 0;
}


/**
 * end of file
 * 
 */
```
