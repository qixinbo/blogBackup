---
title: MOOSE运行过程再分析
tags: [MOOSE]
categories: computational material science
date: 2017-7-28
---
本文是对[前文](http://qixinbo.info/2017/02/14/moose-run/)的一个再次探索。
MOOSE项目太宏大，一点点来，这次先以解析怎样输入为主，还未涉及运行和求解，以后再不定期更新。

# 头文件
```cpp
#include "ExampleApp.h"
```
声明具体问题类。
```cpp
#include "MooseApp.h"
```
用于创建和存储各种对象。
```cpp
#include "AppFactory.h"
```
声明AppFactory类，用于创建各种对象。里面有一个很隐藏的宏定义：
```cpp
#define registerApp(name) AppFactory::instance().reg<name>(#name)
```
在ExampleApp中注册对象实际调用的就是它。
```cpp
#include "Moose.h"
```
该头文件的主要作用是声明Moose这个命名空间，其包含了libMesh的PerfLog头文件。
```cpp
#include "MooseInit.h"
```
该头文件包含了libMesh的libmesh头文件，从LibMeshInit中Public派生出MooseInit类。

# 预备工作  
```cpp
PerfLog Moose::perf_log("Example");
```
PerfLog是libMesh的一个类，用于生成记录性能的日志，给其命名为Example。之所以使用前面要加上Moose，是因为在Moose命名空间中用extern已声明有这么一个变量。具体使用时通过在记录前和记录后加上成对的push和pop来记录。
```cpp
MooseInit init(argc, argv);
```
执行初始化过程。这里先执行libMesh的初始化过程，然后再执行MOOSE的初始化，比如在不同进程中设置统一随机数。
```cpp
ExampleApp::registerApps();
```
ExampleApp继承自MooseApp，它有三个静态成员函数，分别用来注册App、注册对象、关联语法。上面的registerApps实际调用的就是AppFactory类中的registerApp宏定义，且实际传递了ExampleApp作为实参，再来调用AppFactory中的成员函数reg，即：
```cpp
#define registerApp(name) AppFactory::instance().reg<name>(#name)
```
这句话很巧妙，name有两个作用：一是用尖括号括住，这样可以对reg这个函数模板中的类型T用name进行实例化；二是前面加上井号，这是应用了井号在宏定义中的一个特殊作用，可以将name这个类替换成字符串，即后面的这个name是个字符串，正好作为reg的实参传入。看下面的代码就很清楚了：
```cpp
template <typename T>
void reg(const std::string & name)
{
  if (_name_to_build_pointer.find(name) == _name_to_build_pointer.end())
  {
    _name_to_build_pointer[name] = &buildApp<T>;
    _name_to_params_pointer[name] = &validParams<T>;
  }
}
```
注意最后一句话，正是这句话将ExampleApp这样的validParams读入，从而用于后面的create。

另外，注意，这里是对类的操作，用到了static，即static静态成员属于整个类，而不是某个实例化对象。这里还没有创建对象。
```cpp
MooseApp * app = AppFactory::createApp("ExampleApp", argc, argv);
```
这一步才创建对象，同样是使用了AppFactory的static成员creatApp。这里创建的是一个MooseApp指针，注意看MooseApp的构造函数可以发现，其实它是需要一个InputParameters形参的，这个是createApp中隐式传递的，传递的正是reg时读入的validParams`<ExampleApp>`类型的参数。
创建Example对象时，执行Example构造函数时，会注册对象，先是注册Moose公有的，再注册ExampleApp特有的，如下：
```cpp
ExampleApp::ExampleApp(InputParameters parameters) : MooseApp(parameters)
{
  srand(processor_id());

  Moose::registerObjects(_factory);
  ExampleApp::registerObjects(_factory);

  Moose::associateSyntax(_syntax, _action_factory);
  ExampleApp::associateSyntax(_syntax, _action_factory);
}
```
在registerObjects函数中用到了大量的宏定义，所以在Doxygen文档中无法直接搜索到，实际的定义在Factory类的头文件中：
```cpp
#define registerSampler(name) registerObject(name)
#define registerMesh(name) registerObject(name)
#define registerMeshModifier(name) registerObject(name)
#define registerConstraint(name) registerObject(name)
```
在上面的构造函数中还会关联语法associaeSyntax，同样先关联Moose公有的，再关联ExampleApp特有的，这一步牵扯到注册Syntax和各种Action，比如Moose命名空间的associateSyntax：
```cpp
// Transfers
registerSyntax("AddTransferAction", "Transfers/*");

addActionTypes(syntax);
registerActions(syntax, action_factory);
```
其中的registerActions也是一个宏定义。
而ExampleApp的associateSyntax则可以自定义，此例太简化，没有提供这部分的样子，但其他复杂的，比如ContactApp的associateSyntax就是下面这个样子：
```cpp
registerSyntax("NodalAreaAction", "Contact/*");
registerSyntax("NodalAreaVarAction", "Contact/*");

registerAction(ContactAction, "add_aux_kernel");
registerAction(ContactAction, "add_aux_variable");
registerAction(ContactAction, "add_dirac_kernel");
```
# 实际运行
```cpp
app->run();
```
run函数主要由三个函数组成：setupOptions、runInputFile和executeExecutioner。
MooseApp类的对象有若干个数据成员，且各有用处，比如对应于以上三个函数，InputParameters成员来存储命令行参数，ActionWarehouse成员来读取并运行输入文件，Executioner成员来执行求解器。

## setupOptions
该函数是virtual函数，但不是纯虚函数，所以子类ExampleApp可以使用基类MooseApp的实现。
```cpp
std::string hdr(header() + "\n");
```
这句纯粹是为了打印的显示方便，header是MooseApp的一个成员函数，是为了获得一个空字符串。
```cpp
if (multiAppLevel() > 0)
  MooseUtils::indentMessage(_name, hdr);
```
这句是为了获得multiapp的level，master在0层上。这个multiappLevel是通过validParams这个函数模板所取得的。在InputParameters的头文件中有一个总的validParams函数模板的声明，它的返回类型是InputParameters，类型参数是T。后续的每个Moose对象都要在头文件中写明以其名字为类型的validParams函数的定义，这里利用的是函数模板的“显式特化”这个功能，如MooseApp的template `<>` InputParameters validParams`<MooseApp>`(){...}，在这个定义中，该对象会指明要添加什么样的参数。在MooseApp的构造函数的成员初始化器列表中，会对multiappLevel进行初始化。

setupOptions主要是根据命令行的输入，来配置整个系统。比如：
```cpp
if (getParam<bool>("no_timing"))
  _pars.set<bool>("timing") = false;
```
上述语句就是探测命令行中是否输入了--no-timing，一旦输入了，则不输出性能日志。
```cpp
if (isParamValid("trap_fpe"))
  Moose::_trap_fpe = true;
```
上述语句就是探测命令行中是否输入了--trap-fpe，一旦输入了，则捕获浮点数异常。
探测这些命令时，都是使用的libMesh中的函数。
## runInputFile
这里面最重要的一个函数是：
```cpp
_action_warehouse.executeAllActions();
```
该语句将会遍历仓库中的所有actions，然后执行它们。
这一步中一个重要的action是读入网格文件。
## executeExecutioner
```cpp
_executioner->init();
_executioner->execute();
```
注意，_executioner是一个Executioner类型的智能共享指针。同时，init和execute都是纯虚函数，需要在子类中重载，因此，不同的子类有不同的实现。因为ex01是一个稳态问题，所以这里调用的是Steady的init和execute函数。
### init函数
这一步输出了计算参数信息：
```cpp
Framework Information:
MOOSE version:           git commit dacf1cf on 2017-04-01
PETSc Version:           3.7.5
Current Time:            Wed Oct 11 10:53:07 2017
Executable Timestamp:    Wed Apr 26 15:26:37 2017

Parallelism:
  Num Processors:          1
  Num Threads:             1

Mesh: 
  Parallel Type:           replicated
  Mesh Dimension:          3
  Spatial Dimension:       3
  Nodes:                   
    Total:                 3774
    Local:                 3774
  Elems:                   
    Total:                 2476
    Local:                 2476
  Num Subdomains:          1
  Num Partitions:          1

Nonlinear System:
  Num DOFs:                3774
  Num Local DOFs:          3774
  Variables:               "diffused" 
  Finite Element Types:    "LAGRANGE" 
  Approximation Orders:    "FIRST" 

Execution Information:
  Executioner:             Steady
  Solver Mode:             Preconditioned JFNK
```
### execute函数
这一步有几步重要步骤：
```cpp
preExecute(); // 在真正计算之前要做一些提前计算，对于Steady问题没有要做的
_problem.advanceState(); // 获得上一时间步的状态，从而为进入下一时间步做好准备
preSolve(); // 在Solve之前要做的事，对于Steady问题没有要做的
_nl->solve(); // 开始求解!
```
