---
title: 多物理场面向对象模拟环境MOOSE运行过程分析
tags: [MOOSE]
categories: simulation
date: 2017-2-14
---

本文尝试详细地走一遍MOOSE的运行路径，对其加深理解。

# 编译
MOOSE是一个大型工程，从它的编译过程可以看出各个部分之间怎样联系起来。这里以MOOSE的第一个例子为研究对象，分析整个过程。其路径在：
```cpp
moose/examples/ex01_inputfile
```
整个编译过程都写在了Makefile文件中。
## 定义环境变量
首先先定义一下环境变量：
```cpp
EXAMPLE_DIR        ?= $(shell dirname `pwd`)
MOOSE_DIR          ?= $(shell dirname $(EXAMPLE_DIR))
FRAMEWORK_DIR      ?= $(MOOSE_DIR)/framework
```
这些变量都是使用?=操作符来赋值，所以其实可以事先在终端中人为定义，如果之前没有指定，则这些语句就生效，基本思路就是通过调用shell的dirname命令来提取后面给定路径中的目录部分。
本例中，这三个变量经过赋值后就变为：
```cpp
EXAMPLE_DIR = /home/qixinbo/moose/examples
MOOSE_DIR = /home/qixinbo/moose
FRAMEWORK_DIR = /home/qixinbo/moose/framework
```
对于独立的fork出来的程序，一般路径是默认与moose并列，那么那个makefile就能正确地找到moose的路径，如果是放在别的地方，就可以人为定义该环境变量。
## 包含MOOSE主程序的makefile
MOOSE主程序包含基本框架framework和各种衍生模块modules。
基本框架framework肯定是要用到的，这里使用include调用framework下的makefile文件，将其放在此处。
```cpp
# framework
include $(FRAMEWORK_DIR)/build.mk
include $(FRAMEWORK_DIR)/moose.mk
```
将framework的编译另写，是模块化编译的思想，使得Makefile文件更加容易维护和扩展。这部分Makefile执行完后会在framework路径下生成moose的静态和动态链接库，比如如果以opt模式编译，就生成：
```cpp
/home/qixinbo/moose/framework/libmoose-opt.la
/home/qixinbo/moose/framework/libmoose-opt.so
/home/qixinbo/moose/framework/libmoose-opt.so.0
/home/qixinbo/moose/framework/libmoose-opt.so.0.0.0
```
另外，本例是个简单例子，所以没有调用模块。如果是需要基于那些衍生模块进行开发，就需要将其包含进来：
```cpp
# modules
ALL_MODULES := yes
include $(MOOSE_DIR)/modules/modules.mk
```
同时在AnimalApp的源文件中注册所有模块：
```cpp
#include "ModulesApp.h"

StarfishApp::StarfishApp(InputParameters parameters) :
    MooseApp(parameters)
{
  Moose::registerObjects(_factory);
  ModulesApp::registerObjects(_factory);
  StarfishApp::registerObjects(_factory);

  Moose::associateSyntax(_syntax, _action_factory);
  ModulesApp::associateSyntax(_syntax, _action_factory);
  StarfishApp::associateSyntax(_syntax, _action_factory);
}
```
目前从Stork中fork出的程序都是上面的设置，即默认将所有的模块包含。
如果只想包含特定模块，可以这样做，在Makefile中设置：
```cpp
PHASE_FIELD       := yes
SOLID_MECHANICS   := yes
TENSOR_MECHANICS  := yes
HEAT_CONDUCTION   := yes
MISC              := yes
include           $(MOOSE_DIR)/modules/modules.mk
```
然后在本程序的源文件中添加：
```cpp
//#include "ModulesApp.h" 这里要注释掉这个

//Specific Modules
#include "TensorMechanicsApp.h"
#include "PhaseFieldApp.h"
#include "MiscApp.h"

AnimalApp::AnimalApp(const InputParameters & parameters) :
    MooseApp(parameters)
{
  srand(processor_id());

  Moose::registerObjects(_factory);
  Moose::associateSyntax(_syntax, _action_factory);

  // ModulesApp::registerObjects(_factory);
  // ModulesApp::associateSyntax(_syntax, _action_factory);

  TensorMechanicsApp::registerObjects(_factory);
  TensorMechanicsApp::associateSyntax(_syntax, _action_factory);

  PhaseFieldApp::registerObjects(_factory);
  PhaseFieldApp::associateSyntax(_syntax, _action_factory);

  MiscApp::registerObjects(_factory);
  MiscApp::associateSyntax(_syntax, _action_factory);

  AnimalApp::registerObjects(_factory);
  AnimalApp::associateSyntax(_syntax, _action_factory);
}
```

MOOSE主程序部分如果编译过一次以后，假如没有更改，就不会再次编译了，能够有效节省时间。
## 编译本程序
```cpp
# dep apps
APPLICATION_DIR    := $(shell pwd)
APPLICATION_NAME   := ex01
BUILD_EXEC         := yes
DEP_APPS           := $(shell $(FRAMEWORK_DIR)/scripts/find_dep_apps.py $(APPLICATION_NAME))
include            $(FRAMEWORK_DIR)/app.mk
```
这里使用framework下的app.mk这个makefile来编译本程序，涉及该程序的路径、名字等参数会传入该文件中。值得一提的是，如何对某个变量的值不清楚，可以通过自己写一个target来显示出来，比如：
```cpp
showName:
   @echo $(APPLICATION_DIR)
```
然后make showName来执行。
本程序所依赖的源文件和头文件都会在以下路径：
```cpp
$(APPLICATION_DIR)/include
$(APPLICATION_DIR)/src
```
中寻找，目前猜测是遍历两个目录下的所有目录，然后寻找这些文件。
本例因为很简单，所以在这两个目录下仅包含base目录，对于稍大型的项目，牵扯到自己开发的模块时，还可添加其他目录，如：
```cpp
auxkernels base bcs ics kernel materials timesteppers
```
等。这些目录中的头文件用来声明类，源文件用来实现类。
# 入口main函数
所有程序的入口都是main函数，所以从这里入手能比较清晰地把握整个程序的脉络。
## 头文件
```cpp
#include "ExampleApp.h"
//Moose Includes
#include "MooseInit.h"
#include "Moose.h"
#include "MooseApp.h"
#include "AppFactory.h"
```
首先包含该具体程序的头文件，然后再包含MOOSE的基本头文件。这里仅包含ExampleApp这一个头文件即可，之前include目录下的其他头文件是通过src目录下的ExampleApp的源文件来包含，比如Hyrax应用的HyraxApp的源文件：
```cpp
//Kernels
#include "ACBulkCoupled.h"

//Auxiliary Kernels
#include "AuxSupersaturation.h"

//Dirac Kernels

//Boundary Conditions
#include "StressBC.h"

//Materials
#include "PFMobilityLandau.h"

//Initial Conditions
#include "PolySpecifiedSmoothCircleIC.h"

//Dampers

//Executioners
#include "MeshSolutionModify.h"

//Post Processors
#include "NucleationPostprocessor.h"

//TimeSteppers
#include "InitialSolutionAdaptiveDT.h"

//Actions

//UserObjects
#include "NucleationLocationUserObject.h"

//Markers
#include "NucleationMarker.h"
```
所以，整个关系链是：main仅需要ExampleApp这个头文件，且链接ExampleApp这个源文件产生的目标文件，EampleApp又需要自定义的其他头文件，比如ExampleDiffusion等，且链接这些头文件对应的源文件产生的目标文件。所以，在自己开发程序时，不需要修改main文件，仅需要修改私有程序的base源文件以包含正确的头文件即可。

## 创建性能日志
```cpp
PerfLog Moose::perf_log("Example");
```
PerfLog是libMesh的一个类，用来创建性能日志。

# 初始化
```cpp
// Initialize MPI, solvers and MOOSE
MooseInit init(argc, argv);
```
调用MooseInit类来进行初始化，实际上也调用了其父类LibMeshInit，从而对libMesh进行初始化。
# 注册App
```cpp
// Register this application's MooseApp and any it depends on
ExampleApp::registerApps();
```
每个基于MOOSE的应用都是继承自MooseApp这个类，该类提供创建各种对象的工厂factories以及存放它们的仓库warehouses。创建了私有程序后，必须实现几个重要的静态函数：
```cpp
registerApps()
registerObjects()
associateSyntax() (optional)
```
## 注册App
第一个是注册App，这个用简单的通用的一句话即可：
```cpp
void
ExampleApp::registerApps()
{
  registerApp(ExampleApp);
}
```
其中registerApp不是一个函数或类，所以在doxygen中查不到，它是AppFactory中的一个宏定义：
```cpp
#define registerApp(name) AppFactory::instance().reg<name>(#name)
```
## 注册Objects
第二个是注册Objects。本例没有用到自定义的Objects，所以该函数是空白的。但如果自定义了某个object，比如在example2中新增了一个Kernel，那么就需要注册该object：
```cpp
void
ExampleApp::registerObjects(Factory & factory)
{
  // Register any custom objects you have built on the MOOSE Framework
  registerKernel(ExampleConvection);  // <- registration
}
```
除了注册Kernel这种object，还有其他类型的object，比如Hyrax应用中用到的：
```cpp
//Kernels
registerKernel(CHBulkCoupled);

//Auxiliary Kernels
registerAux(AuxSupersaturation);

//Boundary Conditions
registerBoundaryCondition(StressBC);

//Materials
registerMaterial(PFMobilityLandau);

//Initial Conditions
registerInitialCondition(PolySpecifiedSmoothCircleIC);

//Dampers

//Executioners
registerExecutioner(MeshSolutionModify);

//Postprocessors
registerPostprocessor(NucleationPostprocessor);

//TimeSteppers
registerTimeStepper(InitialSolutionAdaptiveDT);

// UserObjects
registerUserObject(NucleationLocationUserObject);

// Markers
registerMarker(NucleationMarker);
```
## 多个App耦合
MOOSE中有多个衍生模块，比如相场、力学等，可以耦合这些模块，在其基础上进行开发。
在上面的编译过程中已经说明了怎样将其他的模块加入。默认是将全部模块加入。
# 创建app
```cpp
// This creates dynamic memory that we're responsible for deleting
MooseApp * app = AppFactory::createApp("ExampleApp", argc, argv);
```
这一步是创建一个动态存储的MooseApp。
# 运行
```cpp
app->run();
```
这个run函数是程序运行的总调度，其除了撰写性能日志，主要执行的动作有三个：
```cpp
setupOptions();
runInputFile();
executeExecutioner();
```
## 选项配置及读入输入文件
setupOptions读取执行程序时在命令行中输入的选项参数以及输入文件中的各个对象的信息。
首先是命令行中的选项参数：
```cpp
Usage: ./ex01-opt [<options>]

Options:
  --allow-deprecated                                Can be used in conjunction with --error to turn off deprecated errors
  --check-input                                     Check the input file (i.e. requires -i <filename>) and quit.
  --distributed-mesh                                The libMesh Mesh underlying MooseMesh should always be a DistributedMesh
  --dump [search_string]                            Shows a dump of available input file syntax.
  --error                                           Turn all warnings into errors
  --error-deprecated                                Turn deprecated code messages into Errors
  -o --error-override                               Error when encountering overridden or parameters supplied multiple times
  -e --error-unused                                 Error when encountering unused input file options
...
```
这里面用到的最重要的一个函数就是getParam，它先调用InputParameters类的getParamHelper函数，然后归根结底是调用了libMesh的parameters类的get函数。
本例中我们执行的命令是：
```cpp
./ex01-opt -i ex01.i
```
根据该命令选项，读取后某些变量的值变为：
```cpp
_distributed_mesh_on_command_line = false;
_half_transient = false;
Moose::_warnings_are_errors = false;
Moose::_deprecated_is_error = false;
Moose::_color_console = true;

_input_filename = "ex01.i"
```
这样就将输入文件成功读取了。MOOSE的输入文件的语法遵循[GetPot](http://getpot.sourceforge.net/)格式。
成功取得输入文件的名字后，再使用解析器来读入输入文件中的各个对象：
```cpp
_parser.parse(_input_filename);
```
解析过程中，会根据语法将各个块的信息读入，比如：
```cpp
section_names = {"Mesh/", "Variables/", "Variables/diffused/", "Kernels/", "Kernels/diff/", "BCs/", "BCs/bottom/", "BCs/top/", "Materials/", "Executioner/", "Outputs/"}
```
然后对所有的块进行遍历，查找每一个块下对应的属性，比如对于Mesh块，其包含的属性有：
```cpp
block_id block_name boundary_id boundary_name construct_side_list_from_node_list
ghosted_boundaries ghosted_boundaries_inflation
```
等等。
然后通过：
```cpp
_action_warehouse.build();
```
来构建一系列动作，这些动作(通过build函数里的getSortedTask来获得)包括：
```cpp
"setup_time_stepper", "setup_predictor", "setup_postprocessor_data", "init_displaced_problem", "add_aux_variable", "add_elemental_field_variable", "add_variable", "setup_variable_complete", "setup_quadrature", "add_function", "add_periodic_bc", "add_user_object", "setup_function_complete", "setup_adaptivity", "set_adaptivity_options", "add_ic", "add_constraint", "add_field_split", "add_preconditioning", "ready_to_init", "setup_dampers", "setup_residual_debug", "add_bounds_vectors", "add_multi_app", "add_transfer", "copy_nodal_aux_vars", "copy_nodal_vars", "add_material", "setup_material_output", "init_problem", "setup_debug", "add_output", "add_postprocessor", "add_vector_postprocessor", "add_aux_kernel", "add_aux_scalar_kernel", "add_bc", "add_damper", "add_dg_kernel", "add_dirac_kernel", "add_indicator", "add_interface_kernel", "add_kernel", "add_marker", "add_nodal_kernel", "add_scalar_kernel", "add_control", "check_output", "check_integrity"}
```
## 执行输入文件
读入输入文件后，runInputFile来执行里面的动作，即上一步build得到的动作列表，对其进行循环遍历，然后运行下面的函数：
```cpp
for (const auto & task : _ordered_names)
  executeActionsWithAction(task);
```
它还会检查输入文件中是否有未辨别出的变量。输入文件中的所有变量通过以下函数获得：
```cpp
all_vars = _parser.getPotHandle()->get_variable_names();
```
得到的结果是：
```cpp
all_vars = 
{"Mesh/type", "Mesh/file", "Variables/diffused/order", "Variables/diffused/family", "Kernels/diff/type", "Kernels/diff/variable", "BCs/bottom/type", "BCs/bottom/variable", "BCs/bottom/boundary", "BCs/bottom/value", "BCs/top/type", "BCs/top/variable", "BCs/top/boundary", "BCs/top/value", "Executioner/type", "Executioner/solve_type", "Outputs/execute_on", "Outputs/exodus"}
```
## 执行Executioner
executeExecutioner就是执行所创建的Executioner对象。
先是初始化：
```cpp
_executioner->init();
```
其实在执行该初始化函数之前，还隐性地执行了该类的构造函数，显然在此之前先执行该构造函数的成员初始化器列表：
```cpp
Executioner::Executioner(const InputParameters & parameters) :
 MooseObject(parameters),
 UserObjectInterface(this),
 PostprocessorInterface(this),
 Restartable(parameters, "Executioners"),
 _fe_problem(*parameters.getCheckedPointerParam<FEProblemBase *>("_fe_problem_base", "This might happen if you don't have a mesh")),
 _initial_residual_norm(std::numeric_limits<Real>::max()),
 _old_initial_residual_norm(std::numeric_limits<Real>::max()),
 _restart_file_base(getParam<FileNameNoExtension>("restart_file_base")),
 _splitting(getParam<std::vector<std::string> >("splitting"))
{}
```
在执行这些构造函数过程中，有一步是调用Console的outputSystemInformation函数：
```cpp
if (_system_info_flags.contains("framework"))
_console << ConsoleUtils::outputFrameworkInformation(_app);

if (_system_info_flags.contains("mesh"))
_console << ConsoleUtils::outputMeshInformation(*_problem_ptr);

if (_system_info_flags.contains("nonlinear"))
{
std::string output = ConsoleUtils::outputNonlinearSystemInformation(*_problem_ptr);
if (!output.empty())
_console << "Nonlinear System:\n" << output;
}
```
还会构建整个有限元问题的系统，比如将问题中的变量读入到libMesh中，设定所使用的单元类型、插值次数等：
```cpp
FEProblemBase::hasVariable (this=0x1028cc0, var_name="diffused")
```
这样就会产生如下的信息：
```cpp
Framework Information:
MOOSE version:           git commit 61910cd on 2016-12-22
PETSc Version:           3.7.4
Current Time:            Mon Feb 13 21:06:54 2017
Executable Timestamp:    Sat Feb 11 04:08:24 2017

Parallelism:
  Num Processors:          1
  Num Threads:             1

Mesh: 
  Parallel Type:           replicated
  Mesh Dimension:          3
  Spatial Dimension:       3
  Nodes:                   
    Total:                 6399
    Local:                 6399
  Elems:                   
    Total:                 30224
    Local:                 30224
  Num Subdomains:          1
  Num Partitions:          1

Nonlinear System:
  Num DOFs:                6399
  Num Local DOFs:          6399
  Variables:               "diffused" 
  Finite Element Types:    "LAGRANGE" 
  Approximation Orders:    "FIRST" 

Execution Information:
  Executioner:             Steady
  Solver Mode:             Preconditioned JFNK
```
这里最重要的一个类就是FEProblemBase类，它包含整个有限元问题的数学信息。
Executioner本身的构造函数中主要就是获得求解器的信息，比如线性迭代和非线性迭代的精度、迭代步数等：
```cpp
// solver params
EquationSystems & es = _fe_problem.es();
es.parameters.set<Real> ("linear solver tolerance")
= getParam<Real>("l_tol");

es.parameters.set<Real> ("linear solver absolute step tolerance")
= getParam<Real>("l_abs_step_tol");

es.parameters.set<unsigned int> ("linear solver maximum iterations")
= getParam<unsigned int>("l_max_its");

es.parameters.set<unsigned int> ("nonlinear solver maximum iterations")
= getParam<unsigned int>("nl_max_its");

es.parameters.set<unsigned int> ("nonlinear solver maximum function evaluations")
 = getParam<unsigned int>("nl_max_funcs");
```

然后就是开始执行，这时Executioner真正开始工作：
```cpp
_executioner->execute();
```
该execute函数是个纯虚函数，需要根据具体的问题来工作，比如对于稳态问题，调用Steady类，对于瞬态问题，则是调用Transient类来求解。
Steady问题的求解过程是：
```cpp
{
 if (_app.isRecovering())
 return;

 preExecute();

 _problem.advanceState();

 // first step in any steady state solve is always 1 (preserving backwards compatibility)
 _time_step = 1;
 _time = _time_step; // need to keep _time in sync with _time_step to get correct output

#ifdef LIBMESH_ENABLE_AMR

 // Define the refinement loop
 unsigned int steps = _problem.adaptivity().getSteps();
 for (unsigned int r_step=0; r_step<=steps; r_step++)
 {
#endif //LIBMESH_ENABLE_AMR
 preSolve();
 _problem.timestepSetup();
 _problem.execute(EXEC_TIMESTEP_BEGIN);
 _problem.outputStep(EXEC_TIMESTEP_BEGIN);

 // Update warehouse active objects
 _problem.updateActiveObjects();

 _problem.solve();
 postSolve();

 if (!lastSolveConverged())
 {
 _console << "Aborting as solve did not converge\n";
  break;
  }
 
  _problem.onTimestepEnd();
  _problem.execute(EXEC_TIMESTEP_END);
 
  _problem.computeIndicators();
  _problem.computeMarkers();
 
  _problem.outputStep(EXEC_TIMESTEP_END);
 
 #ifdef LIBMESH_ENABLE_AMR
  if (r_step != steps)
  {
  _problem.adaptMesh();
  }
 
  _time_step++;
  _time = _time_step; // need to keep _time in sync with _time_step to get correct output
  }
 #endif
 
  postExecute();
 }
```
这里面每一步又有很大的信息量，值得细思。
# 删除
```cpp
delete app;
```
问题求解结束，释放内存，Done！
