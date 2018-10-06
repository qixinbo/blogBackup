---
title: PRISMS-PF v1.0 学习手册——Cahn-Hilliard和Allen-Cahn方程耦合求解
tags: [deal.II, prisms]
categories: computational material science
date: 2016-12-16
---

# 引子
PRISMS-PF的1.0版本相对于最开始的0.8版本，进行了较大的重构，更加模块化，更加容易扩展。同时引入了更加多的功能，比如施加各种边界条件，引入自适应网格等。但带来的一个问题是：文件数量增多，且互相调用，刚开始很容易在文件跳转之间绕晕，所以需要理清整个文件结构和程序运行脉络。

# 文件结构
## 基础头文件
首先是deal.II的各种头文件：
```cpp
include/dealIIheaders.h
```
PDE的无矩阵解法的模板类：
```cpp
include/matrixFreePDE.h
```
包含各种属性的物理场的模板类：
```cpp
include/fields.h
```
模型中用到的变量的模板类：
```cpp
include/model_variables.h
```
默认的模拟参数设置：
```cpp
include/defaultValues.h
```
## 无矩阵解的模板类的实现头文件
```cpp
src/matrixfree/matrixFreePDE.cc"
src/matrixfree/init.cc"
src/matrixfree/initForTests.cc"
src/matrixfree/refine.cc"
src/matrixfree/invM.cc"
src/matrixfree/computeLHS.cc"
src/matrixfree/computeRHS.cc"
src/matrixfree/modifyFields.cc"
src/matrixfree/solve.cc"
src/matrixfree/solveIncrement.cc"
src/matrixfree/outputResults.cc"
src/matrixfree/markBoundaries.cc"
src/matrixfree/boundaryConditions.cc"
src/matrixfree/initialConditions.cc"
src/matrixfree/utilities.cc"
src/matrixfree/calcFreeEnergy.cc"
src/matrixfree/integrate_and_shift_field.cc"
src/matrixfree/getOutputTimeSteps.cc"
```
## 求解模型
不同的问题对应不同的模型。目前提供三种模型供选择：
(1)扩散模型
```cpp
src/models/diffusion/AC.h
src/models/diffusion/CH.h
src/models/diffusion/coupledCHAC.h
src/models/diffusion/coupledCHAC2.h
src/models/diffusion/Fickian.h
```
(2)连续力学模型
```cpp
src/models/mechanics/anisotropy.h
src/models/mechanics/computeStress.h
src/models/mechanics/mechanics.h
```
(3)两者耦合模型
```cpp
src/models/coupled/coupledCHMechanics.h
src/models/coupled/coupledCHACMechanics.h
src/models/coupled/generalized_model.h
src/models/coupled/generalized_model_functions.h
```
## 具体问题
每个具体问题都包含四个文件，分别设定自己所需的：
(1)模拟参数，包括计算域大小、网格大小、时间步、输出间距等
```cpp
parameters.h
```
(2)控制方程，这个是需要重要修改的文件，里面包含方程构建过程以及右端项的计算过程
```cpp
equations.h
```
其中包含一系列变量，用“列表+宏定义”的形式方便将这些变量赋值给之前的通用模型中的一系列容器。
(3)初始条件和边界条件
```cpp
ICs_and_BCs.h
```
(4)入口函数，这个文件非常简单，但其是整个程序的总调度。
```cpp
main.cc
```
# 求解流程
分析源码时一定注意各个类之间及类的成员函数之间传递的参数的类型及结构。

具体求解问题时的步骤：
```cpp
generalizedProblem<problemDIM> problem;

problem.setBCs();
problem.buildFields();
problem.init(); 
problem.solve();
```
## 构造函数初始化
在创建问题类的具体对象时，会调用其构造函数，此时就把手动提供的输入文件中的变量(其用宏定义)传递给类的数据成员上，包括变量名、变量类型、变量所在的控制方程的类型、是否计算值及其梯度和二阶导数的旗标等。
比如，在输入条件中设置：
```cpp
#define variable_name {"c", "n","biharm"}
#define variable_type {"SCALAR","SCALAR","SCALAR"}
#define variable_eq_type {"PARABOLIC","PARABOLIC","PARABOLIC"}

#define need_val {true, true, false}
#define need_grad {true, true, true}
#define need_hess {false, false, false}

#define need_val_residual {true, true, false}
#define need_grad_residual {true, true, true}
```
那么，在初始化时：
```cpp
var_name = variable_name;
var_type = variable_type;
var_eq_type = variable_eq_type;

need_value = need_val;
need_gradient = need_grad;
need_hessian = need_hess;
value_residual = need_val_residual;
gradient_residual = need_grad_residual;
```
## 设定边界条件
如上，设定边界条件的函数是setBCs，它需要在ICs_and_BCs.h中手工编写，调用的是另一个成员函数inputBCs，它根据参数不同有多个重载，可以统一设定边界(参数为4个)，也可以分开设定(参数为2+(2+2)*dim)。这一步的目的就是对BC_list赋值，它是一个容器，里面的元素是varBCs的类，每个varBC包含边界条件的类型和边界条件的值。
## 构建场变量
判断之前读入的各种变量和方程类型，构建出要求解的场变量的类型，这个场就是在基础头文件中定义的场，其需要三个模板参数：变量类型、方程类型、变量名。这里将方程类型分成了两类：抛物型方程和椭圆型方程，前者用于求解与时间有关的方程，即瞬态方程，后者用于求解与时间无关的方程，即稳态方程。
这里就引出来一个非常重要的变量：fields。每个MatrixFreePDE都有一个数据成员fields，它是一个容器，里面盛放的是Field类对象，该类模板在前面提到的基础头文件fields.h中定义：
```cpp
template<int dim>
class Field
{
 public:
  Field(fieldType _type, PDEType _pdetype, std::string _name);
  fieldType type;
  PDEType   pdetype;
  std::string name;
  unsigned int index;
  unsigned int startIndex;
  unsigned int numComponents;

 private:
  static unsigned int fieldCount;
  static unsigned int indexCount;
};
```
因为场分为标量场和矢量场，所以fields里面的场变量索引及每个场分量个数都要依情况而定：
```cpp
switch (type){
case SCALAR:{
  //increment index count by one
  indexCount+=1;
  numComponents=1;
  break;
}
case VECTOR:{
  //increment index count by dim
  indexCount+=dim;
  numComponents=dim;
  break;
}
```
## 初始化
调用MatrixFreePDE类的init函数，做以下事情(注意：这里默认接收的参数是0，意味着都是进行初始化工作，如果非0，则可以读取已计算的所有值，比如用于断点续算等，未详细考察)：
(1)初始化网格
根据输入文件，确定计算域大小、初始网格大小。
```cpp
GridGenerator::subdivided_hyper_rectangle (triangulation, subdivisions, Point<dim>(), Point<dim>(spanX,spanY,spanZ));
```
(2)设置边界标识
这里根据面的中心必须是计算域顶点才设置边界标识，这里自然全是0。
```cpp
for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell;++face_number){
  for (unsigned int i=0; i<dim; i++){
    if ( std::fabs(cell->face(face_number)->center()(i) - (0)) < 1e-12 ){
      cell->face(face_number)->set_boundary_id (2*i);
    }
    else if (std::fabs(cell->face(face_number)->center()(i) - (domain_size[i])) < 1e-12){
      cell->face(face_number)->set_boundary_id (2*i+1);
    }
}
```
(3-7)初始化无矩阵对象
这几步是对MatrixFreePDE的属性进行初始化，注意这几步操作都是在对每个场都操作，所以刚开始要对所有的场进行循环：
```cpp
for(typename std::vector<Field<dim> >::iterator it = fields.begin(); it != fields.end(); ++it)
{}
```
(3)设定方程类型及索引
根据方程类型设定是否是时间相关的，索引数是多少：
```cpp
currentFieldIndex=it->index;

//check if any time dependent fields present
if (it->pdetype==PARABOLIC){
isTimeDependentBVP=true;
parabolicFieldIndex=it->index;
}
else if (it->pdetype==ELLIPTIC){
isEllipticBVP=true;
ellipticFieldIndex=it->index;
}
```
(4)创建有限单元
这里指定有限单元类型，即创建无矩阵对象的FESet属性，它是一个容器，里面的元素是FESystem，之所以用FESystem，也是为了能够处理矢量情形：
```cpp
if (it->type==SCALAR){
fe=new FESystem<dim>(FE_Q<dim>(QGaussLobatto<1>(finiteElementDegree+1)),1);
}
else if (it->type==VECTOR){
fe=new FESystem<dim>(FE_Q<dim>(QGaussLobatto<1>(finiteElementDegree+1)),dim);
}
else{
pcout << "\nmatrixFreePDE.h: unknown field type\n";
exit(-1);
}
FESet.push_back(fe);
}
```
(5)分配自由度
上一步创建了有限单元，这一步就要对它们分配自由度，当然先根据网格创建自由度管理器：
```cpp
DoFHandler<dim>* dof_handler;
if (iter==0){
  dof_handler=new DoFHandler<dim>(triangulation);
  dofHandlersSet.push_back(dof_handler);
  dofHandlersSet_nonconst.push_back(dof_handler);
}
else{
  dof_handler=dofHandlersSet_nonconst.at(it->index);
}
dof_handler->distribute_dofs (*fe);
totalDOFs+=dof_handler->n_dofs();
```
(6)对零通量和周期性边界条件施加限制
(7)施加Dirichlet边界条件，并存储其对应的自由度标识
(8)初始化MatrixFree对象
```cpp
typename MatrixFree<dim,double>::AdditionalData additional_data;
additional_data.mpi_communicator = MPI_COMM_WORLD;
additional_data.tasks_parallel_scheme = MatrixFree<dim,double>::AdditionalData::partition_partition;
additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
QGaussLobatto<1> quadrature (finiteElementDegree+1);
num_quadrature_points=std::pow(quadrature.size(),dim);
matrixFreeObject.clear();
matrixFreeObject.reinit (dofHandlersSet, constraintsOtherSet, quadrature, additional_data);
```
(9)初始化该问题的残差向量和解向量
```cpp
for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
  vectorType *U, *R;
  if (iter==0){
    U=new vectorType; R=new vectorType;
    solutionSet.push_back(U); residualSet.push_back(R); 
    matrixFreeObject.initialize_dof_vector(*R,  fieldIndex); *R=0;
  }
  else{
    U=solutionSet.at(fieldIndex); 
  }
  matrixFreeObject.initialize_dof_vector(*U,  fieldIndex); *U=0;
```
这里就引出了两个很重要的量：残差向量和解向量。两个都是容器，里面的元素是指向向量的指针。这两个向量的大小是问题包含的场的个数，具体到里面的每个元素，即每个场，其大小等于自由度的个数。注意：元素是个指针。
解向量和残差向量参与运算，是在computeRHS函数中进行,它们传入getRHS中：
```cpp
matrixFreeObject.cell_loop (&MatrixFreePDE<dim>::getRHS, this, residualSet, solutionSet);
```
computeRHS相当于step48中的apply函数，getRHS函数相当于step48中的local_apply函数。
(10)根据方程类型不同，初始化一些额外量
比如，针对椭圆型方程，计算dU值：
```cpp
if (fields[fieldIndex].pdetype==ELLIPTIC){
	 matrixFreeObject.initialize_dof_vector(dU,  fieldIndex);
}
```
因为这里的dU不是一个容器，所以仅能允许问题中有一个椭圆型方程。
针对抛物型方程，计算质量矩阵的倒数：
```cpp
if (isTimeDependentBVP){
  computeInvM();
}
```
(11)施加初始条件
根据输入文件中的初始条件进行设置：
```cpp
for (unsigned int var_index=0; var_index < num_var; var_index++){
  if (var_type[var_index] == "SCALAR"){
    VectorTools::interpolate (*this->dofHandlersSet[fieldIndex], InitialCondition<dim>(var_index), *this->solutionSet[fieldIndex]);
    fieldIndex++;
  }
  else {
    VectorTools::interpolate (*this->dofHandlersSet[fieldIndex], InitialConditionVec<dim>(var_index), *this->solutionSet[fieldIndex]);
    fieldIndex += dim;
}
}
```
这里首先区分了标量场和矢量场，然后又因为同一种类型的场中会有多个场，所以还通过指标区分了浓度场、相场。结果就是将初始条件的函数场赋给了解向量。
## 求解
求解过程也分了两种情形：时间相关的边值问题求解和时间无关的边值问题求解。
在计算时可以实现如下功能：
(1)输出计算结果
```cpp
outputResults();
```
(2)计算并输出自由能
```cpp
computeEnergy();                                            
outputFreeEnergy(freeEnergyValues);
```
(3)自适应网格
```cpp
adaptiveRefine(currentIncrement);
```
(4)逐步求解
```cpp
solveIncrement();
```
这步中包含计算右端项:
```cpp
computeRHS();
```
这里面的右端项的构建还是在输入文件中，该文件中的函数值及其偏导的命名与本身这个变量的表达式很相近，比如偏导或下标直接就写上，理解了这样的书写方式就很好读写了。注意这些表达式力的变量名要与文件开头定义的变量名对应。开头是r表示是剩余项，即residual。另外，这里用到了一个带参数的宏定义，用来将变量向量化，用于后续计算。
然后再与质量矩阵的倒数相乘得到最终结果。如果是椭圆型方程，是换用另一种方法直接求解。

# 计算结果
如图：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjhzxkp8ig30hs0f2n3y.gif)

# 后记
(1) make debug 比 make release 严格很多，同时计算也更加慢，因为其要检查很多东西。
(2)prisms-pf的计算右端项时，将residualSet作为了dst，而deal.II的step48中是将solution作为dst，这个地方有些反直觉，但最终顺序没错，因为它在与质量矩阵的倒数相乘时用的是residualSet。
