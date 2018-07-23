---
title: PRISMS-PF v1.0 学习手册——自适应网格
tags: [deal.II, prisms]
categories: computational material science
date: 2016-12-20
---

# 引子
自适应网格是一项很重要的减小计算量、提高计算效率的方法。
# 流程
## 求解步
在求解步时，传入时间步数来判断是否自适应：
```cpp
adaptiveRefine(currentIncrement);
```
如果时间步数满足要自适应的间隔，那么就开始自适应：
```cpp
if ((currentIncrement>0) && (currentIncrement%skipRemeshingSteps==0)){
this->refineMesh(currentIncrement);
}
```
这个refineMesh函数会调用init函数参数是非0的情形：
```cpp
init(_currentIncrement-1);
```
## 初始化参数非零情形
初始化参数如果是0，那么做的就是整个问题的初始化工作，如果非零，那么就是自适应重建整个系统了。
### 自适应网格
如果非零，首先之前的创建网格、边界标定、全局加密等工作都不做了，转而做：
```cpp
refineGrid();
```
此函数会调用adaptiveRefineCriterion来设置细化判据，将解向量暂存在残差向量中：
```cpp
(*residualSet[fieldIndex])=(*solutionSet[fieldIndex]);
```
然后接着执行细化操作。
### 重建系统
这一步在初始化时要做，自适应网格后也要做，但两者也有不同。主要工作就是构建有限单元、分配自由度、施加悬点限制等。
### 重建无矩阵对象
与初始化时相同。
### 重建解向量
此处要需要先把解向量清零，
```cpp
U=solutionSet.at(fieldIndex); 
oldU=oldSolutionSet.at(fieldIndex); 
matrixFreeObject.initialize_dof_vector(*U,  fieldIndex); *U=0;
matrixFreeObject.initialize_dof_vector(*oldU,  fieldIndex); *oldU=0;
```
注意，不能将中间变量清零，否则后面内插的时候就是零向量，即将它们定义在参数为0的地方，而不是参数非零的地方：
```cpp
if (iter==0){
  R=new vectorType;
  tempU=new vectorType;
  residualSet.push_back(R); 
  tempSolutionSet.push_back(tempU);
  matrixFreeObject.initialize_dof_vector(*R,  fieldIndex); *R=0;
  matrixFreeObject.initialize_dof_vector(*tempU,  fieldIndex); *tempU=0;
}
```
然后就是将中间变量的值传递回原变量中：
```cpp
for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
  //interpolate and clear used solution transfer sets
  soltransSet[fieldIndex]->interpolate(*solutionSet[fieldIndex]);
  oldSoltransSet[fieldIndex]->interpolate(*oldSolutionSet[fieldIndex]);

  delete soltransSet[fieldIndex];
  delete oldSoltransSet[fieldIndex];
  
  //reset residual vector
  vectorType *R=residualSet.at(fieldIndex);
  matrixFreeObject.initialize_dof_vector(*R, fieldIndex); *R=0;

  vectorType *tempU=tempSolutionSet.at(fieldIndex);
  matrixFreeObject.initialize_dof_vector(*tempU, fieldIndex); *tempU=0;
}
```
### 重建向量传递函数
```cpp
soltransSet.clear();
oldSoltransSet.clear();
for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
  soltransSet.push_back(new parallel::distributed::SolutionTransfer<dim, vectorType>(*dofHandlersSet_nonconst[fieldIndex]));
  oldSoltransSet.push_back(new parallel::distributed::SolutionTransfer<dim, vectorType>(*dofHandlersSet_nonconst[fieldIndex]));
}
```
### Ghost解向量
```cpp
// Ghost the vectors. Also apply the Dirichet BC's (if any) on the solution vectors 
   for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
     constraintsDirichletSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
     solutionSet[fieldIndex]->update_ghost_values();

     constraintsDirichletSet[fieldIndex]->distribute(*oldSolutionSet[fieldIndex]);
     oldSolutionSet[fieldIndex]->update_ghost_values();

   }
```
## 第一步即加密
这一步稍微修改下自适应过程，让其全局加密后，第一步就加密：
```cpp
template <int dim>
void MatrixFreePDE<dim>::refineMesh(unsigned int _currentIncrement){
#if hAdaptivity==true 
  init(_currentIncrement);
#endif
}
```
即参数不再减1，同时：
```cpp
template <int dim>
void generalizedProblem<dim>::adaptiveRefine(unsigned int currentIncrement){
#if hAdaptivity == true
  if (currentIncrement == 1 || ((currentIncrement>0) && (currentIncrement%skipRemeshingSteps==0))){
    this->refineMesh(currentIncrement);
  }
#endif
}
```
将判断条件也改一下。

# 计算结果
![](http://7xrm8i.com1.z0.glb.clouddn.com/2D-adaptivity.gif)
