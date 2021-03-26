---
title: 使用PRISMS-PF模拟二维和三维枝晶生长
tags: [deal.II, prisms]
categories: simulation 
date: 2016-12-16
---

# 引子
本例将使用PRISMS-PF模拟凝固过程中的枝晶生长。因为原先的PRISMS-PF架构中在构建右端项时只有接收一步解，而枝晶的浓度场控制方程与相场参量的变化率相关，即需要用到相场的前后两步解，所以需要扩展PRISMS-PF的原先架构，本例也主要是说明怎样对原程序进行扩展。

# 获得源码
源码在[这里](https://github.com/qixinbo/phaseField/tree/next/applications/dendriticGrowth)

# 扩展过程
## 定义要使用解的个数
在parameter.h中新增宏定义numSolution，其决定方程中用到几步解，比如在枝晶生长的模拟中用到两步解，其就等于2，之前的所有模拟中都等于1。
```cpp
#define numSolution 2
```
这里涉及头文件的包含顺序，放在后面定义的话，会不认识。
## 新增前一步解和临界解的存储对象
仿照solutionSet的定义，在相同位置(具体就是MatrixFreePDE类声明)增加oldSolutionSet和tempSolutionSet，同时初始化空间，并且根据初始条件初始化值。
注意这个地方初始化时不要跟当前解使用同一个地址，否则它们自动跟着当前解变化，应该另外开辟内存地址：
```cpp
vectorType *U,*oldU, *tempU, *R;
if (iter==0){
  U=new vectorType; R=new vectorType;
  oldU=new vectorType; tempU = new vectorType;
  solutionSet.push_back(U); residualSet.push_back(R); 
  oldSolutionSet.push_back(oldU);
  tempSolutionSet.push_back(tempU);
  matrixFreeObject.initialize_dof_vector(*R,  fieldIndex); *R=0;
}
else{
  U=solutionSet.at(fieldIndex); 
  oldU=oldSolutionSet.at(fieldIndex); 
  tempU=tempSolutionSet.at(fieldIndex); 
}
matrixFreeObject.initialize_dof_vector(*U,  fieldIndex); *U=0;
matrixFreeObject.initialize_dof_vector(*oldU,  fieldIndex); *oldU=0;
matrixFreeObject.initialize_dof_vector(*tempU,  fieldIndex); *tempU=0;
```
## 更新解
每次计算开始前，将solutionSet暂存在tempSolutionSet中，计算完毕后，使用tempSolutionSet对oldSolutionSet赋值(可以在基类中增加这么一个更新解的成员函数)。
注意，用当前步的解给上步解赋值时，不能直接使用等号，像这样：
```cpp
oldSolutionSet = solutionSet;
```
这样会将当前解的地址直接赋给上步解，导致两个解完全同步。
需要只是传递值：
```cpp
  for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
      for (unsigned int dof=0; dof<solutionSet[fieldIndex]->local_size(); ++dof){
	oldSolutionSet[fieldIndex]->local_element(dof) = solutionSet[fieldIndex]->local_element(dof);
    }
  }
```
## 将计算右端项的输入改为盛放两步解的容器并赋值
(1)将computeRHS和getRHS以及residualRHS的输入参数src改为容器，注意需要在多个地方修改：matrixFreePDE原型修改、coupled的函数定义实现、coupled_function中函数实现修改
(2)需要在计算右端项之前，对src容器赋值：通过numSolution判断读入多少个解
这个将原来盛放一步解的变量改成盛放此变量的容器，涉及多个函数及多个地方的修改，一定要仔细。
在getRHS中添加能读取上一步解的功能时，注意学习step48。两者对应关系就是：
```cpp
scalar_vars  VS   current
old_scalar_vars  VS  old
```
在residualRHS中将输入变成容器时，注意将容器中的元素设置为存储指针，这样就直接指向原来的两个解：
```cpp
std::vector<std::vector<modelVariable<dim>>*> modelVarListList;
modelVarListList.push_back(&modelVarList);
if(numSolution == 2)
  modelVarListList.push_back(&oldModelVarList);
```
# 计算结果
二维结果：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvji141btng30hs0f2dra.gif)
三维结果
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvji1ln258g30hs0f2n4t.gif)

# 后记
(1)书写方程各项时一定注意，将其弱形式仔细写下来，尤其是分部积分时对拉普拉斯算子分解时的符号变化。
(2)学会使用天山折梅手，化简为繁；学会使用奥卡姆剃刀，化繁为简。
(3)时间步长和网格间距会导致收敛性
(4)求解精度也会影响枝晶形貌
(5)有限单元的插值方式会直接决定hessian矩阵的计算
