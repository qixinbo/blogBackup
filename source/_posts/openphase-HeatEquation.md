---
title: OpenPhase实例学习系列：HeatEquation算例全解析
tags:
  - openphase
categories: computational material science
date: 2016-03-30 22:00:48
---

# 写在前面——如何阅读OpenPhase源码
OpenPhase的源码真的是挺难读的，它是一整套完全的很成熟的求解相场模型的框架，其模块化完成度很高(尤其设计的数据结构面面俱到，号称能处理所有类型的数据)，所以用它来解方程时写的代码很少，但带来一个问题是你必须先读懂它：高度集成的代码让初学者“一进去就出不来”(从而迷失在各个类中)或“压根进不去”(一行代码就完成一个大功能，根本不知道怎么入手)。
我的建议是：
1. 善用Doxygen
OpenPhase支持Doxygen，其注释也较全面，使用Doxygen生成类之间的关系图，方便理清思路。
附录有Doxygen生成的dot图的涵义说明。
2. 使用gdb调试
刚开始可能不知道OpenPhase里各个参数的值是多少，可以使用gdb逐步调试且输出关键参数的值，增加理性认识。也可以在源码中加cout来输出，但这样有些麻烦。
 
本文尝试对HeatEquation这个算例进行详细解析，试图理清OpenPhase的计算思路。

# 模拟参数
本例模拟参数分布在四个opi文件中：

## ProjectInput.opi文件
此文件中的参数是通过Settings类的ReadInput函数读入。
针对本例，读入的重要参数有：
- 晶粒个数是2: Ngrans=2
- 相的个数是2: Nphses=2
- 格点数目为：Nx=64, Ny=1, Nz=64
- 网格间距为：dx=1.0
- 界面宽度占格点的数目为：iWidth=0
- 系统实际温度：T=500
- OpenMP线程个数：nOMP=2

## BoundaryConditions.opi文件
此文件中的参数通过BoundaryConditions类的ReadInput函数读入。
- 六个边界的边界条件：BC0X=Free, BCNX=Free, BC0Y=Free, BCNY=Free, BC0Z=Free, BCNZ=Free

## Temperature.opi文件
此文件中的参数通过Temperature类的ReadInput函数读入。
- 初始温度梯度所在位置的坐标：R0X=32, R0Y=32, R0Z=32
- 初始温度梯度：DT_DRX=0, DT_DRY=0, DT_DRZ=0
- 冷却速度：DT_Dt=0

## Heat.opi文件
此文件中的参数通过Heat类的ReadInput函数读入。
- 相0的热扩散率：ThermalDiffusivity_0=13.0e-5
- 相0的热容：HeatCapacity_0=0.465e-3
- 相1的热扩散率：ThermalDiffusivity_0=0.0
- 相0的热容：HeatCapacity_0=5.0

# 初始化

## 相场的初始化：
1. 依次读入三个维度的格点数目、网格间距、界面宽度、相的个数
2. 分配Fields场空间
Fields是Storage3D<Node,0>类型的场，通过Allocate函数给其成员变量赋值：
Size_X=Nx+2=66, Size_Y=Ny+2=3, Size_Z=Nz+2=66
同时开辟出一块大小为Size_X*Size_Y*Size_Z=13068的Node类型的内存，将初始地址赋给指针Array。
3. 分配Fractions场空间
Fractions是Storage3D<double,1>类型的场，同样通过Allocate函数给其成员变量(这里是四个)赋值：
Size_X=Nx+2=66, Size_Y=Ny+2=3, Size_Z=Nz+2=66, Size_N=Nphases=2
但它开辟出一块的double类型的内存大小为Size_X\*Size_Y\*Size_Z\*Size_N=26136，同时将初始地址赋给指针Array。
4. 分配Laplacians场空间
Laplacians场的类型及操作与Fields场相同。
5. 分配Gradients场空间
Gradients场的类型是Storage3D<NodeV,0>，其操作与之前的Fields相同，注意Array返回类型不同。
6. 分配Normals场空间
Normals同Gradients场。
7. 分配Flags场空间
Flags场是Storage3D<int,0>类型，操作如前，只是返回类型不同
8. 分配FieldIndex场空间
FieldIndex同Flags场。
9. 分配FieldsStatistics场空间
FieldsStatistics场是GrainInfo类型的场，其Allocate函数传入晶粒个数Ngrains和相个数Nphases，此处初始化就直接将Ngrains设为1，该场的成员变量GrainStorage是一个元素为Grain类型的容器，Allocate函数将该容器的大小(即里面元素的个数)设定为Ngrains\*Nphases=1\*2=2，同时对该场的成员变量Nphases赋值。
10. 对相场的成员变量TotFractions(是元素为double的容器)的尺寸设定为Nphases=2，传入iWidth，对RefVolume赋值，根据Nx等与iWidth的关系判断，此处为0。

## 温度场的初始化
与相场的初始化的逻辑相同，也是读入网格格点、相的个数、网格间距。对场Tx、Txdx、qdot三个场分配空间，注意Txdx的Array指向dVector3类型的变量。初始温度T0=500, 初始温度梯度dT_dr是一个dVector3的变量，此处调用它的set_to_zero成员函数将其成员变量Storage[]清零(这里调用了c++的memset函数)。初始温度梯度所在的位置r0也是一个dVector3，其set_to_zero同理。

## 热传导方程求解器的初始化
与以上的逻辑相同，相继读入网格格点、相的个数、网格间距。成员变量PhaseThermalDiffusivity是一个类型为Storage<double>的变量，其Alllocate函数传入Nphases，将它传给自己的成员变量Size_X=2,同时开辟一个大小为Size_X的指向double的指针Array。PhaseHeatCapacity、PhaseDensity同理。EffectiveThermalDiffusivity、EffectiveHeatCapacity、EffectiveDensity则都是Storage3D<double,0>类型，其Allocate过程之前已描述过。

## 边界条件的初始化
其Initialize函数传入的是Settings类，但此处初始化中直接将其六个int型的成员变量BC0X、BCNX、BC0Y、BCNY、BC0Z、BCNZ都用OpenPhase的整型常量Periodic=0来赋值。

以上可以看出，OpenPhase在设计Initialize函数时并没有完全真的读入相应的参数并初始化，而是采取类似于定义一个变量并赋无意义的初值这样的做法。下面是真正的读取输入文件。

## 边界条件读取输入文件
调用ReadInput函数，传入的形参是FileName字符串，实参是ProjectDir+BCInputFileName，这两个字符串是命令空间OpenPhase的全局变量，默认是ProjectDir = "ProjectInput/"和BCInputFileName = "BoundaryConditions.opi"。
调用Type.h定义的ReadParameterF函数读取.opi中的数据(详见后面的解析)，取得了该文件的字符串，然后通过判断字符串，将相应的边界条件的整型数值传入BCOX、BCNX等六个成员变量：0是周期性边界；1是非流动边界；2是自由边界；3是固定边界。

## 温度场读取输入文件
通过ReadInput函数读取初始温度梯度所在位置的坐标，并赋值给其成员变量r0，即r0[0]=R0X=32,r0[1]=R0Y=32,r0[2]=R0Z=32，起始r0[]是一个运算符重载，实际将参数传给r0的成员变量Storage数组中。同理将初始温度梯度赋值给dT_dr，即dT_dr[0]=DT_DRX=0,dT_dr[1]=DT_DRY=0,dT_dr[2]=DT_DRZ=0。最后将冷却速度赋值给成员变量dT_dt。

## 热传导求解器读取输入文件
Heat的ReadInput函数有些特殊，其会对不同的相进行循环，对于相的指标n，分别将热导率和热容存入PhaseThermalDiffusivity和PhaseHeatCapacity中，注意[]依然是一个运算符重载，实际是将数值存入变量的Array指针指向的地址中。

经过上面的初始化，各个物理场中存储了正确的模拟参数，下面是初始条件设置。


# 相的分布设定
## Initializations::Single函数

该函数首先调用PhaseField类的FieldStatistics成员变量的add_grain函数，传入的参数是PhaseIndex这个重要参数，此处的实参是0，即代表0号相。注意GrainStorage在之前相场的初始化分配FieldsStatistics时已经设定了，其size()=2，经过add_grain后，返回locIndex=0，GrainStorage[0].Exist=true。

然后对格点进行循环，对Flags、Fields、FieldIndex这三个成员变量进行设定。注意到之前初始化时Allocate后开辟的各自的Array指针的地址都是Size_X\*Size_Y\*Size_Z=66\*3\*66=13068大小，故循环是for(int i = 0; i < Nx+2; ++i)这样的类型。注意赋值时使用Storage<T,0>的()运算符重载，返回指向不同偏移量的Array指针，该偏移量为(Size_Y\*x+y)\*Size_Z+z，这是一个神奇的表达式，它起始为0，中间连续加1，最终为(3\*65+2)\*66+65=13067，正好满足13068的大小，其实奥妙在这里：
(Size_Y\*x+y)\*Size_Z+z=(Size_Y\*(Size_X-1)+Size_Y-1)\*Size_Z+Size_Z-1=Size_X\*Size_Y\*Size_Z-1
就这样取得Array指针指向的所有地址后，再对其赋值。对于Flags(i,j,k)，其返回值是int型，直接赋值0。而对于Fields，其返回值是Node类型，其再调用Node类型的set函数(注意是有两个形参的set)。具体的set过程为：Node类的Fields成员变量(注意与前面的Fields区别)是一个元素为FieldEntry的容器，FieldEntry是一个结构体，从Fields.begin()开始做循环时，发现里面根本没有东西，就新建一个NewEntry，并将传入的两个参数,一个是locIndex=0赋给NewEntry的index，一个是1.0赋给NewEntry的value，NewEntry的另外两个值为0，然后将NewEntry压入Fields中。后面对于FieldIndex(i,j,k)就很自然地取得刚刚的index。

最后的Finalize函数由多个部分组成：
1. FinalizeStepOne(BoundaryCondition&)函数。
它使用了OpenMP的并行计算方法，这个地方不好用gdb调试，所以可以直接在源码中用cout输出。首先用omp_get_num_threads()得到线程总数，此处为nOMP=2，然后用omp_get_thread_num()得到线程标识，分别为0和1，然后将x方向的格点分为两段，分别是从1到32和从33到64，然后对三维的格点进行循环，判断Flags(i,j,k)是否为0，因为上面已经设定此值为0，所以此处不进入if处理。
2. SetBoundaryConditions(BoundaryCondition&)函数。
此函数传入BoundaryCondition类的对象，实际是接着调用BC的SetX、SetY、SetZ三个成员函数，注意这里有个函数重载，传入的参数分别是PhaseField类的Fields和FieldIndex成员变量。SetX函数是根据不同的边界条件对Fields和FieldIndex进行设定。如对Periodic边界条件，让0号和64条件相同，1和65号相同。注意这里面设定的对应编号。对于左边界，如果是NoFlux边界条件，则让0号和1号相同，如果是Free边界，则让0号+2号等于2倍的1号。Fixed边界则不做处理。对于右边界，有同样的处理。SetY和SetZ亦同。
3. SetFlags函数
其会对Interface(i,j,k)函数作判断，然后满足某些条件后将Flags(i,j,k)=Interface(i,j,k)+1。此处没有满足if条件，所以不做处理。
4. SetFlagsBC函数
此处调用SetX等对Flags进行边界条件的处理。
5. CalculateFractions函数
计算相分数，此处也会作一些判断，但此时仍然未满足条件，只是将所有地方的Fractions都置为1。
6. CalculateLaplacians函数
计算拉普拉斯算子，仍然没满足条件，所以未分析。
7. CalculateGradients函数
计算梯度，处理同上。
8. SetGradientsBC函数
设置梯度边界条件
9. CalculateNormals函数
计算法线向量
10. CalculateVolumes函数
计算相体积。

## Initializations:Sphere函数
Sphere函数与Single函数逻辑类似，Single相当于Sphere的一个初级版本，很多变量都是0，因此对相场的设置很简单，而Sphere则进一步将不同的相分开，赋予不同的属性，因此更能加深对代码的理解。
这部分的具体步骤不再详述。

# 对条件的进一步设定
此部分对于温度场有了进一步设置，如初始温度Tx.T0和热源Tx.qdot。

# 求解温度场

主要利用Heat类的Solve成员函数。
## 判断稳定性
判断时间步长与网格间距之间的关系是否满足稳定性要求。
这里设定为dt/(dx\*dx)<0.5。
## Temperature.SetBoundaryConditions(BC)函数
传入Tx参数对Ghost Nodes设定边界条件。
## SetEffectiveProperties(Phase)函数
这一步是将相场的热容、热扩散率、密度等材料物性参数传递到Heat类的对应的成员变量中。
## CalculateLaplacian(Temperature)函数
这一步使用有限差分方法计算拉普拉斯项——二阶微分项，具体的差分格式是中心差分。
## UpdateTemperature(Temperature, dt)函数
求解热传导方程,更新温度场


# 输出
相场的输出用Phi.WriteVTK(tStep)函数，格式是标准的VTK格式，格式的具体解析见[此文](http://qixinbo.info/2016/02/22/openphase-vtk/)。
具体输出的内容为：
(1)格点的编号，共Nx\*Ny\*Nz个点，排列次序为：
1  |  1  |  1
2  |  1  |  1
...| ... | ...
64 |  1  |  64 
(2)输出第一个场——Interfaces，1代表界面，0代表体相
(3)输出第二个场——PhaseFields，
(4)输出第三个场——PhaseFraction，该场的个数与相的个数Nphases相等
(5)输出第四个场——ColorScale，
(6)输出第五个场——Flags，

# 注意事项

1. 不同的变量用的编号不同，有的是从0号到63号，有的是从1号到64号，还有65号的，这期间应涉及一个特殊处理——Ghost nodes，暂时没弄明白怎么回事，留待以后解决。

# 附录一：Doxygen的dot图的涵义
![doxygen_dot](https://ws1.sinaimg.cn/large/0072Lfvtly1fvji79gecjj30iz06gmxc.jpg)
## 上图中的box的涵义：
- 实心的灰色box代表绘制此图所基于的结构体或类
- 黑色边界的box意味着此结构体或类有记录
- 灰色边界的box意味着此结构体或类无记录
- 红色边界的box意味着此有记录的结构体或类的关系没有被完全绘制出来
## 上图中的箭头的涵义：
- 深蓝色箭头表示两类之间的public继承关系(上方的类是父类，下方的为子类)
- 深绿色箭头表示两类之间的protected继承关系
- 深红色箭头表示private继承关系
- 紫色虚线箭头表示其所指的类或结构体被包含或使用，上面的label是使用此类的变量
- 黄色虚线箭头表示模板实例和模板类之间的关系，上面的label是该实例的模板参数

# 附录二：ReadParameterF函数
ReadParameterF函数有四个形参：文件流inp、字符串Key、整型变量necessary、字符串defaultVal。传入的实参是BoundaryCondition.opi，将其赋值给inp，将字符串BCOX、BCNX等赋值给Key。
开始读入文件时，截取文件开头一直到字符的内容，赋值给tmp，如果没有这个字符，则抛出一个错误信息；然后再继续读取文件，将空格之前的字符串赋值给ReadKey字符串，如BC0X，将ReadKey与Key通过compare函数对比，如果两者不同的话，则继续读取；如果相同的话，再将后续到字符:的内容赋值给tmp，接着将直到回车符的字符串赋值给tFileName，如Free。为了使格式整齐，使用字符串的erase等函数删除空格，然后调用Info类的GetStandard输出。将tFileName填入ReturnValue中，返回此值。
