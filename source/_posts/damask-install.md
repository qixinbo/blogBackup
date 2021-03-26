---
title: 晶体塑性有限元开源软件DAMASK安装过程记录 
tags: [abaqus,CPFEM,linux]
categories: simulation
date: 2018-1-12
---
2018-1-12更新：
电脑重装后再次安装DAMASK，完善文档，遇到了几个新问题，并附解决方法。

DAMASK是马普钢铁研究所开发的一款用于晶体塑性有限元模拟的开源软件。
安装过程如下：

# 求解器和编译器
DAMASK可以调用三种求解器：MSC.Marc/Mentat，Abaqus和自带的谱方法求解器。鉴于Abaqus用得范围广，这里选择Abaqus作为求解器。DAMASK实际是Abaqus的子程序，因此还需要安装Abaqus的subroutine编译器，即Intel的Fortran编译器。
这两个软件的安装及相互调用在之前的博客中已写过，见[here](http://qixinbo.info/2018/01/12/abaqus-install/)。

# Python及其Modules
DAMASK的安装脚本、前处理和后处理等工具都是由Python写的，所以还要安装Python的编译器及相关Module。
```cpp
sudo apt-get install python python-dev  // 安装Python编译器
sudo pip install numpy  // 安装numpy包
sudo pip install scipy  // 安装scipy包
sudo pip install h5py   // 安装h5py包
sudo apt-get install python-vtk // 安装vtk
```
# 系统配置
用文本编辑器编辑DAMASK文件夹中的CONFIG文件，根据自己的系统配置进行改写（默认配置已基本适用于大部分系统）。注意，该CONFIG文件是在v2.0.1以后才有，之前的v2.0.0没有该文件，是有一个configure的脚本来执行。
```cpp
# "set"-syntax needed only for tcsh (but works with bash and zsh)
# DAMASK_ROOT will be expanded
set DAMASK_BIN=${DAMASK_ROOT}/bin
set DAMASK_NUM_THREADS = 4 // 配置线程数目

set MSC_ROOT=/opt/MSC  // MSC的安装路径
set MARC_VERSION=2015  // MARC的版本

set ABAQUS_VERSION=6.14-1  // Abaqus的版本
```
然后根据自己适用的shell环境来选择不同的文件使配置生效，即使用source命令，如使用bash的话就选择后缀为sh的文件：
```cpp
source DAMASK_env.sh
```
这一步是编译或运行DAMASK的必要条件，因为它让DAMASK知道其使用怎样的配置，同时将DAMASK的bin路径加入到PATH变量中。因此，必须确保它是第一个执行的。或者直接就将该source命令加入到用户配置文件.bashrc中，让其自动生效。

*ATTENTION!: 如果之前abaqus是用sudo安装并运行的，那么还要将该命令加入到root用户的.bashrc中，否则sudo运行abaqus后，会将普通用户的环境变量覆盖，导致找不到DAMASK。*

# 安装前处理和后处理工具
DAMASK有一些前后处理的工具放在了processing路径下，在有Makefile文件的那级目录下使用：
```cpp
make install
```
在DAMASK/bin下创建符号链接来指向这些工具，因此就可以直接调用这些工具。

# Abaqus配置
Abaqus在不同的操作系统Linux和Windows下不同，具体见DAMASK官网配置。
对于Linux来说，需要将Abaqus的环境文件复制到用户路径下或者模型所在路径下：
```cpp
cp DAMASK/installation/mods_Abaqus/abaqus_v6.env  /Dir/of/home/OR/Dir/of/Model/
```
# Abaqus压缩模型
在example路径下有一个运行实例，可以用来测试是否安装和求解成功：
```cpp
cd DAMASK/examples/AbaqusStandard
abaqus job=Job_sx-px user=../../code/DAMASK_abaqus_std.f interactive
abaqus viewer database=Job_sx-px.odb
```
## 可能遇到的问题
(1) Out of memory asking for
(2) Illegal memory reference
第一个问题是出现在Abaqus编译.f文件时，第二个是出现在Abaqus开始运行程序时，原因可能是电脑的运行资源不够。此时的一个方法是重启电脑，然后启动后直接运行该模型，保证程序拥有最多的计算资源。
## 运行结果
两个圆柱体，一个是单晶，一个是多晶，其受压变形：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjjyg1ei7j31b90nimy0.jpg)

# 前处理生成晶粒
用DAMASK生成50个晶粒：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjjypvjb5j312d0li0u9.jpg)



