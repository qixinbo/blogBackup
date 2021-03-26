---
title: deal.II链接PETSc过程记录
tags: [deal.II]
categories: simulation
date: 2016-9-29
---

# 2016-11-16 Update

p4est与deal.II的链接注意事项：
- p4est安装时需要开启mpi，即configure时加上--enable-mpi选项。
- 链接时增添：
```cpp
-DP4EST_DIR=/path/to/installation -DDEAL_II_WITH_P4EST=ON -DDEAL_II_WITH_MPI=ON
```



# 简介
在学习deal.II的Step17和18时，需要用到PETSc。
PETSc，全称Portable-Extensible-Toolkit-for-Scientific-Computation，是美国能源部ODE2000支持开发的20多个ACTS工具箱之一，由Argonne国家实验室开发的可移植可扩展科学计算工具箱，主要用于在分布式存储环境高效求解偏微分方程组及相关问题。PETSc所有消息传递通信均采用MPI标准实现，见[百度百科介绍](http://baike.baidu.com/view/3600627.htm)。
默认安装deal.II时没有与PETSc集成，那么就需要重新编译。

注：这里链接PETSc的过程也同样适用于Trilinos、SLEPc等第三方软件。

# 提前准备
这里链接的是deal.II的8.4.1版本和PETSc的3.5.4版本。
还额外需要MPI库，这里用的是Open MPI的1.6.3版本。还需要hypre，用的是2.9.0版本。
openmpi和hypre之前都安装过，分别安装在/usr/local/openmpi和/usr/local/hypre。这两个的具体安装过程不再详述。

# 安装过程
## 编译PETSc
解压下载的安装包，得到petsc-3.5.4文件夹，然后进入，执行以下命令：
```cpp
export PETSC_DIR=`pwd`
export PETSC_ARCH=x86_64   # or any other identifying text for your machine
./config/configure.py --with-shared-libararies=1 --with-x=0 --with-mpi=1 --with-mpi-dir=/usr/local/openmpi --with-hypre=1 --with-hypre-dir=/usr/local/hypre
make all test
```
要点：
- 这里没有指定prefix路径，也就是在当前安装文件夹下编译。
- 环境变量PETSC_DIR是指定PETSc的安装文件位置，环境变量PETSC_ARCH是指定配置名字，比如x86_64或gnu_intel等等，这样可以生成不同名字的目录，方便切换版本。
- configure的第一个参数是生成动态链接库，这样在lib下就能生成libpetsc.so。
- 一定要使用MPI，并且指定好它的路径，否则跟deal.II不对应也不行。这里务必注意！
- 一定使用hypre，并且指定好路径。否则step17还是不能编译。

编译好PETSc后，将那两个环境变量写入.bashrc中，让其能够始终有效：
```cpp
export PETSC_DIR=/home/qixinbo/program/petsc-3.5.4
export PETSC_ARCH=x86_64
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```
这样同时将PETSc的lib路径加入了全局的动态链接库路径中。

## 编译deal.II
解压下载的tar包，得到dealii-8.4.1文件夹，进入，然后执行以下命令：
```cpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/dealii -DDEAL_II_WITH_PETSC=ON -DDEAL_II_WITH_MPI=ON ..
sudo make install
make test
```
要点：
- 用-DDEAL_II_WITH_PETSC=ON来指明链接PETSC
- 一定要指明用MPI，与前面的PETSc相对应

# Step17编译运行成功
最后附上一张Step17编译运行成功的靓照：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvjjxhizrlj30zk0sgqif.jpg)


# 参考文献
[Interfacing deal.II to PETSc](https://www.dealii.org/developer/external-libs/petsc.html)
[Installation instructions and further information on deal.II](https://www.dealii.org/developer/readme.html)
[PETSc:Documentation: Installation](http://www.mcs.anl.gov/petsc/documentation/installation.html)
[安装 deal.II 7.1.0 心得](http://blog.sina.com.cn/s/blog_684b397d0101h9ov.html)
