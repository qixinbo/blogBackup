---
title: 自适应有限元包AFEPack安装过程
tags: [AFEPack,linux]
categories: programming
date: 2016-3-5
---

安装AFEPack软件包，需要提前安好的软件有gcc、g++、doxygen、automake(1.11版本)、dx、emacs、vim。

# 准备：

我会有一个用户名，比如为qixinbo。在home/qixinbo下建立files，include，lib三个目录。其中include目录是用于存放程序编译时调用的.h头文件，lib目录是用于存放动态链接库.so文件

然后将
boost_1_37_0.tar.bz2,
deal.nodoc-6.3.1.tar.gz,
AFEPack-snapshot.tar.gz,
easymesh.c.gz
四个文件放在files里。

# 安装boost：

安装boost头文件。将boost_1_37_0.tar.bz2解压到files里，即运行tar jxvf boost_1_37_0.tar.bz2命令。解压后会得到一个boost_1_37_0目录，该目录下有一个boost目录，将该目录拷贝到include文件夹里，即

cp -r boost /home/qixinbo/include

# 安装deal.II。

注：个人建议不要下载太高版本，过高版本可能会报Point歧义，原因是由于部分版本deal.II修改了底层网格的一些架构。现deal.II 7.0.0及之前版本已测试可以使用。

(1) 将deal.nodoc-6.3.1.tar.gz解压到files里，即运行tar zxvf deal.nodoc-6.3.1.tar.gz命令。解压后会得到一个deal.II目录。进入该目录，运行

./configure
如果configure通过，就运行
make base lac

在configure过程如果出现提示不存在doxygen，表示它没有被安装在系统里面，请在网上google搜索下载，把这个安装上。

(2) 进入/home/qixinbo/include文件夹，链接头文件。即运行

ln -sf /home/qixinbo/files/deal.II/base/include/base .
ln -sf /home/qixinbo/files/deal.II/lac/include/lac .
ln -sf /home/qixinbo/files/deal.II/contrib/tbb/tbb22_20090809oss/include/tbb .

请注意最后一个点前面有一个空格。

(3) 进入/home/qixinbo/lib文件夹，链接库文件。即运行

ln -sf /home/qixinbo/files/deal.II/lib/lib* .

请注意最后一个点前面有一个空格。

这三步做完以后应该在/home/qixinbo/include目录下存在deal.II 的头文件lac、base和tbb链接头文件，在/home/qixinbo/lib下至少存在libbase.so 和liblac.so动态链接库文件。

将/home/qixinbo/lib加入到缺省的共享链接库路径中，即

在/etc/ld.so.conf中加入一行/home/qixinbo/lib，

然后运行/sbin/ldconfig。

# 安装AFEPack包。

(1) 将AFEPack-snapshot.tar.gz解压缩到files里，即运行tar zxvf AFEPack-snapshot.tar.gz命令。解压后会得到一个AFEPack目录。进入该目录后，首先给定configure找到II的头文件和链接库文件的环境变量。

需要修改一下参数（这一步实际不做也行）：

deal.II 编译参数：

第一种方法：在终端里输入：

gcc -lbase

如果返回：

/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../x86_64-linux-gnu/crt1.o: In function `_start’:
(.text+0x20): undefined reference to `main’

则把 configure.in 文件中 -lbase -llac 修改为 -lbase -llac -ltbb

如果返回：

/usr/bin/ld: cannot find -lbase

则把 configure.in 文件中 -lbase -llac 修改为 -ldeal_II -ltbb

第二种方法：直接查看/Installation/of/deal.II/lib 下的文件

如果有libbase.so, liblac.so 则把 configure.in 文件中 -lbase -llac 修改为 -lbase -llac -ltbb
否则，如果有libdeal_II.so, 则把 configure.in 文件中 -lbase -llac 修改为 -ldeal_II -ltbb
如果上面两个都没有，只能是 deal.II 没装好

gcc 编译参数：

在终端里输入：
gcc -v
会返回 gcc 的版本。

如果版本高于 4.6.0（但小于4.8）， 则修改configure.in文件，一定是这个文件，不能是Make.global_options，因为还牵扯并行mpi下的Makefile。
把 CPPFLAGS=“$EXTRA_INCDIR” 改为 CPPFLAGS=“$EXTRA_INCDIR -std=c++0x”，同理CFLAGS和CXXFLAGS也要加上 -std=c++0x.
注：暂时是需要这样子的，但随着gcc版本升高，可能会有所不同（已证实4.8版本的gcc不适用，会在编译deal.II出现错误。如果已经使用了较新版本的Linux系统，如Ubuntu 14.04LTS，可以手动安装4.6版本的gcc，具体安装过程见文末）。这与deal.II版本和gcc版本及AFEPack的版本有关。

注意：后续的Makefile中的gcc、g++、mpicc、mpicxx等也要注意这个问题。同时还要注意加入并指定base、lac、tbb的库文件路径。

(2)运行

export EXTRA_INCDIR=-I/home/qixinbo/include
export EXTRA_LIBDIR=-L/home/qixinbo/lib

(3)运行

aclocal

autoconf

automake

后产生configure 文件。

可能出现的问题：

如果aclocal时出现提示：

aclocal: main::scan_file() called too early to check prototype

则忽略此信息，继续即可。

(4)运行

./configure

在多个机器上测试发现，在单机版ubuntu下，这样直接configure不会存在问题，而在Redhat Enterprise 服务器上，却是经常提示找不到deal.II的链接库文件

如果能够顺利configure，那么就可以直接执行；
如果configure不通过，就修改configure 对deal.II对链接库文件的检测，让其检测通过，即运行

vi configure

编辑该文件，直接指定lib动态链接库所在的路径，将里面的两行
“-llac -lbase $LIB”
改为
“-llac -lbase $LIB -L/home/qixinbo/lib”

deal_II_library=no
改成
deal_II_library=yes

再运行

./configure

产生makefile后再进入第(5)小步做make编译。

(5) make

make成功后应该在AFEPack/library/lib/目录下产生文件libAFEPack.so

包里面的mpi部分编译会报错，但不影响其它部分的编译，如果要安装，需要修改makefile，由于暂时用不到这一部分，所以可以不予安装。
make也可以分开进行，分别在library、template、example下进行。
即使不分开make，也要在template下make一下，生成需要的模版文件动态库。

(6)进入/home/qixinbo/include文件夹， 链接AFEPack的头文件，运行

ln -sf /home/qixinbo/files/AFEPack/library/include AFEPack

(7)进入/home/qixinbo/lib文件夹， 链接AFEPack的动态链接库文件，运行

ln -sf /home/qixinbo/files/AFEPack/library/lib/libAFEPack.so .
ln -sf /home/qixinbo/files/AFEPack/library/lib/libAFEPack.so libAFEPack.g.so

这时再运行一下/sbin/ldconfig，更新一下ld的缓冲。

# 安装easymesh。

将easymesh.c.gz解压缩到files里，即运行gunzip easymesh.c.gz命令，然后运行gcc -o /home/qixinbo/files/easymesh easymesh.c -lm

这样，四个安装包全部安装好了。

———————————————————————————————–

# 关于并行部分的安装：

AFEPack、openmpi与hypre的相互连接与调用：

## 安装openmpi。

(1) configure过程中，–prefix=/path/指定安装路径：

./configure –prefix=/usr/local/openmpi

然后编译（需要root权限时加上sudo）：

make all install

(2)打开家目录下的.bashrc文件，配置openmpi的动态链接库路径和可执行路径：

vim ~/.bashrc

然后将光标定位到文件末尾，写入:

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib
export PATH=$PATH:/usr/local/openmpi/bin

然后source一下，使配置生效：

source ~/.bashrc

(3)测试openmpi是否安装正确。

进入openmpi安装文件的examples文件夹下，直接编译：

make

然后：

mpirun -np 2 ./hello_c

测试一下。

如果执行正确的话，会输出两行hello,world。
如果有问题，如

./hello_c : error while loading shared libraries:libmpi.so.0:
cannot open shared object file: No such file or directory

则要检查一下是否$LD_LIBRARY_PATH是空的：

echo $LD_LIBRARY_PATH

如果为空，则回到第（2）步，在.bashrc文件中将openmpi的lib路径加入到LD_LIBRARY_PATH中。

## 安装hypre

(1) hypre在configure时使用–prefix指定安装路径（注意路径前不能有空格），需要加上参数–enable-shared以使用动态链接库，还需要让Hypre找到openmpi，故加入参数–with-MPI-include=DIRS、–with-MPI-libs=LIBS、–with-MPI-lib-dirs=DIRS，具体命令为（configure文件在src文件夹中）：

./configure –prefix=/usr/local/hypre -enable-shared
-with-MPI-include=/usr/local/openmpi/include
-with-MPI-libs=”mpi_cxx mpi”
-with-MPI-lib-dirs=/usr/local/openmpi/lib

（有时需要fortran的编译器，如果电脑中没有fortran编译器，此时可以在configure时将这一项忽视，即加上–enable-fortran=no）

然后编译：

sudo make install

此时，有时会出现mpicc未找到的情况，这是因为在root账户的环境变量中没有将mpicc的bin路径加入。解决方法：

切换到root账户下：

su

在root的.bashrc中将PATH变量加入/usr/local/openmpi/bin:

vim /root/.bashrc
export PATH=$PATH:/usr/local/openmpi/bin

然后source一下使其生效：

source /root/.bashrc

然后编译：

make install

(2)将hypre的库文件和头文件加入路径中，两种方法（用第一种即可）：

(2.1)将hypre下的lib文件夹中的两个库文件复制到总的lib文件夹下:

cp /usr/local/hypre/lib/lib* ~/lib/

将hypre的include文件夹同样拷到总的include文件夹中:

cp /usr/local/hypre/include/* ~/include/

(2.2)在实际程序编译时，在Make.global_options中直接指明hypre的头文件和库文件路径，即在Makefile.options文件中指定

HYPRE_LIBS=-L/usr/local/hypre/lib -lHYPRE
HYPRE_INC=-I/usr/local/hypre/include

## 连接

(1)AFEPack包的mpi部分编译需要将mpi的include链接到AFEPack下的include文件下，并改名为mpi。

首先进入AFEPack的include路径下：

cd ~/files/AFEPack/library/include

然后做mpi的链接：

ln -sf ~/files/AFEPack/library/mpi/include mpi

(2)加入openmpi的头文件路径，这里有两种方法（推荐第一种）：

(2.1)将AFEPack下Make.global_options(此文件在configure后产生)里的CFLAGS、CPPFLAGS、CXXFLAGS添加上openmpi的头文件路径：

vim ~/files/AFEPack/Make.global_options

找到CFLAGS、CPPFLAGS、CXXFLAGS，并在原值末尾加上-I/usr/local/openmpi/include，如：

CPPFLAGS = -I/home/qixinbo/include -std=c++0x

变为：

CPPFLAGS = -I/home/qixinbo/include -std=c++0x -I/usr/local/openmpi/include

(2.2)修改gcc的头文件的默认搜索路径C_INCLUDE_PATH和CPLUS_INCLUDE_PATH.

即在.bashrc中添加:

export C_INCLUDE_PATH=$C_INCLUDE_PATH:/path/to/openmpi/include

同理对CPLUS_INCLUDE_PATH修改。

如果之前AFEPack没有编译过，则在AFEPack目录下进行make，并链接AFEPack的头文件和库文件。

如果之前make过，则只需要在/library/mpi下make即可：

cd ~/files/AFEPack/library/mpi

make

(3)make后在AFEPack的lib文件夹下得到libAFEPack_mpi.so，再将它连接到总的lib文件夹下:

ln -sf ~/files/AFEPack/library/lib/libAFEPack_mpi.so ~/lib/
(4)并行计算需要boost的序列化存储的动态链接库，故需要编译boost。编译时最好指定安装目录：

cd ~/files/boost_1_37_0

./configure –prefix=/usr/local/boost

make

sudo make install

将boost编译成功生成的lib文件夹下两个动态链接库拷到AFEPack安装时的总的lib文件夹下，并更名为libboost_serialization-gcc.so和libboost_program_options-gcc.so：

cp /usr/local/boost/lib/libboost_program_options-gcc46-mt-1_37.so.1.37.0 ~/lib/ libboost_program_options-gcc.so
cp /usr/local/boost/lib/libboost_serialization-gcc46-mt-1_37.so.1.37.0 ~/lib/ libboost_serialization-gcc.so

至此，并行库也安装完成了。

——————————————–

# 一些附加注意点：

当将新的库文件加入到默认的库文件搜索路径时，如更改了.bashrc的LD_LIBRARY_PATH及更改了/etc/ld.so.conf中的文件路径下的库文件时，要分别使用source命令及/sbin/ldconfig命令来更新缓冲。

可以在.bashrc文件中写上环境变量：

export MPIDIR=/path/of/openmpi
export PATH=$PATH:$MPIDIR/bin
export LD_LIBRARY_PATH=$MPIDIR/lib:$LD_LIBRARY_PATH：/path/to/totallib
export INCLUDE=$INCLUDE:$MPIDIR/include
export MANPATH=$MPIDIR/share/man:$MANPATH

export AFEPACK_PATH=”/home/qixinbo/files/AFEPack”
export AFEPACK_TEMPLATE_PATH=”$AFEPACK_PATH/template/tetrahedron:$AFEPACK_PATH/template/twin_tetrahedron:$AFEPACK_PATH/template/four_tetrahedron:$AFEPACK_PATH/template/triangle:$AFEPACK_PATH/template/twin_triangle:$AFEPACK_PATH/template/interval”
注意路径作为变量时不能加上空格等正确性。

注意直接从word中拷贝粘贴到vim编辑器中会出现空格的错误，应该先删除空格，再自己加上空格，这个错误比较隐蔽！
