---
title: 手动安装特定版本的gcc编译器
tags: [gcc,linux]
categories: coding
date: 2016-3-23
---

# Update

2016-11-16更新：
在Ubuntu10.10上安装gcc4.8.0时，出现错误：
```cpp
‘CHAR_BIT’ was not declared in this scope
```
解决方法：
```cpp
unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE
```
如果在/etc/profile 或者 ~/.bashrc中设置了ccache的话,也暂时去掉ccache的设置.

参考自[这里](http://www.cnblogs.com/shakin/p/4276434.html)。

另外：
单纯修改gcc的指向后，会造成其他编译器的不匹配，比如gcc虽然指向了4.8，但gfortran还是4.6，两者就不匹配，就会造成如下错误：
```cpp
/usr/bin/ld: cannot find “-lgfortran”
```
这时可以先搜索到4.6版本的gfortran的静态链接库，比如：
```cpp
/usr/lib/gcc/x86_64-linux-gnu/4.6/libgfortran.a
```
然后将它软链接到默认链接库的路径中：
```cpp
sudo ln -sf /usr/lib/gcc/x86_64-linux-gnu/4.6/libgfortran.a /usr/lib/libgfortran.a
```
这样问题就解决了。


# 引子

Linux发行版中一般预装了gcc编译器，版本随系统不同而不同，有时候不想用（或者是不能用）系统默认的gcc编译器，就需要自己编译特定版本的gcc编译器。

这里以在Ubuntu14.04环境（默认gcc为4.8.2）安装gcc4.6.3为例，记录一下安装过程。
整个安装过程很繁琐且漫长，计入发现坑以及填坑的时间，至少需要两个小时（也与机器有关），总之耐心。。。

期间参考了以下网站：
http://blog.csdn.net/wtfmonking/article/details/17577925
http://www.oschina.net/question/12_49423
http://askubuntu.com/questions/251978/cannot-find-crti-o-no-such-file-or-directory
http://blog.csdn.net/gengshenghong/article/details/7498085
https://github.com/couchbase/couchbase-lite-java-native/issues/11

需要提前准备好的软件有：系统自带的gcc（如果没有，需要从软件库中通过apt-get安装）、m4、gmp、mpfr、mpc。
gcc-4.6.3源码和gmp、mpfr、mpc的源码都可以在gcc官网中找到，gcc在release文件夹中，其他的在infrastructure文件夹中。
因为这几个软件之间相互有依赖关系，故它们的安装顺序不要打乱。

# 安装m4

sudo apt-get install m4
（上面命令可能提示找不到m4，此时可以更换软件源试试，也可以使用新立得软件包管理器synpatic安装）

# 安装gmp

tar -xjvf gmp-4.3.2.tar.bz2
cd gmp-4.3.2
./configure --prefix=/usr/local/gmp
make
sudo make install

# 安装mpfr

tar -xjvf mpfr-2.4.2.tar.bz2
cd mpfr-2.4.2
./configure --prefix=/usr/local/mpfr --with-gmp=/usr/local/gmp
make
sudo make install

# 安装mpc

tar -xzvf mpc-0.8.1.tar.gz
cd mpc-0.8.1
./configure --prefix=/usr/local/mpc --with-gmp=/usr/local/gmp --with-mpfr=/usr/local/mpfr
make
sudo make install

# 配置库路径

将三个软件的库文件加入动态链接库中：
vim ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gmp/lib:/usr/local/mpfr/lib:/usr/local/mpc/lib  
保存以后：
source ~/.bashrc
将三个软件的库文件加入共享链接库中：
vim /etc/ld.so.conf
/usr/local/mpc/lib  
/usr/local/gmp/lib  
/usr/local/mpfr/lib  
保存以后：
sudo /sbin/ldconfig  

# 安装gcc-4.6.3

cd gcc-4.6.3
./configure --prefix=/usr/local/gcc-4.6.3 --enable-threads=posix --disable-checking
--disable-multilib   --enable-languages=c,c++ --with-gmp=/usr/local/gmp
--with-mpfr=/usr/local/mpfr --with-mpc=/usr/local/mpc
make
sudo make install

在make中有可能出现以下错误：

（1）/usr/include/features.h:374:25: fatalerror: sys/cdefs.h: 没有那个文件或目录
这是因为在64位机器上生成32位的编译代码，因此需要安装32位的库：
sudo apt-get install gcc-multilib
sudo apt-get install g++-multilib

（2）/usr/bin/ld: cannot find crti.o: No suchfile or directory
此时需要修改LD_LIBRARY_PATH变量，即在.bashrc中设置：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
并且做一个链接：
sudo ln -s /usr/lib/x86_64-linux-gnu /usr/lib64
这里的路径与机器位数有关，64位为x86_64，32位为i386。

# 配置gcc

用gcc-4.6.3替换原来的gcc，需要将原来的gcc屏蔽掉：
sudo mv/usr/bin/gcc /usr/bin/gcc-4.8.2
sudo mv/usr/bin/g++ /usr/bin/g++-4.8.2
然后将gcc-4.6.3的bin路径加入到环境变量中：
vim ~/.bashrc
export PATH=$PATH:/usr/local/gcc-4.6.3/bin
