---
title:  linux系统abaqus安装过程及子程序编译器设置
tags: [abaqus,linux]
categories: simulation
date: 2018-1-12
---
2018-5-2更新：
更改abaqus的执行命令，统一使用绝对路径。
在abaqus cae 后面增加-mesa选项，以防有的电脑显卡不支持。

2018-1-12更新：
笔记本重装系统以后，再次安装Abaqus，修复了一些细节问题。
遇到了几个新的问题，并附解决方法。

2017-6-20更新：
在新笔记本上重新安装了一遍，完善了部分流程，修正了部分错误。

------------------
本文是在新浪博客Install ABAQUS v6.10 in Ubuntu 12.04 基础上所写，希望可以使大家少走弯路。  
原文地址为http://blog.sina.com.cn/s/blog_648bf4210100vxfk.html.  
主要更新为license服务器的设置和子程序上与Intel Fortran编译器的链接上。


# Prerequisites

先需要安装 g++ libstdc++5 csh(貌似没用到)。  


# 安装Abaqus的Documentation（也可以不安装）

## 加载.iso
```cpp
sudo mkdir /mnt/vcdrom
sudo mount -o loop <where-install-files>/Abaqus6-10-2-documentation.iso /mnt/vcdrom

sudo mkdir ~/SIMULIA
cd ~/SIMULIA
sudo /mnt/vcdrom/./setup
```
## 启动abaqus html 帮助文档
启动abaqus html 帮助文档要通过服务器形式，才可以使用搜索功能，即
http://***********:2080/v6.10
如果通过打开本地html文件的形式，则无法使用搜索：
file:///~/SIMULIA/Documentation/docs/v6.10/index.html

# 安装 license

```cpp
sudo umount /mnt/vcdrom
sudo mount -o loop <where-install-files>/Abaqus6-10-2-product.iso /mnt/vcdrom
sudo /mnt/vcdrom/./setup
```
如安装正常，会出现让你输入 scratch path， 就是临时安装路径，可以输入/tmp。然后应该会出现abaqus的gui安装界面，先安装licensed。安装时，注意选择“只安装license server，但是不启动“那一项。
安装完license后，abaqus的安装程序会问你“是否安装其产品”。先不管，但也不要关闭它。
修改破解用的ABAQUS.lic文件，即将里面第一行的this_host改名为本机的主机名（hostname），其他的不用改。将其复制到生成的Lisence文件夹下，运行./lmgrd -c ABAQUS.lic。此时会显示某个TCP port number运行起来了。如果显示有TCP port number已经在运行，则用lmdown下停止，然后再lmgrd重新启动。(或者干脆重启电脑)
至此，license服务器应该就会运行起来。用~/SIMUlIA/License 中 lmstat -c yourhostname@127.0.0.1可查看状态，必须要保证运行正常。

# 安装Product
回到安装Product的窗口，进行安装，输入服务器地址为：27011@主机名，注意27011对于lic文件。如果此时报错也没问题，继续安装。

# 设置abaqus启动时寻找的license服务器
```cpp
sudo gedit ~/SIMULIA/Abaqus/6.10-2/SMA/site/abaqus_v6.env
```
将abaquslm_license_file设置为自己的服务器地址，如”27011@主机名”

# 启动Abaqus
cd ~/SIMULIA  #这是Abaqus安装目录
sudo XLIB_SKIP_ARGB_VISUALS=1 ./Commands/abq6102 cae
XLIB_SKIP_ARGB_VISUALS=1是为了解决abaqus窗口透明的问题
注意除非在Abaqus CAE指定work directory，abaqus会在终端的当前目录下工作
未尽事宜可参见Abaqus的帮助文档

# 建立启动快捷方式

在~/.bashrc里加入：
```cpp
alias abaqus='sudo XLIB_SKIP_ARGB_VISUALS=1 /dir/of/SIMULIA/Abaqus/Commands/./abaqus'
alias abqlm='sudo /dir/of/SIMULIA/License/./lmgrd -c /dir/of/SIMULIA/License/ABAQUS.lic'
alias abqdocserver='sudo /dir/of/SIMULIA/Documentation/installation_info/v6.10/./startServer'
alias abqlmstat='sudo /dif/of/SIMULIA/License/./lmstat'
alias abqlmdown='sudo /dif/of/SIMULIA/License/./lmdown'
```
在root/.bashrc里加入
```cpp
alias abaqus='XLIB_SKIP_ARGB_VISUALS=1 /dif/of/SIMULIA/Abaqus/Commands/./abaqus'
alias abqlm='/dir/of/SIMULIA/License/./lmgrd -c /dir/of/SIMULIA/License/ABAQUS.lic'
alias abqdocserver='/dir/of/SIMULIA/Documentation/installation_info/v6.10/./startServer'
alias abqlmstat='/dir/of/SIMULIA/License/./lmstat'
alias abqlmdown='/dir/of/SIMULIA/License/./lmdown'
```
# Problems
1. when run abaqus cae, if an error “error while loading shared libraries: libjpeg.so.62: cannot open shared object file: No such file or directory” occurs, install libjpeg62:
sudo apt-get install libjpeg62

2. 如果运行abaqus cae时报错：X Error，那么：
abaqus cae -mesa

# 安装intel Fortran编译器。
安装时注意加入正确的lic文件。

# 将intel Fortran与abaqus连接：
将ifort的路径加入到环境变量中：
```cpp
PATH=$PATH:/opt/intel/bin
```


