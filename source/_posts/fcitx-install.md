---
title: linux系统安装fcitx框架和搜狗输入法
tags: [fcitx,linux]
categories: linux
date: 2018-1-19
---
2018-1-19更新：加入自动安装fcitx，以及安装搜狗输入法。

注：在Linux mint系统下，已自带fcitx，但此时安装搜狗后，fcitx并未识别出sogou。
以下是自己的经验：
（1）完全卸载fcitx：
```cpp
sudo apt remove fcitx* 
sudo apt autoremove
```
（2）从搜狗官网下载sogou Linux版，双击安装。
此时其实sogou也能直接用，但经常自己崩溃，所以还要进行下面的步骤。
（3）再次安装fcitx框架：
（3.1）从"菜单"——“Preference”——“input method”中安装fcitx，并安装simplified Chinese语言包。
（3.2）在fcitx配置的input Method中可以看到sogou输入法。


--------------------
以下是手动编译fcitx源码。

参考文章：

（1）http://intijk.com/others/%E6%94%BE%E5%BC%83i-bus%EF%BC%8C%E4%BD%BF%E7%94%A8fcitx.html
（2）http://www.cnblogs.com/jan5/articles/3351574.html

# 下载并编译fcitx源码。

电脑是Ubuntu10.10，这里用的是Fcitx4.0.0版本，太高或太低版本都不好使。
如果系统版本较新，可以直接从fcitx的github页面下载最新的master分支。
注意：旧版的fcitx可以直接make，但新版的fcitx需要cmake，一定注意其依赖关系。

下载列表：https://code.google.com/p/fcitx/downloads/list
然后参照INSTALL文件进行安装。

测试是否安装成功：
输入fcitx -h，如果安装成功，应该能得到帮助文件的，如下：
[root@CentOS ~]# fcitx -h
Usage: fcitx [OPTION]
-d run as daemon(default)
-D don’t run as daemon
-c (re)create config file in home directory and then exit
-n[im name] run as specified name
-v display the version information and exit
-h display this help and exit

# 配置Fcitx为默认输入法

## 方法一：

1、新建配置文件 vim /etc/X11/xinit/xinput.d/fcitx.conf ，内容为：

XIM=fcitx
XIM_PROGRAM=/usr/local/bin/fcitx # 注意这个文件必须存在，请确认它的位置
XIM_ARGS=”-d”  
GTK_IM_MODULE=fcitx   
QT_IM_MODULE =fcitx

2、然后在/etc/alternatives/目录下，将符号链接xinputrc删除，重新建一个：

mv /etc/alternatives/xinputrc /etc/alternatives/xinputrc.bak
ln -s /etc/X11/xinit/xinput.d/fcitx.conf /etc/alternatives/xinputrc

3、最后，注销然后登陆，在菜单 系统—首选项—输入法 里面选择“启用输入法特性”，选择“使用fcitx”，然后“注销”，登录后按“ctrl+空格”就可激活fcitx输入法。

注：如果你使用的桌面是英文环境的，还需要在使用用户的用户目录.bashrc配置文件里添加如下内容：

export LANG=”zh_CN.UTF-8″
export LC_CTYPE=”zh_CN.UTF-8″
export XIM=fcitx
export XIM_PROGRAM=fcitx
export GTK_IM_MODULE=xim
export XMODIFIERS=”@im=fcitx”

## 方法二：

1、新建配置文件：vim /etc/X11/xinit/xinput.d/fcitx，内容为：

XIM=fcitx
XIM_PROGRAM=fcitx
GTK_IM_MOUDLE=fcitx
QT_IM_MOUDLE=fcitx

保存退出，重启电脑

2、查询Fcitx是否开机运行。终端下输入：fcitx，应该是提示：Start FCITX error. Another XIM daemon named SCIM is running？这样就对了，直接到”4“

3、如果没任何提示，则：ln -s /etc/X11/xinit/Xinput.d/fcitx /$HOME/.xinputrc

4、输入： fcitx -nb ，即可看到输入框

默认fcitx启动后，是在后台运行的，因此看不到输入框，用上面的命令就可以调出来了。
ctrl+空格 切换输入法。
配置fcitx输入法修改vim ~/.fcitx/config文件中的相应偏好设置。

—————————————————————————————–
# 可能遇到的问题：

(1)CMake Error at CMakeLists.txt:8 (find_package):
  Could not find a package configuration file provided by “ECM” (requested
  version 1.4.0) with any of the following names:
    ECMConfig.cmake
    ecm-config.cmake
  Add the installation prefix of “ECM” to CMAKE_PREFIX_PATH or set “ECM_DIR”
  to a directory containing one of the above files.  If “ECM” provides a
  separate development package or SDK, be sure it has been installed.
  — Configuring incomplete, errors occurred!

解决方法：到这个页面 https://launchpad.net/ubuntu/+source/extra-cmake-modules/1.4.0-0ubuntu1 下载 extra-cmake-modules_1.4.0.orig.tar.xz

解压后:
	cd extra-cmake-modules-1.4.0
  cmake .
  make
  sudo make install

(2)— Found PkgConfig: /usr/bin/pkg-config (found version “0.26”) 
  — Could NOT find XKBCommon_XKBCommon (missing:  XKBCommon_XKBCommon_LIBRARY XKBCommon_XKBCommon_INCLUDE_DIR) 
CMake Error at /usr/share/cmake-2.8/Modules/FindPackageHandleStandardArgs.cmake:108 (message):
  Could NOT find XKBCommon (missing: XKBCommon_LIBRARIES XKBCommon) (Required
  is at least version “0.5.0”)

处理方法：
	wget http://xkbcommon.org/download/libxkbcommon-0.5.0.tar.xz
  tar xf libxkbcommon-0.5.0.tar.xz
  ./configure —prefix=/usr —libdir=/usr/lib/x86_64-linux-gnu —disable-x11
  make
  sudo make install

  编译libxkbcommon用到yacc，如果没有这个命令，会遇到下面的错误，yacc在 bison软件包中
  sudo apt-get install bison

(3)- package ‘enchant’ not found                                                  
— Could NOT find Enchant (missing:  ENCHANT_LIBRARIES ENCHANT_INCLUDE_DIR ENCHANT_API_COMPATIBLE)

网上搜索并安装enchant包即可。

(4) Could NOT find LibXml2 (missing: LIBXML2_LIBRARIES LIBXML2_INCLUDE_DIR)

网上搜索并安装libxml2包即可。

(5)Could NOT find XkbFile (missing: XKBFILE_LIBRARIES   XKBFILE_MAIN_INCLUDE_DIR)

网上搜索并安装xkbfile包即可。按照软件的文件目录，将相应的文件复制到系统路径中。

(6)如果出现

/usr/local/bin/fcitx-config-gtk: error while loading shared libraries: libfcitx-config.so.4: cannot open shared object file: No such file or directory

只需sudo ldconfig
再不行就加一个符号链接
sudo ln -s /usr/local/lib/libfcitx-config.so /usr/lib/libfcitx-config.so.4
