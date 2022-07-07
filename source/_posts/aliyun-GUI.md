---
title: 使用VNC安装云服务器图形界面的过程记录
tags: [Aliyun]
categories: coding 
date: 2022-7-6
---

# 简介
一般来说，购买云服务器的用途是将其作为服务，比如构建web网站、博客、小程序、API服务等。这些场景都不需要服务器操作系统的图形界面，直接将相应的服务部署在云服务器上即可；而当前遇到的一个问题是：仍然需要通过云服务器来提供服务（即提供API接口），但背后的算法却依赖一个图形界面的桌面端软件。
默认的云服务器只有终端命令行操作，这是为了轻便和资源的高利用率，因此需要在购买的云服务器上安装操作系统的图形界面，以可运行具有图形界面的桌面端软件。
本文就记录通过VNC安装并连接云服务器的图形界面的过程。
# 配置和价格

配置1（轻量应用服务器）：
- 地域和可用性：华北2（北京）
- 镜像：系统镜像`Ubuntu 20.04`
- 套餐配置：`vCPU`：`2`核；内存：`2GB`；每月流量：`800GB`、`ESSD`：`50GB`；限峰值带宽：`4Mbps`。
- 价格：`90元/月`

买了配置1后，发现不够用，又买了一个配置2（换成了云ECS）：
- 地域和可用性：华北2（北京）
- 镜像：系统镜像`Ubuntu 20.04`
- 套餐配置：`vCPU`：`4`核；内存：`8GB`；带宽：`1MB`、`ESSD`：`40GB`。
- 价格：`283元/月`

# 搭建图形界面
轻量应用服务器和云ECS提供的Linux系统均为命令行界面。如果希望通过图形界面管理操作系统，可以使用VNC（Virtual Network Console）实现。
参考文档在[这里](https://help.aliyun.com/document_detail/59330.html)。

## 添加防火墙规则
首先需要在Ubuntu服务器的防火墙中放行VNC服务所需的5900和5901端口。
教程在[这里](https://help.aliyun.com/document_detail/59086.html)。(ECS在安全组规则中设置)

## 搭建图形界面
（1）远程连接服务器
具体操作，请参见通过[管理控制台远程连接Linux服务器](https://help.aliyun.com/document_detail/59083.htm)。
（2）切换root用户：
```sh
sudo su root
```
（3）安装软件包：
（3.1）更新软件源：
```sh
apt-get update
```
（3.2）安装所需的软件包：
软件包包括系统面板、窗口管理器、文件浏览器、终端等桌面应用程序。
```sh
apt install gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal ubuntu-desktop
```
（4）配置VNC
（4.1）安装VNC
Ubuntu 20.04：运行以下命令，安装VNC：
```sh
apt-get install tightvncserver
```
（4.2）启动VNC：
```sh
vncserver
```
第一次启动需要设置VNC的登录密码，输入VNC登录密码和确认密码，并在以下提示中输入`n`，并按`Enter`。
```sh
Password:
Verify:
Would you like to enter a view-only password (y/n)? n
```
注意：如果自定义的密码位数大于8位，系统默认只截取前8位作为您的VNC登录密码。
命令行回显如下信息，表示VNC启动成功：
```sh
New 'X' desktop is iZ2zebq9cg2jxxxxx:1
```
（4.3）运行以下命令，备份VNC的xstartup配置文件：
```sh
cp ~/.vnc/xstartup ~/.vnc/xstartup.bak
```
（4.4）修改该配置文件：
```sh
vi ~/.vnc/xstartup
```
将其内容修改为如下内容：
```sh
#!/bin/sh
export XKL_XMODMAP_DISABLE=1
export XDG_CURRENT_DESKTOP="GNOME-Flashback:GNOME"
export XDG_MENU_PREFIX="gnome-flashback-"
gnome-session --session=gnome-flashback-metacity --disable-acceleration-check &
```
（5）重新启动VNC
（5.1）关闭已启动的VNC：
```sh
vncserver -kill :1
```
（5.2）启动一个新的VNC（端口号仍为1）：
```sh
vncserver -geometry 1920x1080 :1
```

## 测试访问
（1）安装VNC Viewer。
可以访问[VNC Viewer官网](https://www.realvnc.com/en/connect/download/viewer/)获取下载链接以及安装方式。
（2）使用VNC Viewer连接访问。
其中最重要的一个配置项是`VNC Server`，需要输入`<Ubuntu服务器公网IP>:<VNC的端口号>`，例如：`114.55.XX.XX:1`。