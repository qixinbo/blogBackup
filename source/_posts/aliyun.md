---
title: 阿里云服务器：购买、连接和部署
tags: [Aliyun]
categories: coding 
date: 2019-5-8
---
这是记录一下第一次跟阿里云服务器打交道的过程。。
# 购买
阿里云服务器有两种：
（1）轻量应用服务器；
（2）云服务器ECS
两者的区别可以参考如下：
- [ECS 还是轻量应用服务器，看完评测你就知道了](https://yq.aliyun.com/articles/202688)
- [轻量应用服务器和ecs云服务器怎么选?](https://www.chukuangren.com/qingliang.html)

这里购买了Ubuntu服务器，所以下面的操作都是适用于Linux系统，关于Windows系统的配置详见阿里云文档。
配置为：2核 CPU | 2GB 内存 | 80GB SSD | 30Mbps 限制峰值带宽 | 3TB 每月流量 | 香港，
价格为：每月67元。

# 连接
有三种连接方式：

## 浏览器直连：

直接通过浏览器点击网页上的“远程连接”，即可连接到购买的服务器。优点是不需要设置，无需密码，缺点是没法与本地互动，如传输文件等。

## SSH客户端使用账号密码进行连接
这里的SSH客户端比如Windows系统下的Putty、Linux系统下的各种终端等。
首先要先设置管理员密码，具体方法见：
[重置服务器密码](https://help.aliyun.com/document_detail/60055.html?spm=a2c4g.11186623.2.15.5e512eebolLgUZ)
然后SSH客户端中配置Host、端口和密码等。
具体方法见：
[通过账号密码方式连接](https://help.aliyun.com/document_detail/59083.html?spm=5176.10173289.107.4.4db62e774lYrrT)
优点是能在本地操作服务器，且与本地进行互动，如下面的传输文件等。这种使用账号密码的方式的安全性要比下面的使用密钥要低一些。

## SSH客户端使用密钥进行连接：
这种方式要比第二步的使用账号密码多了个密钥验证，所以更安全。
首先要先设置密钥并下载私钥，然后在各种SSH客户端中加载该私钥。具体的设置方式如下：
[使用密钥方式连接](https://help.aliyun.com/document_detail/59083.html?spm=5176.10173289.107.4.4db62e774lYrrT)
注意，开启密钥验证后，默认自动禁止使用账号密码登录。如需开启，也见上面链接。

# 传输文件
本地为Linux系统，可以在终端使用scp命令。
本地为Windows系统，推荐使用有图形化界面的WinSCP软件，下载链接为：
[WinSCP](https://winscp.net/eng/index.php)
跟上面配置SSH相同（注意如果启用了密钥验证，在WinSCP中也要通过Advanced加载私钥文件）。
然后就可以通过自由拖拽在本地和服务器之间方便地传输文件。

# 部署
通过pip安装必要的包，比如flask、opencv-python、opencv-contrib-python、plotly等。
通过gunicorn来启动flask应用，使用方法见：
[flask下 gunicorn在Python中的使用](https://www.jianshu.com/p/e8d125372ca5)
[Gunicorn-配置详解](https://blog.csdn.net/y472360651/article/details/78538188)
（注意这个地方踩了一个坑，主要是因为自己对网络编程不熟：gnicorn会使用多线程，而自己编写的程序中使用的是传统的Python的全局变量，而没有引用flask的那些与request、context相关的全局变量，导致在程序运行时每点一下按钮，会出现不同结果。）
一个较详细的部署教程见：
[通过Gunicorn部署flask应用（阿里云服务器：Ubuntu 16.04）](https://juejin.im/post/5a5a1408518825733060e232)
中间出现了以下几个问题，并给出解决方法：
（1）导入opencv时，报错：
```python
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
ImportError: libXrender.so.1: cannot open shared object file: No such file or directory
```
原因是：服务器在安装时没有安装图形库。解决方法：

```python
apt-get install libsm6 libxrender1
```
参考见下面链接：
[服务器opencv-python使用问题及解决](https://my.oschina.net/u/2422458/blog/1815712)

（2）在执行到imgproc时，出现：
```python
TypeError: Expected cv::UMat for argument 'M'
```
怀疑是python2.7的锅，所以重新配置了pipenv，使用python3环境就好了。以下是具体配置过程：
（a）安装pip3
首先需要升级一下，否则阿里云找不到pip3：
```python
apt-get update
```
然后安装pip3：
```python
apt-get install python3-pip
```
然后使用pip3安装pipenv
```python
pip3 install pipenv
```
使用pipenv创建虚拟环境时，指定python版本：
```python
pipenv --python 3 install
```
（3）启动了flask服务器，但是外部无法连接，总是“time out”：
这是因为阿里云默认只开启几个端口，如果需要额外的端口，需要自己去防火墙那开启。
参考见：
[在ecs上启动flask应用后，无法通过公网ip访问网站](https://yq.aliyun.com/ask/57796)
