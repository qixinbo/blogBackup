---
title:  linux系统Tensorflow GPU版安装过程记录
tags: [linux, tensorflow]
categories: machine learning 
date: 2019-7-26
---

# Prerequisites
Linux系统还是建议选Ubuntu系的，首先是驱动支持得全面，再者是Ubuntu源很给力，安装软件不费劲。
这里选择的Linux发行版是Linux Mint 19，它也是Ubuntu系的，界面比原生的Ubuntu要舒服，且软件源也是用的Ubuntu的。

# 安装Nvidia GPU驱动
首先确认系统是否有Nvidia GPU驱动：
```cpp
nvidia-smi
```
如果显示没有驱动的话，则需要下载安装驱动，可以有多种方式安装，最简单的一种是通过Ubuntu或Linux Mint的Driver Manager。打开该管理器，等待更新缓存后，就会有可用的Nvidia驱动下载，建议下载最新的，比如最新的TF就需要驱动版本在410以上。
也可以直接上Nvidia官网上下载，链接是：
https://www.nvidia.com/Download/index.aspx?lang=en-us
但是安装可能很费劲，比如手动禁用开源的Nvidia驱动nouveau等。
下面是一篇很详细的安装方法，见：
[Linux安装NVIDIA显卡驱动的正确姿势](https://blog.csdn.net/wf19930209/article/details/81877822)

# 安装CUDA工具集
CUDA的版本与TF的版本要对应好，这里要安装的是TF-1.14，它对应的是CUDA10，所以这里下载并安装CUDA10。
下载链接是：
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
依次选择系统、架构、发行版、发行版版本、安装类型后，就会出现下载链接及安装步骤，如：
```cpp
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```
然后再将可执行文件和链接库的路径加入到环境变量中：
```cpp
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
```
如果有其他版本的cuda安装过，比如cuda9.1，那么需要首先完全卸载才行，否则会冲突：
```cpp
sudo apt remove cuda
sudo apt autoclean
sudo apt remove cuda*
sudo rm -rf /usr/local/cuda-9.1
sudo rm -rf /usr/local/cuda
sudo find / -name cuda-9* (then remove all the files related to cuda-9.1)
```

# 安装cuDNN
这里也是安装最新的，要大于7.4.1。
下载链接（要注册）：
https://developer.nvidia.com/rdp/form/cudnn-download-survey
选择好与前面cuda版本相对应的版本。
如果选择的是cuDNN Runtime Library for Ubuntu18.04 (Deb)。
则直接安装即可：
```cpp
sudo dpkg -i xxx.deb
```
如果选择的是cuDNN Library for Linux，则解压后：
```cpp
sudo cp cuda/include/* /usr/local/cuda-10.1/include
sudo cp cuda/lib64/* /usr/local/cuda-10.1/lib64
sudo /sbin/ldconfig
```
# 安装tensorflow-gpu
```cpp
conda create -n tf python=3.7
source activate tf
pip install tensorflow-gpu
```

# 测试
```cpp
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
