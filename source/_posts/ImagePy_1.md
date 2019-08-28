---
title: ImagePy解析：1 -- 简介、安装和启动
tags: [ImagePy]
categories: computational material science 
date: 2019-3-17
---
# ImagePy简介
“ImagePy是一款基于 Python 的可扩展图像处理框架，融合了ImageJ与Python的优势，是一个轻量级的、可扩展的图像处理框架。”（语出ImagePy官网：http://www.imagepy.org/about/）
ImagePy作为一个GUI框架，可以快速接入opencv、scikit-image、mayavi等python的第三方库，因此，在功能性和易用性上都有很好的表现。
ImagePy的作者是闫霄龙yxdragon，目前是成都坐标创意科技有限公司的CEO，真牛人~~
项目的GitHub主页是：[https://github.com/Image-Py/imagepy](https://github.com/Image-Py/imagepy)
GitHub上还有基于ImagePy的更多的扩展，比如opencv扩展、itk扩展、IBook扩展、海冰影像分析项目：[https://github.com/Image-Py](https://github.com/Image-Py)
项目在知乎上的介绍：[https://zhuanlan.zhihu.com/imagepy](https://zhuanlan.zhihu.com/imagepy)
项目在image.sc上的讨论频道是：[https://forum.image.sc/tags/imagepy](https://forum.image.sc/tags/imagepy)
闫霄龙的一个视频直播分享：[B站](https://www.bilibili.com/video/av39218282)或[腾讯视频](https://v.qq.com/x/page/b08204k4bjb.html)

# 安装
如果想直接使用的话，可以直接下载release版本：
[https://github.com/Image-Py/imagepy/releases](https://github.com/Image-Py/imagepy/releases)
如果想二次开发的话，可以自己搭建python环境，并通过pip或conda安装必要的包，然后直接clone源码：
[https://github.com/Image-Py/imagepy](https://github.com/Image-Py/imagepy)

# 设计架构
如图：
![](https://ws4.sinaimg.cn/large/006Xmmmggy1g5nhfqnr12j30qf0e20vv.jpg)

## 展示层
即软件界面，包括菜单栏、工具条、状态栏、图像窗口等组成，ImagePy是基于wxPython来开发GUI。

## 管理层
即系统底层，包括很多管理器，如插件管理器、工具管理器、图像窗口管理器等。

## 逻辑层
即交互逻辑，将用户的界面操作与系统底层联系起来。同时该层也包括两种加载器：插件加载器PluginLoader和工具加载器ToolsLoader，来在软件初始化时来生成菜单栏和工具栏。

# 设计思想
ImagePy采用插件式设计思想，根据功能的形式，制定了多种引擎模板engines，即基类（比如Filter、Free、table、tool等基类）。具体功能都是它们的子类，按照一定方式组织起来，然后由相应的加载器Loader和管理器Manager维护。某种意义上说，功能即文件，菜单即目录。

# 物理文件组织
注意，这里是ImagePy 2019 7月份master分支的结构，不同版本可能有差别。
设计架构和设计思想要体现在具体的物理文件上，这里就体现了上面的“功能即文件，菜单即目录”。更具体地说，Python文件就是一个功能模块，或称功能包；ImagePy通过解析文件目录来生成GUI界面的菜单和目录。
## core包

### wraper包
里面包含了两个ImagePy自定义的图像封装类ImagePlus和表格封装类TablePlus，用于更灵活地操纵图像和表格。
### loader包
里面定义了如何解析插件Plugins和工具tools的函数。
### engine包
里面定义了前面所述的各种引擎模板，用作其他类的基类。
具体的比如有：
> Filter:滤波器基类，用于图像处理类的插件，它将帮助你做一系列的工作，比如处理多通道和图像栈、支持选区等。
> Simple:简单插件基类，不同于Filter，它处理的对象不是图像，而是ImagePlus整体（比如处理图像栈或 ROI）
> Free:自由插件基类，其运行不需要依赖图像，可以在任何情况下执行（比如图像的打开功能）
> Macros:宏插件基类，它可以由一组命令字符串构成，并依次执行。
> Tool:工具基类，定义一组鼠标事件的处理函数，可以在Canvas的鼠标事件中被回调。

### roi包
包含各种选区操作，如点选区、线选区、多边形选区等。

### manager包
包含前面所述的各种管理器，如插件管理器等。

## ui包
包含与UI操作相关的功能包。如：
> Canvas:负责绘制ImagePlus，这其中包括绘制图像数据、绘制选区、以及 Mark 标记。
> CanvasFrame:Canvas的包裹类，使之能以一个窗口的形式展示。
> ParaDialog:参数交互类，图像处理的诸多函数有各种的参数，ParaDialog采用数据驱动视图的方式，可以自动生成各类交互对话框。
> pluginloader:解析菜单栏
> toolsloader：解析工具条

## menus包
存放全部的插件，可以按照任意目录解析

## tools包
存放所有的工具。

## IPy
对一些常用功能的引用类，可以通过Ipy访问ImagePy的许多常用功能，比如打开图像、输出日志等。


# 启动分析
Windows下双击ImagePy.bat或Linux下运行ImagePy.sh，皆可启动ImagePy主程序，两者都是执行：

```cpp
python -m imagepy
```
ImagePy软件是以imagepy这个包来组织代码的，所以上面命令以模块形式运行imagepy，会首先运行该包的__init__.py文件，然后再运行__main__.py文件，这里面会有几个添加并识别路径的语句，保证程序能够被顺利找到。具体地：在__init__.py文件中先将系统路径切换到imagepy路径下，这样就可以通过相对路径来引用内部模块、图片等静态文件，然后在__main__.py中通过相对路径将上一层路径加入到sys.path中，从而让python能找到imagepy这个包，然后调用imagepy.show()这个方法开始真正执行具体的逻辑。

具体地，ImagePy使用wxPython作为GUI框架：

```cpp
app = wx.App(False) #创建application
ImagePy(None).Show() #显示
app.MainLoop() #进入主循环，等待交互

```

ImagePy类是自定义的继承自wx.Frame的框架。
