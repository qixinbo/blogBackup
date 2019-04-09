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
