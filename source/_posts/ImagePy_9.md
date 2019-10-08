---
title: ImagePy解析：9 -- Filter引擎及其衍生的图像取反插件
tags: [ImagePy]
categories: computational material science 
date: 2019-10-6
---

参考文献：
[Filter 插件](https://github.com/Image-Py/demoplugin/blob/master/doc/chinese/filter.md)
[ImagePy开发文档 —— 滤波器引擎](https://zhuanlan.zhihu.com/p/25474961)

Filter引擎是最重要的一类插件，用于对二维图像进行滤波，也是图像处理中最基础、最普遍的一类应用（语出上面的参考文献）。
这一篇分析Filter引擎的功能，并通过基于它所编写的图像取反插件来深入理解。

# Filter引擎
Filter引擎的基本类结构如下：
```python
class Filter:
    title = 'Filter'
    modal = True
    note = []
    para, view = None, None

    def __init__(self, ips=None):
        #...
    def show(self):
        #...
    def run(self, ips, snap, img, para = None):
        #...
    def check(self, ips):
        #...
    def preview(self, para):
        #...
    def load(self, ips):
        #...
    def start(self, para=None):
        #...
```
可以看出，其基本框架与前面分析的Free引擎类似，毕竟都属于引擎一族。
## 初始化函数
先看它的初始化函数：
```python
def __init__(self, ips=None):
    if ips==None:ips = IPy.get_ips()
    self.dialog = None
    self.ips = ips
```
即会首先通过IPy的get_ips()将ImagePlus图像传给self.ips（IPy是调用了框架的ImageManager管理器），如果没有图像，那么显然这个值就是None。

## note属性
```python
note = []
'all, 8-bit, 16-bit, int, rgb, float, not_channel, not_slice, req_roi, auto_snap, auto_msk, preview, 2int, 2float'
```
note属性决定了该Filter的运行特性，其内的字符串可以分为两类：
（1）一类用于检查时：
all表示该滤镜能处理所有类型的图像、8-bit表示该滤镜能处理8位灰度图像、16-bit表示能处理16位灰度图像、int表示能处理32位整型灰度图像、rgb表示能处理RGB彩色图像、float表示能处理浮点图像，req_roi表示该滤镜需要选区。具体检查原理就是在check()方法中形如：
```python
elif ips.get_imgtype()=='8-bit' and not '8-bit' in note:
    IPy.alert('Do not surport 8-bit image')
    return False
```
（2）一类用于运行时：
preview:是否需要提供预览功能、auto_snap:是否需要执行前自动快照、auto_msk:是否要支持选区、not_channel:是否在处理彩色图像时自动处理每个通道（如不填写为是）、not_slice:是否在处理图像栈的时候询问，从而处理每个层（如不填写为是）、2int:如果精度低于16位整数，是否在处理之前把图像转为16位整数（一些运算会产生负数或溢出）、2float:如果精度低于32位浮点，是否在处理之前把图像转为32位浮点（一些运算需要在浮点上做才能保证质量）（语出上面的参考文献）

## modal属性
当 modal 为 True 时，参数对话框将以模态展示，这也是默认情况，这满足大多数的使用场景。
## para和view属性
核心函数需要用到的参数，以及他们的交互方式，默认为 None，代表不需要交互。
## check()方法
check根据note标识，对当前图像是否存在、选区是否存在、图像类型等进行检查，如果不满足，则调用IPy的alert()弹窗警告。
## load()方法
这里默认是没有做什么操作。但是可以在load()里做一些准备工作，比如获取当前图像的像素直方图等，也可以用作进一步的检查，比如检查图像是否为二值图像。返回 True，则继续执行后续流程，返回 False，则终止。
## start()方法
前面分析Free引擎时已提到，start()方法是引擎的启动函数：
```python
def start(self, para=None, callafter=None):
    ips = self.ips
    if not self.check(ips):return
    if not self.load(ips):return
    if 'auto_snap' in self.note:ips.snapshot()

    if para!=None:
        self.ok(self.ips, para, callafter)
    elif self.view==None:
        if not self.__class__.show is Filter.show:
            if self.show():
                self.ok(self.ips, para, callafter)
        else: self.ok(self.ips, para, callafter)
    elif self.modal:
        if self.show():
            self.ok(ips, None, callafter)
        else:self.cancel(ips)
        self.dialog.Destroy()
    else: self.show()
```
可以看到，首先就是调用check()方法做必要的“体检”，检查项目包括图像本身、ROI、类型检查等。然后执行load()方法。
如果note属性里设置了auto_snap，就会调用ImagePlus对象的snapshot()方法将图像的快照存下来。这里插一句，查看这个方法的源码：
```python
def snapshot(self):
    if self.snap is None:
        self.snap = self.img.copy()
    else: self.snap[...] = self.img
```
发现里面有个省略号的符号，Python中的省略号代表未指定的其余数组维度的占位符，可以把它看作是代表它所放置的位置中所有尺寸的完整切片，因此a[...,0]在3d数组中与a[:,:,0]相同，在4d数组中与a[:,:,:,0]相同。具体的参考文章有：
[What does the Python Ellipsis object do?](https://stackoverflow.com/questions/772124/what-does-the-python-ellipsis-object-do)
[Python 的 Ellipsis 对象](https://farer.org/2017/11/29/python-ellipsis-object/)
然后判断para是否为None，参数 para 如果有值，则直接执行 run。（菜单点击的方式下都是传入的 None，而运行宏的时候，可以传入 para 参数）
如果para为None，则还要经过其他判断，比如是否veiw有值、是否模态对话框等，接着会调用self.show()方法（即生成参数对话框）或者self.ok()方法，具体不同以后再分析。下面以最简单的图像取反操作来分析具体流程。

# 图像取反
图像取反插件是基于Filter引擎的最简单的一个例子：
```python
from imagepy.core.engine import Filter

class Plugin(Filter):
    title = 'Invert Demo'
    note = ['all', 'auto_msk', 'auto_snap']

    def run(self, ips, snap, img, para = None):
        return 255-snap
```
我们看看这个插件的运行路径。
因为该插件没有定义para和view，因此，start()就执行到这里了：
```python
print("self.__class__ = ", self.__class__)
if not self.__class__.show is Filter.show:
    print(" __class__.show is not Filter.show")
    if self.show():
        self.ok(self.ips, para, callafter)
else:
    print(" __class__.show is Filter.show")
    print("Execcute directly~~")
    self.ok(self.ips, para, callafter)
```
然后就判断show()方法是该取反插件自己的还是Filter基类的，因为取反插件没有定义show()，所以它就默认调用的是Filter的show，因此实际就执行的else情形的代码。
插一句，关于__class__的用法，见：
[python中的__class__](https://luobuda.github.io/2015/01/16/python-class/)
那么，就看ok()方法干了啥。其实ok()的主要作用就是根据不同情形决定调用process_one()还是process_stack()，即处理单张图像，还是一个图像栈。注意，ips.snap作为这两个函数的src参数，防止对源图像进行污染。
因为这里是一张图像，所以是调用了process_one()方法，那么看这个方法又调用了process_channels()方法，即对通道进行处理：
```python
def process_channels(plg, ips, src, des, para):
    if ips.channels>1 and not 'not_channel' in plg.note:
        for i in range(ips.channels):
            rst = plg.run(ips, src if src is None else src[:,:,i], des[:,:,i], para)
            if not rst is des and not rst is None:
                des[:,:,i] = rst
    else:
        rst = plg.run(ips, src, des, para)
        if not rst is des and not rst is None:
            des[:] = rst
    return des
```
可以看出，在这里调用了插件的run()方法。
