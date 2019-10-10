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
那么，就看ok()方法干了啥。其实ok()的主要作用就是根据不同情形决定调用process_one()还是process_stack()，即处理单张图像，还是一个图像栈。
查看一下这两者的不同。先看两者的原始声明：
```python
def process_one(plg, ips, src, img, para, callafter=None)
def process_stack(plg, ips, src, imgs, para, callafter=None)
```
再看它们实际调用时：
```python
process_one(self, ips, ips.snap, ips.img, para, callafter)
process_stack(self, ips, ips.snap, ips.imgs, para, callafter)
```
分析一下实参和形参的对应关系：
（1）self就是当前插件的指针，传给了它们俩的plg形参，因此可以在plg中调用插件的各个属性和方法；
（2）ips作为整个图像的封装，传给了ips形参，注意ips与下面的src和img的关系，ips是一个统一封装，src和img仅是它的一部分；
（3）ips.snap就是对图像的快照，如前所述，如果在note中设置了auto_snap，则提前就将图像copy一份到snap中，它传给了src形参；
（4）ips.img或ips.imgs充分反映了两个函数的不同点：如果是处理一张图像，则传入ips.img，如前所述，这是调用了ImagePlus的img属性，它与当前的游标self.cur有关；如果是处理一个图像栈，则传入ips.imgs。将它们传给了img或imgs形参。注意src与img的区别，src是图像在处理前的一个copy，在代码中也是使用的copy()函数，所以src创建了一个新的对象，与源图像已没有关系；而img或imgs还是原来对象的引用。这里插一句，拷贝有深拷贝和浅拷贝的区别，同时python的拷贝与numpy的拷贝也有不同（这里查看snapshot()源码可知，snapshot是作用于单张图像上，所以是numpy的copy()，而不是python的copy包的copy()）：
[Python numpy 中的 copy 问题详解](https://blog.csdn.net/u010099080/article/details/59111207)
[理解 Python 引用、浅拷贝和深拷贝](http://wsfdl.com/python/2013/08/16/%E7%90%86%E8%A7%A3Python%E7%9A%84%E6%B7%B1%E6%8B%B7%E8%B4%9D%E5%92%8C%E6%B5%85%E6%8B%B7%E8%B4%9D.html)
（5）para就是参数，注意它与self.para的区别，将它传给了para形参；
（6）callafter：目前看就是默认None，没有对它进行改变，也是正常传给了para形参。

这两个函数又都调用了process_channels()方法，即对通道进行处理。
先来看该方法的声明：
```python
def process_channels(plg, ips, src, des, para)
```
再来看process_one()和process_stack()调用它时：
```python
# 对于一张图像
rst = process_channels(plg, ips, src, buf if transint or transfloat else img, para)
# 对于一个图像栈
rst = process_channels(plg, ips, src, buf if transint or transfloat else i, para)
```
这样的实参与形参对应时，需要注意的就是buf实参与des形参的对应，如果note中表明了2int或2float，即需要转成int型或float型，就要先调用numpy的astype转换一下数据格式。然后实际process_channels中的des就是之前的self.img或self.imgs。

再来看取反插件中的run()是怎样与Filter引擎进行对应的。对于取反插件：
```python
def run(self, ips, snap, img, para = None):
    return 255-snap
```
在Filter引擎中（以单通道为例）：
```python
rst = plg.run(ips, src, des, para)
```
对应关系就是ips传给ips，src传给snap，des传给img，para传给para。
经过上面分析，这四个实参：ips是通过IPy调用ImageManager获得的，src和des都是ips中的属性，para追根溯源是从取反插件的para属性中传入的，即下面这个赋值语句（即如果是菜单点击的话，在插件中直接将para=None就是将插件的para属性传给了run()方法）：
```python
def ok(self, ips, para=None, callafter=None):
    if para == None:
        para = self.para
```
所以对于插件的编写，只需要在重载run()方法时正确地写上这些参数，不需要考虑怎样给它们赋值，而是只定义逻辑操作即可，比如这里的直接用255减去snap的值。
