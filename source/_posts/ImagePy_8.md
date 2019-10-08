---
title: ImagePy解析：8 -- 由新建图像谈起（引出IPy、ImagePlus、Canvas）
tags: [ImagePy]
categories: computational material science 
date: 2019-10-5
---

参考文献：
[创建图像](https://github.com/Image-Py/demoplugin/blob/master/doc/chinese/free.md#创建图像)
[ImagePy开发文档 —— 常用汇总](https://zhuanlan.zhihu.com/p/25474453)
[ImagePy开发文档 —— 图像封装类](https://zhuanlan.zhihu.com/p/25474501)

# 新建图像的插件
最简单的新建图像的插件代码如下：
```python
from imagepy.core.engine import Free
from imagepy import IPy
import numpy as np

class Plugin(Free):
    title = 'New Image Demo'
    para = {'name':'new image','w':300, 'h':300}
    view = [(str, 'name', 'name',''),
            (int, 'w', (1,2048), 0,  'width', 'pix'),
            (int, 'h', (1,2048), 0,  'height', 'pix')]

    def run(self, para = None):
        imgs = [np.zeros((para['h'], para['w']), dtype=np.uint8)]
        IPy.show_img(imgs, para['name'])
```
该插件的写法可以参考之前的解析文章，在run()方法中接收para的宽度w和高度h的数值，然后通过numpy的zeros函数生成了一个全是0的array数组，注意imgs变量又对该array数组加入了一个方括号，即将其转化为python的列表（其实从后面可以看出，该列表的长度就是整个图像栈有多少个slices），所以imgs的具体数值就是：
```python
imgs =  [array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)]
```
然后将imgs和用户输入的图像名称传入IPy的show_img()方法。到这里就可以看出，ImagePy是将传统的numpy数组（一般的图像库都是将图像读作numpy数组）进行了进一步的封装，这个封装类就是ImagePlus，后面再具体分析ImagePlus。

# IPy常用工具集
上面将imgs传入了IPy的show_img()方法，实际IPy是一些常用功能的汇总，比如展示一幅图像，获取当前图像窗口等功能，其实这些功能是分布在各个功能组建中的，IPy只是重新引用和整理，并且IPy处于最顶层目录，随处都可以方便的引用这样以来，提高了开发效率。（语出上面的参看文献）
具体看show_img()方法：
```python
def showimg(imgs, title):
    print('show img')
    from .core import ImagePlus
    ips = ImagePlus(imgs, title)
    showips(ips)

pub.subscribe(showimg, 'showimg')
def show_img(imgs, title):
    if uimode()=='no':
        from .core import manager, ImagePlus
        from .ui.canvasframe import VirturlCanvas
        frame = VirturlCanvas(ImagePlus(imgs, title))
    else:wx.CallAfter(pub.sendMessage, 'showimg', imgs=imgs, title=title)
```
可以看出，该方法又通过Publisher/Subscriber机制调用了showimg()方法，这里就引出了ImagePlus封装类：
```python
ips = ImagePlus(imgs, title)
showips(ips)
```
ImagePlus接收imgs和title，组装成了ImagePlus类型的ips。

# ImagePlus图像封装类
ImagePy 采用 Numpy 数组作为图像数据结构，Numpy 非常强大，但是要让它能够支持我们的全部特性（索引色支持，撤销，选区支持，多通道支持，图像栈支持），还需要一些其他的数据给予必要的辅助，因而我们有了ImagePlus类（语出上面的参考文献）。
下面看看ImagePlus收到imgs后干了什么：
```python
def set_imgs(self, imgs):
    self.is3d = not isinstance(imgs, list)
    self.scrchanged = True
    self.snap = None
    self.imgs = imgs
    self.size = self.imgs[0].shape[:2]
    self.height, self.width = self.size
    self.imgtype = get_img_type(self.imgs)
    if self.imgs[0].ndim==2: self.channels = 1
    else: self.channels = self.imgs[0].shape[2]
    self.dtype = self.imgs[0].dtype
    if self.dtype == np.uint8: self.range = (0, 255)
    else: self.range = self.get_updown('all', 'one')
    if self.dtype == np.uint8:
        self.chan_range = [(0, 255)] * self.channels
    else: self.chan_range = self.get_updown('all', 'all', step=512)
    self.chan = (0, [0,1,2])[self.channels==3]
```
针对于上述图像，经过初始化后，其对应的ImagePlus的各个属性值为：
```python
self.is3d =  False
self.size =  (300, 300)
self.type =  8-bit
self.imgs[0].shape =  (300, 300)
self.channels =  1
self.dtype =  uint8
self.chan =  0
self.chan_range =  [(0, 255)]
self.range =  (0, 255)
```
如果是使用ImagePy源码里的new_plg，即：
```python
import wx,os
from imagepy.ui.canvasframe import CanvasFrame
import numpy as np
from imagepy import IPy

from imagepy.core.engine import Free

class Plugin(Free):
    title = 'New'
    para = {'name':'Undefined','width':300, 'height':300, 'type':'8-bit','slice':1}
    view = [(str, 'name', 'name', ''),
            (int, 'width',  (1,10240), 0,  'width', 'pix'),
            (int, 'height', (1,10240), 0,  'height', 'pix'),
            (list, 'type', ['8-bit','RGB'], str, 'Type', ''),
            (int, 'slice',  (1,2048), 0,  'slice', '')]

    #process
    def run(self, para = None):
        w, h = para['width'], para['height']
        channels = (1,3)[para['type']=='RGB']
        slices = para['slice']
        shape = (h,w,channels) if channels!=1 else (h,w)
        imgs = [np.zeros(shape, dtype=np.uint8) for i in range(slices)]
        IPy.show_img(imgs, para['name'])
```
且此时设定Type为RGB、slice为2，那么此时图像的各个ImagePlus属性值为：
```python
self.is3d =  False
self.size =  (300, 300)
self.type =  rgb
self.imgs[0].shape =  (300, 300, 3)
self.channels =  3
self.dtype =  uint8
self.chan =  [0, 1, 2]
self.chan_range =  [(0, 255), (0, 255), (0, 255)]
self.range =  (0, 255)
```
这里插一句，上面的插件的run()函数中判断channels的值的语句为：
```python
channels = (1,3)[para['type']=='RGB']
```
这是Python的一种特殊的条件表达式，称为元组条件表达式，即(X, Y)[C]，如果C为False，则返回X，如果C为True，则返回Y，这是因为在Python中，True等于1，而False等于0，这就相当于在元组中使用0和1来选取数据。
元组条件表达式在这里是没有问题的，但下面的两篇文章有个提醒，在使用这样的条件表达式时，需要注意如下问题，即元组是提前构建数据，则X和Y会提前运行，然后再用True(1)/False(0)来索引到数据。如果X和Y是比较耗资源的运算，则会开销比较大，另外如果有异常运算，也会直接报错。此时建议使用X if C else Y的条件表达式形式。
[Python中的三元操作](https://oldj.net/blog/2010/10/17/python-ternary-operator/)
[三元运算符](https://wiki.jikexueyuan.com/project/interpy-zh/ternary_operators/ternary_operators.html)

回到两张图像的对比，可以看出对于一张8-bit的只有一个slice的图像与一张RGB的有两个slices的图像来说，两者的图像类型type、图像矩阵的shape、通道标识chan、通道范围chan_range都是不同的。
这里再插一句，对于ImagePlus类的range和img属性，ImagePy采取了既非普通属性、又非方法调用的方式来设定和读取，即property装饰器的方法，这样的好处有两点：（1）相比于普通的属性，这样可以进行参数检查，保证安全性和合理性；（2）相比于set()和get()这样的配对函数，这样的方式又更加简洁。其实原理就是property装饰器将一个getter方法变成了属性，同时生成了一个setter装饰器，负责将setter方法变成属性赋值。对比img和range这两个属性，可以发现，range有property和相应的setter装饰器，而img只有property装饰器，这表明range是可读可写，而img是只读方式。这里的知识点见廖大爷的如下教程：
[使用@property](https://www.liaoxuefeng.com/wiki/1016959663602400/1017502538658208)

另外，这个img属性的具体定义：
```python
@property
def img(self):return self.imgs[self.cur]
```
可以看出，img具体调用的是哪个图像还与当前的游标有关。

# Canvas画布
继续回到IPy，可以看出构造了ImagePlus类型的ips后，就继续调用IPy的showips()函数：
```python
def showips(ips):
    if uimode()=='ipy':
        from .ui.canvasframe import CanvasPanel
        canvasp = CanvasPanel(curapp.canvasnb)
        canvasp.set_ips(ips)
        curapp.canvasnb.add_page( canvasp, ips)
        #canvasp.canvas.initBuffer()
        curapp.auimgr.Update()

    elif uimode()=='ij':
        from .ui.canvasframe import CanvasFrame
        frame = CanvasFrame(curapp)
        frame.set_ips(ips)
        frame.Show()
```
根据当前的UI界面是ImagePy风格还是ImageJ风格，来以不同的方式显示图像。其实ImageJ风格的CanvasFrame也是调用了ImagePy自定义的CanvasPanel，它是一张张图像分成不同的窗口显示，而ImagePy风格的显示方式是以标签页的形式聚合显示不同图像，即传入CanvasPanel的是curapp.canvasnb，查看这个canvasnb即可知道，它是CanvasNoteBook类型的数据，其派生自wx.lib.agw.aui.AuiNotebook。
CanvasPanel又调用了同级目录canvas下的Canvas类，前面已经提到，该类已经从ImagePy中解耦，可以作为独立组件运行。
Canvas类的运行过程乍一看不好理解，其实它是利用了wxPython的SizeEvent：
```python
    self.bindEvents()
def bindEvents(self):
    for event, handler in [ \
            (wx.EVT_SIZE, self.on_size)]:
        self.Bind(event, handler)
```
即当窗口绘制时，就会调用on_size()函数，然后就会执行一系列的绘图等操作。
ImagePy的Canvas是个定制的wxPython Panel，它显示图像的基本原理是利用wxPython的GDI接口，具体是wxPython的BufferedDC，以及其DrawBitmap()函数。
