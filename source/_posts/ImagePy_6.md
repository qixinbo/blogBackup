---
title: ImagePy解析：6 -- wxPython GDI绘图和FloatCanvas
tags: [ImagePy]
categories: computational material science 
date: 2019-9-7
---

序言：
本文的成文与ImagePy没有直接关系，但有相当大的间接关系。起因是我在自己的程序中想集成一个可以自由缩放且绘点的画布工具，发现ImagePy的canvas能够很好地满足需求，但无奈ImagePy的源码看不懂，自己想抽离这个canvas也没抽离出来。后经霄龙提醒，发现wxPython的FloatCanvas也有这些基本功能，所以就有了对FloatCanvas和wxPython GDI绘图进行学习的本文。
更让人欣喜的是，在我弄懂了FloatCanvas的用法、且实现一个基础demo之际，霄龙将ImagePy中的canvas迅速剥离了出来，可以单独调用，同时提供了掩膜模式，可以更好地进行像素标注。官方一出手，就知有没有！
终于可以愉快地进行数据标注了～～


参考文献：
[wxPython graphics](http://zetcode.com/wxpython/gdi/)
[FloatCanvas2 tutorial](https://raw.githubusercontent.com/svn2github/wxPython/master/3rdParty/branches/FloatCanvas/SOC2008_FloatCanvas/floatcanvas2/docs/tutorial/FloatCanvas%20Tutorial.txt)
[wx.lib.floatcanvas](https://wxpython.org/Phoenix/docs/html/wx.lib.floatcanvas.html)
[Phoenix/samples/floatcanvas/](https://github.com/wxWidgets/Phoenix/tree/master/samples/floatcanvas)

# wxPython GDI绘图概述
GDI是图形设备接口Graphics Device Interface的缩写，主要任务是负责系统与绘图程序之间的信息交换，GDI的出现使程序员无需要关心硬件设备及设备正常驱动，就可以将应用程序的输出转化为硬件设备上的输出和构成，实现了程序开发者与硬件设备的隔离，大大方便了开发工作。上述介绍见[百度百科-GDI](https://baike.baidu.com/item/GDI/3123145)。
从编程角度来看，GDI是一组与图像打交道的类和方法，其包括二维矢量图、字体和位图。
在绘图之前，需要先创建一个设备上下文Device Context对象。在wxPython中，称为wx.DC，但它不能直接调用，需要使用它的一系列的派生类才行。比如：
（1）wx.ScreenPC：可以在屏幕的任意位置绘图；
（2）wx.ClientDC：用来在窗口的工作区绘图，即去除窗口的标题和边框；
（3）wx.PaintDC：也在窗口的工作区绘图，但与ClientDC的不同点在于：PaintDC仅能在wx.PaintEvent中使用，而ClientDC不能在wx.PaintEvent中不能；
（4）wx.MemoryDC：用来在位图上绘图；
（5）wx.PostScriptDC：用于输出PostScript文件；

此外，还有wx.BufferedDC、wxBufferedPaintDC等。

## wxPython GDI绘图入门
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZetCode wxPython tutorial
This program draws a line in
a paint event.
author: Jan Bodnar
website: zetcode.com
last edited: May 2018
"""
import wx

class Example(wx.Frame):
    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.SetTitle("Line")
        self.Centre()

    def OnPaint(self, e):
        dc = wx.PaintDC(self)
        dc.DrawLine(50, 60, 190, 60)

def main():
    app = wx.App()
    ex = Example(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
```

## 设备上下文的属性
设备上下文有很多属性，比如画刷、画笔和字体。
（1）画刷wx.Brush是用来填充某个形状的背景，它需要颜色color和样式style，比如设置某一颜色wx.Brush('#c56c00')；
（2）画笔wx.Pen是用来绘制形状的轮廓，它需要colour、width和style，比如wx.Pen('#4c4c4c', 1, wx.LONG_DASH)；
（3）字体wx.Font是用来定义文本的外观。

可以通过设定这些属性来控制所绘图形的样式，比如DrawPoint时点的颜色就是使用了当前画笔的颜色。
wxPython的GDI绘图需要了解很多非常细节的地方，且其通常绘制在一个基础的panel上，能实现基本功能，但功能不强大。下面的FloatCanvas是一个集成在wxPython中的第三方库，试图进一步提高wxPython的绘图功能

# FloatCanvas概述
FloatCanvas是一个用来在任意坐标系中绘制矢量图的窗口类，由Chris Barker编写和维护，其目的是为了提供一个简便地在屏幕上绘图的工具，有很多优点：
（1）有一整套完备的鼠标事件和回调机制用来响应用户的点击等操作，且很容易地转换在不同坐标系中转换坐标；
（2）提供虚拟的无限缩放、平移功能，为用户处理好重绘等操作；
（3）用户无需了解wxWindows的画刷、画笔及色彩等，即FloatCanvas比wx.DC容易使用得多。
该类存在于wx.lib.floatcanvas包中，该包还有另外一个重要模块NavCanvas.py，其是对FloatCanvas的一个封装，额外提供了一个操控画布的工具条，比如缩放、平移等；另外，Resources.py模块中包含了FloatCanvas所需的一些资源，如图标等。

## 绘制图形
```python
import wx
from wx.lib.floatcanvas import FloatCanvas

class DrawFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        self.Canvas = FloatCanvas.FloatCanvas(self, -1,
                                     size=(500, 500),
                                     ProjectionFun=None,
                                     Debug=0,
                                     BackgroundColor="White",
                                     )

        # add a circle
        cir = FloatCanvas.Circle((10, 10), 100)
        self.Canvas.AddObject(cir)

        # add a rectangle
        rect = FloatCanvas.Rectangle((110, 10), (100, 100), FillColor='Red')
        self.Canvas.AddObject(rect)
        self.Canvas.Draw()
app = wx.App()
frame = DrawFrame(None)
frame.Show()
app.MainLoop()
```

运行结果就是在画布上画了一个空心圆和填充红色矩形。
上述代码时先建立某一个FloatCanvas的图形对象，如FloatCanvas.Circle，然后通过AddObject添加进去。
也可以直接采用更简单的API，如：
```python
self.Canvas.AddCircle((10, 10), 100)
self.Canvas.AddRectangle((110, 10), (100, 100), FillColor='Red')
```
其他图形的API可仿照以上样式在github的samples中查找。

## 加上NavCanvas导航条
```python
import wx
from wx.lib.floatcanvas import NavCanvas, FloatCanvas

class DrawFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        # Add the Canvas
        Canvas = NavCanvas.NavCanvas(self,-1,
                                     size = (500,500),
                                     ProjectionFun = None,
                                     Debug = 0,
                                     BackgroundColor = "DARK SLATE BLUE",
                                     ).Canvas

        Rect = Canvas.AddRectangle((50, 20), (40,10), FillColor="Red", LineStyle = None)
        Rect.MinSize = 4 # default is 1
        Rect.DisappearWhenSmall = False # defualt is True

        self.Show()
        Canvas.ZoomToBB()

app = wx.App(False)
F = DrawFrame(None, title="FloatCanvas Demo App", size=(700,700) )
app.MainLoop()
```
其作用就是加上NavCanvans导航条，从而可以缩放、平移画布。
查看NavCanvas源码后就可以发现，它里面其实已经引用了FloatCanvas，所以这里不用显式地调用FloatCanvas。

## 获取鼠标坐标
```python
import wx
from wx.lib.floatcanvas import NavCanvas, FloatCanvas

class DrawFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        self.CreateStatusBar()
        # Add the Canvas
        Canvas = NavCanvas.NavCanvas(self,-1,
                                     size = (500,500),
                                     ProjectionFun = None,
                                     Debug = 0,
                                     BackgroundColor = "DARK SLATE BLUE",
                                     ).Canvas
        self.Canvas = Canvas
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.OnMove )
        Rect = Canvas.AddRectangle((50, 20), (40,10), FillColor="Red", LineStyle = None)
        Rect.MinSize = 4 # default is 1
        Rect.DisappearWhenSmall = False # defualt is True
        self.Show()
        Canvas.ZoomToBB()

    def OnMove(self, event):
        print("Coords = ", event.Coords)
        self.SetStatusText("%.2f, %.2f"%tuple(event.Coords))

app = wx.App(False)
F = DrawFrame(None, title="FloatCanvas Demo App", size=(700,700) )
app.MainLoop()
```
注意，上述代码中使用
```python
event.Coords
```

获取了鼠标的坐标，这其实是鼠标在世界坐标系中的位置。
在立体视觉中，有四个坐标系需要明确，一个详细的说明见：[世界坐标系、相机坐标系、图像坐标系、像素坐标系之间的关系](https://blog.csdn.net/u011574296/article/details/73658560)
摘抄如下：
（1）世界坐标系：
客观三维世界的绝对坐标系，也称客观坐标系。因为数码相机安放在三维空间中，我们需要世界坐标系这个基准坐标系来描述数码相机的位置，并且用它来描述安放在此三维环境中的其它任何物体的位置，用（X, Y, Z）表示其坐标值。
（2）相机坐标系（光心坐标系）：
以相机的光心为坐标原点，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，相机的光轴为Z 轴，用（Xc, Yc, Zc）表示其坐标值。
（3）图像坐标系：
以CCD 图像平面的中心为坐标原点，X轴和Y 轴分别平行于图像平面的两条垂直边，用( x , y )表示其坐标值。图像坐标系是用物理单位（例如毫米）表示像素在图像中的位置。
（4）像素坐标系：
以 CCD 图像平面的左上角顶点为原点，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，用(u , v )表示其坐标值。数码相机采集的图像首先是形成标准电信号的形式，然后再通过模数转换变换为数字图像。每幅图像的存储形式是M × N的数组，M 行 N 列的图像中的每一个元素的数值代表的是图像点的灰度。这样的每个元素叫像素，像素坐标系就是以像素为单位的图像坐标系。

获取像素坐标系的方法就是调用wxpython的原生方法：
```
event.GetPosition()
```
注意GetPosition与上面的Coords的区别：
（1）Coords是事件event的属性，不是一个方法，所以没有括号，且它的返回值是一个长度为2的numpy数组；
（2）GetPosition()是事件event的方法，所以需要括号，且它的返回值是一个wx.Point类型的数据，可以通过wx.Point.x和wx.Point.y来具体获取整型的像素坐标。

FloatCanvas也提供了API进行世界坐标系与像素坐标系的转换，即：
```python
print("mouse in pixel coordinates = ", event.GetPosition())
print("mouse in world coordinates = ", event.Coords)
print("pixel2world = ", self.Canvas.PixelToWorld(event.GetPosition()))
print("world2pixel = ", self.Canvas.WorldToPixel(event.Coords))
```

这个地方需要特别注意的是，图像在这些坐标系中的xy表示顺序与OpenCV中的xy表示顺序是反的，这是因为图像在屏幕上的表示是先列后行，而图像被OpenCV读入后，是一个先行后列的数组，取得某一点时，先取行号，再取列号。比如在屏幕上使用鼠标点击获得的坐标点是(x, y)，那么实际这个像素点在OpenCV中的位置是(y, x)，即是在对角线的另一侧。

## 缩放位图
这个例子介绍了怎样在FloatCanvas中缩放位图：
```python
#!/usr/bin/env python

ImageFile = "white_tank.jpg"

import wx
import random
from wx.lib.floatcanvas import NavCanvas, FloatCanvas

class DrawFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        self.CreateStatusBar()
 
        # Add the Canvas
        Canvas = NavCanvas.NavCanvas(self,
                                     ProjectionFun = None,
                                     BackgroundColor = "White",
                                     ).Canvas
        Canvas.MaxScale=20 # sets the maximum zoom level
        self.Canvas = Canvas
        self.Canvas.Bind(FloatCanvas.EVT_MOTION, self.OnMove )

        # create the image:
        image = wx.Image(ImageFile)
        self.width, self.height = image.GetSize()
        img = FloatCanvas.ScaledBitmap2( image,
                                        (0,0),
                                        Height=image.GetHeight(),
                                        Position = 'tl',
                                        )
        Canvas.AddObject(img)
        self.Show()
        Canvas.ZoomToBB()

    def OnMove(self, event):
        self.SetStatusText("%i, %i"%tuple(event.Coords))

app = wx.App(False)
F = DrawFrame(None, title="FloatCanvas Demo App", size=(700,700) )
app.MainLoop()
```
