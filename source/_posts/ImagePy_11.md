---
title: ImagePy解析：11 -- 使用wxPython设备上下文绘图
tags: [ImagePy]
categories: computational material science 
date: 2019-10-16
---

本文是对《wxPython in Action》一书的第6.1节和第12.2节的翻译理解。
ImagePy很多的组件都用到了wxPython的设备上下文进行绘制，比如histogram panel、curve panel、colormap panel和Canvas等。可以说，对于wxPython不提供的组件，ImagePy都通过DC绘制进行了自己开发，因此这一部分对于理解ImagePy的UI交互会很有帮助。

# wxPython设备上下文
为了能在屏幕上绘图，需要使用wxPython的称为设备上下文Device Context的对象，它能对显示设备进行抽象，给予每个设备一套通用的绘图方法，这样一来，编写的绘图代码就对任意的设备都是相同的。wxPython使用wx.DC及其子类来描述设备上下文。因为wx.DC是个抽象类，因此在实际使用时必须使用它的一个子类。

wx.DC的子类：
wx.BufferedDC：用来缓冲一系列的绘图命令，直到这些命令都完成且准备在屏幕上绘图，这可以防止出现不想要的闪烁现象；
wx.BufferedPaintDC：与wx.BufferedDC相同，但仅能在绘图事件wx.PaintEvent的处理过程中使用，仅能临时地创建该类的实例；
wx.ClientDC：用来在窗口的工作区绘图，即不能在边界或其他装饰区域绘图；该类应该被临时创建，且不能在wx.PaintEvent事件处理中使用；
wx.MemoryDC：用来将图像写入内存中的位图，不用来显示；然后可以选择这张位图，使用wx.DC.Blit()方法将位图绘制在窗口上；
wx.MetafileDC：在Windows操作系统上，该设备上下文可以用来创建标准的Windows元文件数据；
wx.PaintDC：与wx.ClientDC相同，但它仅能用在wx.PaintEvent事件处理中；仅临时创建该类的实例；
wx.PostScriptDC：用来存成PostScript文件；
wx.PrinterDC：在Windows平台上用来写文件给打印机；
wx.ScreenDC：用来直接在屏幕上绘图，仅能临时创建；
wx.WindowDC：用来在整个窗口上绘图，包括边界及其他装饰组件上；非Windows操作系统可能不支持该类。

# 画图板
原始代码见：
[wxPython-In-Action/Chapter-06/example1.py](https://github.com/freephys/wxPython-In-Action/blob/master/Chapter-06/example1.py)

上述代码在Phoenix版的wxPython下无法运行，因为部分API已经改变，以下是经过修改后能够顺利运行的代码：
```python
import wx
 
class SketchWindow(wx.Window):
    def __init__(self, parent, ID):
        wx.Window.__init__(self, parent, ID)
        self.SetBackgroundColour("White")
        self.color = "Black"
        self.thickness = 1
        self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
        self.lines = []
        self.curLine = []
        self.pos = (0, 0)
        self.InitBuffer()

        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def InitBuffer(self):
        size = self.GetClientSize()       
        self.buffer = wx.Bitmap(size.width, size.height)
        dc = wx.BufferedDC(None, self.buffer)       
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.DrawLines(dc)
        self.reInitBuffer = False

    def GetLinesData(self):
        return self.lines[:]

    def SetLinesData(self, lines):
        self.lines = lines[:]
        self.InitBuffer()
        self.Refresh()

    def OnLeftDown(self, event):
        self.curLine = [] 
        self.pos = event.GetPosition().Get()  
        self.CaptureMouse()

    def OnLeftUp(self, event):
        if self.HasCapture():
            self.lines.append((self.color,
                               self.thickness,
                               self.curLine))
            self.curLine = []         
            self.ReleaseMouse()

    def OnMotion(self, event):     
        if event.Dragging() and event.LeftIsDown():
            dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
            self.drawMotion(dc, event)
        event.Skip()

    def drawMotion(self, dc, event):
        dc.SetPen(self.pen)
        newPos = event.GetPosition().Get()
        coords = self.pos + newPos
        self.curLine.append(coords)
        dc.DrawLine(*coords)
        self.pos = newPos

    def OnSize(self, event):
        self.reInitBuffer = True

    def OnIdle(self, event):
        if self.reInitBuffer:
            self.InitBuffer()
            self.Refresh(False)

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self, self.buffer)

    def DrawLines(self, dc):
        for colour, thickness, line in self.lines:
            pen = wx.Pen(colour, thickness, wx.SOLID)
            dc.SetPen(pen)
            for coords in line:
                dc.DrawLine(*coords)

    def SetColor(self, color):
        self.color = color
        self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)

    def SetThickness(self, num):
        self.thickness = num
        self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)

class SketchFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "Sketch Frame",
                size=(800,600))
        self.sketch = SketchWindow(self, -1)

if __name__ == '__main__':
    app = wx.App()
    frame = SketchFrame(None)
    frame.Show(True)
    app.MainLoop()
```

下面对一些必要的知识点说明一下：
```python
self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
```
该行创建了一个wx.Pen实例，从中可以指定画线的颜色、宽度和线型等。
```python
self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
self.Bind(wx.EVT_MOTION, self.OnMotion)
self.Bind(wx.EVT_SIZE, self.OnSize)
self.Bind(wx.EVT_IDLE, self.OnIdle)
self.Bind(wx.EVT_PAINT, self.OnPaint)
```
为这个画图板绑定各种鼠标处理事件，包括鼠标左键按下和弹起、鼠标运动、窗口尺寸变化、窗口重绘，以及空闲时候的处理工作。
```python
self.buffer = wx.Bitmap(size.width, size.height)
dc = wx.BufferedDC(None, self.buffer)  
```
通过两步创建缓冲设备上下文：（1）创建一张空白位图作为缓冲区；（2）使用上述缓冲区创建一个缓冲设备上下文。这个缓冲上下文是为了防止线条的重绘使得屏幕闪烁。
下面的代码还会创建一个dc，注意这两个dc的不同。这里的dc更像是一个看不见的绘图层，它主要是在窗口尺寸变化、需要重绘的时候调用，是通过读取鼠标事件存储下来的一系列的坐标，然后调用self.DrawLines(dc)一次性绘出很多的线条。而第二个dc是一个即时的绘图dc。
```python
dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
dc.Clear()
```
创建一个wx.Brush画刷来设置设备上下文的背景，同时使用该背景画刷来清空上下文的内容。
```python
self.pos = event.GetPosition().Get()
```
得到当前鼠标的精确坐标位置，注意这个地方有API变化，书中的是GetPositionTuple()，同时GetPosition()返回的是wx.Point类型的数据，需要调用它的Get()函数将坐标转化为元组格式。
```python
self.CaptureMouse()
```
CaptureMouse()方法使得所有的鼠标输入局限在该窗口中，即使有时候拖动鼠标超出该窗口的边界。该动作必须在后面通过调用ReleaseMouse()来取消。
```python
def OnLeftUp(self, event):
    if self.HasCapture():
        self.lines.append((self.color,
                           self.thickness,
                           self.curLine))
        self.curLine = []         
        self.ReleaseMouse()
```
定义鼠标左键弹起时的动作，此时是将之前鼠标划过的线条存储到self.lines中，用于窗口重绘时的那个看不见的dc的绘图。同时如上所述，ReleaseMouse()将系统返回到上一个CaptureMouse()之前的状态。wxPython使用一个堆栈来追踪捕获鼠标的窗口，因此ReleaseMouse()和CaptureMouse()的数目必须相等。
```python
if event.Dragging() and event.LeftIsDown():
    dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
    self.drawMotion(dc, event)
```
画线时，要首先判断鼠标拖动是不是画线的一部分，即既要鼠标左键按下，又要鼠标在拖动；如果这两个条件都满足，则进入画线状态。因为wx.BufferedDC是一种临时创建的设备上下文，因此在画线之前要重新创建一个wx.BufferedDC，这里创建了一个wx.ClientDC作为主上下文，然后重新使用了那张空白位图作为缓冲。
```python
def drawMotion(self, dc, event):
    dc.SetPen(self.pen)
    newPos = event.GetPosition().Get()
    coords = self.pos + newPos
    self.curLine.append(coords)
    dc.DrawLine(*coords)
    self.pos = newPos
```
这一步是在设备上下文上进行绘图，注意coords是新旧坐标的综合，两个tuple相加的结果是两者拼接起来，这个语法要注意。同时使用星号变量将coords这个tuple中的四个元素拆分为单个元素传入DrawLine()函数。
```python
def OnSize(self, event):
    self.reInitBuffer = True
```
如果窗口的尺寸被改变，则将self.reInitBuffer属性置为True，然后什么都不用做，直到调用下一个空闲事件。
```python
def OnIdle(self, event):
    if self.reInitBuffer:
        self.InitBuffer()
        self.Refresh(False)
```
当出现空闲事件后，程序就会趁机响应改变尺寸的动作。这里选择将改变尺寸的动作放在空闲事件处理中，而不是放在它本来的尺寸变化事件处理中，是为了允许多个尺寸改变事件能够快速地连续执行，而不用等待每一个的重绘。
```python
def OnPaint(self, event):
    dc = wx.BufferedPaintDC(self, self.buffer)
```
处理重绘要求是比较简单的，即只需要创建一个缓冲绘图设备上下文，注意因为这里是wx.PaintEvent中，因此，需要使用wx.PaintDC，而不是wx.ClientDC。
```python
def DrawLines(self, dc):
    for colour, thickness, line in self.lines:
        pen = wx.Pen(colour, thickness, wx.SOLID)
        dc.SetPen(pen)
       for coords in line:
            dc.DrawLine(*coords)
```
在窗口尺寸改变、需要重绘时，使用那个看不见的dc根据之前存储的线条进行绘制。

下面详细分析这三个dc之间的关系，可以在代码中适时地插入一些print函数和SaveFile函数看一下，如下：

```python
def InitBuffer(self):
    print("@@@ Initializing buffer")
    size = self.GetClientSize()       
    self.buffer = wx.Bitmap(size.width, size.height)
    dc = wx.BufferedDC(None, self.buffer)       
    dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
    dc.Clear()
    self.buffer.SaveFile("buffer_before_DrawLines.jpg", wx.BITMAP_TYPE_JPEG)
    self.DrawLines(dc)
    self.buffer.SaveFile("buffer_after_DrawLines.jpg", wx.BITMAP_TYPE_JPEG)
    self.reInitBuffer = False

def OnMotion(self, event):     
    if event.Dragging() and event.LeftIsDown():
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        self.drawMotion(dc, event)
        self.buffer.SaveFile("buffer_in_OnMotion.jpg", wx.BITMAP_TYPE_JPEG)
    event.Skip()

def OnPaint(self, event):
    print("!!! On painting")
    self.buffer.SaveFile("buffer_in_OnPaint.jpg", wx.BITMAP_TYPE_JPEG)
    dc = wx.BufferedPaintDC(self, self.buffer)
```
详细机理为：当初始化时，首先创建了一个全黑色的wx.Bitmap位图self.buffer，然后将它传给了第一个dc，因为这个dc会设置背景为白色，所以其实这时将self.buffer存成图像，即buffer_before_DrawLines.jpg是一张白色图像，即非常重要的知识点就是dc会改变self.buffer。
在初始化时，InitBuffer()和OnPaint()函数都会调用，那么此时buffer_before_DrawLines.jpg、buffer_after_DrawLines.jpg和buffer_in_OnPaint.jpg都是纯白色图像，而因为鼠标还没有开始绘图，则OnMotion()不会被调用，那么buffer_in_OnMotion.jpg也不会调用。
当鼠标开始绘图时，buffer_in_OnMotion.jpg就会生成，且会将当前有线条的图像存储下来，即self.buffer也会被改变。
如果此时拖动窗口边界，但注意不要松开鼠标，则会发现OnPaint()函数一直执行，但InitBuffer()却没有执行，这就是因为之前将InitBuffer()放在了空闲事件中的结果，如果将它放在wx.EVT_SIZE中，那么也可以，但明显在拖动边界时你会感觉到很卡的感觉。
如果拖动了窗口边界，且松开鼠标后，InitBuffer()就会执行，此时可以发现，buffer_before_DrawLines.jpg因为存储的是初始化后的self.buffer，所以它仍然是白色的，而buffer_after_DrawLines.jpg则会存储有线条且当前窗口形状的图像，注意它是当前窗口形状的图像，而buffer_in_OnMotion.jpg是之前窗口的图像，除非再次用鼠标绘图。
如果不关联wx.EVT_PAINT事件，此时拖动窗口边界，则会发现此时会出现“擦掉”图像的现象，但下一次鼠标再次绘制时，之前的线条又会重现，这会非常让人困扰，所以该事件是非常必要的。

那么总结一下：
self.buffer是穿插在这几个dc间的纽带，每个dc都可以对它进行修改。第一个BufferedDC是为了在窗口尺寸变化时存储之前绘制的线条，第二个BufferedDC是实际在鼠标交互时的绘图层，它让用户能实时看到绘制了什么，然后在窗口重绘时它就销毁，将坐标信息传给第一个DC，第三个BufferedPaintDC是为了取出缓冲self.buffer用来刷新窗口，让前后图像具有一致性。


# 绘制雷达图
原始代码见：
[wxPython-In-Action/Chapter-12/radargraph.py](https://github.com/freephys/wxPython-In-Action/blob/master/Chapter-12/radargraph.py)

```python
import wx
import math
import random

class RadarGraph(wx.Window):
    """
    A simple radar graph that plots a collection of values in the
    range of 0-100 onto a polar coordinate system designed to easily
    show outliers, etc.  You might use this kind of graph to monitor
    some sort of resource allocation metrics, and a quick glance at
    the graph can tell you when conditions are good (within some
    accepted tolerance level) or approaching critical levels (total
    resource consumption).
    """
    def __init__(self, parent, title, labels):
        wx.Window.__init__(self, parent)
        self.title = title
        self.labels = labels
        self.data = [0.0] * len(labels)
        self.titleFont = wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD)
        self.labelFont = wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL)

        self.InitBuffer()

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
 
    def OnSize(self, evt):
        # When the window size changes we need a new buffer.
        self.InitBuffer()

    def OnPaint(self, evt):
        # This automatically Blits self.buffer to a wx.PaintDC when
        # the dc is destroyed, and so nothing else needs done.
        dc = wx.BufferedPaintDC(self, self.buffer)

    def InitBuffer(self):
        # Create the buffer bitmap to be the same size as the window,
        # then draw our graph to it.  Since we use wx.BufferedDC
        # whatever is drawn to the buffer is also drawn to the window.
        w, h = self.GetClientSize()       
        self.buffer = wx.Bitmap(w, h)
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        self.DrawGraph(dc)
 
    def GetData(self):
        return self.data

    def SetData(self, newData):
        assert len(newData) == len(self.data)
        self.data = newData[:]

        # The data has changed, so update the buffer and the window
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        self.DrawGraph(dc)

    def PolarToCartesian(self, radius, angle, cx, cy):
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        return (cx+x, cy-y)
 
    def DrawGraph(self, dc):
        spacer = 10
        scaledmax = 150.0

        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        dw, dh = dc.GetSize()

        # Find out where to draw the title and do it
        dc.SetFont(self.titleFont)
        tw, th = dc.GetTextExtent(self.title)
        dc.DrawText(self.title, (dw-tw)/2, spacer)

        # find the center of the space below the title
        th = th + 2*spacer
        cx = dw/2
        cy = (dh-th)/2 + th

        # calculate a scale factor to use for drawing the graph based
        # on the minimum available width or height
        mindim = min(cx, (dh-th)/2)
        scale = mindim/scaledmax

        # draw the graph axis and "bulls-eye" with rings at scaled 25,
        # 50, 75 and 100 positions
        dc.SetPen(wx.Pen("black", 1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawCircle(cx,cy, 25*scale)
        dc.DrawCircle(cx,cy, 50*scale)
        dc.DrawCircle(cx,cy, 75*scale)
        dc.DrawCircle(cx,cy, 100*scale)

        dc.SetPen(wx.Pen("black", 2))
        dc.DrawLine(cx-110*scale, cy, cx+110*scale, cy)
        dc.DrawLine(cx, cy-110*scale, cx, cy+110*scale)
 
        # Now find the coordinates for each data point, draw the
        # labels, and find the max data point
        dc.SetFont(self.labelFont)
        maxval = 0
        angle = 0
        polypoints = []
        for i, label in enumerate(self.labels):
            val = self.data[i]
            point = self.PolarToCartesian(val*scale, angle, cx, cy)
            polypoints.append(point)
            x, y = self.PolarToCartesian(125*scale, angle, cx,cy)
            dc.DrawText(label, x, y)
            if val > maxval:
                maxval = val
            angle = angle + 360/len(self.labels)

        # Set the brush color based on the max value (green is good,
        # red is bad)
        c = "forest green"
        if maxval > 70:
            c = "yellow"
        if maxval > 95:
            c = "red"

        # Finally, draw the plot data as a filled polygon
        dc.SetBrush(wx.Brush(c))
        dc.SetPen(wx.Pen("navy", 3))
        dc.DrawPolygon(polypoints)
       
class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Double Buffered Drawing",
                          size=(480,480))
        self.plot = RadarGraph(self, "Sample 'Radar' Plot",
                          ["A", "B", "C", "D", "E", "F", "G", "H"])

        # Set some random initial data values
        data = []
        for d in self.plot.GetData():
            data.append(random.randint(0, 75))
        self.plot.SetData(data)

        # Create a timer to update the data values
        self.Bind(wx.EVT_TIMER, self.OnTimeout)
        self.timer = wx.Timer(self)
        self.timer.Start(500)

    def OnTimeout(self, evt):
        # simulate the positive or negative growth of each data value
        data = []
        for d in self.plot.GetData():
            val = d + random.uniform(-5, 5)
            if val < 0:
                val = 0
            if val > 110:
                val = 110
            data.append(val)
        self.plot.SetData(data)


app = wx.App()
frm = TestFrame()
frm.Show()
app.MainLoop()
```
该程序与上面的画图板原理相同，需要注意的是它将InitBuffer()放进了EVT_SIZE事件处理函数中，这样每次窗口变化时就会得到一个新的缓冲，雷达图的绘制也会随着窗口的变化实时改变。
该例子用到了更多的dc的功能，如DrawCircle、DrawText、DrawPolygon等。

# 绘制位图
原始代码见：
[wxPython-In-Action/Chapter-12/draw_image.py](https://github.com/freephys/wxPython-In-Action/blob/master/Chapter-12/draw_image.py)
```python
# This one shows how to draw images on a DC.
import wx
import random
random.seed()

class RandomImagePlacementWindow(wx.Window):
    def __init__(self, parent, image):
        wx.Window.__init__(self, parent)
        self.photo = image.ConvertToBitmap()

        # choose some random positions to draw the image at:
        self.positions = [(10,10)]
        for x in range(50):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            self.positions.append( (x,y) )

        # Bind the Paint event
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, evt):
        # create and clear the DC
        dc = wx.PaintDC(self)
        brush = wx.Brush("sky blue")
        dc.SetBackground(brush)
        dc.Clear()

        # draw the image in random locations
        for x,y in self.positions:
            dc.DrawBitmap(self.photo, x, y, True)

class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Loading Images",
                          size=(640,480))
        img = wx.Image("masked-portrait.png")
        win = RandomImagePlacementWindow(self, img)
       
app = wx.App()
frm = TestFrame()
frm.Show()
app.MainLoop()
```
主要函数就是DrawBitmap()，如果是绘制图标，则用DrawIcon()。
