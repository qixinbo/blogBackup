---
title: ImagePy解析：10 -- Tool引擎及其衍生的画笔工具
tags: [ImagePy]
categories: computational material science 
date: 2019-10-10
---

参考文献：
[ImagePy开发文档 —— 工具](https://zhuanlan.zhihu.com/p/25483900)
[Tool 插件](https://github.com/Image-Py/demoplugin/blob/master/doc/chinese/tool.md)

Filter、Simple 都可以处理图像，但有时候我们需要用鼠标交互对图像进行一些操作。比如我们的选区操作，绘图操作等。ImagePlus 被绘制在一个 Canvas 上，Canvas 是 wxpython 的 Panel 子类，当然我们可以对其添加鼠标事件，但我们并不推荐这样做，原因之一是这样做比较繁琐，其次，多工具同时注册事件，会引起管理混乱和事件冲突。（语出上面的参考文献）
这一篇来解析一下Tool引擎，先分析一下这个基类，然后通过基于它编写的画笔工具深入理解。
注意，Tool引擎与前面的Free、Filter等交互逻辑上有挺多区别，比如入口函数中不触发一系列的连贯操作，而是强烈依赖于鼠标事件，因此需要仔细地区别对待。

# Tool引擎
看一下Tool引擎的全貌：
```python
from ... import IPy
from ...core.manager import ToolsManager

class Tool:
    title = 'Tool'
    view, para = None, None

    def show(self):
        …   
    def config(self):pass
    def load(self):pass
    def switch(self):pass

    def start(self):
        …       
    def mouse_down(self, ips, x, y, btn, **key): pass
    def mouse_up(self, ips, x, y, btn, **key): pass
    def mouse_move(self, ips, x, y, btn, **key): pass
    def mouse_wheel(self, ips, x, y, d, **key): pass
```
然后再细看一下它的具体属性和方法。
## para和view属性
这两位朋友很熟悉了，用于设置输入的参数和交互界面样式，默认为None，代表不需要交互。这里的para与前面Free和Filter中的para有些不一样，它是用来设置工具的自身特性，比如画笔的宽度、颜色等。
## show()方法
Tool引擎的show()方法与Free和Filter的都不同，它不会在入口函数start()中进行触发，而是在鼠标左键双击时触发。那么机理来自于哪里？奥妙就在于在第一步构建主界面的加载工具条时对鼠标事件的绑定，来自于toolsloader.py的add_tools()函数：
```python
def f(plg, e):
    plg.start()
    if isinstance(plg, Tool):
        e.Skip()
def set_info(value):
    IPy.curapp.set_info(value)

def add_tools(bar, datas, clear=False, curids=[]):
    …
        btn.Bind( wx.EVT_LEFT_DOWN, lambda x, p=data[0]:f(p(), x))
        btn.Bind( wx.EVT_RIGHT_DOWN, lambda x, p=data[0]: IPy.show_md(p.title, DocumentManager.get(p.title)))
        btn.Bind( wx.EVT_ENTER_WINDOW,
                  lambda x, p='"{}" Tool'.format(data[0].title): set_info(p))       
        if not isinstance(data[0], Macros) and issubclass(data[0], Tool):
            btn.Bind(wx.EVT_LEFT_DCLICK, lambda x, p=data[0]:p().show())
        btn.SetDefault()
    …
```
可以看出，这里将左键双击事件绑定到了Tool引擎的show()方法上。
同时可以看出，对于界面上的每一个tool，还有其他的鼠标事件绑定：
（1）左键单击：执行f()函数，即执行了Tool引擎的start()函数
（2）右键单击：调用IPy的显示MarkDown帮助文件
（3）鼠标进入窗口：在状态栏显示该tool的title

说回show()方法，看它的源码：
```python
def show(self):
    if self.view == None:return
    rst = IPy.get_para(self.title, self.view, self.para)
    if rst!=None : self.config()
```
可以看出，如果view为None，即不交互，则不做什么动作；反之，则调用IPy的get_para()方法，此时就会蹦出参数对话框用于设置参数。然后接着调用下面的config()方法，该方法默认是什么也不做。

## config()方法
当交互对话框确认时将会被调用，用于生效我们的设置。一些时候参数值的改变本身就会生效，例如调整画笔宽度，但有些时候，需要对外部进行一些通讯，比如设置全局颜色，就需要和 ColorManager 通讯。（语出上面的参考文献）

## load()和switch()方法
load是工具被选中时调用，switch是由当前工具切换到另一个工具时调用（可以处理绘制了一半还没有闭合的多边形选区等）。（语出上面的参考文献）

## start()方法
前已述及，它在鼠标左键单击时调用，具体操作是：
```python
def start(self):
    ips = IPy.get_ips()
    if not ips is None and not ips.tool is None:
        ips.tool = None
        ips.update()
    ToolsManager.set(self)
```
可以看出，首先得到了ips对象，然后还有一个重要操作是调用ToolsManager（ToolsManager在pluginmanager.py中定义）的set()方法，里面就调用了上面的switch()方法。

## mouse_down()、mouse_up()、mouse_move()和mouse_wheel()方法
这四个方法一看名字就知道是与鼠标操作有关，具体地，与鼠标按下、弹起、移动、滚轮相关。
那么，它们是怎样被调用的呢？
从前面的那篇“由新建图像谈起”文章可知，ImagePy中图像是呈现在Canvasl中的。虽然Canvas能够作为独立组件单独调用，但在ImagePy整个框架中，Canvas是通过CanvasPanel来调用的，所以在图像中的鼠标操作事件最开始都是由CanvasPanel所调用的Canvas来捕获和激发的，即下面这句话：
```python
self.canvas = Canvas(self, autofit = IPy.uimode()=='ij')
self.canvas.Bind(wx.EVT_MOUSE_EVENTS, self.on_mouse)
```

这里插一句，为了保证Canvas能够被单独调用，在Canvas类中也写了这么一个鼠标事件绑定，所以如果仅仅是在ImagePy中使用Canvas，可以把那句删掉。
再插一句，Canvas所捕获的鼠标事件与上面提到的加载工具条时捕获的鼠标事件要区分开，这是两个不同的wxPython的Panel。比如，虽然都会捕获鼠标左键按下，但是不同的Panel所做的动作是不一样的。

然后在具体的事件处理函数on_mouse()中，开头是这样的：
```python
def on_mouse(self, me):
    tool = self.ips.tool
    if tool == None : tool = ToolsManager.curtool
```
这样就解释了上面的参考文献中的这句话：
>事件的调用遵循如下规则：
>1.事件最初由 Canvas 类触发。
>2.如果 Canvas 持有的 ImagePlus 有一个 CustomTool，则交给它来处理。（一般来说，比如选区，画笔等工具，是全局有效的，但有时我们需要与单一图像做特定交互，比如当某个图像栈进入了三视图观察状态时，这时CustomTool 就变得很有用。）
>3.如果没有，则交给 ToolsManager 的当前工具来处理。

然后如果真是交给了当前工具，那么Tool引擎的这四个方法就开始调用了：
```python
sta = [me.AltDown(), me.ControlDown(), me.ShiftDown()]
if me.ButtonDown():tool.mouse_down(self.ips, x, y, me.GetButton(),
    alt=sta[0], ctrl=sta[1], shift=sta[2], canvas=self.canvas)
if me.ButtonUp():tool.mouse_up(self.ips, x, y, me.GetButton(),
    alt=sta[0], ctrl=sta[1], shift=sta[2], canvas=self.canvas)
if me.Moving():tool.mouse_move(self.ips, x, y, None,
    alt=sta[0], ctrl=sta[1], shift=sta[2], canvas=self.canvas)
btn = [me.LeftIsDown(), me.MiddleIsDown(), me.RightIsDown(),True].index(True)
if me.Dragging():tool.mouse_move(self.ips, x, y, 0 if btn==3 else btn+1,
    alt=sta[0], ctrl=sta[1], shift=sta[2], canvas=self.canvas)
wheel = np.sign(me.GetWheelRotation())
if wheel!=0:tool.mouse_wheel(self.ips, x, y, wheel,
    alt=sta[0], ctrl=sta[1], shift=sta[2], canvas=self.canvas)
if hasattr(tool, 'cursor'):
    self.canvas.SetCursor(wx.Cursor(tool.cursor))
else : self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
```

具体地分析一下这四个方法所需要的参数：（语出上面的参考文献）
1. mouse_down()
鼠标按下时调用，其参数意义如下：
（1）ips:与事件相关的 ImagePlus，可以通过ips.img获取当前图像，也可以ips.lut, ips.roi, ips.unit获取图像的索引表，roi，比例尺和单位等附加信息。
（2）x:数据坐标系下的 x
（3）y:数据坐标系下的 y
（4）btn:按下的按钮（1：左键，2：中键，3：右键）
（5）key其他参数：可以通过key['alt']、key['ctrl']、key['shift']获取相应功能键是否按下，通过key['canvas']获取触发事件的Canvas对象。
2. mouse_up()
参数和意义与 mouse_down 完全相同
3. mouse_move()
参数和意义与 mouse_down 完全相同，另外附加如果没有按键按下，btn=0
4. mouse_wheel()
参数和意义与 mouse_down 基本相同，除了btn参数变成了d，代表滚动量


下面以画笔工具看看怎样编写基于Tool的插件。
# Pencil画笔工具
```python
from imagepy.core.engine import Tool
from imagepy.core.manager import ColorManager
from skimage.draw import line, circle

def drawline(img, oldp, newp, w, value):
    if img.ndim == 2: value = sum(value)/3
    oy, ox = line(*[int(round(i)) for i in oldp+newp])
    cy, cx = circle(0, 0, w/2+1e-6)
    ys = (oy.reshape((-1,1))+cy).clip(0, img.shape[0]-1)
    xs = (ox.reshape((-1,1))+cx).clip(0, img.shape[1]-1)
    img[ys.ravel(), xs.ravel()] = value
 
class Plugin(Tool):
    title = 'Pencil'
    para = {'width':1}
    view = [(int, 'width', (0,30), 0,  'width', 'pix')]

    def __init__(self):
        self.status = False
        self.oldp = (0,0)

    def mouse_down(self, ips, x, y, btn, **key):
        self.status = True
        self.oldp = (y, x)
        ips.snapshot()

    def mouse_up(self, ips, x, y, btn, **key):
        self.status = False

    def mouse_move(self, ips, x, y, btn, **key):
        if not self.status:return
        w = self.para['width']
        value = ColorManager.get_front()
        drawline(ips.img, self.oldp, (y, x), w, value)
        self.oldp = (y, x)
        ips.update()

    def mouse_wheel(self, ips, x, y, d, **key):pass
```
可以看出，para中定义了线宽。在这个画笔工具中，当鼠标按下时，将status属性置为True，同时将当前坐标点存下来作为oldp，以及也存一个快照。
注意存坐标点时将x和y的顺序互换，这在之前一篇解析文章中已提到，这是因为图像在坐标系中的顺序与实际的Numpy数组的顺序是相反地，前者是先列后行，后者的读取则是先行后列。
然后在鼠标移动过程中绘图，线宽可以通过双击该画笔工具进行设定。
