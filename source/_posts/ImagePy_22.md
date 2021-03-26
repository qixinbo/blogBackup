---
title: ImagePy解析：22 -- 从零搭建一个图像处理软件
tags: [ImagePy]
categories: computer vision 
date: 2020-6-20
---

ImagePy经过一次大的重构后，软件架构有了很大改变，将前端UI和后端数据结构进行了解耦：
- sciapp是一套数据接口，包含图像Image、网格Mesh、表格Table、几何矢量Shape等基础数据结构；
- sciwx是符合sciapp接口标准的可视化组件库；
- ImagePy是后端基于sciapp、前端基于sciwx的一个插件集，包含了大量常用的图像处理算法等。
 
因此可以很容易地基于分离后的sciapp和sciwx构建自定义的独立图像处理软件。

# Sciapp版Hello World
```python
import wx
from sciwx.canvas import CanvasNoteFrame
from sciapp import App

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

app = wx.App()
hello = HelloApp(None)
hello.Show()
app.MainLoop()
```
结果如图：
![helloworld](https://user-images.githubusercontent.com/6218739/84989874-e980a180-b176-11ea-9e9c-3533e6ff8982.png)
可以看出，自定义的app需要继承sciwx库的CanvasNoteFrame以及sciapp的App类，并对其初始化。

# 添加图像
上例是张空白的画布，这里对其添加一张图像，供后面进行操作。
```python
import wx
from sciwx.canvas import CanvasNoteFrame
from sciapp import App
from skimage.data import astronaut
 

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

app = wx.App()
hello = HelloApp(None)

canvas = hello.notebook.add_canvas()
canvas.set_img(astronaut())

hello.Show()
app.MainLoop()
```
如上，添加两行代码即可完全对图像的添加和显示。
结果如图：
![display](https://user-images.githubusercontent.com/6218739/84991154-b63f1200-b178-11ea-8468-00dbbb369f27.png)
首先，获取notebook并对其添加画布：
```python
canvas = hello.notebook.add_canvas()
```
然后，在画布上添加图像：
```python
canvas.set_img(astronaut())
```

如果是展示图像序列，则：
```python
canvas.set_imgs([astronaut(), 255-astronaut()])
```
结果如图：
![multiple-display](https://user-images.githubusercontent.com/6218739/84991303-ec7c9180-b178-11ea-9800-6a76ad413554.png)
注意，这里展示的是图像序列，因此两张图像都显示在同一个标签页下，通过下方的滑动条进行切换。
如果是想多标签页显示，那么：
```python
canvas2 = hello.notebook.add_canvas()
canvas2.set_img(camera())
```
即新添加一个画布，再设置图像，结果如图：
![multi-tab-display](https://user-images.githubusercontent.com/6218739/84992152-20a48200-b17a-11ea-91df-fab6f310b420.png)

# 添加工具
为自己的app添加一个画笔工具：
```python
import wx
from sciwx.canvas import CanvasNoteFrame
from sciapp import App
from sciapp.action import ImageTool
from skimage.data import astronaut
from skimage.draw import line

class Pencil(ImageTool):
    title = 'Pencil'

    def __init__(self):
        self.status = False
        self.oldp = (0,0)

    def mouse_down(self, ips, x, y, btn, **key):
        self.status = True
        self.oldp = (y, x)

    def mouse_up(self, ips, x, y, btn, **key):
        self.status = False

    def mouse_move(self, ips, x, y, btn, **key):
        if not self.status:return
        se = self.oldp + (y,x)
        rs,cs = line(*[int(i) for i in se])
        rs.clip(0, ips.shape[1], out=rs)
        cs.clip(0, ips.shape[0], out=cs)
        ips.img[rs,cs] = (255, 0, 0)
        self.oldp = (y, x)
        key['canvas'].update()

    def mouse_wheel(self, ips, x, y, d, **key):
        pass

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

app = wx.App()
hello = HelloApp(None)
canvas = hello.notebook.add_canvas()
canvas.set_img(astronaut())

tool = hello.add_toolbar()
tool.add_tool('P', Pencil)

hello.Show()
app.MainLoop()
```
效果如图：
![tool](https://user-images.githubusercontent.com/6218739/85089368-15eafb00-b215-11ea-9986-9ddf56794ba3.png)
解析如下：
在app中添加自定义的工具时，需要继承sciapp中提供的ImageTool基类：
```python
from sciapp.action import ImageTool
class Pencil(ImageTool):
    title = 'Pencil'
```
然后再根据自己的需求重载四个鼠标事件：
```python
    def mouse_down(self, ips, x, y, btn, **key): pass
    def mouse_up(self, ips, x, y, btn, **key): pass
    def mouse_move(self, ips, x, y, btn, **key): pass
    def mouse_wheel(self, ips, x, y, d, **key): pass
```
这四个函数的参数意义为：
从中也可以看出自定义工具中可以调用的接口：
（1）ips：即画布承载的Image对象，该对象即sciapp中定义的统一的图像数据结构
（2）x和y：当前鼠标所在的图像像素坐标，水平方向是x方向，垂直方向是y方向。btn是鼠标按键，1为左键按下，2为中键按下，3为右键按下，key是字典，里面有多个字段，key['alt']代表是否按下alt键，key['ctrl']代表是否按下ctrl键，key['shift']代表是否按下shift键，key['px']返回鼠标当前的画布x坐标，key['py']返回画布y坐标，key['canvas']返回该画布自身。

然后在app中添加工具条：
```python
tool = hello.add_toolbar()
tool.add_tool('P', Pencil)
```
如果想添加多个工具，多次调用add_tool即可。

#添加菜单
为自己的app添加菜单（即插件），这里以添加高斯模糊插件为例：
```python
import wx
from sciwx.canvas import CanvasNoteFrame
from sciapp import App
from sciapp.action import ImgAction
from scipy.ndimage import gaussian_filter
from skimage.data import astronaut

class Gaussian(ImgAction):
    title = 'Gaussian'
    note = ['auto_snap', 'preview']

    def run(self, ips, img, snap, para):
        gaussian_filter(snap, 2, output=img)

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

app = wx.App()
hello = HelloApp(None)
canvas = hello.notebook.add_canvas()
canvas.set_img(astronaut())
 
hello.add_img(canvas.image)
hello.add_img_win(canvas)

menu = hello.add_menubar()
menu.load(('menu',[('Filter',[('Gaussian', Gaussian)])]))

hello.Show()
app.MainLoop()
```
效果如图：
![menu-1](https://user-images.githubusercontent.com/6218739/85089717-ed173580-b215-11ea-95da-112f2611301a.png)
解析如下：
首先自定义插件需要继承sciapp提供的ImgAction基类，即：
```python
from sciapp.action import ImgAction
class Gaussian(ImgAction):
    title = 'Gaussian'
    note = ['auto_snap', 'preview']
```
然后在自己的插件类中重载run()方法：
```python
    def run(self, ips, img, snap, para):
        gaussian_filter(snap, 2, output=img)
```
这里的四个参数的意义分别是：
（1）ips即sciapp中定义的Image图像封装；
（2）img即ips中的当前实际的图像；
（3）snap是当前图像在处理之前的拷贝，这样就可以做回退操作；
（4）para是参数对话框，用于与用户进行参数设置的交互。上面代码中没有提供该对话框，稍后添加代码实现这一功能。

编写完自己的插件后，需要添加到app中：
```python
hello.add_img(canvas.image)
hello.add_img_win(canvas)
menu = hello.add_menubar()
menu.load(('menu',[('Filter',[('Gaussian', Gaussian)])]))
```
上面两行是将当前画布的图像及其窗口添加到总控全局的App管理器中，这样插件才能识别到当前图像。
下面两行是添加菜单栏，Filter是一级菜单，Gaussian是二级菜单，如果想添加多个菜单项，可以在相应位置以字典的形式加入，比如：
```python
    menu.load(('menu',[('Filter',[('Gaussian', Gaussian),
                                 ('Unto', Undo)]),
                      ]))
```

前面说了，上述代码中的滤波器的标准差sigma的值是“写死”的，即固定为2，下面添加代码使得显示出参数对话框方便用户交互：
```python
from sciwx.widgets import ParaDialog

class Gaussian(ImgAction):
    title = 'Gaussian'
    note = ['auto_snap', 'preview']
    para = {'sigma':2}
    view = [(float, 'sigma', (0, 30), 1, 'sigma', 'pix')]

    def run(self, ips, img, snap, para):
        gaussian_filter(snap, para['sigma'], output=img)

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

    def show_para(self, title, view, para, on_handle=None, on_ok=None, on_cancel=None, preview=False, modal=True):
        dialog = ParaDialog(self, title)
        dialog.init_view(view, para, preview, modal=modal, app=self)
        dialog.Bind('cancel', on_cancel)
        dialog.Bind('parameter', on_handle)
        dialog.Bind('commit', on_ok)
        return dialog.show()
```
这里应用了sciwx提供的ParaDialog组件，关于ParaDialog的详细解析可以参见之前的一篇文章：
[ImagePy解析：18 -- 参数对话框ParaDialog详解](https://qixinbo.info/2020/03/24/imagepy_18/)

效果如下：
![menu-2](https://user-images.githubusercontent.com/6218739/85090835-a840ce00-b218-11ea-8312-c658d793d1d1.png)

# 添加“打开文件”插件
“打开文件”是一个非常重要的功能，因为它赋予用户通过图形界面打开图像的权利（前面的例子都是在程序中将图像“硬读入”）。
之所以将“打开文件”插件单独拿出来，是因为它涉及的操作比高斯模糊插件更多：获取文件路径并添加图像等，所重载的函数也稍稍不同。见下方源码：
```python
import wx
from sciwx.canvas import CanvasNoteFrame
from sciapp import App
from sciapp.action import ImgAction
from skimage.io import imread
import os

class OpenFile(ImgAction):
    title = "Open File"
    filt = ["png"]
    para = {'path':''}

    def show(self):
        filt = [i.lower() for i in self.filt]
        self.para['path'] = self.app.getpath('Open..', filt, 'open', '')
        print("path = ", self.para['path'])
        return not self.para['path'] is None

    def start(self, app, para = None):
        self.app = app
        if self.show():
            fp, fn = os.path.split(self.para['path'])
            fn, fe = os.path.splitext(fn)
            self.app.show_img(imread(self.para['path']), fn)

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

    def show_img(self, img, title):
        canvas = self.notebook.add_canvas()
        self.remove_img(canvas.image)
        self.remove_img_win(canvas)
        if not title is None:
            canvas.set_img(img)
            canvas.image.name = title
        else: canvas.set_img(img)
        self.add_img(canvas.image)
        self.add_img_win(canvas)

    def getpath(self, title, filt, io, name=''):
        filt = '|'.join(['%s files (*.%s)|*.%s'%(i.upper(),i,i) for i in filt])
        dic = {'open':wx.FD_OPEN, 'save':wx.FD_SAVE}
        dialog = wx.FileDialog(self, title, '', name, filt, dic[io])
        rst = dialog.ShowModal()
        path = dialog.GetPath() if rst == wx.ID_OK else None
        dialog.Destroy()
        return path

app = wx.App()
hello = HelloApp(None)

menu = hello.add_menubar()
menu.load(('menu',[('File', [('Open', OpenFile)])]))

hello.Show()
app.MainLoop()
```
主要增添了如下代码块：
（1）在app中增加路径读取模块，即：
```python
    def getpath(self, title, filt, io, name=''):
```
（2）在插件OpenFile中重载start()方法，而不是run()方法，因为它不涉及图像操作：
```python
    def start(self, app, para = None):
```
（3）在app中增加添加和显示图像的模块，即：
```python
    def show_img(self, img, title):
```
原理就是之前的添加canvas和向App管理器中添加图像。

# 一个完整demo
下面给出一个完整demo，包括“打开文件”和“高斯滤波”这两个菜单插件，及“画笔”和“矩形ROI”这两个工具。
```python
import wx
from sciwx.canvas import CanvasNoteFrame
from sciapp import App
from sciapp.action import ImgAction
from sciapp.action import ImageTool, RectangleROI
from scipy.ndimage import gaussian_filter
from skimage.data import astronaut
from skimage.io import imread
from skimage.draw import line

from sciwx.widgets import ParaDialog
import os

class Gaussian(ImgAction):
    title = 'Gaussian'
    note = ['auto_snap', 'preview']
    para = {'sigma':2}
    view = [(float, 'sigma', (0, 30), 1, 'sigma', 'pix')]

    def run(self, ips, img, snap, para):
        gaussian_filter(snap, para['sigma'], output=img)

class OpenFile(ImgAction):
    title = "Open File"
    filt = ["png"]
    para = {'path':''}

    def show(self):
        filt = [i.lower() for i in self.filt]
        self.para['path'] = self.app.getpath('Open..', filt, 'open', '')
        print("path = ", self.para['path'])
        return not self.para['path'] is None

    def start(self, app, para = None):
        self.app = app
        if self.show():
            fp, fn = os.path.split(self.para['path'])
            fn, fe = os.path.splitext(fn)
            img = imread(self.para['path'])
            self.app.show_img(img, fn)

class Pencil(ImageTool):
    title = 'Pencil'
       
    def __init__(self):
        self.status = False
        self.oldp = (0,0)

    def mouse_down(self, ips, x, y, btn, **key):
        self.status = True
        self.oldp = (y, x)

    def mouse_up(self, ips, x, y, btn, **key):
        self.status = False

    def mouse_move(self, ips, x, y, btn, **key):
        if not self.status:return
        se = self.oldp + (y,x)
        rs,cs = line(*[int(i) for i in se])
        rs.clip(0, ips.shape[1], out=rs)
        cs.clip(0, ips.shape[0], out=cs)
        ips.img[rs,cs] = (255, 0, 0)
        self.oldp = (y, x)
        key['canvas'].update()

    def mouse_wheel(self, ips, x, y, d, **key):
        pass

class HelloApp(CanvasNoteFrame, App):
    def __init__(self, parent):
        App.__init__(self)
        CanvasNoteFrame.__init__ (self, parent, title = 'Hello World')

    def show_img(self, img, title):
        canvas = self.notebook.add_canvas()
        self.remove_img(canvas.image)
        self.remove_img_win(canvas)
        if not title is None:
            canvas.set_img(img)
            canvas.image.name = title
        else: canvas.set_img(img)
        self.add_img(canvas.image)
        self.add_img_win(canvas)

    def getpath(self, title, filt, io, name=''):
        filt = '|'.join(['%s files (*.%s)|*.%s'%(i.upper(),i,i) for i in filt])
        dic = {'open':wx.FD_OPEN, 'save':wx.FD_SAVE}
        dialog = wx.FileDialog(self, title, '', name, filt, dic[io])
        rst = dialog.ShowModal()
        path = dialog.GetPath() if rst == wx.ID_OK else None
        dialog.Destroy()
        return path

    def show_para(self, title, view, para, on_handle=None, on_ok=None, on_cancel=None, preview=False, modal=True):
        dialog = ParaDialog(self, title)
        dialog.init_view(view, para, preview, modal=modal, app=self)
        dialog.Bind('cancel', on_cancel)
        dialog.Bind('parameter', on_handle)
        dialog.Bind('commit', on_ok)
        return dialog.show()

app = wx.App()
hello = HelloApp(None)

menu = hello.add_menubar()
menu.load(('menu',[('File', [('Open', OpenFile)]),
                    ('Filter',[('Gaussian', Gaussian)])]))

tool = hello.add_toolbar()
tool.add_tool('P', Pencil)
tool.add_tool('R', RectangleROI)

hello.Show()
app.MainLoop()
```
效果如图：
![final](https://user-images.githubusercontent.com/6218739/85116340-ba8a2e80-b24f-11ea-9071-ef63059f4fdf.png)

# 更多
管理器、表格、网格等。
