---
title: ImagePy解析：17 -- 重构版ImagePy/sciwx解析
tags: [ImagePy]
categories: computational material science 
date: 2020-2-26
---

新版ImagePy有如下特点：
（1）将原版ImagePy非常特色的可视化组件完全解耦，比如画布、表格、对话框等组件，将其重构为sciwx库，这样第三方开发人员就可以更加方便地使用这些组件而构建自己的特定应用；
（2）新版ImagePy在sciwx库的基础上进行再集成开发，提供一整套完善的管理系统和丰富的插件，从而实现复杂的图像处理功能。

因此，sciwx等价于napari等库，着重于可视化；ImagePy等价于ImageJ等库，着重于图像处理。而这两者因为是“一母双生”，架构思路一脉相承，集成更加自然契合，因此，ImagePy/Sciwx无论是对底层开发人员还是图像处理小白都有着无可比拟的优势：小到开发一个图像处理小工具，大到作为一个大型软件“开箱即用”，都可以轻松应对。


下面是对ImagePy所基于的sciwx库各个组件的demo进行逐步解析（最好是直接运行一下，以获得直观感受）。

# 画布
```python
from skimage.data import camera
from sciwx.canvas import Canvas
import wx

app = wx.App()
frame = wx.Frame(None, title='gray test')
canvas = Canvas(frame, autofit=True)
canvas.set_img(camera())
frame.Show()
app.MainLoop()
```
ImagePy/sciwx展示一张图像所使用的是Canvas类，它是对wxPython的面板Panel类的深度定制，可以实现对图像的区域缩放、拖动、标注等。之前写过一篇对该类的详细解释，见[这里](http://qixinbo.info/2019/10/29/ImagePy_12/)。

# 通道、图像序列展示
```python
from skimage.data import astronaut
from sciwx.canvas import MCanvas
import wx

app = wx.App()
frame = wx.Frame(None, title='gray test')
canvas = MCanvas(frame, autofit=True)
canvas.set_imgs([astronaut(), 255-astronaut()])
canvas.set_cn(0)
frame.Show()
app.MainLoop()
```
上面的Canvas类仅能展示一张图像，这里的MCanvas类则是用于展示图像序列和多通道：
（1）图像序列：将多个图像组合成列表list，然后传入set_imgs()方法中；
（2）通道：对于多通道图像，可以传入单个通道，如0或1或2，这时是单通道灰度显示，也可以组合成(0, 1, 2)，即RGB彩色显示，甚至可以任意按不同顺序组合通道，比如(1, 0, 2)，即将原来的通道1变成现在的通道0，再彩色显示。对于灰度图，则只有通道0。

# 内部图像类Image
```python
from skimage.data import camera
from sciwx.canvas import Canvas, Image
import wx

app = wx.App()
obj = Image()
obj.img = camera()
obj.cn = 0
frame = wx.Frame(None, title='gray test')
canvas = Canvas(frame, autofit=True)
canvas.set_img(obj)
frame.Show()   
app.MainLoop()
```
这个例子是测试ImagePy的内部图像类Image，该类是源图像的一个包装，同时提供多种属性供调用，比如图像名称title、源图像img、通道数channels、整个图像序列中包含图像个数slices、图像尺寸shape、色彩范围range、快照snapshot等。

这里是先构造一个Image对象，然后将camera这张图像传给该对象的img属性，然后再传给Canvas。
之前例子中Canvas是直接接收camera，这两种方式都可以，因为Canvas类中对类型做了判断和处理。

# 自定义鼠标事件
```python
from skimage.data import camera
from sciwx.canvas import Canvas
import wx

class TestTool:
    def mouse_down(self, image, x, y, btn, **key):
        print('x:%d y:%d btn:%d ctrl:%s alt:%s shift:%s'%
              (x, y, btn, key['ctrl'], key['alt'], key['shift']))

    def mouse_up(self, image, x, y, btn, **key):
        pass

    def mouse_move(self, image, x, y, btn, **key):
        pass

    def mouse_wheel(self, image, x, y, d, **key):
        image.img[:] = image.img + d
        key['canvas'].update()

app = wx.App()
frame = wx.Frame(None)
canvas = Canvas(frame, autofit=True)
canvas.set_img(camera())
canvas.set_tool(TestTool())
frame.Show()
app.MainLoop()
```
这一步实际是将画布中的默认绑定的DefaultTool改成了自定义的TestTool，然后再将动作反馈给画布。
从中也可以看出自定义工具中可以调用的接口：
（1）image：即画布承载的Image对象，具体的源图像则是image.img
（2）x和y：当前鼠标所在的坐标，水平方向是x方向，垂直方向是y方向，它的数值没有太多的意义，还有另外一种坐标，是代表鼠标在面板中的像素坐标，该数值更有可解释性；btn是鼠标按键，1为左键按下，2为中键按下，3为右键按下，key是字典，里面有多个字段，key['alt']代表是否按下alt键，key['ctrl']代表是否按下ctrl键，key['shift']代表是否按下shift键，key['px']返回鼠标当前的像素x坐标，key['py']返回像素y坐标，key['canvas']返回该画布自身。

# 工具栏
```python
import wx
from sciwx.widgets import ToolBar
from sciwx.action import Tool

class TestTool(Tool):
    def start(self, app):
        print("i am a tool")

app = wx.App()
frame = wx.Frame(None)
tool = ToolBar(frame)
tool.add_tool('A', TestTool)
tool.add_tools('B', [('A', TestTool), ('C', None)])
tool.Layout()
frame.Fit()
frame.Show()
app.MainLoop()
```
上面的自定义鼠标事件是在后台将默认的鼠标事件进行了重载，无法显示成一个工具。且如果多个工具，每个工具点击后的鼠标事件都不同，是需要将这些事件分开来写的。
这个例子是用来显示工具栏，里面的TestTool纯粹是为了该例子的完整运行，没有任何的实际意义，正规的自定义工具是要重载各种鼠标动作。
实际上的工具栏的鼠标动作是与具体的Canvas相绑定的，所以在没有添加canvas的情况下是无法执行具体工具的。
可以看出：
（1）sciwx现在不仅支持使用icon作为工具图标，也支持使用单个英文字母作为图标，更加方便易用；
（2）sciwx支持一次性添加多个工具。
后面的例子会展示将画布与工具进行绑定。

# 集成面板
CanvasFrame是将画布与菜单栏、工具栏集成显示（实际无法与菜单栏集成，因为菜单栏需要传入app实例，下方具体解释，且后面有具体的集成菜单栏的方法），CanvasNoteFrame进一步地提供标签页功能。
不过这一步仅是测试这两个类能否正确使用，还没有加上特定的菜单栏和工具栏，下面的例子中将具体展示。
同时注意，这一步调用时不再需要像之前的例子那样事先生成一个wxPython的Frame，因为这两个类本身就是Frame，所以省了这一步，相当于定制化了wxPython的frame。
```python
from skimage.data import astronaut, camera
from sciwx.canvas import CanvasFrame, CanvasNoteFrame
import wx

def canvas_frame_test():
    cf = CanvasFrame(None, autofit=True)
    cf.set_imgs([camera(), 255-camera()])
    cf.Show()

def canvas_note_test():
    cnf = CanvasNoteFrame(None)
    cv1 = cnf.add_canvas()
    cv1.set_img(camera())
    cv2 = cnf.add_canvas()
    cv2.set_img(astronaut())
    cv2.set_cn((2,1,0))
    cnf.Show()

if __name__ == '__main__':
    app = wx.App()
    canvas_frame_test()
    canvas_note_test()
    app.MainLoop()
```
CanvasFrame和CanvasNoteFrame归根结底都是调用了Canvas类，且将Canvas一系列的设定方法传递给了它们。
 

# 画布集成工具栏
```python
from skimage.draw import line
from sciwx.canvas import CanvasFrame
from sciwx.action import Tool, DefaultTool

class Pencil(Tool):
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

    def mouse_wheel(self, ips, x, y, d, **key):pass

if __name__=='__main__':
    from skimage.data import camera, astronaut
    from skimage.io import imread

    app = wx.App()
    cf = CanvasFrame(None, autofit=False)
    cf.set_imgs([astronaut(), 255-astronaut()])
    cf.set_cn((0,1,2))
    bar = cf.add_toolbar()
    bar.add_tool('M', DefaultTool)
    bar.add_tool('P', Pencil)
    cf.Show()
    app.MainLoop()
```
这个例子就是将画布与默认工具和画笔工具集成起来。
由于这里是两个工具，那么这个画布是怎样知道该响应哪个工具了吗？原理如下：
（1）ToolBar类中将具体工具绑定了鼠标单击事件，当某一工具被点击后，就会触发它继承自父类Tool的start()方法，将该工具自身传给Tool.default；
（2）Canvas画布类会时刻监听鼠标事件，其中会调用Tool.default，于是，两者就关联了起来。此处Tool.default是种类似“多态”的用法，即直接调用父类，无需知道其具体的子类类型，具体调用则是看运行时类型决定。

# 菜单栏
```python
import wx
from sciwx.widgets import MenuBar

class P:
    def __init__(self, name):
        self.name = name

    def start(self, app):
        print(self.name)

    def __call__(self):
        return self

data = ('menu', [
        ('File', [('Open', P('O')),
                  '-',
                  ('Close', P('C'))]),
        ('Edit', [('Copy', P('C')),
                  ('A', [('B', P('B')),
                         ('C', P('C'))]),
                  ('Paste', P('P'))])])

app = wx.App()
frame = wx.Frame(None)
menubar = MenuBar(frame)
menubar.load(data)
frame.SetMenuBar(menubar)
frame.Show()
app.MainLoop()
```
该例有两个特点：
（1）菜单项是以元组的形式传入MenuBar中，且能多级解析，这就提供了非常大的灵活性，后面ImagePy的丰富的插件系统也能顺利地加载进来；
（2）类P实际就是一个最小的可运行的插件，其重要的一个成员函数就是start()方法，注意到其需要传入一个app参数（这里因为没有用到图像，所以app没有实际用处）。

从菜单栏开始，相比于以前的ImagePy的管理机制，新解耦的sciwx拥有了一个新的管理方式，即App类，如下：
```python
class App:
    def __init__(self):
        self.img_manager = Manager()
        self.wimg_manager = Manager()
        self.tab_manager = Manager()
        self.wtab_manager = Manager()
 
    def show_img(self, img): pass
    def show_table(self, img): pass
    def show_md(self, img, title=''): pass
    def show_txt(self, img, title=''): pass
    def plot(self): pass
    def plot3d(self): pass

    def add_img(self, img):
        print('add', img.name)
        self.img_manager.add(img)

    def remove_img(self, img):
        print('remove', img.name)
        self.img_manager.remove(img)
…
…
```
图像、表格属于基础元素，其管理和展示，归为app自身的功能（以前是通过管理器），创建app实例就可以维护基础元素的信息。这样可以有如下优点：
（1）实现了UI定制，元素的操作与具体的Desktop端、Web端或Headless样式进行解耦。比如sciwx自带一个sciapp模块，它就是一个具体的wxPython的前端实现，同理，也可以自己创建一个web的前端或headless的接口调用；对于headless形式的接口，可以ssh远程登录调用它处理图像，也可以将使用GUI前端形成的处理流程转成headless形式的流程，然后放到服务器上进行运行，这就适用于需要长时间处理图像的大型任务；
（2）创建某个插件时，可以将app对象传入进去，相当于拿着app对象，就可以获取当前打开的各种元素，以显示各种信息。换句话说，插件所需要干的三件事：获取数据、处理数据和展示数据，处理数据时插件自身的功能，而第一个和第三个都是通过与app交互来实现的。与之前相比，不再需要全局的处理函数IPy。

下面就是实际对app的应用实例。
# 集成菜单栏
```python
from scipy.ndimage import gaussian_filter
from sciwx.canvas import CanvasFrame
from sciwx.action import ImgAction
from sciwx.app.manager import App
from sciwx.widgets import MenuBar
 
class Gaussian(ImgAction):
    title = 'Gaussian'
    note = ['auto_snap', 'preview']
    para = {'sigma':2}
    view = [(float, 'sigma', (0, 30), 1, 'sigma', 'pix')]

    def run(self, ips, img, snap, para):
        gaussian_filter(snap, para['sigma'], output=img)

class Undo(ImgAction):
    title = 'Undo'
    def run(self, ips, img, snap, para):
        print(ips.img.mean(), ips.snap.mean())
        ips.swap()

class TestFrame(CanvasFrame, App):
    def __init__ (self, parent):
        CanvasFrame.__init__(self, parent)
        App.__init__(self)

        self.Bind(wx.EVT_ACTIVATE, self.init_image)

    def init_image(self, event):
        self.add_img(self.canvas.image)

    def add_menubar(self):
        menubar = MenuBar(self)
        self.SetMenuBar(menubar)
        return menubar

if __name__=='__main__':
    from skimage.data import camera, astronaut
    from skimage.io import imread

    app = wx.App()
    cf = TestFrame(None)
    cf.set_img(camera())
    cf.set_cn(0)
    bar = cf.add_menubar()
    bar.load(('menu',[('Filter',[('Gaussian', Gaussian),
                                 ('Unto', Undo)]),
                      ]))
    cf.Show()
    app.MainLoop()
```
一个最小可用的例子如上。
需要说明的是：
（1）CanvasFrame定位是带窗口的画布，但因为它默认添加了菜单栏，而菜单栏是插件的组合，所以需要传入app，但当前的CanvasFrame没有提供app接口，所以这里新定义了一个类TestFrame，它继承自原始的CanvasFrame和App，然后将原生的CanvasFrame中的add_menu()方法直接复制并粘贴到新的TestFrame下。
（2）此处的TestFrame是与sciapp等价的，只是TestFrame是最小可用的一个前端实现，而sciapp是大型框架ImagePy的实现。

# 集成工具栏和菜单栏
```python
from scipy.ndimage import gaussian_filter
from skimage.draw import line
from sciwx.canvas import CanvasFrame
from sciwx.action import ImgAction, Tool, DefaultTool
from sciwx.app import App
from sciwx.widgets import MenuBar

class Gaussian(ImgAction):
    title = 'Gaussian'
    note = ['auto_snap', 'preview']
    para = {'sigma':2}
    view = [(float, 'sigma', (0, 30), 1, 'sigma', 'pix')]

    def run(self, ips, img, snap, para):
        gaussian_filter(snap, para['sigma'], output=img)

class Undo(ImgAction):
    title = 'Undo'
    def run(self, ips, img, snap, para): ips.swap()

class Pencil(Tool):
    title = 'Pencil'
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
        se = self.oldp + (y,x)
        rs,cs = line(*[int(i) for i in se])
        rs.clip(0, ips.shape[1], out=rs)
        cs.clip(0, ips.shape[0], out=cs)
        ips.img[rs,cs] = 255
        self.oldp = (y, x)
        key['canvas'].update()

    def mouse_wheel(self, ips, x, y, d, **key):pass

class TestFrame(CanvasFrame, App):
    def __init__ (self, parent):
        CanvasFrame.__init__(self, parent)
        App.__init__(self)
 
        self.Bind(wx.EVT_ACTIVATE, self.init_image)

    def init_image(self, event):
        self.add_img(self.canvas.image)

    def add_menubar(self):
        menubar = MenuBar(self)
        self.SetMenuBar(menubar)
        return menubar

if __name__=='__main__':
    from skimage.data import camera, astronaut
    from skimage.io import imread

    app = wx.App()
    cf = TestFrame(None)
    cf.set_img(camera())
    cf.set_cn(0)

    bar = cf.add_menubar()
    bar.load(('menu',[('Filter',[('Gaussian', Gaussian),
                                 ('Unto', Undo)]),]))

    bar = cf.add_toolbar()
    bar.add_tool('M', DefaultTool)
    bar.add_tool('P', Pencil)
    cf.Show()
    app.MainLoop()
```
这个例子又在上面例子的基础上加了工具栏。

# 全组件集合版
```python
from sciwx.app import SciApp
from sciwx.action import ImgAction, Tool, DefaultTool
from sciwx.plugins.curve import Curve
from sciwx.plugins.channels import Channels
from sciwx.plugins.histogram import Histogram
from sciwx.plugins.viewport import ViewPort
from sciwx.plugins.filters import Gaussian, Undo
from sciwx.plugins.pencil import Pencil
from sciwx.plugins.io import Open, Save

if __name__ == '__main__':
    from skimage.data import camera
   
    app = wx.App(False)
    frame = SciApp(None)
   
    frame.load_menu(('menu',[('File',[('Open', Open),
                                      ('Save', Save)]),
                             ('Filters', [('Gaussian', Gaussian),
                                          ('Undo', Undo)])]))

    frame.load_tool(('tools',[('standard', [('P', Pencil),
                                            ('D', DefaultTool)]),
                              ('draw', [('X', Pencil),
                                        ('X', Pencil)])]), 'draw')

    frame.load_widget(('widgets', [('Histogram', [('Histogram', Histogram),
                                                  ('Curve', Curve),
                                                  ('Channels', Channels)]),
                                   ('Navigator', [('Viewport', ViewPort)])]))

    frame.show_img(camera())
    frame.show_img(camera())
    frame.Show()
    app.MainLoop()
```
该例子可以说是基于sciwx的重构版ImagePy的雏形了，面板、菜单栏、工具栏、直方图、鹰眼灯组件悉数登场，可以说非常完善了。
