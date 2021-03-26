---
title: ImagePy解析：17 -- 重构版ImagePy解析
tags: [ImagePy]
categories: computer vision 
date: 2020-2-26
---

%%%%%%%%
2021.2.14更新:增加了sciapp和sciwx的介绍
%%%%%%%%

新版ImagePy有如下特点：
（1）将原版ImagePy非常特色的可视化组件完全解耦，比如画布、表格、对话框等组件，将其重构为sciwx库，这样第三方开发人员就可以更加方便地使用这些组件而构建自己的特定应用；
（2）创建了一套适用于图像处理的接口标准sciapp，其中定义了图像类Image、表格类Table、几何矢量类Shape，并实现了对这些类的常用操作，即sciapp作为后端支持；
（2）新版ImagePy在sciwx库和sciapp库的基础上进行再集成开发，即底层符合sciapp标准、前端使用sciwx显示，然后提供一整套完善的管理系统和丰富的插件库，从而实现复杂的图像处理功能。

因此，sciwx等价于napari等库，着重于可视化；sciapp等价于ImageJ等库，着重于通用数据接口，ImagePy则是基于两者的插件库。新版ImagePy架构思路清晰，集成更加自然契合，因此，ImagePy/sciapp/sciwx无论是对底层开发人员还是图像处理小白都有着无可比拟的优势：小到开发一个图像处理小工具，大到作为一个大型软件“开箱即用”，都可以轻松应对。

多说一句题外话，多谢龙哥的精辟的总结：对于图像处理问题，图像+矢量+图论三条腿走路。

# 总览
特别地，对sciapp和sciwx包进行一个更为详细的介绍。
如上所述，sciapp负责后端数据操作，sciwx负责前端组件。

## sciapp
sciapp包的介绍主要引用了[官方的介绍](https://github.com/Image-Py/imagepy/blob/master/sciapp/doc/cn_readme.md)。
sciapp包主要有三个重要的模块：Object模块、App模块和Action模块。

### Object模块
Object模块定义了科学计算中常用的基础数据结构封装类，当然，如果仅仅为了计算，绝大多数时候，Numpy，Pandas等数据类型已经可以胜任，这里的封装，主要是面向交互与展示的，例如Image对象是图像数据，里面带了一个lut成员，用于在展示时映射成伪彩色。
（1）Image：多维图像，基于Numpy
（2）Table：表格，基于DataFrame
（3）Shape: 点线面，任意多边形，可与GeoJson，Shapely互转
（4）Surface：三维表面

### App模块
App模块是一个科学容器，里面包含若干管理器managers，用于管理App所持有的上面各类对象object，这里的管理功能包括增加、删除、查询等，即对象object的生命周期都在App管理器中。以图像对象Image为例，App管理器有如下功能：
（1）show_img(self, img, title=None): 展示一个Image对象，并添加到app.img_manager管理器中；
（2）get_img(self, title=None): 根据title获取Image，如果缺省则返回管理器中的第一个Image；
（3）img_names(self): 返回当前app持有的Image对象名称列表；
（4）active_img(self, title=None): 将指定名称的Image对象置顶，以便于get_img可以优先获得；
（5）close_img(self, title=None): 关闭指定图像，并从app.img_manager管理器中移除。

除了这些特定于某种对象的功能，还有一些与用户交互的功能，比如：
（1）alert(self, info, title='sciapp'): 弹出一个提示框，需要用户确认；
（2）yes_no(self, info, title='sciapp'): 要求用户输入True/False；
（3）show_txt(self, cont, title='sciapp'): 对用户进行文字提示；
（4）show_md(self, cont, title='sciapp'): 以MarkDown语法书写，向用户弹出格式化文档；
（5）show_para(self, title, para, view, on_handle=None, on_ok=None, on_cancel=None, on_help=None, preview=False, modal=True): 展示交互对话框，para是参数字典，view指定了交互方式。

但是，需要特别注意的是，这里的App中的这些交互功能，都只是在命令行中print信息，具体使用时需要在子类中用UI框架（比如sciwx）重载这些方法。

### Action模块
Action模块是对App所管理的对象的操作，比如对图像做滤波等。因此，该模块也是后面自定义开发时打交道最多的模块。
该模块与App的交互只需通过它的start函数即可，即将App类的实例app传入即可：
```python
class SciAction:
    '''base action, just has a start method, alert a hello'''
    name = 'SciAction'

    def start(self, app, para=None):
        self.app = app
        app.alert('Hello, I am SciAction!\n')

app = App()
SciAction().start(app)
```
SciAction是所有Action的基类，它定义了最基本的功能，同时，sciapp提供了更高级的模板，供开发者的自定义action用于继承，比如：
（1）ImgAction：用于处理图像，自动获取当前图像，需要重载para、view进行交互，重载run进行图像处理；
（2）Tool：工具，用于在某种控件上的鼠标交互，同时其派生出了图像工具ImageTool、表格工具TableTool、矢量编辑工具ShapeTool（如点线面绘制）。

另外，Advanced目录下有一些高级模板（如支持图像多通道、批量操作、多线程支持等），供扩展插件时使用；Plugins目录下也有一些带有具体功能的、开箱即用的Action。

## sciwx
sciwx提供了一系列基于wxPython的前端可视化组件，其中最重要的就是可视化2D图像的画布功能。
### Canvas画布
Canvas画布是定制化的wxPython的Panel，其详细解析可见[该文](https://qixinbo.info/2019/10/29/imagepy_12/)。

### ICanvas画布、MCanvas组件、CanvasNoteBook组件、CanvasFrame应用和CanvasNoteFrame应用
ICanvas是在Canvas基础上对于位图的展示提供进一步的接口支持，比如默认绑定ImageTool这种Action，提供set_img设置图像、set_rg设置数值范围、set_lut设置快速查找表、set_cn设置通道、set_tool设置工具等接口。
MCanvas是对ICanvas的进一步包装，比如在顶部添加显示图像信息的信息条、在底部增加可以切换某一通道、某一slice的滑动条。
CanvasNoteBook组件是对MCanvas的多标签页管理，即每一个标签页都可以添加一个MCanvas。
以上几个组件实际都是深度定制的前端组件，而接下来的CanvasFrame和CanvasNoteFrame是同时拥有前端和后端的功能，它们的父类同时是wx.Frame和上面的sciapp的App类，应该可以说这两个是可以独立运行的开箱即用的应用。
CanvasFrame是对MCanvas的封装，可以使用上面MCanvas的设置接口，同时还可以增加菜单栏、工具栏以及显示对话框等。
CanvasNoteFrame是对CanvasNoteBook的封装，即增加了标签页管理。

### VCanvas画布、SCanvas组件、VectorNoteBook组件、VectorFrame应用和VectorNoteFrame应用
VCanvas是在Canvas基础上对于矢量形状的展示提供进一步的接口支持，比如默认绑定ShapeTool这种Action，提供set_shp设置形状、set_tool设置工具等接口。
Scanvas是对VCanvas的进一步包装，比如在顶部添加显示形状信息的信息条。
VectorNoteBook组件时对VCanvas的多标签页管理。
同上，VectorFrame和VectorNoteFrame也都是兼具后端和前端功能的应用。

## 前端和后端耦合
如上所述，sciapp负责后端，sciwx负责前端，两者联动的机理如下：
（1）通过sciapp的dataio模块来控制输入输出，将图像等对象添加进App管理器；
（2）将App管理器传入其他action模块的start()入口函数，即可实现对图像等对象的操作；
（3）sciwx前端组件通过set_img()等接口接收App管理器，并将之可视化。
其中，第二步可以通过代码执行，比如：
```python
Gaussian().start(app)
```
但也可以通过前端交互，比如通过菜单命令和工具栏的鼠标操作。那么菜单栏和工具栏又是怎样识别这些命令的呢？
对于菜单栏：
```python
class MenuBar(wx.MenuBar):
    def __init__(self, app):
        wx.MenuBar.__init__(self)
        self.app = app
        app.SetMenuBar(self)
....
....
            f = lambda e, p=vs: p().start(self.app)
            self.Bind(wx.EVT_MENU, f, item)
```
注意两个地方：
一个是MenuBar的初始化函数，需要传入app实例；第二是在添加菜单项时，其功能通过lambda函数调用了命令的start()函数。
对于工具栏：
```python
class ToolBar(wx.Panel):
    def __init__(self, parent, vertical=False):
        self.app = parent

        btn.Bind( wx.EVT_LEFT_DOWN, lambda e, obj=obj: self.on_tool(e, obj))

    def on_tool(self, evt, tol):
        tol.start(self.app)
```
仍然是两个地方：ToolBar的初始化函数也传入parent了，它实际也是app实例；第二也是鼠标按下操作绑定了工具命令的start()函数。

这个作为后端服务的app实例可以额外创建，但是通常做法是将前端和后端联合起来，创建一个组合体，即：
```python
class ImageApp(wx.Frame, App):
    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'ImageApp',
                            size = wx.Size(800,600), pos = wx.DefaultPosition,
                            style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        App.__init__(self)
```
因此，添加菜单栏和工具栏时，传入的都是self，即自身，因为其自身就有App管理器的能力。


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
（2）x和y：当前鼠标所在的图像像素坐标，水平方向是x方向，垂直方向是y方向，这两个坐标是与图像的Numpy数据存储一一对应的：因为Numpy的元素获取是先row后column，所以这里的y对应row，x对应column。另外因为这两个坐标有可能因为缩放、移动而产生负值，此时如果要做画笔这类的应用的话，就需要clip一下，保证画笔始终在图像中；不过另一方面，这样的设置也使得可以draw出超出图像的更大的ROI。还有下面一种坐标，是代表鼠标在面板中的像素坐标。btn是鼠标按键，1为左键按下，2为中键按下，3为右键按下，key是字典，里面有多个字段，key['alt']代表是否按下alt键，key['ctrl']代表是否按下ctrl键，key['shift']代表是否按下shift键，key['px']返回鼠标当前的画布x坐标，key['py']返回画布y坐标，key['canvas']返回该画布自身。

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
