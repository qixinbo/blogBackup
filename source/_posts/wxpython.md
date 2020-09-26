---
title: wxPython知识点
tags: [Image, GUI]
categories: computational material science 
date: 2020-9-26
---

# 简介
本文是对ZetCode上wxPython的摘抄学习，原系列文章见[这里](http://zetcode.com/wxpython/)。

wxPython是一个开发桌面端图形界面的跨平台函数库，开发语言为Python，它是基于C++的函数库wxWidgets的封装。
wxpython有大量组件，它们可以从逻辑上（注意是逻辑上）这样划分：
（1）基础组件
![base](https://user-images.githubusercontent.com/6218739/93730533-9eee4580-fbfb-11ea-8aba-88f4e73ccc42.jpg)
这些组件为其所派生的子组件提供基础功能，通常不直接使用。
（2）顶层组件
![toplevel](https://user-images.githubusercontent.com/6218739/93730576-d2c96b00-fbfb-11ea-9a25-b4c88eccf867.jpg)
这些组件相互独立存在。
（3）容器
![containers](https://user-images.githubusercontent.com/6218739/93730612-ef65a300-fbfb-11ea-9132-1457ffe71d7e.jpg)
这些组件包含其他组件。
（4）动态组件
![dynamic](https://user-images.githubusercontent.com/6218739/93730642-1623d980-fbfc-11ea-8f9d-bbeed738ef73.jpg)
这些组件可以被用户所交互编辑。
（5）静态组件
![staticwidgets](https://user-images.githubusercontent.com/6218739/93730655-276ce600-fbfc-11ea-946c-96378f28d953.jpg)
这些组件用来展示信息，无法被用户所交互编辑。
（6）其他组件
![bars](https://user-images.githubusercontent.com/6218739/93730678-453a4b00-fbfc-11ea-82e4-3060641039e0.jpg)
这些组件包括状态栏、工具栏、菜单栏等。

除了逻辑上的划分，各个组件之间还存在着继承关系，以一个button组件为例：
![inheritance](https://user-images.githubusercontent.com/6218739/93730809-dc070780-fbfc-11ea-84b4-77c4ae347d04.png)
Button是一个小window，具体地，它是继承自wx.Control这一类的window（有些组件是window，但不是继承自wx.Control，比如wx.Dialog，更具体来说，controls这类组件是可放置在containers这类组件上的组件）。同时所有的windows都可以响应事件，button也不例外，因此它还继承自wx.EvtHandler。最后，所有的wxpython对象都继承自wx.Object类。

# wxPython的“你好世界”
这个例子是wxPython的最小可用例子，用来say hello to the world:
```python
import wx

app = wx.App()
frame = wx.Frame(None, title='Hello World')
frame.Show()

app.MainLoop()
```
麻雀虽小五脏俱全，该例子包含了最基本的代码和组件：
（1）首先导入wxPython库：
```python
import wx
```
wx可视为一个命名空间，后面所有的函数和类都以它开头。
（2）创建应用实例：
```python
app = wx.App()
```
每一个wxPython程序都必须有一个应用实例。
（3）创建应用框架并显示：
```python
frame = wx.Frame(None, title='Hello World')
frame.Show()
```
这里创建了一个wx.Frame对象。wx.Frame是一个重要的“容器”组件，它用来承载其他组件，它本身没有父组件（如果我们给组件的parent参数设为None，即代表该组件没有父组件）。创建该对象后，还需调用Show方法才能显示出来。

wx.Frame的构造函数一共有七个参数：
```python
wx.Frame(wx.Window parent, int id=-1, string title='', wx.Point pos=wx.DefaultPosition,
    wx.Size size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, string name="frame")
```
除了第一个parent参数需要显式指定，其余六个都有默认值，包括ID、名称、位置、尺寸和样式等。因此，可以通过改变这些参数来进一步地对该frame进行个性化定制。
（4）启动程序主循环：
```python
app.MainLoop()
```
程序的主循环是一个无限循环模式，它捕获并分发程序生命周期内的所有事件。

# 菜单栏和工具栏
## 菜单栏
菜单栏主要由三部分组成：wx.MenuBar、wx.Menu和wx.MenuItem。
在菜单栏MenuBar中可以添加菜单Menu，在菜单Menu中又可以添加菜单项MenuItem。
添加完后不要忘了使用SetMenuBar来将菜单栏加入到框架中。
进一步地，在某个菜单Menu中，还可以添加子菜单SubMenu，然后继续添加菜单项。
还可以给菜单设置图标、快捷键、对wx.EVT_MENU事件的动作、菜单样式（打勾、单选）等。

## 上下文菜单
上下文菜单有时叫做“弹出菜单”，比如右键某个位置，出现上下文选项。

## 工具栏
工具栏的添加也是类似流程：先添加工具栏CreateToolBar，然后在上面添加工具AddTool。
别忘了使用toolbar.Realize()使之呈现出来（这一步与操作系统有关，Linux上不强制使用，Windows必须使用，为了跨平台性，最好将这一步明确写出）。
对于某个工具，可以设置逻辑使之Enable或Disable，常见的比如undo和redo，这两个按钮不是一直可以点的，在最开始时redo就必须是disabled，因为没有历史操作，所以可以设置具体的逻辑使之disable掉。

## 状态栏
状态栏即底部显示当前状态的状态条。
 

# 布局管理
布局可以分为绝对布局和布局管理器sizer。绝对布局有很多缺点，比如：
（1）组件的尺寸和位置不随窗口的改变而改变；
（2）不同平台上应用程序可能显示不同；
（3）字体的改变可能破坏布局；
（4）如果想改变布局，必须将之前的全部推翻。

因此，推荐使用布局管理器sizer来管理布局。
wxPython常用的sizer有：wx.BoxSizer、wx.StaticBoxSizer、wx.GridSizer、wx.FlexGridSizer、wx.GridBagSizer。
## wx.BoxSizer
wx.BoxSizer是最常见的布局管理器。它的常用设置有：
（1）排列方向：wx.VERTICAL垂直排列还是wx.HORIZONTAL水平排列；
（2）排列比例：一个布局中所包含的组件的尺寸由其比例所决定，比例为0表示在窗口尺寸变化时保持尺寸不变，其他比例系数表示组件在该布局管理器中的尺寸占比；且通常使用wx.EXPAND旗标来使得组件占据管理器分配给它的所有空间；
（3）边界：组件的边界大小可以自定义设置，同时具体哪个边界（上下左右或全部）都可以任意指定；
（4）对齐方式：可以设定左端对齐、右端对齐、顶部对齐、底部对齐、中心对齐等多种对齐方式；
（5）在某一级容器组件中，使用SetSizer()来为其指定布局管理器；
（6）在布局管理器中用Add()方法来添加组件。

wx.StaticBoxSizer是在BoxSizer周围加上了一个静态文本框的显示。

## wx.GridSizer
wx.GridSizer是网格布局管理器，可以设置几排几列以及横纵的间距，网格中的组件尺寸都是相同的。
（如果有的网格不需要添加组件，可以添加没有内容的StaticText作为占位符）

## wx.FlexGridSizer
wx.FlexGridSizer与wx.GridSizer类似，但其更灵活，它不要求网格中所有的组件尺寸都相同，而是在同一行中的所有组件都高度相同，而同一列中的所有组件都宽度相同。
它还可以设置能growable的行和列，即在当前sizer中如果有空间，就将特定的行和列调整相应的大小来占据这个空间（注意将该行或列中的组件设为expandable）。

## wx.GridBagSizer
wx.GridBagSizer是wxPython中最灵活的sizer（不仅仅是wxPython，其他函数库也有类似的配置），它可以显式地指定sizer中组件所占据的区域，比如横跨几行几列等。
它的构造函数很简单：
```python
wx.GridBagSizer(integer vgap, integer hgap)
```
只需设定间距，然后通过Add()方法添加组件：
```python
Add(self, item, tuple pos, tuple span=wx.DefaultSpan, integer flag=0,
    integer border=0, userData=None)
```
pos参数指定组件在这个虚拟网格中的起始位置，(0, 0)就代表左上角，span就指定它横跨几行几列，比如(3, 2)代表占据3行2列。
如果想组件可以随窗口伸缩，别忘了设置expandle属性，及：
```python
AddGrowableRow(integer row)
AddGrowableCol(integer col)
```

## Sizer常见问题
大部分的问题出现在：
（1）设置比例proportional错误，只有需要随窗口变化的组件和sizer才需要设置为非0，其他都设置为0。且sizer和里面的组件可分别设置，比如下面的：
```python
self.panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
vbox = wx.BoxSizer( wx.VERTICAL )

hbox1 = wx.BoxSizer( wx.HORIZONTAL )

self.st1 = wx.StaticText( self.panel, wx.ID_ANY, u"Class Name", wx.DefaultPosition, wx.DefaultSize, 0 )
self.st1.Wrap( -1 )
hbox1.Add( self.st1, 0, wx.RIGHT, 8 )

self.tc = wx.TextCtrl( self.panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
hbox1.Add( self.tc, 1, 0, 5 )

vbox.Add( hbox1, 0, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, 10 )
```
在vbox中添加了hbox1，hbox1中又添加了静态文本框st1和输入框tc，hbox1的比例为0，代表在vbox这一垂直排列的管理器变化时，hbox1尺寸不变化，但tc的比例又为1，所以vbox在垂直变化时，tc按着hbox1不变化，但vbox水平变化时，tc就会随着变化。这样就有非常高的适应性。
（总结起来：看是否expandable要看组件所在的sizer！！）
（2）边界border尺寸设置不统一，导致对不齐
（3）Expandable属性和proportion两个中有一个忘了设置，导致组件不能随窗口伸缩。

这个sizer的编写此处可以借助wxFormBuilder工具来进行设计，实现所想即所得。（wxFormBuilder能够实现即时的改变，但此处遇到一个小问题，在wxGridBagSizer设置了某列进行可伸缩后，在wxFormBuilder中却不能正确伸缩，反而generate code后直接调用能正确伸缩，所以也不能完全相信，但可以99%相信，实在调不通后可以换种运行方式接着调）

# 事件

## 事件
事件是一个图形界面程序的核心部分，任何图形界面程序都是事件驱动的。
事件可以有多重产生方式，大部分是用户触发的，也有可能由其他方式产生，比如网络连接、窗口管理和计时器调用等。
关于事件，这里面有几个过程和要素：
事件循环Event Loop（比如wxPython的MainLoop()方法），它一直在寻找和捕获事件Event（比如wx.EVT_SIZE、wx.EVT_CLOSE等）；当捕获到事件后，就通过分发器dispatcher将事件分发到事件句柄Event Handler（事件句柄是对事件进行响应的动作方法）；事件本身与事件句柄的映射由Event Binder来完成（即Bind()方法）。
对用户编程来说，最常打交道的就是Bind()方法：
```python
Bind(event, handler, source=None, id=wx.ID_ANY, id2=wx.ID_ANY)
```
溯源起来，该Bind()方法是定义在EvtHandler类中，而EvtHandler又派生了Window类，Window类又是绝大多数组件的父类，因此可以在组件中直接使用该方法（如果想将事件解绑，则可以调用Unbind()方法，其参数跟下面的参数相同）。
event参数就是事件对象，它指定了事件类型；
handler就是对该事件的响应方法，这个通常要由编程自定义完成；
source是指该事件来自于哪个组件，比如很多组件都能产生同样的事件，就需要指定具体的来源，比如很多button都能产生鼠标点击事件。这里面就有一个很tricky的地方，假设self是一个panel，该panel上有很多buttons，名为bt1、bt2，那么self.Bind(event, handler, source=self.bt1)和self.bt1.Bind(event, handler)有什么区别呢？两者看起来的效果是相同的，[这里有一个帖子详细说明了两者的区别](https://wiki.wxpython.org/self.Bind%20vs.%20self.button.Bind)；
id是通过ID来指定事件来源，而上面的source是通过直接指定实例，两者目的相同；关于组件的ID，主要有两种创建方式：
（1）让系统自动创建：即使用-1或wx.ID_ANY，系统自动创建的ID都是负数，因此用户自己创建的ID都应该是正数，此种情况通常用于不用改变状态的组件。可以使用GetId()来获取该隐形id；
（2）标准ID：wxPython提供了一些标准IDs，比如wx.ID_SAVE、wx.ID_NEW等；
id2是指定多个IDs，上面的id是一次只能指定单个ID。

这里面有个很好玩的用法，如果想批量给多个同类组件绑定事件，可以用lambda函数，比如：
```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'huangbinghe@gmail.com'

import wx

class TestFrm(wx.Frame):
    """TestFrm"""
    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        panel = wx.Panel(self, -1)
        box = wx.BoxSizer(wx.VERTICAL)
        for i in range(5):
            btn = wx.Button(panel, -1, label="test-{}".format(i))
            btn.Bind(wx.EVT_BUTTON, lambda e, mark=i: self.on_click(e, mark))
            box.Add(btn, 0, wx.LEFT)

        panel.SetSizer(box)

    def on_click(self, event, mark):
        wx.MessageDialog(self, 'click mark:{}'.format(
            mark), 'click btn', wx.ICON_INFORMATION).ShowModal()
 
if __name__ == '__main__':
    app = wx.App()
    frm = TestFrm(None, title="hello world")
    frm.Show()
    app.MainLoop()
```

## 事件传播
有两种类型的事件：basic events和command events。它们两者的区别在于是否传播上。事件的传播是指事件从触发该事件的子组件开始，传递给其父组件，并观察其响应。Basic events不传播，而command events传播。比如wx.CloseEvent就是一个basic event，它不传播，因为如果传播给父组件就很没有道理。
默认情形下，在事件句柄中的事件是阻止传播的，如果想让它继续传播，需要调用skip()方法（这个也解释了上面的self.Bind(event, handler, source=self.bt1)和self.bt1.Bind(event, handler)的区别）。

## 常见事件
窗口移动事件：wx.EVT_MOVE
窗口销毁事件：wx.EVT_CLOSE，发生在点击工具栏的关闭按钮、Alt+F4或从开始菜单关闭计算机时（注意销毁窗口是destroy()方法）
按钮事件：wx.EVT_BUTTON，点击一个按钮时
菜单事件：wx.EVT_MENU，点击一个菜单时
绘图事件：wx.EVT_PAINT，改变窗口尺寸或最大化窗口时（最小化窗口时不会产生该事件）
焦点事件：wx.EVT_SET_FOCUS，当某组件成为焦点时；wx.EVT_KILL_FOCUS，当某组件失去焦点时
键盘事件：wx.EVT_KEY_DOWN，键盘按下；wx.EVT_KEY_UP，键盘弹起；wx.EVT_CHAR，这个应该是为了兼容非英语字符。

# 对话框
对话框是一种非常重要的人机交互的手段，可以使得用户输入数据、修改数据、更改程序配置等。

## 预定义的消息对话框
消息对话框是为了向用户展示消息，可以通过一些预定义的旗标来定制消息对话框的按钮和图标，如下图所示：
![messagebox](https://user-images.githubusercontent.com/6218739/94092464-6b0a5e80-fe4d-11ea-86d0-1ed4627434f5.png)

## 自定义对话框
若想自定义对话框，只需继承wx.Dialog即可。

# 常用组件
## 基础组件
wxPython提供了大量基础组件，如：
基础按钮Button；
图形按钮BitmapButton；
切换按钮ToggleButton（有两种状态可以切换：按下和未按下）；
静态文本框StaticText（展示一行或多行只读文本）；
文本输入框TextCtrl；
富文本输入框RichTextCtrl可以加入图像、文字色彩等效果；
带格式文本输入框StyledTextCtrl；
超链接HyperLinkCtrl；
静态位图：StaticBitmap；
静态分割线StaticLine（可垂直可水平）；
静态框StaticBox（为了装饰用，将多个组件组合在一起显示）；
下拉列表框ComboBox；
可编辑的下拉列表框Choice；
复选框CheckBox（有两个状态：勾选或未勾选）；
单选按钮RadioButton（单选按钮是从一组选项中只能选择一个，将多个单选按钮组合成一个选项组时，只需设定第一个单选按钮style为wx.RB_GROUP，后面跟着的那些单选按钮就自动跟它一组，如果想另开一组，只需再将另一组的第一个单选按钮的style设置为wx.RB_GROUP）；
进度条Gauge；
滑动条Slider；
整数数值调节钮SpinCtrl；
浮点数数值调节钮SpinCtrlDouble；
滚动条ScrollBar。

## 高级组件
列表框ListBox：是对一组选项的展示和交互，它有两个主要的事件，一个是wx.EVT_COMMAND_LISTBOX_SELECTED，即鼠标单击某一项时产生；另一个是wx.EVT_COMMAND_LISTBOX_DOUBLE_CLICKED，即鼠标双击某一项时产生。
列表视图ListCtrl：也是用来展示一组选项，与ListBox不同的是，ListBox仅能展示一列，而ListCtrl能展示多列。ListCtrl有三种视图模式：list、report和icon。向ListCtrl中插入数据需要使用两种方法：首先使用InsertItem()方法获得行号，然后再在当前行中使用SetItem()方法在列中插入数据。
Mixins：Mixins增强了ListCtrl的功能，它们都在wx.lib.mixins.listctrl这个模块中，一共有六种Mixins：
（1）wx.ColumnSorterMixin：使得在report视图中对列进行排序；
（2）wx.ListCtrlAutoWidthMixin：自动调整最后一列的宽度来占据剩余的空间；
（3）wx.ListCtrlSelectionManagerMix：定义了与系统无关的选择策略；
（4）wx.TextEditMixin：使得可以编辑文本；
（5）wx.CheckListCtrlMixin：给每一行增加了一个复选框；
（6）wx.ListRowHighlighter：候选行自动背景高亮。
wx.html.HtmlWindow：用来展示HTML页面。
wx.SplitterWindow：包含两个子窗口（如果使用wxFormBuilder，注意手动添加上两个panel）

另外还有比如：
树状结构TreeCtrl；
表格Grid；
搜索框SearchCtrl；
调色板ColourPickerCtrl；
字体设置器FontPickerCtrl；
文件选择器FilePickerCtrl；
文件目录选择器DirPickerCtrl；
文件树选择器GenericDirCtrl；
日期选择器DatePickerCtrl；
日历CalenderCtrl。

# 绘图
wxPython的绘图之前写过，参见以下两篇：
[ImagePy解析：6 -- wxPython GDI绘图和FloatCanvas](https://qixinbo.info/2019/09/07/imagepy_6/)
[ImagePy解析：11 -- 使用wxPython设备上下文绘图](https://qixinbo.info/2019/10/16/imagepy_11/)

# 自定义组件
如上，wxPython的常用组件已经有很多，但仍然不能涵盖真实情况下的千奇百怪的需求，这时候就要根据自己的需求自定义组件。
自定义组件有两种方式：一种是在现有组件的基础上修改或增强，这种方式仍然有一定的限制；另一种是结合wxPython的GDI绘图，自己从头创建组件，这种方式就具有极大的灵活性。
从头绘制组件一般都是在wx.Panel基础上进行创建。

# 俄罗斯方块
下面给了一个俄罗斯方块的游戏程序代码，可以说是一个使用wxPython编写GUI程序的集大成者：
```python
#!/usr/bin/python

"""
ZetCode wxPython tutorial
This is Tetris game clone in wxPython.
author: Jan Bodnar
website: www.zetcode.com
last modified: April 2018
"""
 
import wx
import random

class Tetris(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, size=(180, 380),
            style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER ^ wx.MAXIMIZE_BOX)

        self.initFrame()

    def initFrame(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText('0')
        self.board = Board(self)
        self.board.SetFocus()
        self.board.start()
        self.SetTitle("Tetris")
        self.Centre()

class Board(wx.Panel):
    BoardWidth = 10
    BoardHeight = 22
    Speed = 300
    ID_TIMER = 1

    def __init__(self, *args, **kw):
        # wx.Panel.__init__(self, parent)
        super(Board, self).__init__(*args, **kw)
        self.initBoard()

    def initBoard(self):
        self.timer = wx.Timer(self, Board.ID_TIMER)
        self.isWaitingAfterLine = False
        self.curPiece = Shape()
        self.nextPiece = Shape()
        self.curX = 0
        self.curY = 0
        self.numLinesRemoved = 0
        self.board = []
        self.isStarted = False
        self.isPaused = False

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_TIMER, self.OnTimer, id=Board.ID_TIMER)

        self.clearBoard()

    def shapeAt(self, x, y):
        return self.board[(y * Board.BoardWidth) + x]

    def setShapeAt(self, x, y, shape):
        self.board[(y * Board.BoardWidth) + x] = shape

    def squareWidth(self):
        return self.GetClientSize().GetWidth() // Board.BoardWidth

    def squareHeight(self):
        return self.GetClientSize().GetHeight() // Board.BoardHeight

    def start(self):
        if self.isPaused:
            return

        self.isStarted = True
        self.isWaitingAfterLine = False
        self.numLinesRemoved = 0
        self.clearBoard()
        self.newPiece()
        self.timer.Start(Board.Speed)

    def pause(self):
        if not self.isStarted:
            return

        self.isPaused = not self.isPaused
        statusbar = self.GetParent().statusbar

        if self.isPaused:
            self.timer.Stop()
            statusbar.SetStatusText('paused')
        else:
            self.timer.Start(Board.Speed)
            statusbar.SetStatusText(str(self.numLinesRemoved))

        self.Refresh()

    def clearBoard(self):
        for i in range(Board.BoardHeight * Board.BoardWidth):
            self.board.append(Tetrominoes.NoShape)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        size = self.GetClientSize()
        boardTop = size.GetHeight() - Board.BoardHeight * self.squareHeight()
 
        for i in range(Board.BoardHeight):
            for j in range(Board.BoardWidth):
                shape = self.shapeAt(j, Board.BoardHeight - i - 1)

                if shape != Tetrominoes.NoShape:
                    self.drawSquare(dc,
                        0 + j * self.squareWidth(),
                        boardTop + i * self.squareHeight(), shape)

        if self.curPiece.shape() != Tetrominoes.NoShape:
            for i in range(4):
                x = self.curX + self.curPiece.x(i)
                y = self.curY - self.curPiece.y(i)

                self.drawSquare(dc, 0 + x * self.squareWidth(),
                    boardTop + (Board.BoardHeight - y - 1) * self.squareHeight(),
                    self.curPiece.shape())

    def OnKeyDown(self, event):
        if not self.isStarted or self.curPiece.shape() == Tetrominoes.NoShape:
            event.Skip()
            return

        keycode = event.GetKeyCode()

        if keycode == ord('P') or keycode == ord('p'):
            self.pause()
            return

        if self.isPaused:
            return

        elif keycode == wx.WXK_LEFT:
            self.tryMove(self.curPiece, self.curX - 1, self.curY)

        elif keycode == wx.WXK_RIGHT:
            self.tryMove(self.curPiece, self.curX + 1, self.curY)

        elif keycode == wx.WXK_DOWN:
            self.tryMove(self.curPiece.rotatedRight(), self.curX, self.curY)

        elif keycode == wx.WXK_UP:
            self.tryMove(self.curPiece.rotatedLeft(), self.curX, self.curY)

        elif keycode == wx.WXK_SPACE:
            self.dropDown()

        elif keycode == ord('D') or keycode == ord('d'):
            self.oneLineDown()

        else:
            event.Skip()

    def OnTimer(self, event):
        if event.GetId() == Board.ID_TIMER:
            if self.isWaitingAfterLine:
                self.isWaitingAfterLine = False
                self.newPiece()
            else:
                self.oneLineDown()
        else:
            event.Skip()

    def dropDown(self):
        newY = self.curY
        while newY > 0:
            if not self.tryMove(self.curPiece, self.curX, newY - 1):
                break
            newY -= 1
        self.pieceDropped()

    def oneLineDown(self):
        if not self.tryMove(self.curPiece, self.curX, self.curY - 1):
            self.pieceDropped()

    def pieceDropped(self):
        for i in range(4):
            x = self.curX + self.curPiece.x(i)
            y = self.curY - self.curPiece.y(i)
            self.setShapeAt(x, y, self.curPiece.shape())

        self.removeFullLines()
        if not self.isWaitingAfterLine:
            self.newPiece()

    def removeFullLines(self):
        numFullLines = 0
        statusbar = self.GetParent().statusbar
        rowsToRemove = []

        for i in range(Board.BoardHeight):
            n = 0
            for j in range(Board.BoardWidth):
                if not self.shapeAt(j, i) == Tetrominoes.NoShape:
                    n = n + 1

            if n == 10:
                rowsToRemove.append(i)

        rowsToRemove.reverse()

        for m in rowsToRemove:
            for k in range(m, Board.BoardHeight):
                for l in range(Board.BoardWidth):
                        self.setShapeAt(l, k, self.shapeAt(l, k + 1))

            numFullLines = numFullLines + len(rowsToRemove)

            if numFullLines > 0:
                self.numLinesRemoved = self.numLinesRemoved + numFullLines
                statusbar.SetStatusText(str(self.numLinesRemoved))
                self.isWaitingAfterLine = True
                self.curPiece.setShape(Tetrominoes.NoShape)
                self.Refresh()
 
    def newPiece(self):
        self.curPiece = self.nextPiece
        statusbar = self.GetParent().statusbar
        self.nextPiece.setRandomShape()

        self.curX = Board.BoardWidth // 2 + 1
        self.curY = Board.BoardHeight - 1 + self.curPiece.minY()
 
        if not self.tryMove(self.curPiece, self.curX, self.curY):
            self.curPiece.setShape(Tetrominoes.NoShape)
            self.timer.Stop()
            self.isStarted = False
            statusbar.SetStatusText('Game over')

    def tryMove(self, newPiece, newX, newY):
        for i in range(4):
            x = newX + newPiece.x(i)
            y = newY - newPiece.y(i)

            if x < 0 or x >= Board.BoardWidth or y < 0 or y >= Board.BoardHeight:
                return False

            if self.shapeAt(x, y) != Tetrominoes.NoShape:
                return False

        self.curPiece = newPiece
        self.curX = newX
        self.curY = newY
        self.Refresh()

        return True

    def drawSquare(self, dc, x, y, shape):
        colors = ['#000000', '#CC6666', '#66CC66', '#6666CC',
                  '#CCCC66', '#CC66CC', '#66CCCC', '#DAAA00']

        light = ['#000000', '#F89FAB', '#79FC79', '#7979FC',
                 '#FCFC79', '#FC79FC', '#79FCFC', '#FCC600']

        dark = ['#000000', '#803C3B', '#3B803B', '#3B3B80',
                 '#80803B', '#803B80', '#3B8080', '#806200']

        pen = wx.Pen(light[shape])
        pen.SetCap(wx.CAP_PROJECTING)
        dc.SetPen(pen)

        dc.DrawLine(x, y + self.squareHeight() - 1, x, y)
        dc.DrawLine(x, y, x + self.squareWidth() - 1, y)

        darkpen = wx.Pen(dark[shape])
        darkpen.SetCap(wx.CAP_PROJECTING)
        dc.SetPen(darkpen)

        dc.DrawLine(x + 1, y + self.squareHeight() - 1,
            x + self.squareWidth() - 1, y + self.squareHeight() - 1)
        dc.DrawLine(x + self.squareWidth() - 1,
        y + self.squareHeight() - 1, x + self.squareWidth() - 1, y + 1)

        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush(colors[shape]))
        dc.DrawRectangle(x + 1, y + 1, self.squareWidth() - 2,
        self.squareHeight() - 2)

class Tetrominoes(object):
    NoShape = 0
    ZShape = 1
    SShape = 2
    LineShape = 3
    TShape = 4
    SquareShape = 5
    LShape = 6
    MirroredLShape = 7

class Shape(object):
    coordsTable = (
        ((0, 0),     (0, 0),     (0, 0),     (0, 0)),
        ((0, -1),    (0, 0),     (-1, 0),    (-1, 1)),
        ((0, -1),    (0, 0),     (1, 0),     (1, 1)),
        ((0, -1),    (0, 0),     (0, 1),     (0, 2)),
        ((-1, 0),    (0, 0),     (1, 0),     (0, 1)),
        ((0, 0),     (1, 0),     (0, 1),     (1, 1)),
        ((-1, -1),   (0, -1),    (0, 0),     (0, 1)),
        ((1, -1),    (0, -1),    (0, 0),     (0, 1))
    )

    def __init__(self):
        self.coords = [[0,0] for i in range(4)]
        self.pieceShape = Tetrominoes.NoShape
        self.setShape(Tetrominoes.NoShape)

    def shape(self):
        return self.pieceShape

    def setShape(self, shape):
        table = Shape.coordsTable[shape]
        for i in range(4):
            for j in range(2):
                self.coords[i][j] = table[i][j]

        self.pieceShape = shape

    def setRandomShape(self):
        self.setShape(random.randint(1, 7))

    def x(self, index):
        return self.coords[index][0]

    def y(self, index):
        return self.coords[index][1]

    def setX(self, index, x):
        self.coords[index][0] = x

    def setY(self, index, y):
        self.coords[index][1] = y

    def minX(self):
        m = self.coords[0][0]
        for i in range(4):
            m = min(m, self.coords[i][0])

        return m

    def maxX(self):
        m = self.coords[0][0]
        for i in range(4):
            m = max(m, self.coords[i][0])

        return m

    def minY(self):
        m = self.coords[0][1]
        for i in range(4):
            m = min(m, self.coords[i][1])

        return m

    def maxY(self):
        m = self.coords[0][1]

        for i in range(4):
            m = max(m, self.coords[i][1])

        return m

    def rotatedLeft(self):
        if self.pieceShape == Tetrominoes.SquareShape:
            return self

        result = Shape()
        result.pieceShape = self.pieceShape

        for i in range(4):
            result.setX(i, self.y(i))
            result.setY(i, -self.x(i))

        return result

    def rotatedRight(self):
        if self.pieceShape == Tetrominoes.SquareShape:
            return self

        result = Shape()
        result.pieceShape = self.pieceShape

        for i in range(4):
            result.setX(i, -self.y(i))
            result.setY(i, self.x(i))

        return result

def main():
    app = wx.App()
    ex = Tetris(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
```
效果如图：
![image](https://user-images.githubusercontent.com/6218739/94236695-4216c600-ff40-11ea-93e8-963195be2d10.png)

不过我在运行上述代码时，出现了无法使用箭头键来控制方块的情形，解决方式在Board这个panel中设置一个旗标：
```python
        super(Board, self).__init__(*args, **kw, style=wx.WANTS_CHARS)
```
该问题的讨论在：
[how to catch arrow keys ?](http://wxpython-users.1045709.n5.nabble.com/how-to-catch-arrow-keys-td2365210.html)
[Stumped: arrows/tab kills keyboard focus](https://discuss.wxpython.org/t/stumped-arrows-tab-kills-keyboard-focus/27163/6)
另外，捕获keycode，如果是判断字母，最好是大小写形式都判断，即里面：
```python
        if keycode == ord('P') or keycode == ord('p'):
```
