---
title: ImagePy解析：4 -- 主界面渲染
tags: [ImagePy]
categories: computer vision 
date: 2019-8-31
---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
更新日志：
2019-9-30 更新：
增加了ImageJ和ImagePy两种UI加载方式的对比

以下为正文
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ImagePy的主界面就是名为ImagePy的类，这一部分详解主界面是如何渲染出来的。

# 父类初始化
ImagePy这个类是继承自wxPython的Frame类，所以首先对这个类进行初始化，包括名称、尺寸、位置、样式等。
```python
wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'ImagePy',
                    size = wx.Size(-1,-1), pos = wx.DefaultPosition,
                    style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
```

# 创建AuiManager管理器，用来管理复杂布局
ImagePy使用AuiManager来管理复杂界面
```python
self.auimgr = aui.AuiManager()
self.auimgr.SetManagedWindow( self )
```
Aui等于Advanced User Interface，它可以用来方便地管理多个面板，尤其是对于可停靠的（dockable）和可悬浮的（floating）的面板。
一个很好的关于AuiManager的教程见：
[Day 4: wxPython AUI介绍](https://perphyliu.wordpress.com/2010/01/02/day-4-wxpython-aui介绍/)
其使用套路如下：
```python
#（1）生成AuiManager实例：以wx.Frame或其子类的实例为参数，用来生成一个AuiManager实例以实现对该Frame实例的管理
self._mgr = wx.aui.AuiManager(self)

#（2）添加面板：AuiManager中可以添加若干个面板；它使用AuiPaneInfo用来控制面板的位置、大小等属性
text1 = wx.TextCtrl(self, -1, 'Pane 1 - sample text',  wx.DefaultPosition, wx.Size(200,150), wx.NO_BORDER | wx.TE_MULTILINE)
self._mgr.AddPane(text1, wx.aui.AuiPaneInfo().Left().MaximizeButton(True))

#（3）更新AuiManager实例：调用Update()方法将AuiManager中的面板添加到其所管理的Frame中
self._mgr.Update()

#（4）卸载AuiManager实例：退出应用程序时应调用UnInit方法以回收资源
self._mgr.UnInit()
```

完整的示例代码如下：
```python
#!/usr/bin/env python
import wx
import wx.aui

class MyFrame(wx.Frame):
      def __init__(self, parent, id=-1, title='wx.aui Test',
                  pos=wx.DefaultPosition, size=(800, 600),
                  style=wx.DEFAULT_FRAME_STYLE):
            wx.Frame.__init__(self, parent, id, title, pos, size, style)
            self._mgr = wx.aui.AuiManager(self)

            # create several text controls
            text1 = wx.TextCtrl(self, -1, 'Pane 1 - sample text',
                                wx.DefaultPosition, wx.Size(200,150),
                                wx.NO_BORDER | wx.TE_MULTILINE)

            text2 = wx.TextCtrl(self, -1, 'Pane 2 - sample text',
                                wx.DefaultPosition, wx.Size(200,150),
                                wx.NO_BORDER | wx.TE_MULTILINE)

            text3 = wx.TextCtrl(self, -1, 'Main content window',
                                wx.DefaultPosition, wx.Size(200,150),
                                wx.NO_BORDER | wx.TE_MULTILINE)

            # add the panes to the manager
            self._mgr.AddPane(text1, wx.aui.AuiPaneInfo().Left().MaximizeButton(True))
            self._mgr.AddPane(text2, wx.aui.AuiPaneInfo().Bottom().MaximizeButton(True))
            self._mgr.AddPane(text3, wx.aui.AuiPaneInfo().Center())

            # Create toolbar
            toolbar = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                             wx.TB_FLAT | wx.TB_NODIVIDER | wx.TB_HORZ_TEXT)
            toolbar.SetToolBitmapSize(wx.Size(16,16))
            toolbar_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, wx.Size(16, 16))
            toolbar.AddTool(101, "Item 1", toolbar_bmp1)
            toolbar.AddTool(101, "Item 2", toolbar_bmp1)
            toolbar.AddTool(101, "Item 3", toolbar_bmp1)
            toolbar.AddTool(101, "Item 4", toolbar_bmp1)
            toolbar.AddSeparator()
            toolbar.AddTool(101, "Item 5", toolbar_bmp1)
            toolbar.AddTool(101, "Item 6", toolbar_bmp1)
            toolbar.AddTool(101, "Item 7", toolbar_bmp1)
            toolbar.AddTool(101, "Item 8", toolbar_bmp1)
            toolbar.Realize()
            self._mgr.AddPane(toolbar, wx.aui.AuiPaneInfo().Name("toolbar").Caption("Toolbar Demo").ToolbarPane().Top().LeftDockable(False).RightDockable(False))

            # tell the manager to 'commit' all the changes just made
            self._mgr.Update()
            self.Bind(wx.EVT_CLOSE, self.OnClose)

      def OnClose(self, event):
            # deinitialize the frame manager
            self._mgr.UnInit()
            # delete the frame
            self.Destroy()

app = wx.App()
frame = MyFrame(None)
frame.Show()
app.MainLoop()

```

# 设置图标
```python
logopath = os.path.join(root_dir, 'data/logo.ico')
self.SetIcon(wx.Icon(logopath, wx.BITMAP_TYPE_ICO))

IPy.curapp = self
# 根据UI的模式，设置窗口的最大、最小尺寸以及变化幅度。这个UI的模式是通过configmanager模块读取preference.cfg配置文件获得的。
self.SetSizeHints( wx.Size(900,700) if IPy.uimode() == 'ipy' else wx.Size( 580,-1 ))
```

# 根据路径添加菜单项
```python
self.menubar = pluginloader.buildMenuBarByPath(self, 'menus', 'plugins', None, True)
self.SetMenuBar( self.menubar )
```
这一部分是核心，详细的解析过程见本系列的第二部分。

# 创建快捷键映射
```python
self.shortcut = pluginloader.buildShortcut(self)
self.SetAcceleratorTable(self.shortcut)
```

# 构建工具条
```python
self.toolbar = toolsloader.build_tools(self, 'tools', 'plugins', None, True)
```
这一部分也是核心，详细解析过程见本系列的第三部分。

# 设置显示样式
根据UI的模式为"ipy"或"ij"来调整UI的样式为仿ImageJ还是自带的ImagePy样式。
```python
print(IPy.uimode())
if IPy.uimode()=='ipy': self.load_aui()
else: self.load_ijui()
self.load_document()
self.Fit()
```
两种样式的相同点和不同点在于：
（1）相同点：都有菜单栏和工具栏，其中菜单栏已经在上面创建，工具栏的创建如下：
```python
self.auimgr.AddPane(self.toolbar, aui.AuiPaneInfo() .Left()  .PinButton( True )
            .CaptionVisible( True ).Dock().Resizable().FloatingSize( wx.DefaultSize ).MaxSize(wx.Size( 32,-1 ))
            . BottomDockable( True ).TopDockable( False ).Layer( 10 ) )
```
（2）不同点：ImagePy会显示navigator、histogram等组件。（ImageJ也会加载这些组件，但是隐藏显示）
首先扫描widgets文件夹下的组件，其扫描方式与plugins、tools相同。
```python
self.widgets = widgetsloader.build_widgets(self, 'widgets', 'plugins')
```
即调用了widgetsloader的build_widgets函数，其中该函数中又接着调用了loader类下的build_widgets函数：
```python
datas = loader.build_widgets(toolspath)
print("widgets = ", datas)
```
因为widgets有两个组件：navigator和histogram，其中histogram又细分为histogram和curve两个组件，因为这一步输出的结果为：
```python
widgets =  (<module 'imagepy.widgets' from 'D:\\imagepy-master\\imagepy\\widgets\\__init__.py'>, [(<module 'imagepy.widgets.histogram' from 'D:\\imagepy-master\\imagepy\\widgets\\histogram\\__init__.py'>, [<class 'imagepy.widgets.histogram.histogram_wgt.Plugin'>, <class 'imagepy.widgets.histogram.curve_wgt.Plugin'>]), (<module 'imagepy.widgets.navigator' from 'D:\\imagepy-master\\imagepy\\widgets\\navigator\\__init__.py'>, [<class 'imagepy.widgets.navigator.navigator_wgt.Plugin'>])])
```
注意，仍然是(pg, subtree)这样的返回格式。UI上使用了wxPython的Choicebook来在同一个组件类的不同插件间进行选择。navigator、histogram和curve这三个组件也都是在wxPython的Panel基础上进行定制，高级，见识了~~

# 创建底部状态栏
```python
self.stapanel = stapanel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
sizersta = wx.BoxSizer( wx.HORIZONTAL )
self.txt_info = wx.StaticText( stapanel, wx.ID_ANY, "ImagePy  v0.2", wx.DefaultPosition, wx.DefaultSize, 0 )
self.txt_info.Wrap( -1 )
#self.txt_info.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_INFOBK ) )
sizersta.Add( self.txt_info, 1, wx.ALIGN_BOTTOM|wx.BOTTOM|wx.LEFT|wx.RIGHT, 2 )
```
# 创建底部进度条
```python
self.pro_bar = wx.Gauge( stapanel, wx.ID_ANY, 100, wx.DefaultPosition, wx.Size( 100,15 ), wx.GA_HORIZONTAL )
sizersta.Add( self.pro_bar, 0, wx.ALIGN_BOTTOM|wx.BOTTOM|wx.LEFT|wx.RIGHT, 2 )
stapanel.SetSizer(sizersta)
```

# 为底部的这个panel增加拖放文件直接打开的功能
```python
stapanel.SetDropTarget(FileDrop())
self.auimgr.AddPane( stapanel,  aui.AuiPaneInfo() .Bottom() .CaptionVisible( False ).PinButton( True )
    .PaneBorder( False ).Dock().Resizable().FloatingSize( wx.DefaultSize ).DockFixed( True )
    . MinSize(wx.Size(-1, 20)). MaxSize(wx.Size(-1, 20)).Layer( 10 ) )
```

# 更新Aui管理器
```python
self.Centre( wx.BOTH )
self.Layout()
self.auimgr.Update()
self.Fit()
self.Centre( wx.BOTH )

if(IPy.uimode()=='ij'):
    self.SetMaxSize((-1, self.GetSize()[1]))
    self.SetMinSize(self.GetSize())
self.update = False
```
# 绑定事件逻辑
```python
# 为关闭按钮绑定on_close逻辑
self.Bind(wx.EVT_CLOSE, self.on_close)

# 为AuiManager管理器所管理的panel的关闭按钮绑定on_pan_close逻辑，但目前发现并没有起到作用。。:(
self.Bind(aui.EVT_AUI_PANE_CLOSE, self.on_pan_close)
```

# 创建并启动多线程
```python
thread = threading.Thread(None, self.hold, ())
thread.setDaemon(True)
thread.start()
```

它所执行的函数是其中的hold函数，具体涉及了ImagePy的任务管理器TaskManager，后面具体分析。
