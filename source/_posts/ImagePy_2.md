---
title: ImagePy解析：2 -- 主界面及菜单加载
tags: [ImagePy]
categories: computational material science 
date: 2019-4-9
---

# 主界面初始化
主界面是继承自wx.Frame类的ImagePy类。它的初始化过程如下：

```python
# 初始化父类，包括名称、尺寸、位置、样式
wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'ImagePy',
                    size = wx.Size(-1,-1), pos = wx.DefaultPosition,
                    style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

# 创建AuiManager管理器，用来管理复杂布局
self.auimgr = aui.AuiManager()

#告诉主界面，现在由AuiManager来管理布局
self.auimgr.SetManagedWindow( self )

# 设置图标
logopath = os.path.join(root_dir, 'data/logo.ico')
self.SetIcon(wx.Icon(logopath, wx.BITMAP_TYPE_ICO))

IPy.curapp = self

# 根据UI的模式，设置窗口的最大、最小尺寸以及变化幅度。这个UI的模式是通过configmanager模块读取preference.cfg配置文件获得的。
self.SetSizeHints( wx.Size(900,700) if IPy.uimode() == 'ipy' else wx.Size( 580,-1 ))

# 根据路径添加菜单项（插件就是菜单），即逐层遍历'menus'路径下的文件夹和文件，找到特定后缀的文件（比如后缀为"plgs.py"），并添加为菜单项（具体过程详见下方）
self.menubar = pluginloader.buildMenuBarByPath(self, 'menus', 'plugins', None, True)
self.SetMenuBar( self.menubar )

# 创建快捷键映射
self.shortcut = pluginloader.buildShortcut(self)
self.SetAcceleratorTable(self.shortcut)

#sizer = wx.BoxSizer(wx.VERTICAL)

# 通过扫描tools路径下的文件夹和文件，构建工具条，这里使用了两个向下三角的BitmapButton按钮
self.toolbar = toolsloader.build_tools(self, 'tools', 'plugins', None, True)

# 根据UI的模式为"ipy"或"ij"来调整UI的样式为仿ImageJ还是自带的ImagePy样式
print(IPy.uimode())
if IPy.uimode()=='ipy': self.load_aui()
else: self.load_ijui()
self.load_document()
self.Fit()

# 创建底部状态栏
self.stapanel = stapanel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
sizersta = wx.BoxSizer( wx.HORIZONTAL )
self.txt_info = wx.StaticText( stapanel, wx.ID_ANY, "ImagePy  v0.2", wx.DefaultPosition, wx.DefaultSize, 0 )
self.txt_info.Wrap( -1 )
#self.txt_info.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_INFOBK ) )
sizersta.Add( self.txt_info, 1, wx.ALIGN_BOTTOM|wx.BOTTOM|wx.LEFT|wx.RIGHT, 2 )

# 创建底部进度条
self.pro_bar = wx.Gauge( stapanel, wx.ID_ANY, 100, wx.DefaultPosition, wx.Size( 100,15 ), wx.GA_HORIZONTAL )
sizersta.Add( self.pro_bar, 0, wx.ALIGN_BOTTOM|wx.BOTTOM|wx.LEFT|wx.RIGHT, 2 )
stapanel.SetSizer(sizersta)

# 为底部的这个panel增加拖放文件直接打开的功能
stapanel.SetDropTarget(FileDrop())
self.auimgr.AddPane( stapanel,  aui.AuiPaneInfo() .Bottom() .CaptionVisible( False ).PinButton( True )
    .PaneBorder( False ).Dock().Resizable().FloatingSize( wx.DefaultSize ).DockFixed( True )
    . MinSize(wx.Size(-1, 20)). MaxSize(wx.Size(-1, 20)).Layer( 10 ) )

self.Centre( wx.BOTH )
self.Layout()
self.auimgr.Update()
self.Fit()
self.Centre( wx.BOTH )

if(IPy.uimode()=='ij'):
    self.SetMaxSize((-1, self.GetSize()[1]))
    self.SetMinSize(self.GetSize())
self.update = False

# 为关闭按钮绑定on_close逻辑
self.Bind(wx.EVT_CLOSE, self.on_close)

# 为AuiManager管理器所管理的panel的关闭按钮绑定on_pan_close逻辑，但目前发现并没有起到作用。。:(
self.Bind(aui.EVT_AUI_PANE_CLOSE, self.on_pan_close)

# 创建并启动多线程，将自定义的hold函数传入，其中调用了TaskManager
thread = threading.Thread(None, self.hold, ())
thread.setDaemon(True)
thread.start()

```
# 构建插件和菜单
前已述及，ImagePy通过下面语句构建菜单：
```python
pluginloader.buildMenuBarByPath(self, 'menus', 'plugins', None, True)

```

下面具体看这个函数做了什么。
首先是调用了加载器loader类的buildplugins函数：

```python
datas = loader.build_plugins(path, report)
```
这一步有如下三个重点：
第一，通过递归得到menus文件夹下的子文件夹和特定后缀的文件：
```python
def build_plugins(path, err=False):
    root = err in (True, False)
    if root: sta, err = err, []
    subtree = []
    cont = os.listdir(os.path.join(root_dir, path))
    for i in cont:
        subp = os.path.join(path,i)
        if os.path.isdir(os.path.join(root_dir, subp)):
            sub = build_plugins(subp, err)
            if len(sub)!=0:subtree.append(sub)
        elif i[-6:] in ('plg.py', 'lgs.py', 'wgt.py', 'gts.py'):
            subtree.append(i)
        elif i[-3:] in ('.mc', '.md', '.wf', 'rpt'):
            subtree.append(i)
    if len(subtree)==0:return []
```
这部分代码会扫描menus目录及其子目录，然后获得后缀为plg.py、lgs.py、wgt.py、gts.py、.mc、.md、.wf和rpt后缀的文件，将其加入到subtree列表中存放起来。
注意这个地方是递归运行，每个（子）目录下都会运行这个函数，即每个目录下的subtree都为空，然后得到该目录下的subtree后，下面的SortPlugins和ExtendPlugins也都会运行，将子目录下的subtree设置为所扫描到的插件的类，然后一并加入到主目录下的subtree。

另外一个注意：每次buildPlugins运行后返回的是(pg, subtree)这样的元组，前者是包名，后者是里面的插件。然后后面通过判断是否是元组来组织菜单，即：

```python
    menu = wx.Menu()
    for item in data[1]:
        if isinstance(item, tuple):
            ## TODO: fixed by auss
            nextpath = curpath + '.' + item[0].title
            LanguageManager.add(item[0].title)
            menu.Append(-1, LanguageManager.get(item[0].title), buildMenu(parent, item,nextpath))
        else:
            buildItem(parent, menu, item)
```

第二，对插件进行顺序调整：
按组织结构来说，插件可分为两类，一类是按文件组织的插件，如插件组，一类是在同一个文件内的不同插件。对于第一类插件，因为上一步中的菜单是依据文件扫描得到，因此默认是按字母顺序排列，但这里可以通过在包的__init__.py文件中定义catlog来调整插件顺序。代码为：
```python
    pg = __import__('imagepy.'+rpath,'','',[''])
    pg.title = os.path.basename(path)
    if hasattr(pg, 'catlog'):
        if 'Personal Information' in pg.catlog:
            print(subtree)
        subtree = sort_plugins(pg.catlog, subtree)
```
注意，这里在catlog列表中加入"-"来映射为菜单分割线。
对于第二类插件，是在下面的ExtendPlugins函数中进行排序，具体地是在该文件的最后定义plgs这个属性来定义顺序。
这个地方是通过动态加载模块的方式实现，相应知识点见：
[Python中__import__()的fromlist参数用法](https://docs.lvrui.io/2017/10/13/Python%E4%B8%AD-import-%E7%9A%84fromlist%E5%8F%82%E6%95%B0%E7%94%A8%E6%B3%95/)

第三，将插件以类的形式加入PluginManager中：
上述subtree列表中存的都是字符串，ExtendPlugins将它们以类的形式加载到PluginManager中（之所以取名为extend，是因为原先扫描到的都是文件，这一步是将文件中定义的类提取出来），其中rpt（report）、.mc（macro宏）、.wf（WorkFlow工作流）、.md（markdown）和.py（plg.py和plgs.py）都加载到PluginManager，而wgt.py和gts.py则加载到WidgetManager中。以plgs.py加载为例：
```python
                rpath = path.replace('/', '.').replace('\\','.')
                plg = __import__('imagepy.'+ rpath+'.'+i[:-3],'','',[''])
                if hasattr(plg, 'plgs'):
                    rst.extend([j for j in plg.plgs])
                    for p in plg.plgs:
                        if not isinstance(p, str):
                            PluginsManager.add(p)
                else:
                    rst.append(plg.Plugin)
                    PluginsManager.add(plg.Plugin)

```

# 启动多线程并等候任务
在主界面初始化的末尾，启动多线程来等候任务：
```python
        thread = threading.Thread(None, self.hold, ())
        thread.setDaemon(True)
        thread.start()
```
它所执行的函数是其中的hold函数，具体涉及了ImagePy的任务管理器TaskManager，后面具体分析。
