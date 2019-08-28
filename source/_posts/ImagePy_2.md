---
title: ImagePy解析：2 -- 插件加载详解
tags: [ImagePy]
categories: computational material science 
date: 2019-8-9
---

在第一部分中已经介绍，ImagePy的插件就是文件，这一篇详细解析ImagePy怎样解析物理文件，然后将其加载到菜单栏中。

# 主界面构建菜单
首先在主界面中根据路径添加菜单项（插件就是菜单），即逐层遍历'menus'路径下的文件夹和文件，找到特定后缀的文件（比如后缀为"plgs.py"），并添加为菜单项。这是整个插件加载的入口函数。
```python
self.menubar = pluginloader.buildMenuBarByPath(self, 'menus', 'plugins', None, True)
self.SetMenuBar( self.menubar )

```
下面详细查看pluginloader的buildMenuBarByPath函数的详细用途。
首先对应好该函数接收的形参和实参，即path形参接收了menus这个文件夹，extends形参接收了plugins这个文件夹（但这个参数后面证明啥也没做），menubar这个形参设为None，report这个形参设为True。
接着看该函数到底干了什么。它首先调用了加载器loader类的buildplugins函数：
```python
datas = loader.build_plugins(path, report)
```

这一步有如下几个重点：

# 通过递归得到menus文件夹下的子文件夹和特定后缀的文件

```python
def build_plugins(path, err=False):
    root = err in (True, False)
    if root: sta, err = err, []
    subtree = []
    # 返回menus文件夹包含的文件和文件夹的名字的列表
    cont = os.listdir(os.path.join(root_dir, path))
    print("cont = ", cont)
    # 遍历这些文件和文件夹
    for i in cont:
        subp = os.path.join(path,i)
        print("subp = ", subp)
        # 判断子路径是否是文件夹
        if os.path.isdir(os.path.join(root_dir, subp)):
            # 如果是文件夹，就做递归操作，即将该子文件夹作为新的path，直到不再是文件夹，而是具体的文件
            sub = build_plugins(subp, err)
            print("sub after building plugins = ", sub)
            if len(sub)!=0:
                subtree.append(sub)
                print("subtree append sub = ", subtree)
        # 如果不再是文件夹后，就判断后缀，如果是以下后缀，则添加进subtree
        elif i[-6:] in ('plg.py', 'lgs.py', 'wgt.py', 'gts.py'):
            subtree.append(i)
            print("subtree append .py = ", subtree)
        # 这些后缀的文件也添加进subtree
        elif i[-3:] in ('.mc', '.md', '.wf', 'rpt'):
            subtree.append(i)
            print("subtree append .mc = ", subtree)
    if len(subtree)==0:
        print("Cont if subtree is empty = ", cont)
        return []
    print("Cont if subtree is NOT empty = ", cont)
    print("path = ", path)
    print("*********Found the directory containing valid files**********")
```
上述代码在源代码基础上加了很多print函数，方便打印结果。
以File文件夹为例，上述print的结果为：
```python
cont =  ['BMP', 'DAT', 'DICOM', 'exit_plg.py', 'Export', 'GIF', 'Import', 'JPG', 'MAT', 'new_plg.py', 'Numpy', 'Open Recent', 'open_plg.py', 'PNG', 'Samples ImageJ', 'Samples Local', 'Samples Online', 'save_plg.py', 'TIF', '__init__.py', '__pycache__']
subp =  menus\File\BMP
cont =  ['bmp_plgs.py', '__init__.py', '__pycache__']
subp =  menus\File\BMP\bmp_plgs.py
subtree append .py =  ['bmp_plgs.py']
subp =  menus\File\BMP\__init__.py
subp =  menus\File\BMP\__pycache__
cont =  ['bmp_plgs.cpython-37.pyc', '__init__.cpython-37.pyc']
subp =  menus\File\BMP\__pycache__\bmp_plgs.cpython-37.pyc
subp =  menus\File\BMP\__pycache__\__init__.cpython-37.pyc
Cont if subtree is empty =  ['bmp_plgs.cpython-37.pyc', '__init__.cpython-37.pyc']
sub after building plugins =  []
Cont if subtree is NOT empty =  ['bmp_plgs.py', '__init__.py', '__pycache__']
*********Found the directory containing valid files**********
```
可以看出，程序先扫描File文件夹下的所有文件和文件夹，然后逐个判断；首先进入子文件夹BMP，扫描得到它下面的所有文件和文件夹，这里面有一个符合后缀的插件文件，就将它加入到subtree中，对于跟它平级的'_pycache_'缓存文件夹，程序也照样会进入，但其里面因为没有有效的插件文件，就返回一个空值给subtree，然后跳出这个文件夹，回到它的父文件夹BMP；前面已提到，BMP中有一个有效插件，因此subtree不为空，这里就是['bmp_plgs.py']，此时就不会终止程序，而是继续往下面走，注意此时的path就是BMP路径，即menus\File\BMP。

接下来是导入BMP这个包，注意这里的术语区别，BMP是包module，表明它还有多个类。接下来一步是导入BMP这个包：
```python
# 将路径中的斜线替换为替换为点号，为后续导入做准备
rpath = path.replace('/', '.').replace('\\','.')
print("rpath = ", rpath)
# 动态加载这个模块
pg = __import__('imagepy.'+rpath,'','',[''])
# 给这个模块的title属性赋值为path变量的basename，title会被后面用作菜单的名字
pg.title = os.path.basename(path)
print("pg = ", pg)
print("pg's title = ", pg.title)
```
输出结果为：
```python
rpath =  menus.File.BMP
pg =  <module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>
pg's title =  BMP
```
注意pg变量是个module，与下面的plgs进行区别。
上述subtree列表中存的都是字符串，extend_plugins将它们以类的形式加载到PluginManager中（之所以取名为extend，是因为原先扫描到的都是文件，这一步是将文件中定义的类提取出来），其中rpt（report）、.mc（macro宏）、.wf（WorkFlow工作流）、.md（markdown）和.py（plg.py和plgs.py）都加载到PluginManager，而wgt.py和gts.py则加载到WidgetManager中。调用接口是：
```python
subtree = extend_plugins(path, subtree, err)
```
此处path就是menus\File\BMP，subtree就是['bmp_plgs.py']。因为这里BMP的例子是插件文件，所以只截取该函数中处理插件文件部分的代码进行分析：
```python
# 将path路径中的斜线替换为点号
rpath = path.replace('/', '.').replace('\\','.')
print("rpath in extend-plugins is ", rpath)
# 动态导入那个插件文件
plg = __import__('imagepy.'+ rpath+'.'+i[:-3],'','',[''])
print("plg in extend-plugins is ", plg)
# 判断这个py文件是否有plgs这个属性，该属性告诉程序该程序有几个类
if hasattr(plg, 'plgs'):
    # 如果有plgs属性的话，就把它添加进rst变量，这个地方兼具插件排序的功能
    rst.extend([j for j in plg.plgs])
    print("rst if plg has attr of plgs = ", rst)
    print("plg.plgs = ", plg.plgs)
    for p in plg.plgs:
        print("p = ", p)
        if not isinstance(p, str):
            # 将插件加入插件管理器，这个add是被classmethod修饰符所修饰的
            print("!!! Add plgs into PluginsManager !!!")
            PluginsManager.add(p)
else:
    #plgs属性是为了一下定义多个类而准备的，也可以使用Plugin属性添加单个插件
    rst.append(plg.Plugin)
    print("rst if plg does NOT have attr of plgs = ", rst)
    PluginsManager.add(plg.Plugin)
```
该函数返回的是rst变量。插件管理器的add方法是classmethod所修饰，一个关于该修饰符的教程见：
[正确理解Python中的 @staticmethod@classmethod方法](https://zhuanlan.zhihu.com/p/28010894)
这里上面的print函数得到的输出为：
```python
rpath in extend-plugins is  menus.File.BMP
plg in extend-plugins is  <module 'imagepy.menus.File.BMP.bmp_plgs' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\bmp_plgs.py'>
rst if plg has attr of plgs =  [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>]
plg.plgs =  [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>]
p =  <class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>
!!! Add plgs into PluginsManager !!!
p =  <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>
!!! Add plgs into PluginsManager !!! 
```
可以看出，rst变量是一个列表，里面包含了该插件的两个类，而插件管理器添加的也是这两个类。
不同后缀的文件经过处理后，该函数返回值会赋值给subtree变量，即：
```python
subtree = extend_plugins(path, subtree, err)
print("subtree after extending = ", subtree)
```
因此，此时subtree变量依然是个列表，但里面的东西不再是字符串，而是动态加载的类，但是注意，subtree里面的东西还会变，注意往下看。
经过后缀处理后，build_plugins这个函数最终返回的是一个(pg, subtree)元组，前者是包名，后者是类名的列表。此处具体为：

```python
sub after building plugins =  (<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>])                                                      
```
注意，build_plugins返回的值是sub变量，如果它不为空的话，它还会添加到subtree中，所以subtree此时仍是一个list，但其中的元素是个tuple，具体值为：
```python
subtree append sub =  [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>])]

```
搜索完File文件夹的子文件夹BMP后，再接着搜索子文件夹DAT，再经过上面一轮操作，得到的subtree又更新了（注意，这个subtree是在menus这个path变量时的全局的subtree，所以它的值可以一直叠加更新，而在子文件夹下的局部的subtree的值则会一直变化）：
```python
subtree append sub =  [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>]), (<module 'imagepy.menus.File.DAT' from 'D:\\imagepy-master\\imagepy\\menus\\File\\DAT\\__init__.py'>, [<class 'imagepy.menus.File.DAT.dat_plgs.OpenFile'>, <class 'imagepy.menus.File.DAT.dat_plgs.SaveFile'>])]

```
上面BMP和DAT都是File的子文件夹，如果遇到了File的子文件，比如exit_plg.py，则subtree会变为：
```python
subtree append .py =  [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>]), (<module 'imagepy.menus.File.DAT' from 'D:\\imagepy-master\\imagepy\\menus\\File\\DAT\\__init__.py'>, [<class 'imagepy.menus.File.DAT.dat_plgs.OpenFile'>, <class 'imagepy.menus.File.DAT.dat_plgs.SaveFile'>]), (<module 'imagepy.menus.File.DICOM' from 'D:\\imagepy-master\\imagepy\\menus\\File\\DICOM\\__init__.py'>, [<class 'imagepy.menus.File.DICOM.dicom_plgs.OpenFile'>]), 'exit_plg.py']
```
虽然subtree在更新过程中是list形式，但最终在最上层的menus路径下执行的build_plugins函数返回的还是元组的形式，这里的具体形式形如：

```python
(<module 'imagepy.menus' from 'D:\\imagepy-master\\imagepy\\menus\\__init__.py'>, [(<module 'imagepy.menus.File' from 'D:\\imagepy-master\\imagepy\\menus\\File\\__init__.py'>, [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>]), (<module 'imagepy.menus.File.DAT' from 'D:\\imagepy-master\\imagepy\\menus\\File\\DAT\\__init__.py'>, [<class 'imagepy.menus.File.DAT.dat_plgs.OpenFile'>, <class 'imagepy.menus.File.DAT.dat_plgs.SaveFile'>]), <class 'imagepy.menus.File.exit_plg.Plugin'>, (<module 'imagepy.menus.File.GIF' from 'D:\\imagepy-master\\imagepy\\menus\\File\\GIF\\__init__.py'>, [<class 'imagepy.menus.File.GIF.gif_plgs.OpenFile'>, <class 'imagepy.menus.File.GIF.gif_plgs.SaveFile'>, '-', <class 'imagepy.menus.File.GIF.animate_plgs.OpenAnimate'>, <class 'imagepy.menus.File.GIF.animate_plgs.SaveAnimate'>]), <class 'imagepy.menus.File.open_plg.OpenFile'>, <class 'imagepy.menus.File.open_plg.OpenUrl'>])])
```
上面的结果是在menus文件夹下只保留File子文件夹，且File子文件夹下只保留BMP、DAT和GIF文件夹以及两个py文件的结果。
更精简地，做这样一个测试，在menus下保留File和Edit两个子文件夹，File下只保留BMP子文件夹，Edit下仅有edit_plg.py文件，且其中的类只有Undo、分隔符和ClearOut，那么此时最终返回结果为：
```python
(<module 'imagepy.menus' from 'D:\\imagepy-master\\imagepy\\menus\\__init__.py'>, [(<module 'imagepy.menus.File' from 'D:\\imagepy-master\\imagepy\\menus\\File\\__init__.py'>, [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>])]), (<module 'imagepy.menus.Edit' from 'D:\\imagepy-master\\imagepy\\menus\\Edit\\__init__.py'>, [<class 'imagepy.menus.Edit.edit_plg.Undo'>, '-', <class 'imagepy.menus.Edit.edit_plg.ClearOut'>])])

```
可以看出，整个返回值是一个整体有2个元素的tuple，第一个元素是导入的menus这个包，第二个元素是一个list，这个list又是含有2个元素，分别对应File和Edit两个文件夹，第一个元素又是一个tuple，这个tuple的第一个元素导入的是File这个包，第二个元素是一个list，包含BMP子文件夹所形成的tuple。那么规律就是：对于一个文件夹（包括menus这个总文件夹）层级，每个文件夹就是一个Python包，该层上的返回的结构就是一个含有2个元素的tuple，第一个元素是导入的这个Python包信息，第二个元素就是一个list，包含该文件夹下的插件信息，如果有子文件夹，那么该子文件夹又会重复这样的层级表示。构思相当之巧妙～～

# 对插件进行顺序调整
首先明确插件的类型有两种：一种是按物理文件组织的插件，如插件组，即多个插件文件在同一个文件夹中；一种是在同一个文件内的不同插件，即一个插件文件中有多个类。
先看第一种插件，上面的BMP和DAT文件夹中都只有一个插件，而GIF文件夹中有两个插件，因此这里面还会涉及插件的排序问题，这个排序问题存在于“一个文件夹中有多个插件”的情形，因此File这个文件夹也会存在该问题。默认情况下，插件是通过字母顺序排序的，这是因为python扫描文件时就是按字母顺序来的。也可以通过在初始化文件中定义catlog属性来人为规定插件顺序。具体代码为：

```python
if hasattr(pg, 'catlog'):
    if 'Personal Information' in pg.catlog:
        print(subtree)
    subtree = sort_plugins(pg.catlog, subtree)
    print("subtree with catlog = ", subtree)
subtree = extend_plugins(path, subtree, err)
print("subtree after extending = ", subtree)
```
注意，这里在catlog列表中加入”-“来映射为菜单分割线。
调整顺序后的局部的subtree变量为：
```python
subtree after extending =  [<class 'imagepy.menus.File.GIF.gif_plgs.OpenFile'>, <class 'imagepy.menus.File.GIF.gif_plgs.SaveFile'>, '-', <class 'imagepy.menus.File.GIF.animate_plgs.OpenAnimate'>, <class 'imagepy.menus.File.GIF.animate_plgs.SaveAnimate'>]
```
再看第二种插件，其实这种插件的排序是在extend_plugins函数中隐含进行的，具体地是在该文件的最后定义plgs这个属性，然后extend_plugins这个函数读取该属性，然后逐个添加插件到插件管理器中。

# python脚本解析为wxPython菜单
使用wxPython菜单栏的一个基本教程为：
[创建和使用wxPython菜单](https://wizardforcel.gitbooks.io/wxpy-in-action/15.html)
基本套路为：
```python
# 创建菜单栏
menuBar = wx.MenuBar()
# 把菜单栏附加给框架：使用SetMenuBar()方法将它附加给一个wx.Frame（或其子类），通常这些都在框架的__init__或OnInit()方法中实施
self.SetMenuBar(menuBar)
# 创建单个的菜单
menu = wx.Menu()
# 把菜单附加给菜单栏或一个父菜单，即可以用Append添加一个子菜单，但这个API在最新版本中是deprecated，需要使用AppendSubMenu
menuBar.Append(menu, "Left Menu")
# 创建单个的菜单项，并附加给某个菜单 （即：菜单栏包含多个菜单，某个菜单包含多个菜单项）
exit = menu.Append(-1, "Exit")
# 为每个菜单项创建一个事件绑定
self.Bind(wx.EVT_MENU, self.OnExit, exit)
```

一个基础例子为：
```python
import wx
class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1,
                          "Sub-menu Example")
        p = wx.Panel(self)
        menu = wx.Menu()
        submenu = wx.Menu()
        submenu.Append(-1, "Sub-item 1")
        submenu.Append(-1, "Sub-item 2")
        menu.Append(-1, "Sub-menu", submenu)
        menu.AppendSeparator()
        exit = menu.Append(-1, "Exit")
        self.Bind(wx.EVT_MENU, self.OnExit, exit)
        menuBar = wx.MenuBar()
        menuBar.Append(menu, "Menu")
        self.SetMenuBar(menuBar)

    def OnExit(self, event):
        self.Close()

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
```

回到ImagePy的解析菜单功能，调用的函数如下，其中的形参datas的值为：
```python
datas =  (<module 'imagepy.menus' from 'D:\\imagepy-master\\imagepy\\menus\\__init__.py'>, [(<module 'imagepy.menus.File' from 'D:\\imagepy-master\\imagepy\\menus\\File\\__init__.py'>, [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>])]), (<module 'imagepy.menus.Edit' from 'D:\\imagepy-master\\imagepy\\menus\\Edit\\__init__.py'>, [<class 'imagepy.menus.Edit.edit_plg.Undo'>, '-', <class 'imagepy.menus.Edit.edit_plg.ClearOut'>])])

```
即还是跟之前一节的值一样。
```python
def buildMenuBar(parent, datas, menuBar=None):
    if menuBar==None:
        # 创建wxPython菜单栏对象
        menuBar = wx.MenuBar()
    for data in datas[1]:
        print("data = ", data)
        if len(data[1]) == 0:
            continue
        print("data.title = ", data[0].title)
        LanguageManager.add(data[0].title)
        # 这个地方又调用了下面的buildMenu来创建菜单
        menuBar.Append(buildMenu(parent, data, data[0].title), LanguageManager.get(data[0].title))
    return menuBar
 
def buildMenu(parent, data, curpath):
    # 创建wxPython菜单，这是为了后面将它加入MenuBar
    menu = wx.Menu()
    for item in data[1]:
        # 如果item是个tuple，就会进行递归构建
        if isinstance(item, tuple):
            print("When item is a tuple")
            print("item = ", item)
            nextpath = curpath + '.' + item[0].title
            print("item's title = ", item[0].title)
            print("nextpath = ", nextpath)
            LanguageManager.add(item[0].title)

            # 这个地方也可用于子菜单的构建
            menu.Append(-1, LanguageManager.get(item[0].title), buildMenu(parent, item,nextpath))
        else:
            # 如果item不是个tuple，是个具体的class，那么就会进入构建菜单项（注意是菜单项）并绑定事件
            print("When item is NOT a tuple")
            print("item = ", item)
            buildItem(parent, menu, item)
    return menu
```
针对于传入的datas那个值，上面的输出为：
```python
data =  (<module 'imagepy.menus.File' from 'D:\\imagepy-master\\imagepy\\menus\\File\\__init__.py'>, [(<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>])])
data.title =  File
When item is a tuple
item =  (<module 'imagepy.menus.File.BMP' from 'D:\\imagepy-master\\imagepy\\menus\\File\\BMP\\__init__.py'>, [<class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>, <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>])
item's title =  BMP
nextpath =  File.BMP
When item is NOT a tuple
item =  <class 'imagepy.menus.File.BMP.bmp_plgs.OpenFile'>
When item is NOT a tuple
item =  <class 'imagepy.menus.File.BMP.bmp_plgs.SaveFile'>
data =  (<module 'imagepy.menus.Edit' from 'D:\\imagepy-master\\imagepy\\menus\\Edit\\__init__.py'>, [<class 'imagepy.menus.Edit.edit_plg.Undo'>, '-', <class 'imagepy.menus.Edit.edit_plg.ClearOut'>])
data.title =  Edit
When item is NOT a tuple
item =  <class 'imagepy.menus.Edit.edit_plg.Undo'>
When item is NOT a tuple
item =  -
When item is NOT a tuple
item =  <class 'imagepy.menus.Edit.edit_plg.ClearOut'>
```
从上面的输出可以清楚地看到解析过程。当item是一个具体的class时，就会进入下面的buildItem函数，进行菜单项的创建，以及绑定事件。这个地方又有一个小绕绕，即它绑定的事件这里使用了lambda表达式，即匿名函数，所以比较难看懂，见：
```python
parent.Bind(wx.EVT_MENU, lambda x, p=item:p().start(), mi)
```
其实这个函数就是将item传给p，然后调用了p的start()函数，那么start()函数在哪？
这就牵扯到了ImagePy所定义的各种引擎engine，比如基础引擎Simple、自由引擎Free、宏引擎Macros等，这些引擎都在engine文件夹下定义。而之前的插件类实际都是这些引擎的派生，自然也就继承了每个引擎的start()函数。
