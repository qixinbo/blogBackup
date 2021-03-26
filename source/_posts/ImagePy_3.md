---
title: ImagePy解析：3 -- 工具条加载详解
tags: [ImagePy]
categories: computer vision 
date: 2019-8-29
---

这一部分详解ImagePy的工具条是如何加载的。

# 构建工具条入口
通过build_tools这个函数来构建工具条：

```python
self.toolbar = toolsloader.build_tools(self, 'tools', 'plugins', None, True)
```
这几个实参所对应的该函数的形参依次为：tools传入toolpath, plugins传入extends（这个参数目前看没有用处），None传入bar，True赋给report。
下面详细看这个函数做了什么。

# 递归获得所有工具的类文件和图标文件
上面的build_tools函数又会调用loader的build_tools函数，这个函数所做的东西跟前面加载插件时的逻辑基本是相同的，都是为了递归获得所有工具的类文件，不过这里还会获得图标文件。
```python
datas = loader.build_tools(toolspath, report)
print("datas = ", datas)
```
如果tools文件夹下仅保留部分文件，那么上面的返回结果如下：
```python
datas =  (<module 'imagepy.tools' from 'D:\\imagepy-master\\imagepy\\tools\\__init__.py'>, [(<module 'imagepy.tools.Measure' from 'D:\\imagepy-master\\imagepy\\tools\\Measure\\__init__.py'>, [(<class 'imagepy.tools.Measure.distance_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/distance.gif'), (<class 'imagepy.tools.Measure.profile_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/profile.gif')]), (<module 'imagepy.tools.Standard' from 'D:\\imagepy-master\\imagepy\\tools\\Standard\\__init__.py'>, [(<class 'imagepy.tools.Standard.point_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Standard/point.gif')])])
```
可以看出，上述数据结构的层级类似于构建菜单时得到的结果。

#解析为wxpython工具条
使用的函数为：
```python
toolsbar = buildToolsBar(parent, datas, bar)
```
即，将上面得到的datas元组传入，然后该函数首先创建了wxpython的penel对象，并用BoxSizer进行组织：
```python
if toolsbar is None:
    box = wx.BoxSizer( wx.HORIZONTAL )
    toolsbar = wx.Panel( parent, wx.ID_ANY,  wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
    toolsbar.SetSizer( box )
else:
    box = toolsbar.GetSizer()
    toolsbar.DestroyChildren()
    box.Clear()
```
注意，ImagePy创建工具条不是使用的wxPython默认的ToolBar对象，而是使用的wxPython的panel对象，应该是为了考虑更加灵活的控制。
然后将获得工具条层级关系传入下面的add_tools函数中，注意，这个地方传入的是datas这个tuple的第三维度，即：
```python
print("datas[1][0][1] = ", datas[1][0][1])
add_tools(toolsbar, datas[1][0][1], clear=True)
```
原因可以从上面的输出中进行追究：
```python
datas[1][0][1] =  [(<class 'imagepy.tools.Measure.distance_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/distance.gif'), (<class 'imagepy.tools.Measure.profile_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/profile.gif')]
```
可以看出，第一个index选择1，是因为选择它之前扫描到的tools文件夹下的所有工具形成的list；第二个index选择0，代表第一个文件夹，即Measure文件夹；第二个index选择1，即选择Measure文件夹下的所有工具形成的list。其实，看下方的代码，ImagePy还会加载第二个文件夹，但此时源代码中有一个bug，即如果tools文件夹只有一个子文件夹，那么就会报错，无法启动，因此这里对ImagePy源码可以修改，增加一个判断，如果有多于两个文件夹，才会加载下方的代码，即：
```python
if len(datas[1]) > 1:
    ...
    print("datas[1][1][1] = ", datas[1][1][1])
    add_tools(toolsbar, datas[1][1][1])
```
GitHub上我提了一个Pull Request，见：
[fix the startup error when only there is only one toolbar](https://github.com/Image-Py/imagepy/pull/65)

然后下面进入add_tools函数：
```python
def add_tools(bar, datas, clear=False, curids=[]):
    box = bar.GetSizer()
    if not clear:
        for curid in curids:
            curid.Destroy()
    del curids[:]

    for data in datas:
        print("data = ", data)
        # 将扫描到的icon图像文件制作成wxPython的BitmapButton
        btn = wx.BitmapButton(bar, wx.ID_ANY, make_bitmap(wx.Bitmap(data[1])), wx.DefaultPosition, (32,32), wx.BU_AUTODRAW|wx.RAISED_BORDER )       
        if not clear:curids.append(btn)       
        # 判断是否在界面上直接显示，还是收缩起来，点击后再显示
        if clear:
            print("--- It is clear ---")
            # 将这个button直接添加进入BoxSizer中
            box.Add(btn)
        else:
            print("--- It is not clear")
            # 如果是收缩起来，则在特定位置加入，而不是直接加入，这里的特定位置是box已有的所有的item数-2，之所以是减去2，与之前box添加的东西有关，具体可以看下面的输出，即在box添加的分割线后添加
            print("length of box's children = ", len(box.GetChildren()))
            for child in box.GetChildren():
                print("box's children = ", child.GetWindow())
            box.Insert(len(box.GetChildren())-2, btn)

        # 为该button绑定“左键按下”的鼠标事件   
        btn.Bind( wx.EVT_LEFT_DOWN, lambda x, p=data[0]:f(p(), x))

        # 为该button绑定“右键按下”的鼠标事件
        btn.Bind( wx.EVT_RIGHT_DOWN, lambda x, p=data[0]: IPy.show_md(p.title, DocumentManager.get(p.title)))

        # 为该button绑定“鼠标移入”的鼠标事件
        btn.Bind( wx.EVT_ENTER_WINDOW, lambda x, p='"{}" Tool'.format(data[0].title): set_info(p))       

        if not isinstance(data[0], Macros) and issubclass(data[0], Tool):
            btn.Bind(wx.EVT_LEFT_DCLICK, lambda x, p=data[0]:p().show())

        btn.SetDefault()
    box.Layout()
    bar.Refresh()
```
上面的输出为：
```python
datas[1][0][1] =  [(<class 'imagepy.tools.Measure.distance_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/distance.gif'), (<class 'imagepy.tools.Measure.profile_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/profile.gif')]
data =  (<class 'imagepy.tools.Measure.distance_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/distance.gif')
--- It is clear ---
data =  (<class 'imagepy.tools.Measure.profile_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Measure/profile.gif')
--- It is clear ---
datas[1][1][1] =  [(<class 'imagepy.tools.Standard.point_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Standard/point.gif')]
data =  (<class 'imagepy.tools.Standard.point_tol.Plugin'>, 'D:\\imagepy-master\\imagepy\\tools\\Standard/point.gif')
--- It is not clear
length of box's children =  5
box's children =  <wx._core.BitmapButton object at 0x0000020B2C24E558>
box's children =  <wx._core.BitmapButton object at 0x0000020B2C24E708>
box's children =  <wx._core.StaticLine object at 0x0000020B2C24E8B8>
box's children =  None
box's children =  <wx._core.BitmapButton object at 0x0000020B2C24E828>

```

# 工具按钮绑定鼠标事件
前面代码已经简略描述了如何为工具绑定鼠标事件：
```python
btn.Bind( wx.EVT_LEFT_DOWN, lambda x, p=data[0]:f(p(), x))
btn.Bind( wx.EVT_RIGHT_DOWN, lambda x, p=data[0]: IPy.show_md(p.title, DocumentManager.get(p.title)))
btn.Bind( wx.EVT_ENTER_WINDOW, lambda x, p='"{}" Tool'.format(data[0].title): set_info(p))

if not isinstance(data[0], Macros) and issubclass(data[0], Tool):
    btn.Bind(wx.EVT_LEFT_DCLICK, lambda x, p=data[0]:p().show())
```
首先还是使用了lambda算子，即匿名函数，进行事件绑定。
这里的工具都是继承了Tool这个引擎，事件触发的也都是Tool引擎本身的函数或者该特定工具重载的同名函数。
值得注意的是，这里的右键按下事件，会调用IPy的显示markdown的功能，再次说明IPy将常用功能抽象出来的作用。
另外，工具的加载不像插件的加载那样，会加入到一个管理器Manager中，工具的加载就是对wxPython的panel的渲染，然后绑定鼠标事件。
