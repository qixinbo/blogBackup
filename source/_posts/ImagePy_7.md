---
title: ImagePy解析：7 -- ImagePy插件运行解析
tags: [ImagePy]
categories: computational material science 
date: 2019-9-30
---

%%%%%%%%%%%% 更新日志 %%%%%%%%%%%%%
2019-10-13 更新：修正之前关于自定义控件的Bind()函数的解释、增加对GUI界面绘制的理解
2019-10-4 更新：增加参数对话框ParaDiglog部分的解析
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

参考文献：
[Demo Plugin](https://github.com/Image-Py/demoplugin/blob/master/READMECN.md)
[ImagePy开发文档 —— 自由引擎](https://zhuanlan.zhihu.com/p/25483752)

这一篇解析ImagePy的插件的编写规则、运行机理。

# ImagePy的自由引擎Free engine解析
在[第一篇解析](http://qixinbo.info/2019/03/17/ImagePy_1/)中已提到，ImagePy提前定义了多种引擎，来作为插件的基类，如Filter类作为滤波器基类。这里通过解析Free引擎来研究ImagePy的插件运行原理。Free引擎是与图像本身相关性最弱的一个引擎，它仅仅是用来完成一个任务，如打开、新建图像，打开主题帮助，查看版本信息等操作（语出上面的参考文献）。

Free引擎的源码为：
```python
class Free:
    title = 'Free'
    view = None
    para = None
    prgs = (None, 1)
    asyn = True

    def progress(self, i, n):
        self.prgs = (i, n)

    def run(self, para=None):
        print('this is a plugin')

    def runasyn(self, para, callback=None):
        TaskManager.add(self)
        start = time()
        self.run(para)
        IPy.set_info('%s: cost %.3fs'%(self.title, time()-start))
        TaskManager.remove(self)
        if callback!=None:callback()

    def load(self):return True

    def show(self):
        if self.view==None:return True
        with ParaDialog(WindowsManager.get(), self.title) as dialog:
            dialog.init_view(self.view, self.para, False, True)
            doc = self.__doc__ or '### Sorry\nNo document yet!'
            dialog.on_help = lambda : IPy.show_md(self.title, DocumentManager.get(self.title))
            return dialog.ShowModal() == wx.ID_OK

    def start(self, para=None, callback=None):
        if not self.load():
            return

        if para!=None or self.show():
            if para==None:
                para = self.para
            win = WidgetsManager.getref('Macros Recorder')
            if win!=None:
                win.write('{}>{}'.format(self.title, para))
            if self.asyn and IPy.uimode()!='no':
                threading.Thread(target = self.runasyn, args = (para, callback)).start()
            else:
                self.runasyn(para, callback)
```
## title属性
title是插件的标题，将作为菜单栏的显示、交互对话框的标题，以及插件管理器中的主键。
比如作为菜单栏的显示时，就是pluginloader中的以下代码：
```python
def buildItem(parent, root, item):
    LanguageManager.add(item.title)
    title = LanguageManager.get(item.title) if sc==None else LanguageManager.get(item.title)+'\t'+sc
    mi = wx.MenuItem(root, -1, title)
```
## para和view属性
定义函数需要的参数及交互方式，两者默认都为None，表示没有参数，也不需要交互

## show()方法
弹出交互对话框，一般时候我们只需要设定para和view，Filter会自动调用ParaDialog 生成交互对话框，但必要时，可以覆盖 show()方法。
这里默认的show()方法中会先查看self.view是否为None，如果view为None，即不交互，则返回True，即使view有值，即有交互，那么在蹦出对话框后，如果后面点击了OK按钮，那么也会使得self.show()返回true。

## run()方法
核心函数，para 将作为参数传入，在这里做你想要做的。

## load()方法
如果return结果为False，插件将中止执行。默认返回True，如有必要，可以对其进行重载，进行一系列条件检验，如不满足，IPy.alert弹出提示，并返回False。

## start()方法
启动函数。之所以会启动，就是因为之前在创建菜单时的按键绑定：
```python
def buildItem(parent, root, item):
    ...
    mi = wx.MenuItem(root, -1, title)
    parent.Bind(wx.EVT_MENU, lambda x, p=item:p().start(), mi)
    root.Append(mi)
```
具体看一下start函数是怎么运行的：
```python
def start(self, para=None, callback=None):
    if not self.load():return
    if para!=None or self.show():
        if para==None:para = self.para
        win = WidgetsManager.getref('Macros Recorder')
        if win!=None:
            win.write('{}>{}'.format(self.title, para))
        if self.asyn and IPy.uimode()!='no':
            threading.Thread(target = self.runasyn, args = (para, callback)).start()
        else:
            self.runasyn(para, callback)
```
首先判断self.load()，如果特定插件有重载这个函数，那么就会执行。继续往下，判断para是否不为None或者self.show()是否为True，如果两者有其一满足，则继续往下走，然后就会以新开一个线程或在这个线程下直接运行self.runasyn()，这个函数就会继续调用self.run()方法。

下面通过三个基于Free引擎的三个事例插件详细探究。 

# 跟世界打招呼
下面是最简单的ImagePy向世界打招呼的方式：
```python
from imagepy.core.engine import Free
from imagepy import IPy
 
class Plugin(Free):
    title = 'Hello World'

    def run(self, para=None):
        IPy.alert('Hello World, I am ImagePy!')
```
这里就是重定义了title属性，以及run()方法，调用了IPy的alert()方法，以publisher和subscriber的方式发送和接收一个alert消息，蹦出一个wxPython的MessageDialog对话框。
注意这个地方除了蹦出这个对话框，还有一个变化，即因为是通过runasyn()方法调用的run()方法，所以runasyn()方法中的另外的语句也会执行。

```python
def runasyn(self, para, callback=None):
    TaskManager.add(self)
    start = time()
    self.run(para)
    IPy.set_info('%s: cost %.3fs'%(self.title, time()-start))
    TaskManager.remove(self)
    if callback!=None:callback()
```
即还会调用IPy的set_info()方法，然后因为curapp已传入当前的ImagePy这个Frame，所以会调用它的set_info()方法，效果就是在最下面的状态栏上打出"Hello World: cost 0.000s"的字样。

# 你是谁
“你是谁”这个插件可以展示怎样与插件进行交互。
```python
from imagepy.core.engine import Free
from imagepy import IPy

class Plugin(Free):
    title = 'Who Are You'
    para = {'name':'', 'age':0}
    view = [(str, 'name', 'name', 'please'),
            (int, 'age', (0,120), 0, 'age', 'years old')]

    def run(self, para=None):
        IPy.alert('Name:\t%s\r\nAge:\t%d'%(para['name'], para['age']))
```
注意此时设置了para和view都不为None，那么在start()方法中，就会因为view有值而进入交互模式，弹出一个对话框，这个对话框也是定制的，基类是wxPython的Diglog类。para是接收的具体参数，view来控制这个对话框的显示样式。

那么这个对话框是如何自定义接收参数的界面并传递参数的呢，一切要从free引擎的show()方法入手：
```python
def show(self):
    if self.view==None:
        return True
    with ParaDialog(WindowsManager.get(), self.title) as dialog:
        dialog.init_view(self.view, self.para, False, True)
        return dialog.ShowModal() == wx.ID_OK
```
ParaDialog就是那个定制的基于wxPython Dialog类的对话框类，它负责解析“你是谁”插件中的para和view参数。
首先看它的入口函数：
```python
def init_view(self, items, para, preview=False, modal = True):
    print("*********Enter init_view ********")
    print("para = ", para)
    print("items = ", items)
    self.para = para

    for item in items:
        print("item = ", item)
        print("Ctrl = ", widgets[item[0]])
        print("key = ", item[1])
        print("pre/postfix = ", item[2:])
        self.add_ctrl_(widgets[item[0]], item[1], item[2:])
```

针对于“你是谁”插件，上面的一系列print输出就是：
```python
*********Enter init_view ********
para =  {'name': '', 'age': 0} 
items =  [(<class 'str'>, 'name', 'name', 'please'), (<class 'int'>, 'age', (0, 120), 0, 'age', 'years old')]
item =  (<class 'str'>, 'name', 'name', 'please')
Ctrl =  <class 'imagepy.ui.widgets.normal.TextCtrl'>
key =  name
pre/postfix =  ('name', 'please')
item =  (<class 'int'>, 'age', (0, 120), 0, 'age', 'years old')
Ctrl =  <class 'imagepy.ui.widgets.normal.NumCtrl'>
key =  age
pre/postfix =  ((0, 120), 0, 'age', 'years old')
```
可以看出，它将view的每个元组的第一项都解析成了它自定义的类，比如imagepy.ui.widgets.normal.TextCtrl和imagepy.ui.widgets.normal.NumCtrl，具体的解析关系就是在panelconfig.py的最上面的：
```python
widgets = { 'ctrl':None, 'slide':FloatSlider, int:NumCtrl,
            float:NumCtrl, 'lab':Label, bool:Check, str:TextCtrl,
            list:Choice, 'img':ImageList, 'tab':TableList, 'color':ColorCtrl,
            'any':AnyType, 'chos':Choices, 'fields':TableFields,
            'field':TableField, 'hist':HistCanvas, 'cmap':ColorMap}
```
它是一个字典，因此很容易就将key-value对应起来。
注意，这个地方在接收输入参数时，还有一个特别灵巧的地方：
```python
def add_ctrl_(self, Ctrl, key, p):
    ctrl = Ctrl(self, *p)
    if not p[0] is None:
        self.ctrl_dic[key] = ctrl
```
因为不同类型的输入框需要不同数目的参数，比如接收字符串文本时，需要有名称和单位这样的提示语，那么就需要定义prefix和suffix，如果接收数值，还需要定义数值范围、精度等信息，因此参数的数目是不固定的，这里就运用了Python的可变参数的功能，这里使用一个星号来接收不定长度的元组参数（还可以用两个星号来接收不同长度的字典参数）。关于可变长参数，如下是一篇很好的教程：
[Python 优雅的使用参数 - 可变参数（*args & **kwargs)](https://n3xtchen.github.io/n3xtchen/python/2014/08/08/python-args-and-kwargs)
同时add_ctrl_()的接下来一句话将不同的组件与key相联系起来，key参数是view与para联系的纽带，如reset()方法中所示：
```python
def reset(self, para=None):
    if para!=None:self.para = para
    #print(para, '====')
    for p in list(self.para.keys()):
        if p in self.ctrl_dic:
            self.ctrl_dic[p].SetValue(self.para[p])
```
通过查看是否两者都有相同的key，来读取para中的value大小并赋值给相应的对话框。
再总结一下“你是谁”插件中view的写法：
str：用于接收文本字符串，view中的用法为：(str, key, prefix, suffix)，key是para中的key，prefix和suffix分别是输入框前后的提示内容，如title和unit；
int：用于接收整型数值，view中的用法为：(int, key, (lim1, lim2), accu, 'prefix', 'suffix')，其中key是para中的key，limit用于限定输入数值的范围，accu限定小数点位数，prefix和suffix用作输入框前后的提示内容。

以下部分的解析之前有误，感谢霄龙哥的讲解~~

ImagePy自定义的比如TextCtrl、NumCtrl等都是在wxPython标准组件上的组合。比如，自定义的TextCtrl是StaticText、标准组件TextCtrl和另外一个StaticText的组合。查看这个自定义的TextCtrl，可以发现有两个Bind()绑定：
```python
    self.ctrl.Bind(wx.EVT_KEY_UP, self.ontext)
def Bind(self, z, f):self.f = f  
def ontext(self, event):
    self.f(event)
```
第一个Bind是对该自定义的TextCtrl中的标准TextCtrl组件的绑定，第二个是对该自定义的TextCtrl自身的绑定。
这个地方的运行原理是这样的：标准TextCtrl等输入框都绑定了按键弹起EVT_KEY_UP这个事件，即任何一个键弹起时，都会触发ontext()方法，这个方法又调用了self.f()方法，那么self.f()方法是怎样的？其实就是第二行的Bind()函数，它在ParaDiglog类中是这样执行的：
```python
def add_ctrl_(self, Ctrl, key, p):
    ctrl = Ctrl(self, *p)
    if not p[0] is None:
        self.ctrl_dic[key] = ctrl
    if hasattr(ctrl, 'Bind'):
        ctrl.Bind(None, self.para_changed)
```
即如果该面板中添加的组件有Bind()函数，那么就调用它的Bind()函数，将ParaDialog的para_changed()方法传入，即将它赋值给了自定义TextCtrl的self.f()方法，所以，上面的ontext就是实际执行了para_changed()方法。
因为自定义的TextCtrl基于wxPython的Panel类，其实仔细查看Panel类的继承关系就知道，wx.Panel继承自wx.Window，wx.Window又继承自wx.EventHandler，而wx.EventHandler中有Bind()函数，其定义为

```python
Bind(self, event, handler, source=None, id=wx.ID_ANY, id2=wx.ID_ANY)
```
因此，自定义TextCtrl中的这个Bind()方法就是对它的重载：
```python
def Bind(self, z, f):self.f = f
```
即z就是个事件event，f就是句柄或称事件处理函数handler。因此，调用Bind()的时候就是给定两个参数，一个是event，一个是handler。

这个地方龙哥又延伸了一下：
> 其实所有的控件，本质上都是GDI draw出来的（查看ImagePy的Histgram的panel就知道都是调用了GDI绘图），包括按钮、选项卡和文本框等等。
> 所有的控件本质上都是一个panel，可以draw自己（根据数据进行draw），也可以响应事件（基础事件也只有鼠标点击和键盘，具体是哪个键则是操作系统维护的）。所以，控件的核心就是数据关联，然后draw和Bind。
> 举个例子：ImagePy的表格控件，也只是一个panel，要做的事情就是关联数据、维护表格行列宽度、绘制滚动条，然后计算显示的范围、绘制行列分割线、绘制每个单元格的值。然后重载鼠标单击事件，对外暴露成一个Bind，关联一个特殊的标识符，比如叫wx.EVT_Cell_Clicked，然后根据鼠标点击的x和y，计算出对应的单元格，然后高亮绘制，看起来就像是选中了。
> 没有现成的控件，就要从panel开始重载（自己动手、丰衣足食）。


# 问卷调查
这个插件详细说明了怎样设置参数和定制对话框样式。
```python
from imagepy.core.engine import Free
from imagepy import IPy

class Plugin(Free):
    title = 'Questionnaire'

    para = {'name':'yxdragon', 'age':10, 'h':1.72, 'w':70, 'sport':True, 'sys':'Mac', 'lan':['C/C++', 'Python'], 'c':(255,0,0)}

    view = [('lab', 'lab', 'This is a questionnaire'),
            (str, 'name', 'name', 'please'),
            (int, 'age', (0,150), 0, 'age', 'years old'),
            (float, 'h', (0.3, 2.5), 2, 'height', 'm'),
            ('slide', 'w', (1, 150), 0, 'kg'),
            (bool, 'sport', 'do you like sport'),
            (list, 'sys', ['Windows','Mac','Linux'], str, 'favourite', 'system'),
            ('chos', 'lan', ['C/C++','Java','Python'], 'lanuage you like(multi)'),
            ('color', 'c', 'which', 'you like')]

    def run(self, para=None):
        rst = ['Questionnaire Result',
            'Name:%s'%para['name'],
            'Age:%s'%para['age'],
            'Height:%sm'%para['h'],
            'Weight:%skg'%para['w'],
            'Like Sport:%s'%para['sport'],
            'Favourite System:%s'%para['sys'],
            'Like lanuage:%s'%para['lan'],
            'Favourite Color:%s'%str(para['c'])]

        IPy.alert('\r\n'.join(rst))
```
首先，para就是一个python字典，里面就是一系列的key-value键值对。
view是一个列表，具体格式得按照一定的规则。明天就是70周年国庆了，这里先将格式要求抄录如下，后面再详细分析。
lab: para类型：不需要参数, view用法：('lab', 'lab', 'what you want to show')
str: para类型：str, view用法：(str, key, prefix, suffix)，其中key要和para中的key对应，prefix，suffix用作输入框前后的提示内容。
int: para类型：int，view用法：(int, key, (lim1, lim2), accu, 'prefix', 'suffix')，其中key要和para中的key对应，limit用于限定输入数值的范围，accu限定小数点位数(0)，prefix，suffix用作输入框前后的提示内容。
float: para类型：float，view用法：(int, key, (lim1, lim2), accu, 'prefix', 'suffix')，其中key要和para中的key对应，limit用于限定输入数值的范围，accu限定小数点位数，prefix，suffix用作输入框前后的提示内容。
slider: para类型：int/float，view用法：('slide', key, (lim1, lim2), accu, 'prefix')，其中key要和para中的key对应，limit用于限定输入数值的范围，accu限定小数点位数，prefix用作输入框前后的提示内容。
bool: para类型：bool，view用法：(bool, 'key', 'label')，其中key要和para中的key对应，label用作提示。
list: para类型：any type，view用法：(list, key, [choices], type, prefix, suffix)，其中key要和para中的key对应，choices是字符选项，type是期望输出类型，如str, int，prefix，suffix用作选择框前后的提示内容。
choices: para类型：str list，view用法：('chos', key, [choices], prefix, suffix)，与list类似，不同的是choices可以支持多选，选项以list of string形式记录。
color: para类型：(r,g,b) 0-255，用法：('color', key, prefix, suffix)，其中key要和para中的key对应，prefix，suffix用作输入框前后的提示内容。
