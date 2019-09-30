---
title: ImagePy解析：7 -- ImagePy插件运行解析
tags: [ImagePy]
categories: computational material science 
date: 2019-9-30
---

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
首先判断self.load()，从前面代码可以看出，这条语句不会执行。继续往下，判断para是否不为None或者self.show()是否为True，如果两者有其一满足，则继续往下走，然后就会以新开一个线程或在这个线程下直接运行self.runasyn()，这个函数就会继续调用self.run()方法。

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
注意此时设置了para和view都不为None，那么在start()方法中，就会因为view有值而进入交互模式，弹出一个对话框，这个对话框也是定制的，基类是wxPython的Diglog类。para是接收的具体参数，view来控制这个对话框的显示样式。这个插件也是重写了run()函数，将之前对话框中输入的数值显示出来。

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
