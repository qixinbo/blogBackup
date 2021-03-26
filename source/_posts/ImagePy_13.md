---
title: ImagePy解析：13 -- Macros引擎及宏录制
tags: [ImagePy]
categories: computer vision 
date: 2019-11-4
---

参考文献：
[ImagePy开发文档 —— 宏引擎](https://zhuanlan.zhihu.com/p/25483846)
[Macros 插件](https://github.com/Image-Py/demoplugin/blob/master/doc/chinese/macros.md#Macros)

Macros 是一个宏执行器引擎，它负责将一串 ImagePy 命令依次执行。 事实上我们几乎不会去继承 Macros，它仅仅是 ImagePy 为了实现宏功能，并统一为一种引擎接口而设计的辅助类。
因此，Macros引擎常见的用法是：首先通过宏录制器来完成记录，宏录制器在**Plugins > Macros> Macros Recorder**， 然后将录制的命令保存到menus或其子文件夹里，以mc作为后缀，重启即可加载到对应位置。（语出上面的参考文献）

# Macros引擎
先来看一下Macros类的全貌：
```python
def stepmacros(plg, callafter=None):
    plg._next(callafter)
pub.subscribe(stepmacros, 'stepmacros')

class Macros:
    def __init__(self, title, cmds):
        …
    def _next(self, callafter=None):
        …
    def next(self):
        …
    def run(self):
        …
    def __call__(self):
        …
    def start(self, para=None, callafter=None):
        …
```
下面具体看一下Macros类的属性和方法。
## 初始化函数
```python
def __init__(self, title, cmds):
    self.title = title
    self.cmds = cmds
```
可以看出，在初始化函数中需要传入宏命令的名称title和具体的执行操作。
这一块实际调用是发生在主界面解析插件目录时，具体是：
```python
def extend_plugins(path, lst, err):
    ……
        elif i[-3:] == '.mc':
            pt = os.path.join(root_dir, path)
            f = open(pt+'/'+i, 'r', 'utf-8')
            cmds = f.readlines()
            f.close()
            rst.append(Macros(i[:-3], [getpath(pt, i) for i in cmds]))
            PluginsManager.add(rst[-1])
```
可以看出，先判断脚本文件是否为mc后缀，如果是的话，则按行读取文件内容，然后根据根据这些信息创建一个Macros对象，传入PluginsManager管理器中。

因此，整个宏文件形成一个插件，会被解析成一个菜单项，然后如果点击的话，就是执行的Macros类的start()函数。

##_next()函数
```python
def _next(self, callafter=None):
    if self.cur==len(self.cmds):
        if self.callafter!=None:
            self.callafter()
        return
    if len(self.cmds[self.cur])<3 or self.cmds[self.cur][0] == '#':
        self.cur += 1
       return self._next(callafter)
    title, para = self.cmds[self.cur].split('>')
    self.cur += 1
    plg = PluginsManager.get(title)()
    plg.start(eval(para), self.next)
```
这个函数的作用是实际读取宏文件中的命令并执行。它是通过游标self.cur来逐行读取。
第一个判断语句是判断是否将命令列表读完，如果读完了，则返回。
第二个判断语句是判断该行是否是小于三个字符或以井号开头，如果是，则读下一行。以井号开头说明该行是条注释。小于三个字符是为了忽略空行，这个地方在不同系统上有一个微小不同，见[彻底解读剪不断理还乱的\r\n和\n, 以Windows和Linux为例](https://blog.csdn.net/stpeace/article/details/45767245)。
> Windows系统中有如下等价关系： 用enter换行 <====> 程序写\n  <====> 真正朝文件中写\r\n(0x0d0x0a) <====>程序真正读取的是\n
> linux系统中的等价关系： 用enter换行 <====> 程序写\n  <====> 真正朝文件中写\n(0x0a)  <====> 程序真正读取的是\n
 
后面的语句是真正执行命令：
```python
plg = PluginsManager.get(title)()
plg.start(eval(para), self.next)
```
## next()函数
```python
def next(self):
    if IPy.uimode() == 'no':
        self._next(self)
    else: wx.CallAfter(pub.sendMessage, 'stepmacros', plg=self)
```
采用pub-sub模式，然后调用上面的_next()函数。

## run()函数
```python
def run(self):self.next()
```
这是为了与其他插件进行统一，run()也就是调用了next()函数。

## start()函数
```python
def start(self, para=None, callafter=None):
    self.callafter = callafter
    self.cur = 0
    self.run()
```
前面已经说过，这是入口函数。

# 宏录制
## 录制操作步骤
Plugins > Macros > Macros Recorder
## 录制机制
首先这个Macros Recorder是个wgt.py文件，在初始解析插件时就会被加入到组件管理器WidgetManager中。然后它有一个write()方法：
```python
def write(self, cont):
if not self.recording: return
self.txt_cont.AppendText((cont+'\n'))
```
即在该panel的文本框中记录执行的命令名称。
 
然后仔细查看Free、Simple和Filter引擎类中，都可以发现这么两行代码：
```python
win = WidgetsManager.getref('Macros Recorder')
if win!=None:
    win.write('{}>{}'.format(self.title, para))
```
即后面在执行某个插件时，都会调用宏录制器，将插件的名字title和参数para传给它，使其记录下来，记录形式为：插件名称>参数字典。
## 执行机制
（语出上面的参考文献）
所有的插件都被 PluginsManager 所管理，PluginsManager 内部实际上是维护了一个以插件的 title 为主键的键值对。所以我们这样做：
1.解析宏命令，用 ‘>’ 进行字符串分割
2.用分割的 title 作为主键去 PluginsManager 中查找，得到滤波器的实例。
3.调用 eval 函数，把 para 重新解析成 python 对象（这里充分发挥了脚本语言的优势）
4.执行获取的滤波器的 start 方法，把 para 当作参数输入。
（还记得引起的 start() 方法特性，当 para 为 None 时进行交互，否则直接执行 run）

最后一条值得注意，如果该有参数的地方设置为None，则会跳出参数对话框供手动设置，也可以直接传入参数。
