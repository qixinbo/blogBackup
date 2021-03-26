---
title: ImagePy解析：21 -- 管理器
tags: [ImagePy]
categories: computer vision 
date: 2020-6-19
---

ImagePy中的管理器分两类：Source里的管理器维护全局静态数据，比如读写器、配置文件等， App里面的管理器维护运行时数据，比如图像、表格。

# 静态管理器
## 创建管理器
这里创建一个money管理器，里面可以添加美元USD、欧元EUR、人民币RMB，比如：
```python
Source.manager('money').add('USD', MoneyReader, 'MoneyDisplay')
Source.manager('money').add('EUR', MoneyReader, 'MoneyDisplay')
Source.manager('money').add('RMB', MoneyReader, 'MoneyDisplay')
```
意思就是管理器名为money，添加的成员有USD、EUR、RMB，处理方式是MoneyReader，显示方式是MoneyDisplay。
add方法的这三个参数分别对应的形参为name、obj和tag，可以这样统一理解：name表示对象名，obj表示处理方式，MoneyDisplay是显示方式。可以这样来感性认识，也可以认为这三个参数的地位是平齐的，因为有这三个参数可以用于索引，所以可表示的范围会非常大，导致manager的用处也非常广。

## 读取管理器
money管理器中添加元素以后，可以再在全局读取出来。
最重要的读取方式就是Manager类的gets()方法，即：
```python
    def gets(self, name=None, tag=None, obj=None):
        rst = [i for i in self.objs if name is None or name == i[0]]
        rst = [i for i in rst if obj is None or obj is i[1]]
        return [i for i in rst if tag is None or tag == i[2]]
```
可以看出，可以根据name、tag和obj来读取。从上面代码可以看出，如果不明确指定某一参数的话，就认为该参数不做过滤条件。
以tag过滤为例，假设读取设置为：
```python
print("gets = ", Source.manager('money').gets(tag='MoneyDisplay'))
```
那么返回结果就是：
```python
gets =  [('RMB', <function MoneyReader at 0x000002E0C3C6F510>, 'MoneyDisplay'), ('EUR', <function MoneyReader at 0x000002E0C3C6F510>, 'MoneyDisplay'), ('USD', <function MoneyReader at 0x000002E0C3C6F510>, 'MoneyDisplay')]
```
即将tag为MoneyDisplay的所有对象都返回。
还有一个直接获取管理器中的对象名称的快捷方式，即：
```python
print("names = ", Source.manager('money').names())
```
返回结果为：
```python
names =  ['RMB', 'EUR', 'USD']
```

## 管理器持久化
管理器中的对象可以通过持久化将内存中的数据存储到磁盘上，该功能对于配置类文件非常重要，因为可以及时存储软件设置。
管理器的持久化使用的是write函数，如下：
```python
Source.manager('money').write("1.txt")
```
这里需要注意的是上面我们设定的MoneyReader是一个函数，所以无法json化，所以我们这里将其设为None，才能正确存储。
这样该txt文件中的内容就是：
```python
[["RMB", null, "MoneyDisplay"], ["EUR", null, "MoneyDisplay"], ["USD", null, "MoneyDisplay"]]
```
持久化以后还可以读取回来：
```python
Source.manager('money').read("1.txt")
```

## App中的静态管理器
App类中也有一个与Source类似的静态管理器，用来管理color和roi，即颜色管理器和roi管理器，这样就能在全局来调用颜色和roi。
具体用法见上面的静态管理器。

# 动态管理器
App类中的动态管理器用来管理ImagePy所打开的图像、图像窗口（即画布）、表格、表格窗口、网格、网格窗口和任务，即：
```python
class App():
    def __init__(self):
        self.img_manager = Manager()
        self.wimg_manager = Manager()
        self.tab_manager = Manager()
        self.wtab_manager = Manager()
        self.mesh_manager = Manager()
        self.wmesh_manager = Manager()
        self.task_manager = Manager()
        self.managers = {}
```
下面以图像管理器为例，看一下动态管理器的运行机制。
## 创建管理器并添加元素
动态管理器的创建实际在App类创建时就在初始化时创建。
下面是添加图像。
```python
    app = wx.App(False)
    frame = ImagePy(None)
    frame.Show()
    frame.show_img([np.zeros((512, 512), dtype=np.uint8)], 'zeros')
    frame.show_img([np.ones((512, 512), dtype=np.uint8)], 'ones')

    app.MainLoop()
```
这里我们通过ImagePy框架中的show_img添加了两张图像，一张名为zeros，一张名为ones。
实际查看该函数的源码：
```python
    def _show_img(self, img, title=None):
        canvas = self.canvasnb.add_canvas()
        self.remove_img(canvas.image)
        self.remove_img_win(canvas)
        if not title is None:
            canvas.set_imgs(img)
            canvas.image.name = title
        else: canvas.set_img(img)
        self.add_img(canvas.image)
        self.add_img_win(canvas)

    def show_img(self, img, title=None):
        wx.CallAfter(self._show_img, img, title)
```
可以看出，它是调用了App类的add_img和add_img_win来添加图像及图像窗口（画布）。
注意，这里之所以frame能调用App类中的这两个方法，因为frame是ImagePy类的实例对象，而ImagePy类既继承了wx.Frame类，也继承了App类。
## 读取管理器
在插件中如果需要获取图像或其窗口，则可以使用：
```python
self.app.get_img_win()
self.app.get_img()
```
这里之所以这样调用，是因为App类的实例化对象app是贯穿全局的，任何一个tool或menu在start()启动的时候都需要传入app，所以app能统领全局。
