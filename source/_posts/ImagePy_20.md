---
title: ImagePy解析：20 -- 几何矢量Shape
tags: [ImagePy]
categories: computer vision 
date: 2020-6-14
---

# 前言

ImagePy中表示几何矢量的结构类是Shape，最直观的一个应用就是各种ROI操作，这里通过一个小例子看看各种几何图形是怎样操纵和显示的。

# 最小demo
```python
from sciapp.object import mark2shp
from sciwx.canvas import VCanvas as Canvas
import wx

circle = {'type':'circle', 'color':(255,0,0), 'fcolor':(255,255,0), 'fill':False, 'body':(100,100,50)}

def mark_test(mark):
    frame = wx.Frame(None, title='gray test')
    canvas = Canvas(frame, autofit=False, up=True)
    canvas.set_shp(mark2shp(mark))
    frame.Show()

if __name__ == '__main__':
    app = wx.App()
    mark_test(circle)
    app.MainLoop()
```
上述是个可运行的最小demo，运行结果为：
![shape](https://user-images.githubusercontent.com/6218739/84457021-2985e700-ac94-11ea-90e7-b38f2869b1ed.png)
可以看出，成功绘制出了一个红色轮廓的圆形。
下面逐步解析一下。

# mark格式
```python
circle = {'type':'circle', 'color':(255,0,0), 'fcolor':(255,255,0), 'fill':False, 'body':(100,100,50)}
```

可以看出，这里的圆形是通过imagepy的mark格式来定义的，即通过一个特定的字典来定义，具体写法见之前的关于mark的解析，在[这里](https://qixinbo.info/2020/03/28/imagepy_19/)。

之所以使用mark格式，是因为它的可读性非常高，如果直接写Shape类会非常不直观。

# mark转Shape
```python
mark2shp(mark)
```
这一步是将上面的mark格式的定义转为imagepy内置的Shape类型的对象。具体的函数定义为：
```python
def mark2shp(mark):
    style = mark.copy()
    style.pop('body')
    keys = {'point':Point, 'points':Points, 'line':Line, 'lines':Lines,
            'polygon':Polygon, 'polygons':Polygons, 'circle':Circle,
            'circles':Circles, 'rectangle':Rectangle, 'rectangles':Rectangles,
            'ellipse':Ellipse, 'ellipses':Ellipses, 'text':Text, 'texts':Texts}

    if mark['type'] in keys: return keys[mark['type']](mark['body'], **style)
    if mark['type']=='layer':
        return Layer([mark2shp(i) for i in mark['body']], **style)
    if mark['type']=='layers':
        return Layers(dict(zip(mark['body'].keys(),
            [mark2shp(i) for i in mark['body'].values()])), **style)
```
里面的Point、Circle、Rectangle、Ellipse就是ImagePy内置的几何类，它们有一个共同的基类，即Shape类，里面有三个重要的属性和方法，这也是它的不同子类之间需要进行的重载实现（以Circle为例）：

（1）body属性：
```python
self.body = np.array(body, dtype=np.float32)
```
将mark格式的body传入numpy的array数组中，然后赋给Shape对象的body属性。
这里使用numpy数组的原因有如下几个（源自龙哥的答疑）：
- 在draw的时候，需要根据canvas的位移和比例，进行一个加乘运算，得到最后需要draw的画布坐标，即下面代码：
```python
	if pts.dtype == 'circles':
		lst = []
		x, y, r = pts.body.T
		x, y = f(x, y)
		r = r * key['k']
		lst = np.vstack([x-r, y-r, r*2, r*2]).T
		dc.DrawEllipseList(lst)
```
- Shape也可以自动计算边界，就用数组的min、max，带上axis参数就可以实现
- 编辑的时候，一个snap，其实也要判断所有的点，距离鼠标最近的，也有必要用numpy广播。


另外，Shape类的style也是mark传入，也可以转为JSON格式的数据，具体可以详看Shape类的代码。
（2）转为mark格式：
```python
    def to_mark(self):
        return Shape.to_mark(self, tuple(self.body.tolist()))
```
即将Shape的body转为mark格式。

（3）转为shapely的geom格式：
```python
    def to_geom(self):
        return geom.Point(self.body[:2]).buffer(self.body[2])
```

shapely是一个对几何矢量几何进行操作和分析的python库。
上面这条语句就是将body的前两个数作为点的坐标生成shapely中的Point，然后将body的第三个数（即半径）生成该Point的缓冲区，即形成一个圆形区域。
转为shapely的geometry结构后，就可以进行复杂的几何运算。比如编辑时候的拖拽，判断鼠标是否点击在图形的内部，就需要将Shape转成shapely的geometry。

关于shapely的教程可以参考：
[矢量数据的空间分析：使用Shapely](https://www.osgeo.cn/pygis/shapely.html)
[基于Python的缓冲区分析](https://zhuanlan.zhihu.com/p/24782733)


# 将Shape传入画布
```python
canvas.set_shp(mark2shp(mark))
```
注意，这里的canvas对象其实是VCanvas类的对象，实际做的是：
```python
class VCanvas(Canvas):
    def __init__(self, parent, autofit=False, ingrade=True, up=True):
        Canvas.__init__(self, parent, autofit, ingrade, up)

    def set_shp(self, shp):
        self.marks['shape'] = shp
        self.update()
```
可以看出，VCanvas继承了Canvas，所以就是将Circle对象传给了Canvas的marks属性（这个属性是个字典）的shape这个key。
并且调用Canvas的update进行更新。

# 画布绘制几何图形
那么在画布中是怎样绘制几何图形的呢？
具体代码实现如下：
```python
        for i in self.marks.values():
            if i is None: continue
            if callable(i):
                i(dc, self.to_panel_coor, k = self.scale)
            else:
                drawmark(dc, self.to_panel_coor, i, k=self.scale, cur=0,
                    winbox=self.winbox, oribox=self.oribox, conbox=self.conbox)
```
首先通过字典的values方法返回marks属性中的所有的值values，以这里绘制Circle为例，那么返回的i就是：
```python
<class 'sciapp.object.shape.Circle'>
```
即i是Circle对象。注意这里不要使用print来直接打印i，而需要使用type来显示，因为在Shape类中有一个方法：
```python
    def __str__(self):
        return str(self.to_mark())
```
即如果想看i的值时，会先将其转为mark格式再打印出来，但实际i是个Shape对象。
因为i不是callable的，所以就会调用drawmark来显示，最终是使用dc的DrawCircle来绘图。具体绘制过程也可以参见mark模式解析那一篇。

# 添加shape动作
这一部分是shape对象进阶，主要看怎样在画布中实时绘制shape，涉及了shape动作和鼠标事件。
最小可用的demo如下：
```python
from sciapp.object import mark2shp
from sciapp.action import EllipseEditor
from sciwx.canvas import VCanvas as Canvas
import wx

circle = {'type':'circle', 'color':(255,0,0), 'fcolor':(255,255,0), 'fill':False, 'body':(100,100,50)}

layer = {'type':'layer', 'num':-1, 'color':(0,0,255), 'fcolor':(255,255,255), 'fill':False, 'body':[circle]}

def mark_test(mark):
    frame = wx.Frame(None, title='gray test')
    canvas = Canvas(frame, autofit=False, up=True)
    canvas.set_shp(mark2shp(mark))
    frame.Show()

if __name__ == '__main__':
    app = wx.App()
    EllipseEditor().start(None)
    mark_test(layer)
    app.MainLoop()
```
可以看出，就是在最上面例子上添加了一个自由绘制椭圆的动作。
具体分析一下：
首先添加椭圆编辑器：
```python
from sciapp.action import EllipseEditor
EllipseEditor().start(None)
```
这个椭圆编辑器实际是一个工具Tool，它最开始的源头可视为Tool类，即：
```python
class Tool(SciAction):
    title = 'Base Tool'
    default = None
    cursor = 'arrow'

    def mouse_down(self, canvas, x, y, btn, **key): pass
    def mouse_up(self, canvas, x, y, btn, **key): pass
    def mouse_move(self, canvas, x, y, btn, **key): pass
    def mouse_wheel(self, canvas, x, y, d, **key): pass

    def start(self, app):
        self.app, self.default = app, self
        if not app is None: app.tool = self
```
Tool中定义了鼠标动作，最原始的Tool中只是提供了鼠标动作定义入口，并没有具体的动作。
Tool派生了DefaultTool，可以实现最朴素的移动画布和缩放画布功能（具体见DefaultTool代码）。
DefaultTool派生了ShapeTool，不过这个派生并没有实质性的扩展，只是为了与ImageTool、TableTool进行区分。
ShapeTool派生了BaseEditor，该工具对Shape对象进行了深度的动作定制：
（1）鼠标中键拖动；
（2）alt+右键：删除一个shape
（3）shift+右键：合并shape
（4）右键：将shape根据当前区域大小缩放
（5）alt+ctrl：显示锚点（注意这个地方得移动一下鼠标，因为没有定义单独的键盘事件，否则无法触发动作）
（6）alt+ctrl+鼠标拖动锚点：改变shape

BaseEditor派生了EllipseEditor，该工具又对椭圆形状进行了自定义：
（1）鼠标左键按下并拖动：新建一个椭圆；
（2）alt+新建椭圆：两者做差；
（3）shift+新建椭圆：两者取并集
（4）alt+shift+新建椭圆：两者取交集

那么，画布是怎样获取shape和tool的呢？答案就在Canvas中的这两行：
```python
obj, tol = self.get_obj_tol()
btn, tool = me.GetButton(), self.tool or tol
```
第一行得到了当前的对象，对于shape对象，注意VCanvas的这两个方法和属性：
```python
class VCanvas(Canvas):
    def __init__(self, parent, autofit=False, ingrade=True, up=True):
        Canvas.__init__(self, parent, autofit, ingrade, up)

    def get_obj_tol(self):
        return self.shape, ShapeTool.default

    def set_shp(self, shp):
        self.marks['shape'] = shp
        self.update()

    def set_tool(self, tool): self.tool = tool

    @property
    def shape(self):
        if not 'shape' in self.marks: return None
        return self.marks['shape']
```
即VCanvas重载了Canvas的获取对象的方法，同时shape属性又获得了之前的Shape对象。
另外需要注意的是在EllipseEditor中添加椭圆时用的是：
```python
shp.body.append(self.obj)
```
所以这也就是为什么在程序中又新加了一个layer，即：
```python
layer = {'type':'layer', 'num':-1, 'color':(0,0,255), 'fcolor':(255,255,255), 'fill':False, 'body':[circle]}
```
否则body中无法append进去。
