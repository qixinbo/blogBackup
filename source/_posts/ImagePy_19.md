---
title: ImagePy解析：19 -- Mark模式
tags: [ImagePy]
categories: computational material science 
date: 2020-3-28
---

ImagePy/sciwx有个Mark模式，即在图像上面可以再绘制其他图形，比如矩形、文本、ROI标识等，本质即是利用GDI绘图。
本文是对该Mark模式的解析。

# demo
先给出一个小的demo，主要就是为了看Mark模式的输入输出，方便后面在此基础上二次开发。
```python
from sciwx.mark.mark import GeometryMark

mark =  {'type': 'layer', 'body': [{'type': 'rectangle', 'body': (117, 133, 96, 96), 'color': (255, 0, 128)}, {'type': 'circle', 'body': (117, 133, 2), 'color': (255, 0, 128)}, {'type': 'text', 'body': (69, 85, 'S:30 W:48'), 'pt': False, 'color': (255, 0, 128)}]}

geometry_mark = GeometryMark(mark)

def f(x, y):
    return x, y

class Example(wx.Frame):
    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)

        self.Bind(wx.EVT_PAINT, self.DrawMarks)

    def DrawMarks(self, e):
        dc = wx.ClientDC(self)
        geometry_mark.draw(dc, f, k=1)

app = wx.App()
ex = Example(None)
ex.Show()
app.MainLoop()
```

一定要使用这种PaintEvent事件来调用画图（或者调用CallLater），否则会看不到所绘制的图形，具体原因见下方链接：
[wxPython graphics](http://zetcode.com/wxpython/gdi/)

效果如图：
![mark](https://user-images.githubusercontent.com/6218739/77822795-eace9f00-7130-11ea-9033-a28077379552.png)

可以看出，整个程序的逻辑很简单，
（1）创建一个mark配置（具体写法后面介绍），作为GeometryMark对象的参数传入
（2）创建一个坐标映射函数f
（3）通过wx.ClientDC创建一个设备上下文dc
（4）将dc、f和缩放因子k传入GeometryMark对象的draw方法，从而进行绘制

下面分步具体解析。
# mark配置
```python
mark =  {'type': 'layer', 'body': [{'type': 'rectangle', 'body': (117, 133, 96, 96), 'color': (255, 0, 128)}, {'type': 'circle', 'body': (117, 133, 2), 'color': (255, 0, 128)}, {'type': 'text', 'body': (69, 85, 'S:30 W:48'), 'pt': False, 'color': (255, 0, 128)}]}
```
mark的写法是这样的：
（1）mark是个字典，里面的重要的键值比如type、body等；
（2）第一层的type是layer，表明这是多个图形的结合，那么这一层的body就是所包含的图形的list
（3）具体到某个具体所绘制的图形，以上面的配置为例：
（3.1）矩形，其键值对有：type是rectangle，body是四个整型数值，即x、y、w和h，其中x和y是矩形的中心，w和h是高和宽，但实际绘制时还要乘以缩放因子k，color是绘制的颜色。
（3.2）圆形：其键值对有：type是circle，body是三个整型数值，即x、y和r，即圆形中心和半径。
（3.3）文字：其键值对有：type是text，body是两个整型数值和一个文本，即x、y和文本内容，x、y是绘制文本的左上角，pt是指定是否在该左上角绘制一个小圆点。

# 坐标映射函数和缩放因子
这个函数f存在的意义是在imagepy/sciwx的canvas中，其需要将图像坐标系中的坐标转换到面板坐标系中，所以在canvas的源码中可以看到，该函数f就是传入的转换到面板坐标的函数：
```python
drawmark(dc, self.to_panel_coor, self.marks[i], k=self.scale)
```
同理，缩放因子也是为了适应canvas画布的缩放而需要传入的。
因为这里没有使用到canvas，所以f中没有做任何变换，而k也是为1。

# GeometryMark类
## 初始化函数
```python
class GeometryMark:
def __init__(self, body):
self.body = body
```

可以看出，GeometryMark类初始化时需要传入一个body参数，即mark配置。

## draw方法
```python
def drawmark(dc, f, body, **key):
	pen, brush, font = dc.GetPen(), dc.GetBrush(), dc.GetFont()
	pen.SetColour(ConfigManager.get('mark_color') or (255,255,0))
	brush.SetColour(ConfigManager.get('mark_fcolor') or (255,255,255))
	brush.SetStyle((106,100)[ConfigManager.get('mark_fill') or False])
	pen.SetWidth(ConfigManager.get('mark_lw') or 1)
	dc.SetTextForeground(ConfigManager.get('mark_tcolor') or (255,0,0))
	font.SetPointSize(ConfigManager.get('mark_tsize') or 8)
	dc.SetPen(pen); dc.SetBrush(brush); dc.SetFont(font);
	draw(body, dc, f, **key)

class GeometryMark:
	def __init__(self, body):
		self.body = body

	def draw(self, dc, f, **key):
		drawmark(dc, f, self.body, **key)
```

该方法就是先获得设备上下文的画笔pen、画刷brush和字体font，然后通过SetColour设置颜色、SetStyle设置风格、SetWidth设置笔宽、SetPointSize设置字体尺寸等对上述工具进行属性设置。然后再调用全局的draw()函数。

Attention!!!:之所以将drawmark独立出来，是因为canvas画布中可以调用它，实际上将它独立出来后，完全就可以只使用这个函数，而不使用GeometryMark类。
 
# 全局draw()函数
```python
draw_dic = {'points':plot, 'point':plot, 'line':plot, 'polygon':plot, 'lines':plot, 'polygons':plot, 'circle':draw_circle, 'circles':draw_circle, 'ellipse':draw_ellipse, 'ellipses':draw_ellipse, 'rectangle':draw_rectangle, 'rectangles':draw_rectangle, 'text':draw_text, 'texts':draw_text}

def draw(obj, dc, f, **key):
	draw_dic[obj['type']](obj, dc, f, **key)

draw_dic['layer'] = draw_layer
```
可以看出，draw()函数中先从传入的obj中提取它的type这个key，从而找到obj中对应的key-value，然后这个value作为draw_dic字典中的键，找到所绘制的类型，从而定位到具体的绘制函数。

对于layer这个type，可以把layer认为是多个图形的集合，会调用draw_layer这个函数，该函数里也会遍历该集合里的所有图形来再调用draw绘制：

```python
def draw_layer(pts, dc, f, **key):
         …
        for i in pts['body']:draw(i, dc, f, **key)
```
以绘制矩形为例，会调用draw_rectangle函数：
```python
def draw_rectangle(pts, dc, f, **key):
	if pts['type'] == 'rectangle':
		x, y, w, h = pts['body']
		x, y = f(x, y)
		w, h = w*key['k'], h*key['k']
		dc.DrawRectangle(x-w/2, y-h/2, w, h)
```

可以看出，它会提取传入的配置的body键，获取x、y、w、h四个值，然后：
（1）将函数f作用在x和y上，这是对这两个坐标进行某种坐标系的转换
（2）对w和h乘以key参数中的k值，即缩放因子
（3）这里注意x和y不是所绘矩形的左上角，而是它的中心，因为传入DrawRectangle函数时两者分别减去了宽和高的二分之一。

# 保存mark
如果想在ImagePy中保存mark，可以参考我提交的一个pull request，在[这里](https://github.com/Image-Py/imagepy/pull/96/commits/770342625d320659d2c5c406b1a78a809be086b0)。
原理就是将当前的dc中的内容都save出来。
