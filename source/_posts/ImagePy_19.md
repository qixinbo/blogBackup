---
title: ImagePy解析：19 -- Mark模式
tags: [ImagePy]
categories: computational material science
date: 2020-3-28
---

%%%%%%%%%%%%%%
2021-1-31更新：基于最新的sciwx修改了以前的失效代码。
%%%%%%%%%%%%%%

ImagePy/sciwx有个Mark模式，即在图像上面可以再绘制其他图形，比如矩形、文本、ROI标识等，本质即是利用GDI绘图。
本文是对该Mark模式的解析。

# demo
先给出一个小的demo，主要就是为了看Mark模式的输入输出，方便理解它的运行本质。
```python
from sciwx.canvas.mark import drawmark
from sciapp.object import mark2shp

mark =  {'type': 'layer', 'body': [{'type': 'rectangle', 'body': (50, 50, 200, 200), 'color': (255, 0, 0)}, {'type': 'circle', 'body': (150, 150, 5), 'color': (0, 0, 255)}, {'type': 'text', 'body': (75, 75, 'S:30 W:48'), 'pt': False, 'color': (0, 255, 0)}]}

shape = mark2shp(mark)

def f(x, y):
    return x, y

class Example(wx.Frame):
    def __init__(self, parent, obj):
        super().__init__(parent)
        self.obj = obj

        self.Bind(wx.EVT_PAINT, self.DrawMarks)

    def DrawMarks(self, e):
        dc = wx.PaintDC(self)
        drawmark(dc, f, self.obj, k=1)

app = wx.App()
ex = Example(None, shape)
ex.Show()
app.MainLoop()
```
一定要使用这种PaintEvent事件来调用画图（或者调用CallLater），否则会看不到所绘制的图形，具体原因见下方链接：
[wxPython graphics](http://zetcode.com/wxpython/gdi/)

效果如图（注意这张是旧图，最新代码生成的图略有不同）：
![mark](https://user-images.githubusercontent.com/6218739/77822795-eace9f00-7130-11ea-9033-a28077379552.png)

可以看出，整个程序的逻辑很简单，
（1）创建一个mark配置（具体写法后面介绍），传入mark2shp函数对其转换一下
（2）创建一个坐标映射函数f，这是为了坐标变换（这里不涉及坐标变换，所以直接原样返回）
（3）通过wx.PaintDC创建一个设备上下文dc
（4）将dc、f和缩放因子k传入drawmark方法，从而进行绘制
 

下面分步具体解析。
# mark配置
mark写法：
```python
mark =  {'type': 'layer', 'body': [{'type': 'rectangle', 'body': (50, 50, 200, 200), 'color': (255, 0, 0)}, {'type': 'circle', 'body': (150, 150, 5), 'color': (0, 0, 255)}, {'type': 'text', 'body': (75, 75, 'S:30 W:48'), 'pt': False, 'color': (0, 255, 0)}]}
```
（之所以这样书写，是因为这样写是人类友好的，后面会将这一阅读良好的字典转换成sciapp内部的特有的Shape数据结构。）

mark的写法是这样的：
（1）mark是个字典，里面的重要的键值比如type、body等；
（2）第一层的type是layer，表明这是多个图形的结合，那么这一层的body就是所包含的图形的list
（3）具体到某个具体所绘制的图形，以上面的配置为例：
（3.1）矩形，其键值对有：type是rectangle，body是四个整型数值，即x、y、w和h，其中x和y是矩形的左上角（这里跟之前旧代码不同，原先的是矩形中心），w和h是高和宽，color是绘制的颜色。
（3.2）圆形：其键值对有：type是circle，body是三个整型数值，即x、y和r，即圆形中心和半径。
（3.3）文字：其键值对有：type是text，body是两个整型数值和一个文本，即x、y和文本内容，x、y是绘制文本的左上角，pt是指定是否在该左上角绘制一个小圆点。

# 坐标映射函数和缩放因子
这个函数f存在的意义是在imagepy/sciwx的canvas中，其需要将图像坐标系中的坐标转换到面板坐标系中，所以在canvas的源码中可以看到，该函数f就是传入的转换到面板坐标的函数：
```python
drawmark(dc, self.to_panel_coor, self.marks[i], k=self.scale)
```
同理，缩放因子也是为了适应canvas画布的缩放而需要传入的。
因为这里没有使用到canvas，所以f中没有做任何变换，而k也是为1。

# drawmark函数
这里看一下该函数的源代码：
```python
def drawmark(dc, f, body, **key):
	default_style = body.default
	pen, brush, font = dc.GetPen(), dc.GetBrush(), dc.GetFont()
	pen.SetColour(default_style['color'])
	brush.SetColour(default_style['fcolor'])
	brush.SetStyle((106,100)[default_style['fill']])
	pen.SetWidth(default_style['lw'])
	dc.SetTextForeground(default_style['tcolor'])
	font.SetPointSize(default_style['size'])
	dc.SetPen(pen); dc.SetBrush(brush); dc.SetFont(font);
	draw(body, dc, f, **key)
```
该方法就是先获得设备上下文的画笔pen、画刷brush和字体font，然后通过SetColour设置颜色、SetStyle设置风格、SetWidth设置笔宽、SetPointSize设置字体尺寸等对上述工具进行属性设置。然后再调用全局的draw()函数。
通过该函数也可以看出来，它接收的body需要具有一定的格式，比如有default属性。因此，最开始的mark变量没法直接传入drawmark中，需要使用mark2shp方法来转换一下。

# mark2shp函数
仍然看一下该函数的源代码：
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
这里就引出了sciapp中的自定义的矢量类Shape这一数据结构。
Shape类是这类对象的基类，上述代码中的Point、Line、Rectangle类等都是该类的派生类。
Shape类会在下一篇文章中详细解释，见[这里](https://qixinbo.info/2020/06/14/imagepy_20/)。
总之，该函数就是将mark字典转换成了sciapp内置的Shape数据结构。

# 全局draw()函数
```python
draw_dic = {'points':plot, 'point':plot, 'line':plot,
'polygon':plot, 'lines':plot, 'polygons':plot,
'circle':draw_circle, 'circles':draw_circle,
'ellipse':draw_ellipse, 'ellipses':draw_ellipse,
'rectangle':draw_rectangle, 'rectangles':draw_rectangle,
'text':draw_text, 'texts':draw_text}
 
def draw(obj, dc, f, **key): 
	if len(obj.body)==0: return
	draw_dic[obj.dtype](obj, dc, f, **key)

draw_dic['layer'] = draw_layer
```
可以看出，draw()函数中先从传入的obj中提取它的dtype这个key，从而找到obj中对应的key-value，然后这个value作为draw_dic字典中的键，找到所绘制的类型，从而定位到具体的绘制函数。

对于layer这个type，可以把layer认为是多个图形的集合，会调用draw_layer这个函数，该函数里也会遍历该集合里的所有图形来再调用draw绘制：
```python
def draw_layer(pts, dc, f, **key):
         …
        for i in pts['body']:draw(i, dc, f, **key)
```
以绘制矩形为例，会调用draw_rectangle函数：
```python
def draw_rectangle(pts, dc, f, **key):
	pen, brush = dc.GetPen(), dc.GetBrush()
	width, color = pen.GetWidth(), pen.GetColour()
	fcolor, style = brush.GetColour(), brush.GetStyle()
	
	if not pts.color is None: 
		pen.SetColour(pts.color)
	if not pts.fcolor is None:
		brush.SetColour(pts.fcolor)
	if not pts.lw is None:
		pen.SetWidth(pts.lw)
	if not pts.fill is None:
		brush.SetStyle((106,100)[pts.fill])

	dc.SetPen(pen)
	dc.SetBrush(brush)

	if pts.dtype == 'rectangle':
		x, y, w, h = pts.body
		w, h = f(x+w, y+h)
		x, y = f(x, y)
		dc.DrawRectangle(x.round(), (y).round(), 
			(w-x).round(), (h-y).round())
	if pts.dtype == 'rectangles':
		x, y, w, h = pts.body.T
		w, h = f(x+w, y+h)
		x, y = f(x, y)
		lst = np.vstack((x,y,w-x,h-y)).T
		dc.DrawRectangleList(lst.round())

	pen.SetWidth(width)
	pen.SetColour(color)
	brush.SetColour(fcolor)
	brush.SetStyle(style)
	dc.SetPen(pen)
	dc.SetBrush(brush)
```

可以看出，在绘制矩形时最基本的步骤就是：
（1）将之前的设备上下文传入；
（2）提取特定的Rectangle这一Shape数据结构中的body信息，即绘制矩形的坐标点；
（3）调用最底层的设备上下文的DrawRectangle方法将其绘制出来。

# 保存mark
上面的mark都是在最原始的wxPython的frame上进行绘制，是为了展示mark的本质，实际应用时这些mark都是在Canvas上进行绘制。
（以下内容已合并在ImagePy的master分支中）
如果想在ImagePy中保存mark，可以参考我提交的一个pull request，在[这里](https://github.com/Image-Py/imagepy/pull/96/commits/770342625d320659d2c5c406b1a78a809be086b0)。
原理就是将当前的dc中的内容都save出来。
