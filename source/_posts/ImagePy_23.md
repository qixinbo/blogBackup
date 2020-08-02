---
title: ImagePy解析：23 -- ROI操作
tags: [ImagePy]
categories: computational material science 
date: 2020-8-2
---

前面有两篇文章介绍了ImagePy/sciwx的[Mark模式](https://qixinbo.info/2020/03/28/imagepy_19/)和[几何矢量](https://qixinbo.info/2020/06/14/imagepy_20/)，这两个的结合就是图像处理中经典的ROI(Region Of Interest)操作，即选定一个范围（矩形、圆形、自由区域），然后对该区域进行进一步的操作。
这个过程说起来非常简单，但实际实现起来却是非常不容易，因为这里面涉及到了图像这一位图格式和几何这一矢量格式的统一。
这一篇文章就着重剖析一下ImagePy/sciwx是怎样实现的。

本文选定的入手案例是“绘制矩形ROI，然后裁剪”。

# 矩形ROI
首先看矩形ROI的绘制时怎样实现的。
```python
from sciapp.action import RectangleROI as Plugin
```
可以看出，就是从sciapp的action包中直接导入了RectangleROI模块。

## RectangleROI
再深入看一下RectangleROI是怎样的。
```python
class RectangleROI(BaseROI):
	title = 'Rectangle ROI'
	def __init__(self): 
		BaseROI.__init__(self, RectangleEditor)
```
即，RectangleROI的父类是BaseROI，然后给RectangleROI一个特定的名称。另外一个非常重要的点就是在RectangleROI初始化函数中，对父类BaseROI的初始化中传入了RectangleEditor，而这正是RectangleROI与其他ROI的本质区别，比如EllipseROI传入的是EllipseEditor，PointROI传入的是PointEditor，而这些Editor实际又是BaseEditor的子类。

换句话说，这些ROI是两个重要的类（BaseROI和BaseEditor）的组合，具备这两个类的综合特性；这也呼应了文章开头所说的ROI操作需要兼具“位图”和“矢量图”的特点。

接下来分别深入这两个重要的类。

## BaseROI
首先是BaseROI。
```python
class BaseROI(ImageTool):
	def __init__(self, base): 
		base.__init__(self)
		self.base = base

	def mouse_down(self, img, x, y, btn, **key):
		if img.roi is None: img.roi = ROI()
		else: img.roi.msk = None
		self.base.mouse_down(self, img.roi, x, y, btn, **key)

	def mouse_up(self, img, x, y, btn, **key):
		self.base.mouse_up(self, img.roi, x, y, btn, **key)
		if not img.roi is None:
			if len(img.roi.body)==0: img.roi = None
			else: img.roi.msk = None

	def mouse_move(self, img, x, y, btn, **key):
		self.base.mouse_move(self, img.roi, x, y, btn, **key)

	def mouse_wheel(self, img, x, y, d, **key):
		self.base.mouse_wheel(self, img.roi, x, y, d, **key)
```

BaseROI的源代码对它的来源讲得一目了然，它的父类是ImageTool，即它本质是ImageTool。为什么这一点是如此重要。因为无论是自定制的图像处理工具，还是现成的ImagePy，其画布都是对最底层的ICanvas类的封装，而ICanvas中绑定的tool就是ImageTool，见：

```python
class ICanvas(Canvas):
    def __init__(self, parent, autofit=False):
        Canvas.__init__(self, parent, autofit)
        self.images.append(Image())
        #self.images[0].back = Image()
        self.Bind(wx.EVT_IDLE, self.on_idle)

    def get_obj_tol(self):
        return self.image, ImageTool.default
```
当然这里所说的画布是具有常规用途的对位图的图像处理，如果纯粹是对矢量图的画布，则是对最底层的VCanvas的封装（具体可以见sciwx关于shape的各种demo），该类绑定的Tool则是ShapeTool：
```python
class VCanvas(Canvas):
   def __init__(self, parent, autofit=False, ingrade=True, up=True):
        Canvas.__init__(self, parent, autofit, ingrade, up)

    def get_obj_tol(self):
        return self.shape, ShapeTool.default
```

说回BaseROI，可以看出其在初始化函数中需要传入base，比如它的子类RectangleROI在初始化时给它传入的RectangleEditor。

进一步地，可以看出BaseROI的鼠标事件都是调用的该base的鼠标事件。

这个地方需要注意的是，因为BaseROI本质是ImageTool，所以它的鼠标事件函数的第二个形参所传入的是Image对象，而base其实是个ShapeTool（后面详细解析），所以base的鼠标事件函数的第二个形参是个shape对象。两者的结合是在鼠标按下这个事件中进行的：

```python
	def mouse_down(self, img, x, y, btn, **key):
		if img.roi is None: img.roi = ROI()
		else: img.roi.msk = None
		self.base.mouse_down(self, img.roi, x, y, btn, **key)
```
即首先判断一下img的roi属性是否为None，如果为None，则将一个ROI类型的变量赋给它：
```python
class ROI(Layer):
	default = {'color':(255,255,0), 'fcolor':(255,255,255), 
    'fill':False, 'lw':1, 'tcolor':(255,0,0), 'size':8}

	def __init__(self, body=None, **key):
		if isinstance(body, Layer):  body = body.body
		if not body is None and not isinstance(body, list):
			body = [body]
		Layer.__init__(self, body, **key)
		self.fill = False
		self.msk = None
```

这个ROI类的父类是Layer，而Layer的父类又是Shape类，所以ROI本质是个Shape对象，那么它的具体属性和操作就可以参见之前那篇专门的文章了，在[这里](https://qixinbo.info/2020/06/14/imagepy_20/)。

如果img的roi属性不为None的话，就将roi的msk属性设为None。
然后将该roi传入base工具中。

## BaseEditor
前面已经说到RectangleEditor的父类是BaseEditor，BaseEditor本身写了详细的鼠标事件函数。

```python
class BaseEditor(ShapeTool):
	def __init__(self, dtype='all'):
		self.status, self.oldxy, self.p = '', None, None
		self.pick_m, self.pick_obj = None, None

	def mouse_down(self, shp, x, y, btn, **key):
		self.p = x, y
		if btn==2:
			self.status = 'move'
			self.oldxy = key['px'], key['py']
		if btn==1 and self.status=='pick':
			m, obj, l = pick_point(shp, x, y, 5)
			self.pick_m, self.pick_obj = m, obj
		if btn==1 and self.pick_m is None:
			m, l = pick_obj(shp, x, y, 5)
			self.pick_m, self.pick_obj = m, None
		if btn==3:
			obj, l = pick_obj(shp, x, y, 5)
			if key['alt'] and not key['ctrl']:
				if obj is None: del shp.body[:]
				else: shp.body.remove(obj)
				shp.dirty = True
			if key['shift'] and not key['alt'] and not key['ctrl']:
				layer = geom2shp(geom_union(shp.to_geom()))
				shp.body = layer.body
				shp.dirty = True
			if not (key['shift'] or key['alt'] or key['ctrl']):
				key['canvas'].fit()

	def mouse_up(self, shp, x, y, btn, **key):
		self.status = ''
		if btn==1:
			self.pick_m = self.pick_obj = None
			if not (key['alt'] and key['ctrl']): return
			pts = mark(shp)
			if len(pts)>0: 
				pts = Points(np.vstack(pts), color=(255,0,0))
				key['canvas'].marks['anchor'] = pts
			shp.dirty = True

	def mouse_move(self, shp, x, y, btn, **key):
		self.cursor = 'arrow'
		if self.status == 'move':
			ox, oy = self.oldxy
			up = (1,-1)[key['canvas'].up]
			key['canvas'].move(key['px']-ox, (key['py']-oy)*up)
			self.oldxy = key['px'], key['py']
		if key['alt'] and key['ctrl']:
			self.status = 'pick'
			if not 'anchor' in key['canvas'].marks: 
				pts = mark(shp)
				if len(pts)>0: 
					pts = Points(np.vstack(pts), color=(255,0,0))
					key['canvas'].marks['anchor'] = pts
			if 'anchor' in key['canvas'].marks:
				m, obj, l = pick_point(key['canvas'].marks['anchor'], x, y, 5)
				if not m is None: self.cursor = 'hand'
		elif 'anchor' in key['canvas'].marks: 
			self.status = ''
			del key['canvas'].marks['anchor']
			shp.dirty = True
		if not self.pick_obj is None and not self.pick_m is None:
			drag(self.pick_m, self.pick_obj, x, y)
			pts = mark(self.pick_m)
			if len(pts)>0:
				pts = np.vstack(pts)
				key['canvas'].marks['anchor'] = Points(pts, color=(255,0,0))
			self.pick_m.dirty = True
			shp.dirty = True
		if self.pick_obj is None and not self.pick_m is None:
			offset(self.pick_m, x-self.p[0], y-self.p[1])
			pts = mark(self.pick_m)
			if len(pts)>0:
				pts = np.vstack(pts)
				key['canvas'].marks['anchor'] = Points(pts, color=(255,0,0))
			self.p = x, y
			self.pick_m.dirty =shp.dirty = True

	def mouse_wheel(self, shp, x, y, d, **key):
		if d>0: key['canvas'].zoomout(x, y, coord='data')
		if d<0: key['canvas'].zoomin(x, y, coord='data')
```

可以看出，对应不同的情形，有很多种处理方式：

（1）鼠标中键按下：将status设为move，同时记录当前坐标。关于x和kx的区别，可以看之前[这篇解析](https://qixinbo.info/2020/02/26/imagepy_17/#%E8%87%AA%E5%AE%9A%E4%B9%89%E9%BC%A0%E6%A0%87%E4%BA%8B%E4%BB%B6)；
（2）鼠标左键按下且状态为pick：选取锚点，这个状态为pick目前只能通过同时按住ctrl+Alt，以及移动一下鼠标才能激活（见下面的鼠标拖动事件）
（3）鼠标左键按下且pick_m属性为None：选取ROI对象
（4）鼠标右键按下且alt按下、ctrl未按：删掉ROI
（5）鼠标右键按下且shift按下、alt和ctrl未按：合并ROI（具体操作是将Shape格式转为Shapely的geometry格式，然后几何操作，再转为Shape格式）
（6）只按下鼠标右键：画布尺寸适配
（7）鼠标弹起且alt和ctrl未按：将status置为空
（8）鼠标左键弹起且同时按住alt和ctrl：显示锚点（这里在画布上显示是通过对画布的marks字典进行更改）
（9）鼠标中键按下且鼠标拖动：画布移动
（10）选择锚点后拖动：可以更改锚点位置
（11）选择对象后拖动：可以更改对象位置
（12）鼠标滚轮：画布缩放

## RectangleEditor
RectangleEditor针对矩形这一特定形状的区域对鼠标事件进行了重载，比如鼠标左键按下创建Rectangle对象，并添加进shape的body中；鼠标左键弹起是，将最终点的坐标添加进之前Rectangle的范围中。

经过上面的操作，使得Image对象的roi属性的body发生变化，而在画布显示端是通过修改canvas的marks字典来实现。具体呈现时注意，Image和Shape对象有个dirty属性，如果它为True的话，就会调用canvas的update来对画布进行刷新。这个dirty的监控是在EVT_IDLE事件中进行的，因为IDLE是系统无时无刻不停运行的，即随时监听，必要时刷新。

# 裁剪
看一下Image菜单下的Crop插件：
```python
class Crop(Simple):
    title = 'Crop'
    note = ['all', 'req_roi']

    def run(self, ips, imgs, para = None):
        sc, sr = ips.rect
        if ips.isarray: imgs = imgs[:, sc, sr].copy()
        else: imgs = [i[sc,sr].copy() for i in imgs]
        ips.set_imgs(imgs)
        if not ips.back is None:
            if ips.back.isarray: imgs = ips.back.imgs[:, sc, sr].copy()
            else: imgs = [i[sc,sr].copy() for i in ips.back.imgs]
            ips.back.set_imgs(imgs)
        offset(ips.roi, ips.roi.box[0]*-1, ips.roi.box[1]*-1)
```
可以看出，在该插件的note里明确表明了需要ROI。
然后，
## 取得ROI矩形范围
之所以说是ROI矩形范围，不仅仅是因为该例中是矩形ROI，而是如果使用的是其他形状的ROI，比如椭圆、自由区域等，都是获得该ROI的矩形范围，即最终裁剪后的整个图形仍然是矩形。
```python
        sc, sr = ips.rect
```
可以看出，矩形范围是通过Image对象的rect属性获得，那么rect又是怎样的：
```python
    @property
    def rect(self):
        if self.roi is None: return slice(None), slice(None)
        box, shape = self.roi.box, self.shape
        l, r = max(0, int(box[0])), min(shape[1], int(box[2]))
        t, b = max(0, int(box[1])), min(shape[0], int(box[3]))
        return slice(t,b), slice(l,r)
```

在rect属性中，先取得ROI的box属性和Image本身的shape，然后再对比该box（注意是ROI的box，而不是Image的box）和Image的大小，获得上下左右四个角点，返回的是垂直和水平两个方向的对应的切片对象。
那么再看看ROI的box：
```python
    @property
    def box(self):
        if self._box is None or self.dirty:
            self._box = self.count_box()
        return self._box

    def count_box(self, body=None, box=None):
        if body is None:
            box = [1e10, 1e10,-1e10,-1e10]
            self.count_box(self.body, box)
            return box

        if isinstance(body, np.ndarray):
            body = body.reshape((-1,2))
            minx, miny = body.min(axis=0)
            maxx, maxy = body.max(axis=0)
            newbox = [minx, miny, maxx, maxy]
            box.extend(merge(box, newbox))
            del box[:4]

        else:
            for i in body: self.count_box(i, box)

```
## 显示裁剪区域
```python
        if ips.isarray: imgs = imgs[:, sc, sr].copy()
        ips.set_imgs(imgs)
```
即切片后再通过set_imgs显示裁剪区域。
## 更新ROI
图像显示区域更新后，ROI也要更新到新的图像上：
```python
        offset(ips.roi, ips.roi.box[0]*-1, ips.roi.box[1]*-1)
```
