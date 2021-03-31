---
title: ImagePy解析： 26 -- 矢量图形的操作
tags: [ImagePy]
categories: computer vision 
date: 2021-3-31
---

# 简介
本文是对ImagePy的矢量图形绘制工具进行深度解析。
矢量图形相对于位图来说，有其特有的操作，比如两个矢量进行求交集、求并集、求差等。
阅读本文之前，可以先参考[之前的这篇文章](https://qixinbo.info/2020/06/14/imagepy_20/)，以对ImagePy的矢量图形有初步了解。

# 功能函数
## 将矢量图形转化为点集
该函数的作用是将矢量图形转换为点集，这里的点作为锚点，可以供后续编辑。
比如对于矩形这一矢量，在shp中定义了它的起始点和长宽，通过该函数，可以将该矩形转为9个点的点集，即将该矩形分成田字格。
（具体到语法上，使用了numpy的mgrid函数，其中的步长设为了复数的形式，具体可以参考[这里](https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html)）
```python
def mark(shp, types = 'all'):
	pts = []
	if not (types=='all' or shp.dtype in types): return pts
	if shp.dtype == 'point':
		pts.append([shp.body])
	if shp.dtype == 'points':
		pts.append(shp.body)
	if shp.dtype == 'line':
		pts.append(shp.body)
	if shp.dtype == 'lines':
		pts.extend(shp.body)
	if shp.dtype == 'polygon' and len(shp.body)==1:
		pts.append(shp.body[0])
	if shp.dtype == 'polygons':
		for i in shp.body:
			if len(i) != 1: continue
			pts.append(i[0])
	if shp.dtype == 'rectangle':
		l,t,w,h = shp.body
		ps = np.mgrid[l:l+w:3j, t:t+h:3j].T.reshape((-1,2))
		pts.append(ps)
	if shp.dtype == 'rectangles':
		for i in range(len(shp.body)):
			l,t,w,h = shp.body[i]
			ps = np.mgrid[l:l+w:3j, t:t+h:3j].T.reshape((-1,2))
			pts.append(ps)
	if shp.dtype == 'ellipse':
			x0, y0, l1, l2, ang = shp.body
			mat = np.array([[np.cos(-ang),-np.sin(-ang)],
						  [np.sin(-ang),np.cos(-ang)]])
			ps = np.mgrid[-l1:l1:3j, -l2:l2:3j].T.reshape((-1,2))
			pts.append(mat.dot(ps.T).T + (x0, y0))
	if shp.dtype == 'ellipses':
		for i in range(len(shp.body)):
			x0, y0, l1, l2, ang = shp.body[i]
			mat = np.array([[np.cos(-ang),-np.sin(-ang)],
						  [np.sin(-ang),np.cos(-ang)]])
			ps = np.mgrid[-l1:l1:3j, -l2:l2:3j].T.reshape((-1,2))
			pts.append(mat.dot(ps.T).T + (x0, y0))
	if shp.dtype == 'layer':
		minl, obj = 1e8, None
		for i in shp.body:
			pts.extend(mark(i, types))
	return pts
```

## 选择对象
该函数的功能是选择与鼠标点击位置距离在一定范围内且距离最近的那个矢量对象。
```python
def pick_obj(shp, x, y, lim, types='all'):
	obj, minl = None, lim
	if not (types=='all' or shp.dtype in types): 
		return m, obj, minl
	# 如果是layer类型，那么就遍历里面的元素
	if shp.dtype == 'layer':
		for i in shp.body:
			o, l = pick_obj(i, x, y, lim, types)
			if l < minl: 
				obj, minl = o, l
	elif shp.dtype in 'polygons':
		b = shp.to_geom().contains(Point([x, y]).to_geom())
		if b : return shp, 0
	else:
		# 首先将鼠标位置传给ImagePy的Point这一结构
		# 然后调用to_geom方法就转换为shapely的Point对象
		# 然后通过distance函数计算shp中的矢量与鼠标所在位置的Point矢量的距离
		d = shp.to_geom().distance(Point([x, y]).to_geom())
		# 找到最近的或小于阈值minl的矢量，然后返回它
		if d<minl: obj, minl = shp, d
	return obj, minl
```

## 选择锚点
该函数的功能是选择与鼠标所在位置小于某个距离的那个锚点。
如果锚点被选中，就会返回该锚点所在的矢量对象，同时表示该锚点的一个标识。比如对于椭圆上一个锚点，有“lt”左上、“rt”右上、“o”中心点等多种锚点。
```python
def pick_point(shp, x, y, lim, types='all'):
	m, obj, minl = None, None, lim
	if not (types=='all' or shp.dtype in types): 
		return m, obj, minl
	if shp.dtype == 'point':
		l = ((shp.body-(x, y))**2).sum()
		if l < minl: 
			m, obj, minl = shp, shp.body, l
	if shp.dtype == 'points':
		l = norm(shp.body-(x,y), axis=1)
		n = np.argmin(l)
		l = l[n]
		if l < minl: 
			m, obj, minl = shp, shp.body[n], l
	if shp.dtype == 'line':
		l = norm(shp.body-(x,y), axis=1)
		n = np.argmin(l)
		l = l[n]
		if l < minl: 
			m, obj, minl = shp, shp.body[n], l
	if shp.dtype == 'lines':
		for line in shp.body:
			l = norm(line-(x,y), axis=1)
			n = np.argmin(l)
			l = l[n]
			if l < minl: 
				m, obj, minl = shp, line[n], l
	if shp.dtype == 'polygon' and len(shp.body)==1:
		l = norm(shp.body[0]-(x,y), axis=1)
		n = np.argmin(l)
		l = l[n]
		if l < minl: 
			m, obj, minl = shp, shp.body[0][n], l
	if shp.dtype == 'polygons':
		for i in shp.body:
			if len(i) != 1: continue
			l = norm(i[0]-(x,y), axis=1)
			n = np.argmin(l)
			l = l[n]
			if l < minl: 
				m, obj, minl = shp, i[0][n], l
	if shp.dtype == 'rectangle':
		l,t,w,h = shp.body
		pts = np.mgrid[l:l+w:3j, t:t+h:3j].T.reshape((-1,2))
		names = ['lt','t','rt','l','o','r','lb','b','rb']
		l = norm(pts-(x,y), axis=1)
		n = np.argmin(l)
		if l[n] < minl:
			m, obj, minl = shp, names[n], l[n]
	if shp.dtype == 'rectangles':
		for i in range(len(shp.body)):
			l,t,w,h = shp.body[i]
			pts = np.mgrid[l:l+w:3j, t:t+h:3j].T.reshape((-1,2))
			names = ['lt','t','rt','l','o','r','lb','b','rb']
			l = norm(pts-(x,y), axis=1)
			n = np.argmin(l)
			if l[n] < minl:
				m, obj, minl = shp, (names[n], i), l[n]
	if shp.dtype == 'ellipse':
			x0, y0, l1, l2, ang = shp.body
			mat = np.array([[np.cos(-ang),-np.sin(-ang)],
						  [np.sin(-ang),np.cos(-ang)]])
			pts = np.mgrid[-l1:l1:3j, -l2:l2:3j].T.reshape((-1,2))
			pts = mat.dot(pts.T).T + (x0, y0)
			names = ['lt','t','rt','l','o','r','lb','b','rb']
			l = norm(pts-(x,y), axis=1)
			n = np.argmin(l)
			if l[n] < minl:
				m, obj, minl = shp, names[n], l[n]
	if shp.dtype == 'ellipses':
		for i in range(len(shp.body)):
			x0, y0, l1, l2, ang = shp.body[i]
			mat = np.array([[np.cos(-ang),-np.sin(-ang)],
						  [np.sin(-ang),np.cos(-ang)]])
			pts = np.mgrid[-l1:l1:3j, -l2:l2:3j].T.reshape((-1,2))
			pts = mat.dot(pts.T).T + (x0, y0)
			names = ['lt','t','rt','l','o','r','lb','b','rb']
			l = norm(pts-(x,y), axis=1)
			n = np.argmin(l)
			if l[n] < minl:
				m, obj, minl = shp, (names[n], i), l[n]
	if shp.dtype == 'layer':
		# minl, obj = 1e8, None
		for i in shp.body:
			h, o, l = pick_point(i, x, y, lim, types)
			if l < minl: 
				m, obj, minl = h, o, l
	return m, obj, minl
```

## 拖动锚点
这个函数接收当前的矢量对象、它的某个锚点以及当前鼠标位置，然后通过该锚点的类型，来对该矢量对象的范围进行调整。
```python
def drag(shp, pt, x, y, types='all'):
	if not (types=='all' or shp.dtype in types): return
	if shp.dtype == 'rectangle':
		body = shp.body
		if pt == 'o':body[:2] = (x, y) - body[2:]/2
		if 'l' in pt:body[[0,2]] = x, body[0]+body[2]-x
		if 'r' in pt:body[2] = x - body[0]
		if 't' in pt:body[[1,3]] = y, body[1]+body[3]-y
		if 'b' in pt:body[3] = y - body[1]
	elif shp.dtype == 'rectangles':
		pt, i = pt
		body = shp.body[i]
		if pt == 'o':body[:2] = (x, y) - body[2:]/2
		if 'l' in pt:body[[0,2]] = x, body[0]+body[2]-x
		if 'r' in pt:body[2] = x - body[0]
		if 't' in pt:body[[1,3]] = y, body[1]+body[3]-y
		if 'b' in pt:body[3] = y - body[1]
	elif shp.dtype == 'ellipse':
		if pt == 'o': 
			shp.body[:2] = x, y
			return
		x0, y0, l1, l2, ang = shp.body
		v1, v2 = (np.array([[np.cos(-ang),-np.sin(-ang)],
			[np.sin(-ang),np.cos(-ang)]]) * (l1, l2)).T
		l, r, t, b = np.array([-v1, v1, -v2, v2]) + (x0, y0)
		if 'l' in pt: l = v1.dot([x-x0, y-y0])*v1/l1**2+(x0, y0)
		if 'r' in pt: r = v1.dot([x-x0, y-y0])*v1/l1**2+(x0, y0)
		if 't' in pt: t = v2.dot([x-x0, y-y0])*v2/l2**2+(x0, y0)
		if 'b' in pt: b = v2.dot([x-x0, y-y0])*v2/l2**2+(x0, y0)
		k = np.linalg.inv(np.array([-v2,v1]).T).dot((l+r-t-b)/2)
		shp.body[:2] = (l+r)/2 + v2*k[0]
		shp.body[2:4] = np.dot(r-l, v1)/l1/2, np.dot(b-t, v2)/l2/2
	elif shp.dtype == 'ellipses':
		pt, i = pt
		body = shp.body[i]
		if pt == 'o': 
			body[:2] = x, y
			return
		x0, y0, l1, l2, ang = body
		v1, v2 = (np.array([[np.cos(-ang),-np.sin(-ang)],
			[np.sin(-ang),np.cos(-ang)]]) * (l1, l2)).T
		l, r, t, b = np.array([-v1, v1, -v2, v2]) + (x0, y0)
		if 'l' in pt: l = v1.dot([x-x0, y-y0])*v1/l1**2+(x0, y0)
		if 'r' in pt: r = v1.dot([x-x0, y-y0])*v1/l1**2+(x0, y0)
		if 't' in pt: t = v2.dot([x-x0, y-y0])*v2/l2**2+(x0, y0)
		if 'b' in pt: b = v2.dot([x-x0, y-y0])*v2/l2**2+(x0, y0)
		k = np.linalg.inv(np.array([-v2,v1]).T).dot((l+r-t-b)/2)
		body[:2] = (l+r)/2 + v2*k[0]
		body[2:4] = np.dot(r-l, v1)/l1/2, np.dot(b-t, v2)/l2/2
	else: pt[:] = x, y
```

## 移动对象
该函数目的是对矢量对象进行移动。
```python
def offset(shp, dx, dy):
	if shp.dtype in {'rectangle', 'ellipse', 'circle'}:
		shp.body[:2] += dx, dy
	elif shp.dtype in {'rectangles', 'ellipses', 'circles'}:
		shp.body[:,:2] += dx, dy
	elif isinstance(shp, np.ndarray):
		shp += dx, dy
	elif isinstance(shp.body, list):
		for i in shp.body: offset(i, dx, dy)
```

# BaseEditor鼠标动作
## 鼠标中键拖动
```python
def mouse_down(self, shp, x, y, btn, **key):
	self.p = x, y
	if btn==2:
		self.status = 'move'
		self.oldxy = key['px'], key['py']
def mouse_move(self, shp, x, y, btn, **key):
	self.cursor = 'arrow'
	if self.status == 'move':
		ox, oy = self.oldxy
		up = (1,-1)[key['canvas'].up]
		key['canvas'].move(key['px']-ox, (key['py']-oy)*up)
		self.oldxy = key['px'], key['py']
```
## alt+右键以删除一个shape
```python
def mouse_down(self, shp, x, y, btn, **key):
	if btn==3:
		obj, l = pick_obj(shp, x, y, 5)
		if key['alt'] and not key['ctrl']:
			if obj is None: del shp.body[:]
			else: shp.body.remove(obj)
			shp.dirty = True
```

## shift+右键以合并shape
```python
def mouse_down(self, shp, x, y, btn, **key):
	if btn==3:
		if key['shift'] and not key['alt'] and not key['ctrl']:
			layer = geom2shp(geom_union(shp.to_geom()))
			shp.body = layer.body
			shp.dirty = True
```
## 右键根据当前区域大小缩放
```python
def mouse_down(self, shp, x, y, btn, **key):
	if btn==3:
		if not (key['shift'] or key['alt'] or key['ctrl']):
			key['canvas'].fit()
```

## alt+ctrl以显示锚点
（注意该组合键是放在鼠标移动这个事件中，所以此时要鼠标移动一下，才会看到锚点）
```python
def mouse_move(self, shp, x, y, btn, **key):
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
```
最开始时，画布中是没有锚点的，此时就会将矢量对象通过mark函数转为锚点的点集，然后在画布上显示出来（具体原理可以见上面的mark函数解析）。
当画布中有了锚点后，如果鼠标靠近了某个锚点，通过pick_point这个函数捕捉到该锚点，就会将鼠标的样式设置为“手形”。

## alt+ctrl+鼠标左键拖动锚点
需要提前非常注意的一点是，当同时按住alt和ctrl后，就会在鼠标移动事件中将此时的status设为pick模式：
```python
self.status = 'pick'
```
此时在鼠标按下事件中：
```python
def mouse_down(self, shp, x, y, btn, **key):
	self.p = x, y
	if btn==1 and self.status=='pick':
		m, obj, l = pick_point(shp, x, y, 5)
		self.pick_m, self.pick_obj = m, obj
```
如果是捕捉到了某锚点，那么self.pick_m和self.pick_obj都会有值。
此时如果移动鼠标，那么：
```python
def mouse_move(self, shp, x, y, btn, **key):
	if not self.pick_obj is None and not self.pick_m is None:
		drag(self.pick_m, self.pick_obj, x, y)
		pts = mark(self.pick_m)
		if len(pts)>0:
			pts = np.vstack(pts)
			key['canvas'].marks['anchor'] = Points(pts, color=(255,0,0))
		self.pick_m.dirty = True
		shp.dirty = True
```
就会触发drag这个函数来对锚点进行拖动。

## alt+ctrl+鼠标左键拖动整个矢量对象
上面拖动锚点，是因为在鼠标按下时能够捕捉到锚点，而如果捕捉不到锚点（即与锚点离得较远），此时就会尝试选择整个对象，即：
```python
def mouse_down(self, shp, x, y, btn, **key):
	if btn==1 and self.pick_m is None:
		m, l = pick_obj(shp, x, y, 5)
		self.pick_m, self.pick_obj = m, None
```
（注意到此时self.pick_m是None，即没有捕捉到锚点的前提下）
此时如果探测到了矢量对象，那么self.pick_m就会有值，但self.pick_obj没有值。
此时如果移动鼠标，那么：
```python
def mouse_move(self, shp, x, y, btn, **key):
	if self.pick_obj is None and not self.pick_m is None:
		offset(self.pick_m, x-self.p[0], y-self.p[1])
		pts = mark(self.pick_m)
		if len(pts)>0:
			pts = np.vstack(pts)
			key['canvas'].marks['anchor'] = Points(pts, color=(255,0,0))
		self.p = x, y
		self.pick_m.dirty =shp.dirty = True

```

# 特定形状Editor的鼠标动作
## 调用BaseEditor
BaseEditor中有预置的鼠标动作，何时调用它。
```python
def inbase(key, btn):
	status = key['ctrl'], key['alt'], key['shift']
	return status == (1,1,0) or btn in {2,3}
```
即同时按住Ctrl和alt，或点击了鼠标中键或右键，就先响应BaseEditor中的行为。

## 自定义动作
有几个特定的矢量图形绘制时都有如下动作，即：
（1）按住alt，求差集；
（2）按住shift，求并集；
（3）同时按住shift和alt，求交集。
```python
if key['alt'] or key['shift']:
	obj = shp.body.pop(-1)
	rst = geom_union(shp.to_geom())
	if key['alt'] and not key['shift']:
		rst = rst.difference(obj.to_geom())
	if key['shift'] and not key['alt']:
		rst = rst.union(obj.to_geom())
	if key['shift'] and key['alt']:
		rst = rst.intersection(obj.to_geom())
	layer = geom2shp(geom_flatten(rst))
	shp.body = layer.body
```
