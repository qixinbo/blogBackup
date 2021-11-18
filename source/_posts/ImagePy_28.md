---
title: ImagePy解析： 28 -- 三维可视化
tags: [ImagePy]
categories: computer vision 
date: 2021-11-18
---

本文解析一下ImagePy的三维画布。
以如下例子入手：
![demo](https://user-images.githubusercontent.com/6218739/141955577-f4a7c5c5-0a9a-409b-9fde-399a92f0c1d8.png)
首先，原始图像是一个5乘5的方形图像，其中间是4乘4的白色，周围是一圈黑色。
由这张原始图根据距离变换得到右上角的高程图，继而对该高程图做三维可视化。

# 渲染插件
二维平面的三维可视化插件是这样写的：
```python
class Surface2D(Simple):
    title = '2D Surface'
    note = ['8-bit', '16-bit', 'float']
    para = {'name':'undifine', 'sample':2, 'sigma':2,'h':0.3, 'cm':'gray'}
    view = [(str, 'name', 'Name', ''),
            (int, 'sample', (1,10), 0, 'down sample', 'pix'),
            (int, 'sigma', (0,30), 0, 'sigma', ''),
            (float, 'h', (0.1,10), 1, 'scale z', ''),
            ('cmap', 'cm', 'color map')]

    def run(self, ips, imgs, para = None):
        ds, sigma, cm = para['sample'], para['sigma'], ColorManager.get(para['cm'])
        mesh = Surface2d(ips.img, sample=ds, sigma=sigma, k=para['h'], cmap=cm)
        self.app.show_mesh(mesh, para['name'])
```
其界面为：
![interface](https://user-images.githubusercontent.com/6218739/141955719-7c81041c-9c56-4f10-ba0c-1d5d68643f20.png)
即设定名字、下采样率、平滑率和z轴伸缩率，以及渲染所用的colormap。(在该demo中，就按如图中的参数进行设置)
然后将这些参数传给Surface2d这个ImagePy定义的Mesh对象。
最后调用show_mesh方法将其呈现出来。
下面是一步步分析这个Mesh对象及其绘制方法。

# Mesh对象
如前所述，二维高程图传给了Surface2d这一类，具体看一下其代码实现：
```python
class Surface2d(Mesh):
	def __init__(self, img=None, sample=1, sigma=0, k=0.3, **key):
		self.img, self.sample, self.sigma, self.k = img, sample, sigma, k
		Mesh.__init__(self, **key)
		self.set_data(img, sample, sigma, k)

	def set_data(self, img=None, sample=None, sigma=None, k=None, **key):
		if not img is None: self.img = img
		if not sample is None: self.sample = sample
		if not sigma is None: self.sigma = sigma
		if not k is None: self.k = k
		if sum([not i is None for i in (img, sample, sigma, k)])>0:
			from ..util import meshutil
			vert, fs = meshutil.create_surface2d(self.img, self.sample, self.sigma, self.k)
			Mesh.set_data(self, verts=vert, faces=fs.astype(np.uint32), colors=vert[:,2], **key)
		else: Mesh.set_data(self, **key)
```
可以看到，在它的初始化函数中调用了set_data方法。进一步地，在该方法中有两个核心方法：将图像转化为顶点和面，然后再转为Mesh对象。
## 位图提取格点坐标和像素值
即如下方法：
```python
vert, fs = meshutil.create_surface2d(self.img, self.sample, self.sigma, self.k)
```
源码及注释为：
```python
def create_surface2d(img, sample=1, sigma=0, k=0.3):
    from scipy.ndimage import gaussian_filter
    #start = time()
    # 以采样率为步长进行图像的重新提取
    img = img[::sample, ::sample].astype(np.float32)
    # 如果指定了平滑率，则使用高斯滤波进行平滑
    if sigma>0: img = gaussian_filter(img, sigma)
    # 根据采样后的图像形状生成网格格点
    xs, ys = np.mgrid[:img.shape[0],:img.shape[1]]
    # 根据采样率，将格点范围伸缩到之前的大小
    xs *= sample; ys *= sample
    # 将图像像素值乘以伸缩大小，作为z轴的值，与格点坐标xy传入下面的方法
    return create_grid_mesh(xs, ys, img*k)
```
在此例中，依照上面的参数，来看一下各个中间结果：
首先高程图在降采样后，图像矩阵为：
```python
[[0. 0. 0.]
 [0. 2. 0.]
 [0. 0. 0.]]
```
然后在$\sigma=1$的高斯滤波后，图像矩阵为：
```python
[[0.17534617 0.2415005  0.17534617]
 [0.2415005  0.3326134  0.2415005 ]
 [0.17534617 0.2415005  0.17534617]]
```
其再经过k倍的伸缩，变为：
```python
[[0.8767308 1.2075025 0.8767308]
 [1.2075025 1.6630671 1.2075025]
 [0.8767308 1.2075025 0.8767308]]
```
同时xs和ys即网格格点坐标，也经过了降采样，以及范围伸缩，变为：
```python
[[0 0 0]
 [2 2 2]
 [4 4 4]]
```

## 获取格点和面的信息
```python
def create_grid_mesh(xs, ys, zs):
    h, w = xs.shape
    # 将xy坐标位置和z值合并起来
    vts = np.array([xs, ys, zs], dtype=np.float32)
    # 这一步是定义以某格点为参考点的坐标系下它与哪些点形成面
    # 在局部坐标系下，参考点索引为0，那么它所构成的面有两个
    # 分别与(1, 1+w)这两个点构成一个面，与(1+w, w)这两个点构成一个面
    # 比如在此例下，did的值就是[[0 1 4 0 4 3]]
    did = np.array([[0, 1, 1+w, 0, 1+w, w]], dtype=np.uint32)
    # rcs由两部分构成
    # 第一部分是获取全局坐标系下每一排的第一个元素的索引，所以是以w为步长
    # 注意排除最后一排，即是w*h还要减去w，因为最后一排元素所参与的面可以通过倒数第二排来获得
    # 对于此例，就是[[0], [3]]，注意这里使用None来增加一个维度
    # 第二部分是获取全局坐标系下每一列的索引，所以是以1为步长
    # 注意是排除最后一列，即w要减去1，这也是因为最后一列参与的面可以通过前一列得到
    # 对于此例，就是[0, 1]
    # 最终rcs就是numpy数组的[[0],[3]]+[0, 1]
    # 这里用到了numpy的广播： https://qixinbo.info/2019/10/20/python-indexing/
    # [[0, 0],[3, 3]] + [[0, 1], [0, 1]] = [[0, 1], [3, 4]]
    # 代表的意思就是在全局坐标系中，第一排取索引为0和1的格点，在第二排取索引为3和4的格点
    rcs = np.arange(0,w*h-w,w)[:,None] + np.arange(0,w-1,1)
    # 接下来就是根据上面取得的格点，得到每个格点上所形成的面
    # 首先第一部分是将rcs拉直为[[0], [1], [3], [4]]，即这四个格点索引拉平到一个维度上
    # 然后加上上面的局部坐标系下形成面的格点索引[[0 1 4 0 4 3]]
    # 同样根据广播原则，就得到了每个格点与相邻点所形成的面
    # 结果为：[[0 1 4 0 4 3], [1 2 5 1 5 4], [3 4 7 3 7 6], [4 5 8 4 8 7]] 
    # 即每个格点上都参与形成两个面，具体每一个面的格点组成看上面的序列
    faces = rcs.reshape(-1,1) + did
    # 返回值是两个
    # 第一个就是格点坐标及其上面的值，并按这三个值合并起来算一个重新改变形状
    # 第二个就是由格点所形成的面的信息
    return vts.reshape(3,-1).T.copy(), faces.reshape(-1,3)
```
具体的解析过程见上面源码。
最后说一下最终返回的格点信息和面信息，分别是：
```python
[[0.        0.        0.8767308]
 [0.        2.        1.2075025]
 [0.        4.        0.8767308]
 [2.        0.        1.2075025]
 [2.        2.        1.6630671]
 [2.        4.        1.2075025]
 [4.        0.        0.8767308]
 [4.        2.        1.2075025]
 [4.        4.        0.8767308]]
```
以第一个格点为例，它是在(0, 0)坐标，同时上面的值是0.8767308。
以及面信息：
```python
[[0 1 4]
 [0 4 3]
 [1 2 5]
 [1 5 4]
 [3 4 7]
 [3 7 6]
 [4 5 8]
 [4 8 7]]
```
以第一个面为例，它由(0, 1, 4)号格点组成。

## 构建Mesh对象
```python
# 传入上面的格点、面、颜色（这里取的是格点上的z值）以及cmap
Mesh.set_data(self, verts=vert, faces=fs.astype(np.uint32), colors=vert[:,2], **key)
```
上述代码是调用了Mesh对象的set_data方法。
```python
class Mesh:
	# 在初始化函数中传入一个Mesh对象所需要的信息
	def __init__(self, verts=None, faces=None, colors=None, cmap=None, **key):
		# 如果有格点信息，但没有面信息
		if faces is None and not verts is None: 
			# 则直接按格点个数-1生成面，即两个相邻格点相连，就成为面
			faces = np.arange(len(verts), dtype=np.uint32)
		# 传入格点信息
		self.verts = verts.astype(np.float32, copy=False) if not verts is None else None
		# 传入面信息
		self.faces = faces.astype(np.uint32, copy=False) if not faces is None else None
		# 传入颜色信息
		self.colors = colors
		# 设置模式、可见性和dirty属性等
		self.mode, self.visible, self.dirty = 'mesh', True, 'geom'
		# 设置alpha透明度和边信息
		self.alpha = 1; self.edges = None
		# 设置高光、colormap
		self.high_light = False; self.cmap = 'gray' if cmap is None else cmap
		# 调用set_data方法
		self.set_data(**key)

	def set_data(self, verts=None, faces=None, colors=None, **key):
		# 同上面的初始化功能近似，区别是可以直接调用它来配置信息
		if faces is None and not verts is None: 
			faces = np.arange(len(verts), dtype=np.uint32)
		if not verts is None: self.verts = verts.astype(np.float32, copy=False)
		if not faces is None: self.faces = faces.astype(np.uint32, copy=False)
		if not colors is None: self.colors = colors
		if not faces is None: self.edge = None
		if sum([i is None for i in [verts, faces, colors]])<3: self.dirty = 'geom'
		if not self.faces is None and self.faces.ndim==1: key['mode'] = 'points'
		elif not self.faces is None and self.faces.shape[1]==2: 
			if key.get('mode', self.mode)=='mesh': key['mode'] = 'grid'
		if key.get('mode', self.mode) != self.mode: self.dirty = 'geom'
		self.mode = key.get('mode', self.mode)
		self.visible = key.get('visible', self.visible)
		self.alpha = key.get('alpha', self.alpha)
		self.high_light = key.get('high_light', False)
		self.cmap = key.get('cmap', self.cmap)
		self.dirty = self.dirty or True
```

# 可视化Mesh
即将Mesh对象通过三维画布展示出来：
```python
self.app.show_mesh(mesh, para['name'])
```
这里就是调用了app的show_mesh方法。
ImagePy的三维画布是基于VisPy的，同时又进行了封装，最底层的是如下这个类：
```python
class Canvas3D(scene.SceneCanvas):
    def __new__(cls, parent, scene3d=None):
        self = super().__new__(cls)
        scene.SceneCanvas.__init__(self, app="wx", parent=parent, keys='interactive', show=True, dpi=150)
        canvas = parent.GetChildren()[-1]
        self.unfreeze()
        self.canvas = weakref.ref(canvas)
        self.view = self.central_widget.add_view()
        self.set_scene(scene3d or Scene())
        self.visuals = {}
        self.curobj = None
        self.freeze()
        canvas.Bind(wx.EVT_IDLE, self.on_idle)
        canvas.tool = None
        canvas.camera = scene.cameras.TurntableCamera(parent=self.view.scene, fov=45, name='Turntable')
        canvas.set_camera = self.set_camera
        canvas.fit = lambda : self.set_camera(auto=True)
        canvas.at = self.at
        self.view.camera = canvas.camera
        return canvas
```
VisPy的教程略微有点少，留坑待填。

