---
title: ImagePy解析：16 -- 细胞分析Cell Analysis
tags: [ImagePy]
categories: computational material science 
date: 2020-2-10
---

本文是对ImagePy的IBook中的[Cell Analysis](https://github.com/Image-Py/IBook/blob/master/menus/IBook/Chapter7%20Binary-Image/Show%20Cell%20Analysis.mc)的解析。
该例子用到了很多功能，从中可以一窥图像处理的常用流程和ImagePy常用组件的用法。
该文件是个录制的宏文件，因此，命令都是“插件名称>参数字典”的形式。

# 打开图像
```python
cell>None
```
该宏命令就是打开了cell图像，如图：
![cell](https://user-images.githubusercontent.com/6218739/71238417-f7c44100-233e-11ea-8ec2-5bd1066ff0b6.JPG)
之所以能这样打开图像，是因为在生成ImagePy主界面时，已经将该图像解析成了插件，具体过程可以参见[之前解析主界面渲染的文章](http://qixinbo.info/2019/08/31/ImagePy_4/)。
主要原理通过查看IBook下的Image Referenced文件夹下的image_plgs文件就一目了然：
```python
class Data(Free):
    def __init__(self, name):
        self.name = name
        self.title = name.split('.')[0]
    def run(self, para = None):
        root_dir = osp.abspath(osp.dirname(__file__))
        img = imread(osp.join(root_dir, self.name))
        if img.ndim==3 and img.shape[2]==4:
            img = img[:,:,:3].copy()
        IPy.show_img([img], self.title)
    def __call__(self):
        return self

datas = ['Angkor.jpg','qrcode.png','street.jpg','road.jpg','house.jpg',
'neubauer.jpg','windmill.jpg','sunglow.jpg','ailurus.jpg','Yuan.jpg',
'saltpepper.jpg','dem.jpg', 'honeycomb.jpg', 'necklace.jpg','universe.jpg',
'towel-far.jpg','towel-near.jpg','trafficsign.jpg','bee.png','insect.png',
'game.jpg','gear.png','block.png','distance.png','points.png', 'qin.png', 
'img.png','pointline.png','marble.png','cell.jpg','rust.jpg', 'rose.jpg',
'roses.jpg']
plgs = [Data(i) for i in datas]
```

# 复制图像
```python
Duplicate>{'name': 'cell-gray', 'stack': True}
```
将上述图像复制一份，便于后续处理。
调用的Duplicate命令的原理见menus的Image的duplicate_plg.py文件：
```python
class Duplicate(Simple):
    title = 'Duplicate'
    note = ['all']
    para = {'name':'Undefined','stack':True}
    def run(self, ips, imgs, para = None):
        name = para['name']
        if ips.get_nslices()==1 or self.para['stack']==False:
            if ips.roi == None:
                img = ips.img.copy()
                ipsd = ImagePlus([img], name)
                ipsd.back = ips.back
            ipsd = ImagePlus(imgs, name)
        ipsd.chan_mode = ips.chan_mode
        IPy.show_ips(ipsd)
```
注意将img复制完后，还要组装成ImagePy特有的ImagePlus类型的结构。

# 图像灰度化
```python
8-bit>None
```
这一步就是将RGB图像转为8位灰度图像
```python
class To8bit(Simple):
    title = '8-bit'
    note = ['all']
 
    def run(self, ips, imgs, para = None):
        if ips.imgtype == '8-bit': return
        n = ips.get_nslices()
        if ips.is3d:
        else:
            img8 = []
            minv, maxv = ips.get_updown()
            for i in range(n):
                self.progress(i, len(imgs))
                if ips.imgtype == 'rgb':
                    img8.append(imgs[i].mean(axis=2).astype(np.uint8))
        ips.set_imgs(img8)
```
可以看出，这里灰度化所使用的方法是取三通道的平均值。
这一步处理得到的结果如图：
![grey](https://user-images.githubusercontent.com/6218739/72133839-5f5d3180-33bd-11ea-9867-8b5b79f620b0.png)

# 复制图像
```python
Duplicate>{'name': 'cell-msk', 'stack': True}
```
将灰度化的图像又复制了一份。。

# 阈值分割
```python
Threshold>{'thr1': 193, 'thr2': 255}
```
将图像根据上下阈值进行二值化分割。
```python
class Plugin(Filter):
    modal = False
    title = 'Threshold'
    note = ['all', 'auto_msk', 'auto_snap', 'not_channel', 'preview']
    arange = (0,255)

    def run(self, ips, snap, img, para = None):
        if para == None: para = self.para
        ips.lut = self.lut
        img[:] = 0
        img[snap>=para['thr2']] = 255
        img[snap<para['thr1']] = 255
        ips.range = (0, 255)
```
可以看出，首先将img都置为0，然后将原图中大于thr2和小于thr1的像素在img中都置为255。具体地，这里thr2是255， thr1是193，所以这里就是将小于193的像素值都置为白色，大于193的像素都置为黑色。
这一步处理后结果如图：
![threshold](https://user-images.githubusercontent.com/6218739/72133901-8582d180-33bd-11ea-9e38-d8c47f3f633e.png)

# 填充孔洞
```python
Fill Holes>None
```
从上图可以看出，二值分割后有很多细胞中间都有孔洞，因此这一步目的是将这些孔洞填充。
```python
class FillHoles(Filter):
    """FillHoles: derived from imagepy.core.engine.Filter """
    title = 'Fill Holes'
    note = ['8-bit', 'auto_msk', 'auto_snap','preview']

    def run(self, ips, snap, img, para = None):
        ndimg.binary_fill_holes(snap, output=img)
        img *= 25
```
具体操作是调用了scipy的ndimage模块的binary_fill_holes函数，结果如下：
![fill-holes](https://user-images.githubusercontent.com/6218739/72134539-55d4c900-33bf-11ea-83d5-3009575d3aa2.png)

# 几何过滤
```python
Geometry Filter>{'con': '4-connect', 'inv': False, 'area': 1100.0, 'l': 0.0, 'holes': 0, 'solid': 0.0, 'e': 0.0, 'front': 255, 'back': 0}
```
由上图可以看出，填充孔洞后的图像依然有很多杂点，这一步是通过面积来过滤掉这些杂点。
该插件名为“几何过滤”，因此不只是可以通过面积来过滤，还可以通过周长、偏心率等来过滤：
```python
class RegionFilter(Filter):
    title = 'Geometry Filter'
    note = ['8-bit', '16-bit', 'int', 'auto_msk', 'auto_snap','preview']
    para = {'con':'4-connect', 'inv':False, 'area':0, 'l':0, 'holes':0, 'solid':0, 'e':0, 'front':255, 'back':100}
    view = [(list, 'con', ['4-connect', '8-connect'], str, 'conection', 'pix'),
            (bool, 'inv', 'invert'),
            ('lab', None, 'Filter: "+" means >=, "-" means <'),
            (int, 'front', (0, 255), 0, 'front color', ''),
            (int, 'back', (0, 255), 0, 'back color', ''),
            (float, 'area', (-1e6, 1e6), 1, 'area', 'unit^2'),
            (float, 'l', (-1e6, 1e6), 1, 'perimeter', 'unit'),
            (int, 'holes', (-10,10), 0, 'holes', 'num'),
            (float, 'solid', (-1, 1,), 1, 'solidity', 'ratio'),
            (float, 'e', (-100,100), 1, 'eccentricity', 'ratio')]

    #process
    def run(self, ips, snap, img, para = None):
        k, unit = ips.unit
        strc = generate_binary_structure(2, 1 if para['con']=='4-connect' else 2)
        lab, n = label(snap==0 if para['inv'] else snap, strc, output=np.uint32)
        idx = (np.ones(n+1)*(0 if para['inv'] else para['front'])).astype(np.uint8)
        ls = regionprops(lab)

        for i in ls:
            if para['area'] == 0: break
            if para['area']>0:
                if i.area*k**2 < para['area']: idx[i.label] = para['back']
            if para['area']<0:
                if i.area*k**2 >= -para['area']: idx[i.label] = para['back']

        for i in ls:
            if para['l'] == 0: break
            if para['l']>0:
                if i.perimeter*k < para['l']: idx[i.label] = para['back']
            if para['l']<0:
                if i.perimeter*k >= -para['l']: idx[i.label] = para['back']

        for i in ls:
            if para['holes'] == 0: break
            if para['holes']>0:
                if 1-i.euler_number < para['holes']: idx[i.label] = para['back']
            if para['holes']<0:
                if 1-i.euler_number >= -para['holes']: idx[i.label] = para['back']

        for i in ls:
            if para['solid'] == 0: break
            if para['solid']>0:
                if i.solidity < para['solid']: idx[i.label] = para['back']
            if para['solid']<0:
                if i.solidity >= -para['solid']: idx[i.label] = para['back']

        for i in ls:
            if para['e'] == 0: break
            if para['e']>0:
                if i.minor_axis_length>0 and i.major_axis_length/i.minor_axis_length < para['e']:
                    idx[i.label] = para['back']
            if para['e']<0:
                if i.minor_axis_length>0 and i.major_axis_length/i.minor_axis_length >= -para['e']:
                    idx[i.label] = para['back']

        idx[0] = para['front'] if para['inv'] else 0
       img[:] = idx[lab]
```
具体解析一下：
## 生成结构单元
首先生成一个结构单元，该单元的形式由con参数决定，如果选择了4-connect，则代表处理四邻域，否则就是处理八邻域：
```python
strc = generate_binary_structure(2, 1 if para['con']=='4-connect' else 2)
```
该结构是为了下面的连通域的标记，即在标记区域是否连通时，考察中心像素与周围像素的连通性，该参数非常重要，因为有时考察四邻域不连通，但考察八邻域就连通了。

## 连通域标记
然后就是对连通域进行标记：
```python
lab, n = label(snap==0 if para['inv'] else snap, strc, output=np.uint32)
```
这个地方还提供了一个选项inv来选择处理哪一部分。如果inv是false，那么就是处理原二值图中的1，如果inv是true，那么就通过"snap==0"将原图中的0转化为1。

## 连通域属性计算
这一步就是skimage的measure模块的regionprops函数对上面的连通域的属性进行计算，常用的属性有：
- area：区域内像素点总数
- perimeter: 区域周长
- euler_number:区域欧拉数，可以用来计算一个连通域内的孔洞数，公式为1-euler_number，比如B字符这个连通域的欧拉数为-1，那么孔洞数就是1-（-1）=2
- solidity: 坚实度，区域内像素点数目与其凸包图像的像素点数目的比值
- eccentricity: 离心率，这里是长轴与短轴的长度的比值
 
这里的参考文献有：
[OpenCV轮廓层次分析实现欧拉数计算](https://cloud.tencent.com/developer/article/1357073)
[skimage.measure.label和skimage.measure.regionprops()](https://blog.csdn.net/pursuit_zhangyu/article/details/94209489)

然后调用regionprops函数：
```python
idx = (np.ones(n+1)*(0 if para['inv'] else para['front'])).astype(np.uint8)
ls = regionprops(lab)
```
这里首先又创建了一个idx来临时存储过滤结果，如果没有设置inv的话，就用参数front的灰度值来填充它。
然后在不同的过滤器起作用时，将参数back的灰度值填入被过滤掉的像素中。

经过上面的处理后，结果为：
![geom-filter](https://user-images.githubusercontent.com/6218739/72135032-77828000-33c0-11ea-9574-c3e0c5edbf11.png)

# 开运算
```python
Binary Opening>{'w': 3, 'h': 3}
```
开运算是一种形态学计算，计算步骤是先腐蚀，后膨胀。通过腐蚀运算能去除小的非关键区域，也可以把离得很近的元素分割开，再通过膨胀填补过度腐蚀留下的空隙。因此，通过开运算能去除一些孤立的、细小的点，平滑毛糙的边缘线，同时原区域面积也不会有明显的改变，类似于“去毛刺”的效果。（以上摘抄自《机器视觉算法原理与编程实战》一书）
开运算后的结果如下：
![open](https://user-images.githubusercontent.com/6218739/72236302-3b922980-3611-11ea-814d-7d1d3ebde463.png)

# 二值分水岭
```python
Binary Watershed>{'tor': 1, 'con': False}
```
这一步是做二值分水岭。之前有过对该算法的详细解析，见[这里](http://qixinbo.info/2019/12/20/ImagePy_15/)
这一步的目的就是将黏连的细胞分割开：
![watershed-cell](https://user-images.githubusercontent.com/6218739/72236532-3e414e80-3612-11ea-9955-c3f9615166fb.png)

# 几何分析
```python
Geometry Analysis>{'con': '4-connect', 'center': True, 'area': True, 'l': True, 'extent': False, 'cov': True, 'slice': False, 'ed': False, 'holes': False, 'ca': False, 'fa': False, 'solid': False}
```
这一步“几何分析”与上面的“几何过滤”本质操作是一样的，都是先生成结构单元，然后标记连通域，再调用regionprops计算各个连通域的各种属性。只不过这里没有过滤功能，是对最后的连通域的分析。
```python
class RegionCounter(Simple):
    title = 'Geometry Analysis'
    note = ['8-bit', '16-bit', 'int']
    para = {'con':'8-connect', 'center':True, 'area':True, 'l':True, 'extent':False, 'cov':False, 'slice':False,
            'ed':False, 'holes':False, 'ca':False, 'fa':False, 'solid':False}
    view = [(list, 'con', ['4-connect', '8-connect'], str, 'conection', 'pix'),
            (bool, 'slice', 'slice'),
            ('lab', None, '=========  indecate  ========='),
            (bool, 'center', 'center'),
            (bool, 'area', 'area'),
            (bool, 'l', 'perimeter'),
            (bool, 'extent', 'extent'),
            (bool, 'ed', 'equivalent diameter'),
            (bool, 'ca', 'convex area'),
            (bool, 'holes', 'holes'),
            (bool, 'fa', 'filled area'),
            (bool, 'solid', 'solidity'),
            (bool, 'cov', 'cov')]

    #process
    def run(self, ips, imgs, para = None):
        if not para['slice']:imgs = [ips.img]
        k = ips.unit[0]
        titles = ['Slice', 'ID'][0 if para['slice'] else 1:]
        if para['center']:titles.extend(['Center-X','Center-Y'])
        if para['area']:titles.append('Area')
        if para['l']:titles.append('Perimeter')
        if para['extent']:titles.extend(['Min-Y','Min-X','Max-Y','Max-X'])
        if para['ed']:titles.extend(['Diameter'])
        if para['ca']:titles.extend(['ConvexArea'])
        if para['holes']:titles.extend(['Holes'])
        if para['fa']:titles.extend(['FilledArea'])
        if para['solid']:titles.extend(['Solidity'])
        if para['cov']:titles.extend(['Major','Minor','Ori'])

        buf = imgs[0].astype(np.uint32)
        data, mark = [], {'type':'layers', 'body':{}}
        strc = generate_binary_structure(2, 1 if para['con']=='4-connect' else 2)
        for i in range(len(imgs)):
            label(imgs[i], strc, output=buf)
            ls = regionprops(buf)

            dt = [[i]*len(ls), list(range(len(ls)))]
            if not para['slice']:dt = dt[1:]

            layer = {'type':'layer', 'body':[]}
            texts = [(i.centroid[::-1])+('id=%d'%n,) for i,n in zip(ls,range(len(ls)))]
            layer['body'].append({'type':'texts', 'body':texts})
            if para['cov']:
                ellips = [i.centroid[::-1] + (i.major_axis_length/2,i.minor_axis_length/2,i.orientation) for i in ls]
                layer['body'].append({'type':'ellipses', 'body':ellips})
            mark['body'][i] = layer

            if para['center']:
                dt.append([round(i.centroid[1]*k,1) for i in ls])
                dt.append([round(i.centroid[0]*k,1) for i in ls])
            if para['area']:
                dt.append([i.area*k**2 for i in ls])
            if para['l']:
                dt.append([round(i.perimeter*k,1) for i in ls])
            if para['extent']:
                for j in (0,1,2,3):
                    dt.append([i.bbox[j]*k for i in ls])
            if para['ed']:
                dt.append([round(i.equivalent_diameter*k, 1) for i in ls])
            if para['ca']:
                dt.append([i.convex_area*k**2 for i in ls])
            if para['holes']:
                dt.append([1-i.euler_number for i in ls])
            if para['fa']:
                dt.append([i.filled_area*k**2 for i in ls])
            if para['solid']:
                dt.append([round(i.solidity, 2) for i in ls])
            if para['cov']:
                dt.append([round(i.major_axis_length*k, 1) for i in ls])
                dt.append([round(i.minor_axis_length*k, 1) for i in ls])
                dt.append([round(i.orientation*k, 1) for i in ls])

            data.extend(list(zip(*dt)))
        ips.mark = GeometryMark(mark)
        IPy.show_table(pd.DataFrame(data, columns=titles), ips.title+'-region')
```

可以从最后面两行代码看出，几何分析比几何过滤还多了两个功能，一个是对连通域的绘制，一个是对连通域属性的表格统计。
这两个功能值得深究一下，因为它们提供了怎样统一管理各个连通域并可视化呈现的思路。

## 连通域绘制
```
        data, mark = [], {'type':'layers', 'body':{}}
        for i in range(len(imgs)):
            label(imgs[i], strc, output=buf)
            ls = regionprops(buf)

            layer = {'type':'layer', 'body':[]}
            texts = [(i.centroid[::-1])+('id=%d'%n,) for i,n in zip(ls,range(len(ls)))]
            layer['body'].append({'type':'texts', 'body':texts})
            if para['cov']:
                ellips = [i.centroid[::-1] + (i.major_axis_length/2,i.minor_axis_length/2,i.orientation) for i in ls]
                layer['body'].append({'type':'ellipses', 'body':ellips})
            mark['body'][i] = layer
        ips.mark = GeometryMark(mark)
```
首先是对整个图像栈进行循环，因为这里只有一张图像，所以imgs的length是1。
然后就是对该图像使用label()函数进行标记，及使用reginprops()函数进行连通域属性分析。得到的ls变量为：
```python
[<skimage.measure._re...CFF7AF8D0>, <skimage.measure._re...CFE7B2D30>, <skimage.measure._re...CFE7B25C0>, <skimage.measure._re...CFE7B25F8>, <skimage.measure._re...CFE7B2978>, <skimage.measure._re...CFE7B2AC8>, <skimage.measure._re...CFE7B29E8>, <skimage.measure._re...CFE7B2BA8>, <skimage.measure._re...CFE7B2DA0>, <skimage.measure._re...CFE7B2898>, <skimage.measure._re...CFE7B2DD8>, <skimage.measure._re...CFE7B2828>, <skimage.measure._re...CFE7B27F0>, <skimage.measure._re...CFE7B2780>, ...]
```
其中第0个元素为：
```python
<skimage.measure._regionprops._RegionProperties object at 0x000001ECFF7AF8D0>
```
可以看出ls变量是一个这种RegionProperties类型的列表，该类型又包含了area、bbox、centroid等连通域的具体属性。针对于该图像，ls的length为35，表明有35个连通域。

再建立一个字典layer，用来盛放各个连通域的标识：
```python
            layer = {'type':'layer', 'body':[]}
            texts = [(i.centroid[::-1])+('id=%d'%n,) for i,n in zip(ls,range(len(ls)))]
            layer['body'].append({'type':'texts', 'body':texts})
```
texts中是提取了各个连通域的centroid及附上'id=0'类似的标识，然后在layer的body中再添加一个字典。所以，layer是一个嵌套型的字典，最上层的字段type是"layer"，字段body中又包含一个字典，该字典的字段type是"texts"，字段body是一个列表，里面包含了35个连通域的位置和标识。

从接下来代码可以看出，如果para中设置了conv，那么再在layer的body字段中添加一个字典，其字段type为ellipse，字段body则是centroid与半长轴、半短轴和取向的加和。

然后将layer加入到最高级别的mark的body字段中：
```python
mark['body'][i] = layer
```
mark字典的字段type则是layers。
而这个mark则会被当做GeometryMark类的参数被封装一下后传给ips图像类的mark属性，即：
```python
ips.mark = GeometryMark(mark)
```
这一步的信息量非常大，其包含了多个操作：
（1）将mark传入GeometryMark类中，形成一个该类的对象，并传给ips的mark属性：
```python

def drawmark(dc, f, body, **key):
	pen, brush, font = dc.GetPen(), dc.GetBrush(), dc.GetFont()
	pen.SetColour(default_color or (255,255,0))
	brush.SetColour(default_face or (255,255,255))
	brush.SetStyle((106,100)[default_fill or False])
	pen.SetWidth(default_lw or 1)
	dc.SetTextForeground(default_tcolor or (255,0,0))
	font.SetPointSize(default_tsize or 8)
	dc.SetPen(pen); dc.SetBrush(brush); dc.SetFont(font);
	draw(body, dc, f, **key)

class GeometryMark:
	def __init__(self, body):
		self.body = body

	def draw(self, dc, f, **key):
		drawmark(dc, f, self.body, key)
```
可以看出，该类有一个draw()函数，接着调用了drawmark()函数，即调用了该python文件上面的一系列的自定义的绘图方法，如draw_text()、draw_ellipse()、draw_circle()等等，具体的绘图参数也是按照上面mark变量的层级进行提取和使用。

（2）ips的mark属性的draw方法会传入画布canvas的marks属性中：
这一步是发生在canvasframe.py文件的CanvasPanel类的on_idle()函数中：
```python
        if self.ips.dirty != False:
            if self.ips.mark is None: self.canvas.marks['mark'] = None
            else:
                draw = lambda dc, f, **key: self.ips.mark.draw(dc, f, cur=self.ips.cur, **key)
                self.canvas.marks['mark'] = draw
```
注意，该CanvasPanel类实际会继续调用Canvas类来进行显示图像，上述函数中调用了Canvas类的update()函数进行画布更新显示：
```python
self.canvas.update()
```
（3）画布更新的同时进行标注显示
在Canvas类的update()函数中：
```python
    def update(self):
…
        for i in self.marks:
            if self.marks[i] is None: continue
            if callable(self.marks[i]):
                self.marks[i](dc, self.to_panel_coor, k = self.scale)
            else:
                drawmark(dc, self.to_panel_coor, self.marks[i], k=self.scale)
```
可以看出，有一个callable判定（对于函数、方法、lambda 函式、 类以及实现了 __call__ 方法的类实例, 它都返回 True。），在这里进行那些标识的绘制。
结果如图：
![mark](https://user-images.githubusercontent.com/6218739/74136499-1a7d2280-4c29-11ea-9381-080d05bb4e75.png)

## 连通域属性的表格统计
```python
            dt = [[i]*len(ls), list(range(len(ls)))]
            if not para['slice']:dt = dt[1:]
```
根据上面ls变量的长度构建一个新的容器dt：
```python
dt = [[i]*len(ls), list(range(len(ls)))]
```
具体地，dt的值为：
```python
[[0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]]
```
如果在para没有选定slice，那么就只取dt的后一部分：
```python
if not para['slice']:dt = dt[1:]
```
即，dt变为：
```python
[[0, 1, 2, 3, 4, 5, 6, 7, 8, ...]]
```
然后，再依次将各个连通域的各个属性值依次提取到dt中，比如：
```python
            if para['center']:
                dt.append([round(i.centroid[1]*k,1) for i in ls])
                dt.append([round(i.centroid[0]*k,1) for i in ls])
```
最后将dt打包成data：
```python
data.extend(list(zip(*dt)))
```
data中的数据形如：
```python
[(0, 243.6, 28.1, 1785, 155.4, 49.3, 46.2, -0.9), (1, 45.4, 48.4, 1443, 141.1, 44.1, 41.7, 1.5), ...]
```
即每个连通域的各个属性打包在一块。
然后使用IPy的show_table()函数将其显示：
```python
IPy.show_table(pd.DataFrame(data, columns=titles), ips.title+'-region')
```
ImagePy也提供了专门的TablePlus类来存储table数据和TablePanel类来显示table数据。

结果如图：
![tables](https://user-images.githubusercontent.com/6218739/74136683-79db3280-4c29-11ea-8309-86c575fdbbc0.png)

# 做散点图
```python
Scatter Chart>{'x': 'Center-X', 'y': 'Center-Y', 's': 5, 'alpha': 1.0, 'rs': 'Perimeter', 'c': (0, 0, 255), 'cs': 'Area', 'cm': 'Red_Hot', 'grid': True, 'title': 'Cells Scatter'}
```

这里的散点图绘制用的是Table引擎，该引擎原理类似于之前的引擎，不再详述。
作图是调用的matplotlib库，结果为：
![scatter](https://user-images.githubusercontent.com/6218739/74138224-30d8ad80-4c2c-11ea-8410-120807b8548f.png)

这一篇太长了。。等着再接着写吧。。
