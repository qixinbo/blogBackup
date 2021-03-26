---
title: ImagePy解析： 25 -- 智能画笔
tags: [ImagePy]
categories: computer vision 
date: 2021-3-26
---

# 简介
本文介绍ImagePy中的智能画笔工具，它能够很方便地对图像进行像素级标注，尤其是在复杂图像上进行多种类别的标注时，用好这个智能画笔，能使效率飞升。
先来一连串的功能介绍镇楼：
（1）鼠标左键
左键：具有联想功能的局部标注
Ctrl+左键单击：吸取颜色
Ctrl+左键：普通画笔
Shift+左键：落笔选定保护色，在矩形框内闭合且非保护色区域被填充
Alt+左键：落笔选定保护色，在矩形框内对该色进行描边
Ctrl+Alt+左键：落笔选定保护色，对任意的非保护色的区域进行标注

（2）鼠标右键
右键：全局填充
Shift+右键：落笔所在的保护色的内部闭合区域被填充
Ctrl+右键：落笔所在的保护色的外边缘被描边
Alt+右键：落笔所在的保护色的内边缘被描边
Ctrl+Alt+右键：落笔选定保护色，外部非保护色区域被填充

（3）鼠标中键/滚轮
中键/滚轮：缩放，按住则拖动画布
Shift+滚轮：调节联想功能的粘性系数
Ctrl+滚轮：调节画笔的宽度
Ctrl+Alt+滚轮：调节矩形框大小

那么主角就是下面这个工具了：
![aipen](https://user-images.githubusercontent.com/6218739/111566949-1e3e8980-87d9-11eb-94d0-04e023349d42.png)
对应源码在[这里](https://github.com/Image-Py/imagepy/blob/master/imagepy/tools/Draw/aibrush_tol.py)。

# 控制参数
智能画笔工具有如此强大的功能，自然有相应的参数可以供调节：
![para-view](https://user-images.githubusercontent.com/6218739/111739860-b5314180-88be-11eb-89d0-dfc741c7de40.png)
（1）win：矩形框窗口尺寸，控制局部标注区域的大小
（2）color：矩形框窗口颜色
（3）stickiness：粘性系数
（4）radius：画笔宽度
（5）tolerance：填充的容忍度
（6）connect：邻域

# 功能解析
## 鼠标左键
### 吸取颜色
Ctrl+左键单击来吸取颜色的代码如下：
（代码逻辑就是左键按下时记录下鼠标位置，鼠标弹起时判断是否是左键+Ctrl，同时鼠标未移动，符合条件后就将当前图像中的颜色数值传递给颜色管理器中的前景色）：
```python
    def mouse_down(self, ips, x, y, btn, **key):
        self.oldp = self.pickp = (y, x)

    def mouse_up(self, ips, x, y, btn, **key):
        if btn==1 and (y,x)==self.pickp and key['ctrl']:
            x = int(round(min(max(x,0), ips.img.shape[1])))
            y = int(round(min(max(y,0), ips.img.shape[0])))
            self.app.manager('color').add('front', ips.img[y, x])
```

### 获取局部区域
鼠标左键相关的动作是与局部标注相关联的，因此在介绍鼠标左键功能实现时非常重要的一点就是先介绍这个局部区域是怎样获得的。
奥秘就在如下代码中：
```python
    def mouse_move(self, ips, x, y, btn, **key):
        # 获取当前鼠标位置
        x = int(round(min(max(x,0), img.shape[1])))
        y = int(round(min(max(y,0), img.shape[0])))

      # 获取当前位置与上一个位置之间的连续坐标，注意这里的加号是作用在tuple上，即连接两个元组，而不是数值相加
        rs, cs = line(*[int(round(i)) for i in self.oldp + (y, x)])
        # 确保这些坐标都在图像内部
        np.clip(rs, 0, img.shape[0]-1, out=rs)
        np.clip(cs, 0, img.shape[1]-1, out=cs)

        for r,c in zip(rs, cs):
            start = time()
            w = self.para['win']
            # 以上述坐标为中心，获取其上下左右间距为w的矩形窗口，同时还保证这个窗口也在图像内部
         # 这样一来，如果在图像中抠出该窗口，那么窗口中心就是w
            sr = (max(0,r-w), min(img.shape[0], r+w))
            sc = (max(0,c-w), min(img.shape[1], c+w))
            # 如果上述坐标本身数值就小于窗口大小，那么窗口中心就得是实际的这个坐标
            r, c = min(r, w), min(c, w)
            # 从原图中抠出该窗口
            backclip = imgclip = img[slice(*sr), slice(*sc)]
            if not ips.back is None: 
               # 从背景图中同步抠出该窗口
                backclip = ips.back.img[slice(*sr), slice(*sc)]
```

### 局部区域和画笔展示
在执行鼠标左键操作时，会涉及局部区域和画笔在界面上的展示。
局部区域是用矩形框显示，画笔是用一个小的圆形显示，以及会有一个文本显示矩形框尺寸和粘度大小。源码为：
```python
    def mouse_move(self, ips, x, y, btn, **key):
        if self.status == None and ips.mark != None:
            ips.mark = None
            ips.update()
        if not self.status in ['local_pen','local_brush',
            'local_sketch','local_in','local_out','move']:  return
        img, color = ips.img, self.app.manager('color').get('front')
        x = int(round(min(max(x,0), img.shape[1])))
        y = int(round(min(max(y,0), img.shape[0])))

      # 绘制的形状传给ips的mark属性，在ips.update时就会将其绘制出来
        ips.mark = self.make_mark(x, y)
        self.oldp = (y, x)
        ips.update()

   # 具体绘制的函数
    def make_mark(self, x, y):
        wins = self.para['win']
        rect = {'type':'rectangle', 'body':(x-wins, y-wins, wins*2, wins*2), 'color':self.para['color']}
        # 只需将矩形框、文本、圆形的属性用非常简单的语句描述出来即可
        mark = {'type':'layer', 'body':[rect]}
        r = 2 if self.status=='local_brush' else self.para['r']/2
        mark['body'].append({'type':'circle', 'body':(x, y, r), 'color':self.para['color']})
        mark['body'].append({'type':'text', 'body':(x-wins, y-wins, 
            'S:%s W:%s'%(self.para['ms'], self.para['win'])), 'pt':False, 'color':self.para['color']})
        return mark2shp(mark)
```

### 普通画笔功能
Ctrl键和鼠标左键实现普通画笔功能：
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==1 and key['ctrl']:
            self.status = 'local_pen'

    def mouse_move(self, ips, x, y, btn, **key):
        for r,c in zip(rs, cs):
            backclip = imgclip = img[slice(*sr), slice(*sc)]
            if not ips.back is None: 
                backclip = ips.back.img[slice(*sr), slice(*sc)]

            if self.status == 'local_pen':
                local_pen(imgclip, r, c, self.para['r'], color)

def local_pen(img, r, c, R, color):
    img = img.reshape((img.shape+(1,))[:3])
    # 以r和c为中心，R/2为半径进行画圆，获取该圆内的坐标
    rs, cs = circle(r, c, R/2+1e-6, shape=img.shape)
    img[rs, cs] = color
```

### 有联想功能的画笔
左键点击后拖动就会对区域进行有联想功能的填充，使用的算法是Felzenszwalb算法，原理是使用像素之间的颜色距离来衡量两者的相似性，具体原理可以参照[该博客](https://blog.csdn.net/ttransposition/article/details/38024557)。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==1:
            self.status = 'local_brush'

    def mouse_move(self, ips, x, y, btn, **key):
        for r,c in zip(rs, cs):
            backclip = imgclip = img[slice(*sr), slice(*sc)]
            if not ips.back is None: 
                backclip = ips.back.img[slice(*sr), slice(*sc)]

            if self.status == 'local_brush':
                if (imgclip[r,c] - color).sum()==0: continue
                local_brush(imgclip, backclip, r, c, color, 0, self.para['ms'])

def local_brush(img, back, r, c, color, sigma, msize):
    # 使用felzenszwalb算法对该局部区域进行分割
    lab = felzenszwalb(back, 1, sigma, msize)
    # 对分割后的区域进行泛洪填充，将区域分为True和False，True的地方即标注颜色
    msk = flood(lab, (r, c), connectivity=2)
    img[msk] = color
```
### 落笔选定保护色，在矩形框内闭合且非保护色区域被填充
Shift+左键用来实现上述功能。
这里要注意有三个关键点：保护色、矩形框内闭合区域、非保护色区域。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        # 设定保护色
        self.pickcolor = ips.img[y, x]
        elif btn==1 and key['shift']:
            self.status = 'local_in'

    def mouse_move(self, ips, x, y, btn, **key):
        img, color = ips.img, self.app.manager('color').get('front')
        color = (np.mean(color), color)[img.ndim==3]
        for r,c in zip(rs, cs):
            backclip = imgclip = img[slice(*sr), slice(*sc)]
            if not ips.back is None: 
                backclip = ips.back.img[slice(*sr), slice(*sc)]

            if self.status == 'local_in':
                local_in_fill(imgclip, r, c, self.para['r'], self.pickcolor, color)

def local_in_fill(img, r, c, R, color, bcolor):
    img = img.reshape((img.shape+(1,))[:3])
    # 判断当前颜色是否是保护色，返回True和False矩阵
    msk = (img == color).min(axis=2)
    # 将上面判断保护色后的结果进行“填充孔洞”，如果不是保护色区域的孔洞，则不会填充
    filled = binary_fill_holes(msk)
    # 将填充孔洞后的结果与原掩膜进行“异或”操作，两者相异的地方结果为1
   # 这样就把孔洞内非保护色的区域给提取了出来，
   # 因此，即使非保护色，如果不属于保护色内的孔洞，也不会被提取出来
    filled ^= msk
    # 获得当前画笔描绘的区域
    rs, cs = circle(r, c, R/2+1e-6, shape=img.shape)
    # 将原掩膜全部置为0
    msk[:] = 0
   # 将当前画笔描绘的区域置为1
    msk[rs, cs] = 1
    # 将孔洞内非保护色区域与现在的掩膜进行“与”操作，两者都为1，结果才为1
    msk &= filled
    # 将最终的掩膜处的图像置为当前的前景色
    img[msk] = bcolor
```

### 落笔选定保护色，在矩形框内对该色进行描边
Alt+左键用来实现上述功能。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        # 设定保护色
        self.pickcolor = ips.img[y, x]
        elif btn==1 and key['alt']:
            self.status = 'local_sketch'

    def mouse_move(self, ips, x, y, btn, **key):
        img, color = ips.img, self.app.manager('color').get('front')
        color = (np.mean(color), color)[img.ndim==3]
        for r,c in zip(rs, cs):
            backclip = imgclip = img[slice(*sr), slice(*sc)]
            if not ips.back is None: 
                backclip = ips.back.img[slice(*sr), slice(*sc)]

            if self.status == 'local_sketch':
                local_sketch(imgclip, r, c, self.para['r'], self.pickcolor, color)

def local_sketch(img, r, c, R, color, bcolor):
    img = img.reshape((img.shape+(1,))[:3])
    # 判断当前颜色是否是保护色，返回True和False矩阵
    msk = (img == color).min(axis=2)
    # 对该掩膜进行膨胀操作
    dilation = binary_dilation(msk, np.ones((3,3)))
    # 对膨胀后的掩膜与原掩膜进行“异或”操作，这样就提取出了保护色的边界
    dilation ^= msk
    rs, cs = circle(r, c, R/2+1e-6, shape=img.shape)
    msk[:] = 0
    msk[rs, cs] = 1
    # 取出既是边界，同时又是画笔所描绘的地方
    msk &= dilation
    # 将这个地方赋以前景色
    img[msk] = bcolor
```

### 落笔选定保护色，对任意的非保护色的区域进行标注
Ctrl+Alt+左键实现如上功能：
```python
    def mouse_down(self, ips, x, y, btn, **key):
        # 设定保护色
        self.pickcolor = ips.img[y, x]
        if btn==1 and key['ctrl'] and key['alt']:
            self.status = 'local_out'

    def mouse_move(self, ips, x, y, btn, **key):
        img, color = ips.img, self.app.manager('color').get('front')
        color = (np.mean(color), color)[img.ndim==3]
        for r,c in zip(rs, cs):
            backclip = imgclip = img[slice(*sr), slice(*sc)]
            if not ips.back is None: 
                backclip = ips.back.img[slice(*sr), slice(*sc)]

            if self.status=='local_out':
                local_out_fill(imgclip, r, c, self.para['r'], self.pickcolor, color)

def local_out_fill(img, r, c, R, color, bcolor):
    img = img.reshape((img.shape+(1,))[:3])
    # 判断当前颜色是否不是保护色（注意这里选择“非保护色”），返回True和False矩阵
    msk = (img != color).max(axis=2)
    rs, cs = circle(r, c, R/2+1e-6, shape=img.shape)
    buf = np.zeros_like(msk)
    buf[rs, cs] = 1
    # 挑选出既是非保护色，然后又是鼠标描绘的地方
    msk &= buf
    img[msk] = bcolor
```

## 鼠标右键
鼠标右键的操作都是与全局标注相关的。

### 全局填充
右键单击就实现了全局填充的功能。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==3:
            if (ips.img[y, x] - color).sum()==0: return
            conn = {'4-connect':1, '8-connect':2}
            conn = conn[self.para['con']]
            tor = self.para['tor']
            fill_normal(ips.img, y, x, color, conn, tor)
            ips.update()

def fill_normal(img, r, c, color, con, tor):
    img = img.reshape((img.shape+(1,))[:3])
    msk = np.ones(img.shape[:2], dtype=np.bool)
    # 将多通道的图像拆分成每一个通道来泛洪填充
    for i in range(img.shape[2]):
        # 以鼠标点击的像素为中心，以tolerance为容差，以及设定邻域范围，然后在单一通道上进行填充
      # 虽然是每一通道分别处理，但是所有通道都是与msk进行“与”操作
      # 所以msk最终是满足所有通道的填充结果
        msk &= flood(img[:,:,i], (r, c), connectivity=con, tolerance=tor)
    img[msk] = color
```

### 落笔所在保护色的内部闭合区域被填充
Shift+右键完成上述功能：
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==3 and key['shift']:
            self.status = 'global_in_fill'
            global_in_fill(ips.img, y, x, color)
            ips.update()

def global_in_fill(img, r, c, color):
    img = img.reshape((img.shape+(1,))[:3])
    msk = np.ones(img.shape[:2], dtype=np.bool)
    # 以鼠标所在的颜色为种子点，进行泛洪填充
   # 但与上面的填充不同的是，这里不设定tolerance，即严格地与该保护色相等地地方才填充
    for i in range(img.shape[2]):
        msk &= flood(img[:,:,i], (r, c), connectivity=2)
    # 填充的地方如果有孔洞，那么就填充起来
    filled = binary_fill_holes(msk)
    # 填充孔洞后的图像与原先泛洪后的图像进行“异或”操作，就得到了孔洞区域
    filled ^= msk
    # 将孔洞区域赋值为前景色
    img[filled] = color
```

### 落笔所在的保护色的外边缘被描边
Ctrl+右键完成上述功能。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==3 and key['ctrl']:
            self.status = 'global_out_line'
            global_out_line(ips.img, y, x, color)
            ips.update()

def global_out_line(img, r, c, color):
    img = img.reshape((img.shape+(1,))[:3])
    msk = np.ones(img.shape[:2], dtype=np.bool)
    # 同样的严格按照鼠标所在的颜色进行泛洪填充
    for i in range(img.shape[2]):
        msk &= flood(img[:,:,i], (r, c), connectivity=2)
    # 对内部孔洞进行填充
    msk = binary_fill_holes(msk)
    # 膨胀一下
    dilation = binary_dilation(msk, np.ones((3,3)))
    # 膨胀的图像与未膨胀的图像进行异或，得到外边缘
    dilation ^= msk
    img[dilation] = color
```

### 落笔所在保护色的内边缘被描边
Alt+右键完成上述功能。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==3 and key['alt']:
            self.status = 'global_in_line'
            global_in_line(ips.img, y, x, color)
            ips.update()

def global_in_line(img, r, c, color):
    img = img.reshape((img.shape+(1,))[:3])
    msk = np.ones(img.shape[:2], dtype=np.bool)
    # 同样的严格按照鼠标所在的颜色进行泛洪填充
    for i in range(img.shape[2]):
        msk &= flood(img[:,:,i], (r, c), connectivity=2)
    # 对内部孔洞进行填充
    inarea = binary_fill_holes(msk)
    # 填充孔洞后的图像与未填充孔洞的图像进行异或，得到孔洞区域
    inarea ^= msk
    # 对孔洞区域腐蚀一下，然后与原孔洞进行异或，从而得到这个孔洞的外边缘，也就是保护色填充区域的内边缘。
    inarea ^= binary_erosion(inarea, np.ones((3,3)))
    img[inarea] = color
```

### 落笔选定保护色，外部非保护色区域被填充
Ctrl+Alt+右键完成该功能。
```python
    def mouse_down(self, ips, x, y, btn, **key):
        elif btn==3 and key['ctrl'] and key['alt']:
            self.status = 'global_out_fill'
            global_out_fill(ips.img, y, x, color)
            ips.update()

def global_out_fill(img, r, c, color):
    img = img.reshape((img.shape+(1,))[:3])
    # 同样的严格按照鼠标所在的颜色进行泛洪填充，注意如果与保护色不连续的地方，也是不会被填充的，因为水流不过去
    ori = np.ones(img.shape[:2], dtype=np.bool)
    for i in range(img.shape[2]):
        ori &= flood(img[:,:,i], (r, c), connectivity=2)
    # 对内部孔洞进行填充
    filled = binary_fill_holes(ori)
    # 膨胀一下
    dilation = binary_dilation(ori)
    # 获得泛洪填充区域的外边缘
    dilation ^= filled
    # 获得这些外边缘像素的坐标序列
    rs, cs = np.where(dilation)
    if len(rs)==0: return
    # 挑选图像中与鼠标所在的保护色相等的地方，返回1和0的掩膜矩阵
    msk = ((img == img[r,c]).min(axis=2)).astype(np.uint8)
    # 对该掩膜进行泛洪填充，种子点就是外边缘像素中的一个（注意这里只是用了这个外边缘像素的位置，填充时参考的数值是掩膜中的0）
   # 这里用的是flood_fill，原理与flood相同，只是flood返回的是掩膜，而flood_fill则是对原矩阵进行直接数值填充
   # 这里填充的新值设为2
   # 这里还有一个隐藏功能，就是保护色的内部孔洞区域仍然是0，所以这些内部孔洞在后面也不会被填充为前景色。
    flood_fill(msk, (rs[0], cs[0]), 2, connectivity=2, inplace=True)
    # 将值为2的地方设为前景色
    img[msk==2] = color
```


## 鼠标中键
鼠标中键的功能非常容易理解，比如滚动滚轮进行缩放、按住滚轮来拖动画布等都是常规的画布操作。
而与Shift、Ctrl和Alt等的组合用法，其实就是对上面参数（win、stickiness和radius）的调节。
鼠标中键相关操作的源码对应实现分别为：
### 拖动画布
```python
    def mouse_down(self, ips, x, y, btn, **key):
        if btn==2: 
            self.oldp = key['canvas'].to_panel_coor(x,y)
            self.status = 'move'
            return

    def mouse_move(self, ips, x, y, btn, **key):
        x = int(round(min(max(x,0), img.shape[1])))
        y = int(round(min(max(y,0), img.shape[0])))
        
        if self.status == 'move':
            x,y = key['canvas'].to_panel_coor(x,y)
            key['canvas'].move(x-self.oldp[0], y-self.oldp[1])
            self.oldp = x, y
            ips.update()
            return
```
### 通过滚轮调节参数
```python
    def mouse_wheel(self, ips, x, y, d, **key):
        if key['shift']:
            if d>0: self.para['ms'] = min(50, self.para['ms']+1)
            if d<0: self.para['ms'] = max(10, self.para['ms']-1)
            ips.mark = self.make_mark(x, y)
        elif key['ctrl'] and key['alt']:
            if d>0: self.para['win'] = min(64, self.para['win']+1)
            if d<0: self.para['win'] = max(28, self.para['win']-1)
            ips.mark = self.make_mark(x, y)
        elif key['ctrl']:
            if d>0: self.para['r'] = min(30, self.para['r']+1)
            if d<0: self.para['r'] = max(2, self.para['r']-1)
            ips.mark = self.make_mark(x, y)
```
### 缩放画布
```python
        elif self.status == None:
            if d>0:key['canvas'].zoomout(x, y, 'data')
            if d<0:key['canvas'].zoomin(x, y, 'data')
        ips.update()
```

