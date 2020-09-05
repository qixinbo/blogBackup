---
title: ImagePy解析：12 -- 画布Canvas类详解
tags: [ImagePy]
categories: computational material science 
date: 2019-10-29
---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
2020-9-5 更新：
加入draw_image()和mix_img()方法的详解
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


前面系列文已经提到，ImagePy的画布Canvas类已经被抽象出来，可以被单独使用。
本节就对该类做一个详细解析：因为Canvas类又调用了同一路径下的boxutil和imutil，所以本文的思路是先整体介绍它的运行机理，然后具体到某一功能时可以再详细查看下方的详细解释，这样不至于迷失在代码中。。

该类的源码地址在[这里](https://github.com/Image-Py/imagepy/tree/master/imagepy/ui/canvas)。

# 画布Canvas运行机理
总体看一下它的调用代码：
```python
if __name__=='__main__':
    msk = np.zeros((512,512), dtype=np.uint8)
    msk[100:200,100:200] = 1
    msk[200:300,200:300] = 2
    msk[300:400,300:400] = 3

    lut = np.array([(0,0,0),(255,0,0),(0,255,0),(0,0,255)], dtype=np.uint8)

    from skimage.data import astronaut, camera
    app = wx.App()
    frame = wx.Frame(None)
    canvas = Canvas(frame)
    canvas.set_img(msk)
    canvas.set_lut(lut)
    canvas.set_cn(0)
    canvas.set_back(astronaut())
    canvas.set_cn('rgb', 1)
    canvas.set_mode('msk')
    frame.Show(True)
    app.MainLoop()
```
这里是查看单独调用该画布时的用法，该画布也可以直接用在ImagePy框架中用。
然后分步看一下具体设置和运行。

## 创建掩膜
```python
msk = np.zeros((512,512), dtype=np.uint8)
msk[100:200,100:200] = 1
msk[200:300,200:300] = 2
msk[300:400,300:400] = 3
```
这里的掩膜msk是一个512乘以512大小的黑色图像，然后将它的三个区域分别由0变为1、2或3。
掩膜之所以设置成512乘以512，是因为下面的astronaut()是这么个形状，所以一般掩膜应该是设置成与背景图像相同大小的黑色图像。（也可以任意设置掩膜大小，但那样只会显示原图的一部分，没有什么实际意义。）

## 创建查找表
```python
lut = np.array([(0,0,0),(255,0,0),(0,255,0),(0,0,255)], dtype=np.uint8)
```
这里的查找表lut是一个array数组，里面盛放了四种颜色。具体使用它的地方是imutil中lookup()方法，可以详细查看本文下面的说明。一言以蔽之，是使用了Numpy的Fancy Indexing，与上面的msk进行呼应，当msk中的像素值为1时，就取lut的(255, 0, 0)红色。

## 创建并运行Canvas对象
```python
app = wx.App()
frame = wx.Frame(None)
canvas = Canvas(frame)
...
...
frame.Show(True)
app.MainLoop()

```
上面创建了Canvas的对象，并通过它所依赖的frame的Show()方法进行渲染展示。当Show()方法调用时，就会生成Canvas的窗口，然后就会触发它的窗口改变wx.EVT_SIZE事件，然后就执行一系列的绘图动作，具体做了啥见下方说明。

## 配置Canvas：设置图像
上面创建了Canvas后，还要对其进行一番配置。这一步及下面的五步都是Canvas的核心配置步骤。
```python
canvas.set_img(msk)
```
这一步是设置前景图（下面还有一个背景图），因为这个例子是有点特别，是用于标注，所以此时Canvas要显示的图像是掩膜msk与原图的在掩膜模式下的混合（看起来就像是前景可以任意涂抹，背景则不受影响）。
那么再重新理一下这几张图：在这个例子中，前景图是一张黑色图像（大部分像素值为0，但里面的一些像素值已被更改为非0），背景图是彩色原图，实际上画布Canvas显示的是一张图像，即两者的混合，根据不同的混合模式显示不同的混合图像。

还需要注意一点（感谢龙哥的语音指导）：
（1）在单纯这个Canvas显示中，前景图和背景图需要分辨清楚，因为它是用于标注，需要明确哪个在前哪个在后，而在其他混合模式下，如最大值、最小值混合，两者效果相同。
（2）如果将Canvas放入ImagePy中，两者也是要分辨清楚，因为前景永远指的是滤波器要处理的这一张图，而背景图仅用于衬托。

## 配置Canvas：设置查找表
```python
canvas.set_lut(lut)
```
这一步是设置查找表。这里因为只传递了lut，所以该方法的第二个参数用的是False，即这里设置的是self.lut属性，而self._lut保持默认值（即黑色到白色的渐变）不变。
这里也提前说明，不带下划线的属性（包括self.lut、self.rg和self.cn）都是与前景图相关，而带下划线的属性（包括self._lut、self._rg和self._cn）都是与背景图相关。

## 配置Canvas：设置图像通道
```python
canvas.set_cn(0)
```
这一步是设置图像通道，这里因为只传递了0，所以第二个参数用的是默认值False，即这里不是针对背景图设置的，而是针对前景图，即self.cn设为0。这里有两点需要注意：首先因为是cn，不带下划线，说明是对前景图的设置；再者一定要设置为0，这是通道标识，因为掩膜msk是一个灰度值，所以只有一个通道，那么它的标识就是0。

可以再往下看，后面又调用了set_cn()方法，传入的是：
```python
canvas.set_cn('rgb', 1)
```
即此时将'rgb'赋给了self._cn，即对背景图的通道进行设置，即想让背景图以RGB彩色图的形式显示。

## 配置Canvas：设置背景图像
```python
canvas.set_back(astronaut())
```
这一步是设置背景图像，是将numpy数组形式的图像传入。

## 配置Canvas：设置混合模式
```python
canvas.set_mode('msk')
```
这里将模式设为msk，即掩膜模式。
因为如前所述，该例子是为了标注，设置为掩膜模式后，画布上呈现的图就是掩膜不为0的地方的前景图与掩膜为0的地方的背景图的叠加。
关于mode的详述，见本文下面的set_mode()方法。

# Canvas类及其属性和方法

## set_img()方法
这一步就是指定前景图，再强调一下，这个前景图并不是画布呈现的图，画布呈现的是前景与背景的混合图像。
```python
def set_img(self, img):
    self.img = img
    self.conbox = [0, 0, *img.shape[1::-1]]
    self.oribox = [0, 0, *img.shape[1::-1]]
```
注意，这个地方用到了一个星号变量，是为了将shape由元组拆分为单个参数，具体用法可以看我转载的一篇讲星号变量的文章，那么这个地方的运算顺序实际是：
```python
self.conbox = [0, 0, *(img.shape[1::-1])]
```
同时，注意对shape元组切片时的步长是-1，即它是反着切的，即只取第0个元素和第1个元素。
得到的conbox和oribox也是：
```python
conbox =  [0, 0, 512, 512]
```
通过设置前景图，给Canvas的img属性赋值，同时根据图像尺寸得到了Canvas的conbox和oribox。
这里面的几个box要明白它们的含义（因为这里没涉及旋转，oribox未作分析）：
（1）winbox：即盛放整个panel窗口的box；
（2）conbox：即承载图像的box；
（3）csbox：即上面两个box重叠的部分，也是真正绘图的画布box。
比如一张图像为[512, 1024]大小，呈现给用户的窗口是[1000, 1000]，那么conbox就是512乘以1024大小，winbox就是1000乘以1000大小，csbox大小则是512乘以1000，那么看起来就是整个窗口上，图像水平方向能全部显示，且左右都有空白填充，而垂直方向仅能显示大部分图像，没有空白填充。

## set_lut()方法
```python
def set_lut(self, lut, b=False):
    if b: self._lut = lut
    else: self.lut = lut
```
这一步是定义查找表，除了传入查找表数组，还传入一个旗标b（就是background），如果是False（即不是背景，可以把旗标b命名为isBackgroud，否则可能有点绕），就设置self.lut属性，如果是True（即是背景），就设置self._lut属性。

这两个属性默认值是：
```python
self.lut = np.array([np.arange(256)]*3, dtype=np.uint8).T
self._lut = np.array([np.arange(256)]*3, dtype=np.uint8).T
```
即是从黑色到白色渐变的一个颜色索引（注意后面的转置）。

## set_cn()方法
```python
def set_cn(self, cn, b=False):
    if b: self._cn = cn
    else: self.cn = cn
```
这一步是设置图像通道，除了传入通道，还传入一个旗标b，如果是False（即不是背景），就设置self.cn，如果是True（即是背景），就设置self._cn。

这两个属性默认值都是0：
```python
self.cn = 0
self._cn = 0
```
这里传入的cn是通道标识，如果图像只有一个通道，一定要设置为0；如果有多个通道，比如RGB，若想显示彩图，则设置为RGB；若只想显示其中一个通道，则设置通道标识为0或1或2。

## set_back()方法
```python
def set_back(self, back): self.back = back
```
这一步是设置背景图像，为self.back属性赋值。

## set_mode()方法
```python
def set_mode(self, mode): self.mode = mode
```
这一步是设置混合模式，为self.mode属性赋值。
该属性的默认值为：
```python
self.mode = 'set'
```
有多种混合模式：
（1）set模式：默认模式，直接将图像设为要显示的图像（即不混合）
（2）min模式：逐个对比背景图和前景图两张图像中的像素，选择数值小的像素保存进混合图像
（3）max模式：逐个对比背景图和前景图两张图像中的像素，选择数值大的像素保存进混合图像
（4）msk模式：根据掩膜将背景图和前景图进行混合，在掩膜msk不为0的地方，将背景图设为0；在msk为0的地方，保留原值（因为对msk作了逻辑非操作，然后下面在元素相乘时，True为1，False为0）。然后将两图进行相加合成存入混合图像。
（5）按比例混合模式：mode可以是一个小数，即将背景图与1-mode相乘，前景图与mode相乘，然后两者相加合成存入混合图像。

## set_rg()
```python
def set_rg(self, rg, b=False):
    if b: self._rg = rg
    else: self.rg = rg
```
这一步是设置像素值范围，默认两张图都是：
```python
self.rg = (0, 255)
self._rg = (0, 255)
```
## initBuffer()方法
```python
def initBuffer(self):
    box = self.GetClientSize()
    self.buffer = wx.Bitmap(*box)
    self.winbox = [0, 0, *box]
```
可以看出来，初始化缓冲时就是先创建一个初始尺寸的wxPython的位图Bitmap，然后将它赋给self.buffer变量。
同时也将整个面板的大小self.winbox设定一下。
## update()方法
```python
def update(self):
    start = time()
    lay(self.winbox, self.conbox)
    dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
    dc.Clear()
    self.draw_image(dc, self.img, self.back, 0)
    dc.UnMask()
    print('frame rate:',int(1/max(0.001, time()-start)))
```
首先通过lay()方法根据winbox的位置来更新conbox的位置（该方法的解析见本文下方），然后创建一个BufferedDC与用户交互，然后调用draw_image()方法。

## draw_image()方法
```python
def draw_image(self, dc, img, back, mode):
    out, rgb = self.outimg, self.outrgb
    csbox = cross(self.winbox, self.conbox)
    shp = csbox[3]-csbox[1], csbox[2]-csbox[0]
    o, m = mat(self.oribox, self.conbox, csbox)
    shp = tuple(np.array(shp).round().astype(np.int))
    if out is None or (out.shape, out.dtype) != (shp, img.dtype):
        self.outimg = np.zeros(shp, dtype=img.dtype)
    if rgb is None or rgb.shape[:2] != shp:
        self.outrgb = np.zeros(shp+(3,), dtype=np.uint8)
        self.outint = np.zeros(shp, dtype=np.uint8)
        buf = memoryview(self.outrgb)
        self.outbmp = wx.Bitmap.FromBuffer(*shp[::-1], buf)

    mix_img(back, m, o, shp, self.outimg,
          self.outrgb, self.outint,
          self._rg, self._lut, cns=self._cn, mode='set')
 
    mix_img(img, m, o, shp, self.outimg,
          self.outrgb, self.outint,
          self.rg, self.lut, cns=self.cn, mode=self.mode)

    self.outbmp.CopyFromBuffer(memoryview(self.outrgb))
    dc.DrawBitmap(self.outbmp, *csbox[:2])
```

首先通过cross()方法取得了winbox和conbox这两个矩形框重叠的部分csbox（仍然见下方解析），然后计算csbox的宽度和高度，形成shape这个元组，这个大小是真正要作图的区域，而不是winbox和conbox这两个的大小。
这里根据shape的大小，会创建几个变量：
self.outimg：大小为shape、值全为0、类型为img.dtype
self.outrgb：大小为shape+3通道（注意，是3通道）、值全为0、类型为np.uint8
self.outint：大小为shape（注意，是单通道）、值全为0、类型为np.uint8
然后就会调用两次图像混合。
通过mat()方法得到偏移量和缩放因子，这两个参数都是后面仿射变换的重要参数。
可以看出mix_img()方法调用了两次，分别是为了设置背景图和设置前景图。两者的顺序不能调换，因为第一次的调用返回的self.outrgb是作为背景图在第二次调用时进行图像混合。

## 事件响应
```python
def bindEvents(self):
    for event, handler in [ \
            (wx.EVT_SIZE, self.on_size),
            (wx.EVT_MOUSE_EVENTS, self.on_mouseevent),
            (wx.EVT_IDLE, self.on_idle),
            (wx.EVT_CLOSE, self.on_close),
            (wx.EVT_PAINT, self.on_paint)]:
        self.Bind(event, handler)
```
### 窗口尺寸改变
这个事件就是wx.EVT_SIZE，即窗口尺寸变化时触发，比如用鼠标拖动窗口边界。
实际上，刚开始创建该窗口时也会触发该事件，即该事件实际就是该Canvas的入口函数。
```python
def on_size(self, event):
    self.initBuffer()
    self.update()
```
可以看出，该事件处理函数就是调用初始化方法和更新方法。
### 鼠标事件
这个事件就是wx.EVT_MOUSE_EVENTS，包括鼠标按下和释放、鼠标移动、鼠标滚动等。
```python
def on_mouseevent(self, me):
    if me.ButtonDown():
       if me.GetButton()==1:
            self.oldxy = me.GetX(), me.GetY()
        if me.GetButton()==3:
            self.fit()
    wheel = np.sign(me.GetWheelRotation())
    if wheel!=0:
        if wheel == 1:
            self.zoomout(me.GetX(), me.GetY())
        if wheel == -1:
            self.zoomin(me.GetX(), me.GetY())
    if me.Dragging():
        x, y = self.oldxy
        self.move(me.GetX()-x, me.GetY()-y)
        self.oldxy = me.GetX(), me.GetY()
```
可以看出，首先对鼠标按键按下做判断：1就是左键，3就是右键，2就是中键，如果想更清楚地表示的话，可以分别用wx.MOUSE_BTN_LEFT、wx.MOUSE_BTN_MIDDLE和wx.MOUSE_BTN_RIGHT来表示。
如果是左键按下，就要捕捉当前鼠标所在的像素坐标系（以图像左上角为原点）的坐标。然后下面如果接着出了Dragging()事件，则记录新的坐标点，同时根据新旧坐标点移动conbox，并重新绘图。
如果是右键按下，则会调用fit()方法，对比oribox与winbox的相对大小，然后寻找哪个收缩比例可以使得oribox小于winbox，然后调用zoom()，传给zoom()的是收缩因子以及中心点(0,0)，更新一下conbox。
```python
scales = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]

for i in self.scales[6::-1]:
    if oriw*i<winw and orih*i<winh:
        break
self.scaidx = self.scales.index(i)
```
scales属性本身存放了很多的缩放因子，fit()方法寻找收缩因子的时候，是从中间的比例1往前查找，找到第一个能使图像完整呈现的因子。
如果是鼠标滚轮动了，则：
```python
if wheel == 1:
    self.zoomout(me.GetX(), me.GetY())
if wheel == -1:
    self.zoomin(me.GetX(), me.GetY())
```
判断到底是放大还是缩小，然后分别调用zoomout()和zoomin()，注意将当前鼠标的坐标点传入，这样就能够实现以当前鼠标点为中心进行缩放，妙~~~

### 空闲事件
目前该事件处理函数是空的。
 
### 绘图事件
这个事件就是wx.EVT_PAINT：
```python
def on_paint(self, event):
    wx.BufferedPaintDC(self, self.buffer)
```
这个事件发生在拉动窗口边界时，此时创建一个临时的BufferedPaintDC，然后将当前的缓冲呈现出来。

# boxutil模块及其方法
## cross()方法
```python
def cross(winbox, conbox):
    two = np.array([winbox, conbox])
    x1, y1 = two[:,:2].max(axis=0)
    x2, y2 = two[:,2:].min(axis=0)
    return [x1, y1, x2, y2]
```
cross()方法就是取winbox和conbox这两个矩形框交叠的部分。
比如，winbox是[0, 0, 520, 211]，conbox是[4.0, -2, 516.0, 510]，那么cross()返回的就是[4.0, 0.0, 516.0, 211.0]。
## lay()方法
lay()的功能就是根据图像与窗口之间的相对大小来设定conbox的大小。
```python
def layx(winbox, conbox):
    conw = conbox[2]-conbox[0]
    winw = winbox[2]-winbox[0]  
    if conw<winw:
        mid = (winbox[0]+winbox[2])/2
        conbox[0] = mid-conw/2
        conbox[2] = mid+conw/2
    elif conbox[0] > winbox[0]:
        conbox[0] = winbox[0]
        conbox[2] = conbox[0] + conw
    elif conbox[2] < winbox[2]:
        conbox[2] = winbox[2]
        conbox[0] = conbox[2] - conw

def layy(winbox, conbox):
    winh = winbox[3]-winbox[1]
    conh = conbox[3]-conbox[1]
    if conh<winh:
        mid = (winbox[1]+winbox[3])/2
        conbox[1] = mid-conh/2
        conbox[3] = mid+conh/2
    elif conbox[1] > winbox[1]:
        conbox[1] = winbox[1]
        conbox[3] = conbox[1] + conh
    elif conbox[3] < winbox[3]:
        conbox[3] = winbox[3]
        conbox[1] = conbox[3] - conh

def lay(winbox, conbox):
    layx(winbox, conbox)
    layy(winbox, conbox)
```
可以分为以下几种情况，以x方向的大小为例（从layx和layy可以看出，两者代码是一致的），主要是判断宽度的相对大小、左端点和右端点的相对位置：
（1）如果conbox宽度小于winbox，即整张图像无法完全填充整个窗口，需要使用空白来填充窗口时，则：
```python
if conw<winw:
    mid = (winbox[0]+winbox[2])/2
    conbox[0] = mid-conw/2
    conbox[2] = mid+conw/2
```
即先得到winbox的中心mid，然后这个中心mid在左右分别减去和加上conbox的半宽，就得到新的conbox的位置，但保持其宽度不变。
（2）如果conbox宽度大于winbox，即conbox能覆盖住winbox，在此前提下，如果conbox的左侧端点大于winbox的左侧，即conbox相比于winbox太靠右了，则：
```python
elif conbox[0] > winbox[0]:
    conbox[0] = winbox[0]
    conbox[2] = conbox[0] + conw
```
即将conbox挪回到winbox的左侧，同时还要保证原来的宽度不变。
（3）另外，在conbox能覆盖住winbox前提下，如果conbox的右侧端点小于winbox的右侧，即conbox相比于winbox太靠左了，则：
```python
elif conbox[2] < winbox[2]:
    conbox[2] = winbox[2]
    conbox[0] = conbox[2] - conw
```
则将conbox挪回到winbox的右侧，同时还要保证原来的宽度不变。
后面这两种情形都与鼠标动作的拖动有关，即捕捉当前鼠标位置，拖动后再记录当前新位置，两者相减得到移动量，得到更新的conbox位置。

## mat()方法
mat()方法是为了得到图像conbox与绘图区域csbox的旋转缩放矩阵matrix及偏移量offset。
```python
o, m = mat(self.oribox, self.conbox, csbox)
def mat(ori, vir, cros):
    kx = (ori[2]-ori[0])/(vir[2]-vir[0])
    ky = (ori[3]-ori[1])/(vir[3]-vir[1])
    ox = (cros[1]-vir[1])*ky
    oy = (cros[0]-vir[0])*kx
    return (ox, oy), (kx, ky)
```
传入的分别是oribox（这里oribox一直没变）、conbox和csbox，然后返回偏移量和缩放因子。

# imutil模块及其方法
## stretch()方法
```python
def stretch(img, out, rg, rgb=None, mode='set'):
    if img.dtype==np.uint8 and rg==(0,255):
        out[:] = img
    else:
        np.subtract(img, rg[0], out=out, casting='unsafe')
        np.multiply(img, 255.0/np.ptp(rg), out=out, casting='unsafe')
```
这一方法实际在这个canvas中只运行了if中的语句，else中的并没有执行，所以其实并没有深刻理解它干了啥。
目前看，if中实现了用img来填充out中的内容。注意这个赋值语句中，左边是out的切片，千万不能将这个切片去掉，否则img和out的地址是一样的，即out也指向了img，而如果使用了切片，out还是保留原来的地址指向，只是内容改成了img。知识点可以参见如下：
[How assignment works with Python list slice?](https://stackoverflow.com/questions/10623302/how-assignment-works-with-python-list-slice)
 
## lookup()方法
```python
def lookup(img, lut, out, mode='set'):
    blend(lut[img], out, img, mode)
```
可以看出，lookup()方法实际是调用了下面的blend()方法，所以具体干了啥还要看一下blend()干了啥。
但这一步中有一个非常重要的操作，即：
```python
lut[img]
```
这一步就是将img作为参数传入了lut查找表中，其实是用到了Numpy的Fancy Indexing，具体知识点可以查看之前一篇博文。
值得注意的是，Fancy Indexing返回的数组的shape是索引的shape，所以这一步其实返回的是一个跟img相同shape的另一张图像。
所以，这一步的效果就是将图像按照查找表中的颜色进行了重新着色，即当img中的像素值为1时，那么就取lut中的(255, 0, 0)红色。

## blend()方法
blend()方法有很多种模式：
```python
def blend(img, out, msk, mode):
    if mode=='set': out[:] = img
    if mode=='min': np.minimum(out, img, out=out)
    if mode=='max': np.maximum(out, img, out=out)
    if mode=='msk':
        msk = np.logical_not(msk)
        out.T[:] *= msk.T
        out += img
    if isinstance(mode, float):
        np.multiply(out, 1-mode, out=out, casting='unsafe')
        np.multiply(img, mode, out=img, casting='unsafe')
        out += img
```
最终目的就是看按哪一种方式将img和out混合起来：
（1）set模式：直接将out里的内容设为img
（2）min模式：逐个对比out和img两张图像中的像素，选择数值小的像素存入out
（3）max模式：逐个对比out和img两张图像中的像素，选择数值大的像素存入out
（4）msk模式：在掩膜msk不为0的地方，将out设为0；在msk为0的地方，保留原值（因为对msk作了逻辑非操作，然后下面主元素相乘时，True为1，False为0）。然后将img和out进行相加合成。
（5）按比例混合模式：mode可以是一个小数，即将out与1-mode相乘，img与mode相乘，然后两者相加合成。

## mix_img()方法
```python
def mix_img(img, m, o, shp, buf, rgb, byt, rg=(0,255), lut=None, cns=0, mode='set'):
    if img is None: return
    img = img.reshape((img.shape[0], img.shape[1], -1))
    if isinstance(rg, tuple): rg = [rg]*img.shape[2]

    if isinstance(cns, int):
        affine_transform(img[:,:,cns], m, o, shp, buf, 0, 'nearest')
        stretch(buf, byt, rg[cns])
        return lookup(byt, lut, rgb, mode)

    irgb = [cns.index(i) if i in cns else -1 for i in 'rgb']
    for i,v in enumerate(irgb):
        if v==-1: rgb[:,:,i] = 0
        elif mode=='set' and buf.dtype==np.uint8 and rg[v]==(0,255):
            affine_transform(img[:,:,v], m, o, shp, rgb[:,:,v], 0, prefilter=False)
        else:
            affine_transform(img[:,:,v], m, o, shp, buf, 0, prefilter=False)
            stretch(buf, byt, rg[v])
            blend(byt, rgb[:,:,v], byt, mode)
```
mix_img()方法接收的参数特别多：
```python
def mix_img(img, m, o, shp, buf, rgb, byt, rg=(0,255), lut=None, cns=0, mode='set'):
```
第一个是img图像，从前面的调用可知，可以传入前景图或背景图；
第一个是m，旋转缩放矩阵；
第三个是shp，即最终要绘图的区域大小；
第四个是buf，其在显示某个通道（即单通道图像）时作为仿射变换后的图像；
第五个是rgb，其在显示RGB彩色图时作为仿射变换后的图像，在显示单通道图像时作为混合图像的背景；
第六个是byt，这张图像没分析出有什么作用。。；
第七个是rg，像素值范围，默认是(0, 255)；
第八个是lut，查找表；
第九个是cns，通道标识；
第十个是mode，混合模式。

具体看一下mix_img的操作：
```python
def mix_img(img, m, o, shp, buf, rgb, byt, rg=(0,255), lut=None, log=True, cns=0, mode='set'):
    if img is None: return
    img = img.reshape((img.shape[0], img.shape[1], -1))
```
首先是对img进行一下reshape，这里是将img的格式统一一下，将原来的两维或三维数组统一为三维数组。
```python
    if isinstance(rg, tuple): rg = [rg]*img.shape[2]
```
这里也是对范围range进行一下统一，如果rg为元组，则将其转化为与img的通道数相匹配的形式，比如原来rg是(0, 255)，若图像是三通道的，则经过转换后，rg变为：
```python
[(0, 255), (0, 255), (0, 255)]
```
接下来是对通道的处理：
如果cns是一个数，比如就选了一张彩色图中的一个通道显示，或者直接就是一张灰度图，那么：
```python
    if isinstance(cns, int):
        if np.iscomplexobj(buf):
            affine_transform(img[:,:,0].real, m, o, shp, buf.real, 0, prefilter=False)
            affine_transform(img[:,:,0].imag, m, o, shp, buf.imag, 0, prefilter=False)
            buf = complex_norm(buf, buf.real, buf.imag, buf.real)
        else:
            affine_transform(img[:,:,cns], m, o, shp, buf, 0, prefilter=False)
        stretch(buf, byt, rg[cns], log)
        return lookup(byt, lut, rgb, mode)
```
（1）首先判断一下buf是不是复数，这里应该是支持傅里叶变换的图像显示，暂且不分析；
（2）否则，就对该图像（灰度图或彩图的某一通道）进行仿射变换，输出到buf中，接着通过stretch转移到byt中，然后再调用lookup，根据上面的分析，lookup实际调用了blend，只是中间转了一下，输入到blend中的是lut(byt)，即将原图用lut转一下，然后再图像混合到rgb，即此种情况下rgb中的颜色是原图在查找表lut中的值。

如果通道标识cns是一个元组形式传入的，则：
```python
    for i,v in enumerate(cns):
        if v==-1: rgb[:,:,i] = 0
        elif mode=='set' and img.dtype==np.uint8 and rg[v]==(0,255) and not log:
            affine_transform(img[:,:,v], m, o, shp, rgb[:,:,i], 0, prefilter=False)
        else:
            affine_transform(img[:,:,v], m, o, shp, buf, 0, prefilter=False)
            stretch(buf, byt, rg[v], log)
            blend(byt, rgb[:,:,i], byt, mode)
```
对通道标识cns进行遍历，这里有多个判断条件：
（1）如果某个标识为-1，则将rgb中的相应通道置为0，
（2）否则，如果mode为set，且满足dtype、rg的相应条件，则进行仿射变换，注意此时改变的是rgb的相应通道，即将img的相应通道的值经过放射变换赋值给rgb的相应通道。这种情况就是用来显示RGB彩色图；
（3）若上述条件都不满足，则进行以下操作，这种情形就是用来显示彩色图的某些通道，但又与单独显示某通道不同：
（3.1）仿射变换：注意这里的output是buf，因为buf是单一通道，所以这里直接就是对其本身进行作用；
（3.2）stretch：目前这里的测试效果就是将buf的值同样赋值给byt，即byt也是img的仿射变换
（3.3）blend：这里是混合模式的设置，注意这里是将byt和rgb的相应通道进行混合，改变的是rgb的值，注意，后面画布绘制出的也是基于self.rgb。

经过上面对于通道标识cns的分析，就可以知道以下两者的不同：
```python
    image.cn = (0, -1, -1)
    image.cn = 0
```
假设图像是一张RGB彩色图，前者是将绿色通道和蓝色通道都置为0，红色通道照常处理，最终呈现的还是三者的综合；而后者是仅选了红色通道，即将红色通道单独摘出来作为一张灰度图显示。

mix_img()方法主要操作就是仿射变换和图像混合。关于仿射变换的具体细节见下面一小节。

以下面的代码为例，看看具体是怎么搞的：
```python
    image = Image()
    msk = np.zeros(astronaut().shape[:2], dtype=np.uint8)
    msk[100:200,100:200] = 1
    msk[200:300,200:300] = 2
    msk[300:400,300:400] = 3

    image.img = msk
 
    bak = Image([astronaut()])
    bak.cn = (0, 1, 2)

    image.back = bak
    image.mode = 'msk'

    canvas.images.append(image)
```
注意，前景图是一个有个别位置不为0的掩膜，背景图是宇航员图，前景图的混合模式是msk。

第一次调用“图像混合”方法时：
```python
            mix_img(back.img, m, o, shp, self.outbak,
                self.outrgb, self.outint, back.rg, back.lut,
                back.log, cns=back.cn, mode='set')
```
以宇航员astronaut()图像为例，传入的这些参数实际值是：
back.img的shape是(512, 512, 3)，
m是(4.0, 4.0)，这个值（包括下面的o和shp）不是固定的，与具体操作有关，这里不是绝对值，
o是(0.0, 0.0)，
shp是(128, 128)，
self.outbak对应形参中的buf，即缓冲，是shape为(128, 128)的全为0的数组，
self.outrgb对应形参中的rgb，即彩色图，是shape为(128, 128, 3)的全为0的数组，
self.outint对应形参中的byt，即灰度图，是shape为(128, 128)的全为0的数组，
back.rg是(0, 255)，即数值范围，
back.lut是查找表，这里默认的是shape为(256, 3)的数组，由以下语句构建：
```python
default_lut = np.arange(256*3, dtype=np.uint8).reshape((3,-1)).T
```
具体的值为：
```python
array([[  0,   0,   0],
       [  1,   1,   1],
       [  2,   2,   2],
       [  3,   3,   3],
…
       [ 254,  254,  254],
       [ 255,  255,  255],
```
back.log是False，
back.cn是通道标识，这里是(0, 1, 2)，
mode为'set'。
经过这一步的mix_img，改变的只有self.outrgb，其大小仍是(128, 128, 3)，但数值已经变成了img经过仿射变换后的数值。

第二次调用“图像混合”方法时：
```python
        mix_img(img.img, m, o, shp, self.outimg,
            self.outrgb, self.outint, img.rg, img.lut,
            img.log, cns=img.cn, mode=img.mode)
```
注意，这一次是基于img.img，即前景图，相应的buf、rg、lut、log也都是前景图的属性，值得注意的是这一次仍然调用了self.outrgb，它的值经过了前面背景图的处理，已经变成了背景图的仿射变换。
然后，这里因为是msk模式，先是将掩膜（再次强调掩膜是这里的前景图）仿射变换到buf，然后将buf和rgb进行混合，在掩膜不为0的地方，会将rgb的值置为掩膜的值，比如某些位置置为掩膜中的1，然后在显示时，就会去寻找lut中位置为1的颜色。

## 仿射变换
[使用python对2D坐标点进行仿射变换](https://sparkydogx.github.io/2018/09/03/affine-with-python/)
仿射变换就是对原图进行缩放、旋转、平移等操作，其示意图如下图（取自上面的文献）：
![](https://user-images.githubusercontent.com/6218739/69029568-54ff6680-0a10-11ea-879e-e1df857d0df2.jpg) 
可以通过下面的代码进行探究：
```python
import numpy as np
from scipy.ndimage import affine_transform
from skimage import io
from skimage.data import astronaut
 
img = astronaut()
matrix = (0.5, 0.1)
# theta = np.pi/4
# matrix = ((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta)))
offset = (0, 100)
shape = img.shape
output = np.zeros(shape, dtype=np.uint8)

for i in range(3):
    affine_transform(img[:, :, i], matrix, offset, output.shape[0:2], output[:,:,i])

io.imshow(output)
io.show()
```
对于图中的九种操作，可以按如下参数进行设定得到：
（1）no change:
```python
matrix = (1.0, 1.0)
offset = 0
```
（2）Translate:
```python
matrix = (1.0, 1.0)
offset = (100, 200)
# or offset = 100
```
这个地方可以单独为x、y设置偏移，也可以只设置一个数，表示x、y都偏移多少。注意，前面的是y偏移量。
（3）Scale about origin:
```python
matrix = (0.5, 0.1)
offset = 0
```
小于1为放大倍数，大于1则为缩小倍数。
（4）Rotate about origin:
```python
theta = np.pi/4
matrix = ((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta)))
offset = 0
```
theta就是旋转角度。
（5）Shear in x direction：
```python
phi = np.pi/6
matrix = ((1.0, 0.0), (np.tan(phi), 1.0))
offset = 0
```
（6）Shear in y direction:
```python
phi = np.pi/6
matrix = ((1.0, np.tan(phi)), (0.0, 1.0))
offset = 0
```
（7）Reflect about origin:
```python
matrix = (-1.0, -1.0)
offset = 0
```
实测这样操作后，因为还是显示的第一象限的数据，此时全是黑色。
其实可以通过：
```python
offset = img.shape[0:2]
```
将图像给挪到第一象限中，从而正确显示。
（8）Reflect about x-axis
```python
matrix = (1.0, -1.0)
```
同样，通过平移将它挪到第一象限中：
```python
offset = (0, img.shape[0])
```
（9）Reflect about y-axis
```python
matrix = (-1.0, 1.0)
```
同样，通过平移将它挪到第一象限中：
```python
offset = (img.shape[0], 0)
```
另外，上面代码中的shape是取的跟原有图像一样的shape，这里可以任意设定大小，那么就可以得到特定形状的输出图像，比如：
```python
shape = (100, 200, 3)
```
上述代码中img和output都是3通道，也可以output只有1通道，那么相应的输出shape也要改一下，即：
```python
shape = (512, 512)
output = np.zeros(shape, dtype=np.uint8)
for i in range(3):
    affine_transform(img[:, :, i], matrix, offset, output.shape[0:2], output)
```
此时虽然做了三次仿射变换，但只有img的第三个通道作用在了output上，它把前两个通道的作用给覆盖了。
即输入图像和输出图像可以不是一样的通道数，但affine_transform()的第四个参数shape和第五个参数output它俩一定要对应好。
