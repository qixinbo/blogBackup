---
title: ImageJ的Python脚本编程
tags: [ImageJ]
categories: programming
date: 2018-9-15
---

# 开篇说明
原生ImageJ仅支持JS脚本，而ImageJ的衍生版本Fiji支持Python脚本编程，所以这里的ImageJ实际是Fiji。
本文是对这个[Tutorial](http://www.ini.uzh.ch/~acardona/fiji-tutorial/)的翻译。
Fiji官方的Jython指南在[这里](https://imagej.net/Jython_Scripting)。
# 上手
有两种方式可以打开脚本编辑器：
- 通过File-New-Script打开。
- 使用Command finder：具体就是按字母“l”，然后输入script，然后选择下面的script。

打开编辑器后，选择Language为Python。

# 你的第一个Fiji脚本
首先随便打开一个图片。
## 获取打开的图片
在编辑器中输入以下代码：
```python
from ij import IJ

imp = IJ.getImage()
print imp
```
然后点击“Run”，或者使用快捷键“Ctrl+R”，这样程序就会运行，且在下方的窗口中打印出结果。
值得一提的是，imp是一个常用的名字，用来表示ImagePlus类的一个实例。ImagePlus是ImageJ表示一张图片的抽象，实际上在ImageJ中打开的每一个图像都是一个ImagePlus的实例。

注意：从下面可以看到，也可以通过IJ.openImage直接在脚本中获取某个未打开的图像。

## 通过一个文件对话框保存图像
这里在图像上的第一个操作是“保存”它。
在编辑器中增加以下代码：
```python
from ij.io import FileSaver
fs = FileSaver(imp)
fs.save()
```
这部分代码是导入FileSaver这个命名空间，然后创建一个参数为上面imp的FileSaver的实例，然后调用它的save函数。

## 将图像直接写入文件
编写脚本的一个目的是避免与人交互，因此这里直接将图像写入文件中：
```python
folder = "/home/qixinbo/temp/"
filepath = folder + "boats.tif"
fs.saveAsTiff(filepath)
```
这里指定了文件保存路径，然后调用FileSaver的saveAsTiff方法，同理还有saveAsPng、saveAsJpeg等。

## 保存文件时事先检查路径和文件
FileSaver将会覆盖给定路径下的文件，因此需要事先检查一下，防止误操作。
```python
folder = "/home/qixinbo/temp/"

from os import path

if path.exists(folder) and path.isdir(folder):
	print "folder exists:", folder
	filepath = path.join(folder, "boats.tif")
	if path.exists(filepath):
		print "File exists! Not saving the image, would overwrite a file!"
	elif fs.saveAsTiff(filepath):
		print "File saved successfully at ", filepath	
else:
	print "Folder does not exist or it's not a folder!"
```
首先检查上面的folder是否存在及它是否是一个路径；然后检查该路径下是否有与要写入的文件有相同名字的文件，如果有，则提示，且不覆盖；最后检查FileSaver的saveAsTiff函数是否工作正常，如果该方法工作正常，将会返回true，因此这里会打印成功提示，否则则报错。

# 探究图像的属性和像素
如前所述，ImageJ或Fiji中的一张图像实际是ImagePlus类的一个实例。

## 输出图片基本信息
```python
from ij import IJ, ImagePlus

imp = IJ.getImage()

print "title:", imp.title
print "width:", imp.width
print "height:", imp.height
print "number of pixels:", imp.width*imp.height
print "number of slices:", imp.getNSlices()
print "number of channels:", imp.getNChannels()
print "number of time frames:", imp.getNFrames()

types = {ImagePlus.COLOR_RGB: "RGB",
	ImagePlus.GRAY8: "8-bit",
	ImagePlus.GRAY16: "16-bit",
	ImagePlus.GRAY32: "32-bit",
	ImagePlus.COLOR_256: "8-bit color"}

print "image type:", types[imp.type]
```
ImagePlus包含诸如图像标题、尺寸（宽度、高度、slices数目、时间帧数目、通道数目），以及像素（被包装在ImageProcessor实例中）。这些数据都是ImagePlus的field，可以通过点号来获取，如img.title就能得到图像标题，如果没有发现某些fields，可以通过get方法得到，如getNChannels得到通道数。
对于image的type，上述代码是定义了一个types字典，这样就通过键值对的方式更加直观地显示image的type。

## 获得图像的像素统计
ImageJ/Fiji提供了ImageStatistics类做像素统计，其中又使用了这个类的getStatistics的静态方法。
```python
from ij import IJ
from ij.process import ImageStatistics as IS

imp = IJ.getImage()

ip = imp.getProcessor()

options = IS.MEAN | IS.MEDIAN | IS.MIN_MAX
stats = IS.getStatistics(ip, options, imp.getCalibration())

print "Image statistics for", imp.title
print "Mean:", stats.mean
print "Median:", stats.median
print "Min and max:", stats.min, "-", stats.max
```
注意，用getProcessor得到像素信息。如果图像是stacks，那么使用StackStatistics。

如果是很多图片，该怎样做统计？可以像下面这样写一个函数：
```python

from ij import IJ
from ij.process import ImageStatistics as IS
import os

options = IS.MEAN | IS.MEDIAN | IS.MIN_MAX

def getStatistics(imp):
	""" Return statistics for the given ImagePlus """
	global options
	ip = imp.getProcessor()	
	stats = IS.getStatistics(ip, options, imp.getCalibration())	
	return stats.mean, stats.median, stats.min, stats.max


folder = "/home/qixinbo/temp/"

for filename in os.listdir(folder):
	if filename.endswith(".png"):
		print "Processing", filename
		imp = IJ.openImage(os.path.join(folder, filename))
		if imp is None:
			print "Could not open image from file:", filename
			continue
		mean, median, min, max = getStatistics(imp)
		print "Image statistics for", imp.title
		print "Mean:", mean
		print "Median:", median
		print "Min and max:", min, "-", max
	else:
		print "Ignoring", filename
```

## 像素迭代
像素迭代是一个低层操作，很少用到。但如果真需要用到，下面提供了三种迭代方式：
```python
from java.lang import Float
from ij import IJ

imp = IJ.getImage()

ip = imp.getProcessor().convertToFloat()
pixels = ip.getPixels()

print "Image is", imp.title, "of type", imp.type

# Method 1: the for loop, C style
minimum = Float.MAX_VALUE
for i in xrange(len(pixels)):
	if pixels[i] < minimum:
		minimum = pixels[i]
print "1. Minimum is:", minimum

# Method 2: iterate pixels as a list
minimum = Float.MAX_VALUE
for pix in pixels:
	if pix < minimum:
		minimum = pix
print "2. Minimum is:", minimum

# Method 3: apply the built-in min function
# to the first pair of pixels,
# and then to the result of that and the next pixel, etc.
minimum = reduce(min, pixels)
print "3. Minium is:", minimum
```
对于上面的pixels，根据不同类型的图像，可以有不同的形式，具体可以打印出来看一下。对于RGB图像，可以调用getBrightness来看看哪个像素最亮，可以通过ip.toFloat(0,None)来获得第一个通道。
上面使用了三种风格来处理像素列表，一种是C风格的形式，第二种是列表形式，第三种是函数式编程风格。

## 对列表或集合的迭代或循环操作
对于上面三种处理列表的风格，前两种都使用了for循环来进行迭代，好处是特别容易理解，坏处是性能上会有点弱，且不简洁；第三种采用reduce这种函数式编程的风格，其他还有map、filter等操作，好处是性能好且形式简洁，但坏处是较难理解。
下面通过几个例子更详细地介绍这几种风格的对比。
### map操作
map操作接收一个长度为N的列表，然后施加一个函数，返回一个长度也为N的列表。比如，你想得到Fiji中打开的一系列的图片，如果用for循环，需要先显式创建一个列表，然后一个一个地把图片附加进去。
如果使用python的列表解析的语法糖，列表可以直接创建，然后在中括号里直接写上for循环的逻辑。因此，本质上它仍然是个序列化的循环，无法并行操作。
通过使用map，则可直接在ID列表中对每个ID进行WM.getImage作用。
```python
from ij import WindowManager as WM

# Method 1: with a for loop
images = []
for id in WM.getIDList():
	images.append(WM.getImage(id))

# Method 2: with list comprehension
images = [WM.getImage(id) for id in WM.getIDList()]

# Method 3: with a map openration
images = map(WM.getImage, WM.getIDList())
```

### filter操作
filter操作接收一个长度为N的列表，然后返回一个更短的列表，只有通过特定条件的元素才能进入到新列表中。比如下面这个列子就是找到Fiji打开的图片中标题中有特定字符串的图片。
```python
from ij import WindowManager as WM

imps = map(WM.getImage, WM.getIDList())

def match(imp):
	""" Returns true if the image title contains the word 'boat' """
	return imp.title.find("boat") > -1

# Method 1: with a for loop
matching = []
for imp in imps:
	if match(imp):
		matching.append(imp)

# Method 2: with list comprehension
matching = [imp for imp in imps if match(imp)]

# Method 3: with a filter operation
matching = filter(match, imps)
```

### reduce操作
reduce操作接收一个长度为N的列表，然后返回一个单值。比如你想找到Fiji当前打开的图片中面积最大的那个。如果使用for循环，需要设置临时变量，以及判断列表中是否有至少一个元素等，而使用reduce则不需要临时变量，只需要一个辅助函数（其实甚至这里只需要一个匿名lambda函数，但这里显式定义，从而更好理解及可用性更好）。
```python
from ij import IJ
from ij import WindowManager as WM

imps = map(WM.getImage, WM.getIDList())

def area(imp):
	return imp.width * imp.height

# Method 1: with a for loop
largest = None
largestArea = 0
for imp in imps:
	if largest is None:
		largest = imp
	else:
		a = area(imp)
		if a > largestArea:
			largest = imp
			largestArea = a

# Method 2: with a reduce operation
def largestImage(imp1, imp2):
	return imp1 if area(imp1) > area(imp2) else imp2

largest = reduce(largestImage, imps)
```

## 对每一个像素减去最小值
首先使用上面的reduce方法得到最小的像素值，然后使用下面两种方法对每个像素减去该最小值：
```python
from ij import IJ, ImagePlus
from ij.process import FloatProcessor

imp = IJ.getImage()
ip = imp.getProcessor().convertToFloat()
pixels = ip.getPixels()

minimum = reduce(min, pixels)

# Method 1: subtract the minimum from every pixel
# in place, modifying the pixels array
for i in xrange(len(pixels)):
	pixels[i] -= minimum

imp2 = ImagePlus(imp.title, ip)
imp2.show()

# Method 2: subtract the minimum from every pixel
# and store the result in a new array
pixels3= map(lambda x: x-minimum, pixels)
ip3 = FloatProcessor(ip.width, ip.height, pixels3, None)
imp3 = ImagePlus(imp.title, ip3)
imp3.show()
```

## 将像素列表概括为单值
### 统计在阈值以上的像素个数
本例将计算有多少像素在特定阈值之上，使用reduce函数。
```python
from ij import IJ

imp = IJ.getImage()
ip = imp.getProcessor().convertToFloat()
pixels = ip.getPixels()

mean = sum(pixels) / len(pixels)

n_pix_above = reduce(lambda count, a: count+1 if a > mean else count, pixels, 0)

print "Mean value", mean
print "% pixels above mean:", n_pix_above / float(len(pixels)) * 100
```
### 统计在阈值以上的所有像素的坐标，并计算重心
所有像素值在阈值（这里是均值）以上的坐标点都通过filter函数收集在一个列表中。注意，在ImageJ中，一张图像的所有像素都存储在一个线性数组中，数组的长度是width与height的乘积。因此，某个像素的索引除以width的余数（即模）是该像素的X坐标，索引除以width的商是该像素的Y坐标。
同时，给出了四种计算中心的方法，第一种是使用for循环，第二种是使用map和sum，第三种是使用reduce，第四种是对第三种的简化，使用了functools包中的partial函数。
```python
from ij import IJ

imp = IJ.getImage()
ip = imp.getProcessor().convertToFloat()
pixels = ip.getPixels()

mean = sum(pixels) / len(pixels)

above = filter(lambda i: pixels[i] > mean, xrange(len(pixels)))

print "Number of pixels above mean value:", len(above)

width = imp.width

# Method 1: with a for loop
xc = 0
yc = 0
for i in above:
	xc += i % width
	yc += i / width
xc = xc / len(above)
yc = yc / len(above)
print xc, yc

# Method 2: with sum and map
xc = sum(map(lambda i: i % width, above)) / len(above)
yc = sum(map(lambda i: i / width, above)) / len(above)
print xc, yc

# Method 3: iterating the list "above" just once
xc, yc = [d / len(above) for d in 
		reduce(lambda c, i: [c[0] + i%width, c[1] + i/width], above, [0,0])]
print xc, yc

# Method 4: iterating the list "above" just once, more clearly and performant
from functools import partial

def accum(width, c, i):
	c[0] += i % width
	c[1] += i / width
	return c
xc, yc = [d/len(above) for d in reduce(partial(accum, width), above, [0,0])]
print xc, yc
```

# 运行ImageJ/Fiji插件
下面是一个对当前图片施加中值滤波的例子：
```python
from ij import IJ, ImagePlus
from ij.plugin.filter import RankFilters

imp = IJ.getImage()
ip = imp.getProcessor().convertToFloat()

radius = 2
RankFilters().rank(ip, radius, RankFilters.MEDIAN)

imp2 = ImagePlus(imp.title + " median filtered", ip)
imp2.show()
```
## 查找哪个类用了哪个插件
现在的问题是怎样知道上面的中值滤波在RankFilters中。可以通过Command Finder来查找。
## 查找插件所需要的参数
可以通过Macro Recorder来查看一个插件必要的参数。
- 打开Plugins-Macros-Record
- 运行命令，比如上面的Process-Filters-Median。此时会出现一个对话框，要求输入半径，点击OK
- 查看Recorder窗口中的显示：run("Median...", "radius=2");

这说明中值滤波的一个必要参数就是radius。
## 运行命令command
上面的插件plugin其实可以通过刚才录制的宏命令来替代：
```python
IJ.run(imp, "Median...", "radius=2")
```

# 创建图像及其ROI
## 从零开始创建一张图像
一张ImageJ/Fiji图像至少包含三个部分：
- 像素数组：一个含有原始类型数据的数组，数据类型可以是byte、short、int或float
- ImageProcessor子类实例：用来承载像素数组
- ImagePlus实例：用来承载ImageProcessor实例

下面的例子中创建了一个空的float型数组，然后用随机的浮点数填充，然后把它传给FloatProcessor，最后传给ImagePlus：
```python
from ij import ImagePlus
from ij.process import FloatProcessor
from array import zeros
from random import random

width = 1024
height = 1024
pixels = zeros('f', width * height)

for i in xrange(len(pixels)):
	pixels[i] = random()

fp = FloatProcessor(width, height, pixels, None)
imp = ImagePlus("White noise", fp)

imp.show()
```
## 用给定值填充一个ROI
为了填充一个ROI，我们可以对像素进行迭代，知道在ROI内部的像素，然后将它们的值设为指定值。但这个过程是繁杂且容易出错。更有效的方式是创建一个Roi类的实例或它的子类PolygonRoi、OvalRoi、ShapeRoi等的实例，然后告诉ImageProcessor来填充这个区域。
下面的例子使用了上面创建的白噪声图像，然后在其中定义了一个矩形区域，然后用2填充。
```python
from ij.gui import Roi, PolygonRoi

roi = Roi(400, 200, 400, 300)
fp.setRoi(roi)
fp.setValue(2.0)
fp.fill()

imp.show()
```
因为原来的白噪声的值是0到1的随机数，现在加上的值是2，是新的最大值，所以看起来是白色的。
在上面的代码基础上再增加一个新的多边形区域，用-3填充：
```python
xs = [234, 174, 162, 102, 120, 123, 153, 177, 171,  
      60, 0, 18, 63, 132, 84, 129, 69, 174, 150,  
      183, 207, 198, 303, 231, 258, 234, 276, 327,  
      378, 312, 228, 225, 246, 282, 261, 252]  
ys = [48, 0, 60, 18, 78, 156, 201, 213, 270, 279,  
      336, 405, 345, 348, 483, 615, 654, 639, 495,  
      444, 480, 648, 651, 609, 456, 327, 330, 432,  
      408, 273, 273, 204, 189, 126, 57, 6]
proi = PolygonRoi(xs, ys, len(xs), Roi.POLYGON)
fp.setRoi(proi)
fp.setValue(-3)
fp.fill(proi.getMask()) # Attention here!

imp.show()
```

# 创建和操作图像的stacks和hyperstacks
## 加载一张多彩图像stack并提取绿色通道
```python
from ij import IJ, ImagePlus, ImageStack

imp = IJ.openImage("/home/qixinbo/temp/flybrain.zip")
stack = imp.getImageStack()

print "Number of slices:", imp.getNSlices()

greens = []
for i in xrange(1, imp.getNSlices()+1):
	cp = stack.getProcessor(i) # Get the ColorProcessor slice at index i
	fp = cp.toFloat(1, None) # Get its green channel as a FloatProcessor
	greens.append(fp) # store it in a list

stack2 = ImageStack(imp.width, imp.height) # Create a new stack with only the green channel
for fp in greens:
	stack2.addSlice(None, fp)

imp2 = ImagePlus("Green channel", stack2) # Create a new image with stack of green channel slices
IJ.run(imp2, "Green", "") # Set a green look-up table
imp2.show()
```
首先加载名为"Fly Brain"的样例。然后对它的slices进行迭代。每个slices都是一个ColorProcessor，它有一个toFloat的方法，可以得到一个特定颜色通道的FloatProcessor。将颜色通道用浮点数表示是最方便的处理像素值的方法，它不像字节类型那样会溢出。最后一个创建绿色的LUT命令，可以通过录制宏命令的形式得到具体的写法。如果不加这个绿色的LUT，整个图像虽然是绿色通道的像素值，但是以灰度图呈现的。
## 将一个RGB stack转换为双通道的32-bit hyperstack
```python
from ij import IJ, ImagePlus, ImageStack, CompositeImage

imp = IJ.openImage("/home/qixinbo/temp/flybrain.zip")
stack = imp.getImageStack()

stack2 = ImageStack(imp.width, imp.height) # Create a new stack with only the green channel

for i in xrange(1, imp.getNSlices()+1):
	cp = stack.getProcessor(i) # Get the ColorProcessor slice at index i
	red = cp.toFloat(0, None) # Get the red channel as a FloatProcessor
	green = cp.toFloat(1, None) # Get its green channel as a FloatProcessor
	stack2.addSlice(None, red)
	stack2.addSlice(None, green)

imp2 = ImagePlus("32-bit 2-channel composite", stack2)
imp2.setCalibration(imp.getCalibration().copy())

nChannels = 2
nSlices = stack.getSize()
nFrames = 1
imp2.setDimensions(nChannels, nSlices, nFrames)
comp = CompositeImage(imp2, CompositeImage.COLOR)
comp.show()
```
其中比较重要的是告诉hyperstack怎样解释它的图像，即setDimensions中的参数，即有多少个通道、多少个slices和多少个时间帧。

# 人机交互：文件和选项对话框、消息、进度条
## 询问打开文件夹
```python
from ij.io import DirectoryChooser

dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()

if folder is None:
	print "User cancelled the dialog!"
else:
	print "Selected folder:", folder
```
## 询问打开文件
```python
from ij.io import OpenDialog

od = OpenDialog("Choose a file", None)
filename = od.getFileName()

if filename is None:
	print "User canceled the dialog!"
else:
	directory = od.getDirectory()
	filepath = directory + filename
	print "Selected file path:", filepath
```
## 询问输入参数
```python
from ij.gui import GenericDialog

def getOptions():
	gd = GenericDialog("Options")
	gd.addStringField("name", "Untitled")
	gd.addNumericField("alpha", 0.25, 2)
	gd.addCheckbox("Optimize", True)
	types = ["8-bit", "16-bit", "32-bit"]
	gd.addChoice("Output as", types, types[2])
	gd.addSlider("Scale", 1, 100, 100)
	gd.showDialog()

	if gd.wasCanceled():
		print "User canceled dialog!"
		return

	name = gd.getNextString()
	alpha = gd.getNextNumber()
	optimize = gd.getNextBoolean()
	output = gd.getNextChoice()
	scale = gd.getNextNumber()
	return name, alpha, optimize, output, scale

options = getOptions()
if options is not None:
	name, alpha, optimize, output, scale = options
	print name, alpha, optimize, output, scale
```

## 显示进度条
```python
from ij import IJ

imp = IJ.getImage()
stack = imp.getImageStack()

for i in xrange(1, stack.getSize()+1):
	IJ.showProgress(i, stack.getSize()+1)
	ip = stack.getProcessor(i)

IJ.showProgress(1)
```

# 将脚本转换为插件
将脚本存储在Fiji插件文件夹或其子文件夹下，注意：
- 文件名中要有一个下划线
- 后缀是.py

比如"my\_script.py"。然后运行“Help-Update Menus”，或者重启Fiji。
脚本就会作为"Plugins"中的菜单命令出现，也可以使用Command Finder找到。
插件文件夹在哪呢：对于Ubuntu和Windows系统，在Fiji.app文件夹中。

# 列表、原生数组及与Java类的交互
## Jython列表以只读数组的形式传给Java类
```python
from java.awt.geom import AffineTransform
from array import array

x = 10
y = 40

aff = AffineTransform(1, 0, 0, 1, 45, 56)

p = [x, y]
aff.transform(p, 0, p, 0, 1)
print p

q = array('f', [x, y])
aff.transform(q, 0, q, 0, 1)
print q
```
上面的p没有被更新，而q被更新了，所以Jython列表只是可读，而array类型可写。

## 创建原生数组
array包有两个函数：
- zeros：创建空的原生数组
- array：从列表或另一个相同类型的array中创建数组

array的第一个参数是type。对于原生类型，如char、short、int、float、long和double，可以使用带引号的单个字母。下面的例子中创建了ImagePlus类型的array。
```python
from ij import IJ
from array import array, zeros
from ij import ImagePlus

a = zeros('f', 5)
print a

b = array('f', [0, 1, 2, 3, 4])
print b

imps = zeros(ImagePlus, 5) # An empty native ImagePlus array of length 5
print imps

imps[0] = IJ.getImage() # Assign the current image to the first element of the array
print imps

print "length:", len(imps)
```

# 可作用于任意类型图像的通用算法：ImgLib库
Imglib是一个处理n维数据的通用库，主要是面向图像处理。用Imglib编程能够极大地简化在不同类型图片上的操作。
## 对图片进行数学操作
主要有以下几类的操作：
- script.imglib.math：提供操作每一个像素的函数。这些函数是可组合的，即某个函数的输出可作为另一个的输入。
- script.imglib.color：提供创建和操纵彩色图片的函数，比如在RGB色彩空间中提取特定的色彩通道。
- script.imglib.algorithm：提供诸如Gauss、Scale3D、Affine3D、Resample、Downsample等函数，一次对多个像素进行操作，而不是逐个像素的操作。一些函数可能改变图像的尺寸。这些算法全部都返回图像，换句话说，是对原始图像施加这些算法后得到的新图像。
- script.imglib.analysis：提供提取或测量图像或者评价图像的函数。

```python
from script.imglib.math import Compute, Subtract
from script.imglib.color import Red, Green, Blue, RGBA
from script.imglib import ImgLib
from ij import IJ

imp = IJ.openImage("/home/qixinbo/temp/flybrain.zip")

img = ImgLib.wrap(imp) # Wrap it as an Imglib image

sub = Compute.inFloats(Subtract(Green(img), Red(img)))

ImgLib.wrap(sub).show()

rgb = RGBA(Red(img), Subtract(Green(img), Red(img)), Blue(img)).asImage()

ImgLib.wrap(rgb).show()
```
## 使用图像数学进行平场校正
```python
from script.imglib.math import Compute, Subtract, Divide, Multiply
from script.imglib.algorithm import Gauss, Scale2D, Resample
from script.imglib import ImgLib
from ij import IJ

img = ImgLib.wrap(IJ.openImage("/home/qixinbo/temp/bridge.gif"))

brightfield = Resample(Gauss(Scale2D(img, 0.25), 20), img.getDimensions())  
  
# 3. Simulate a perfect darkfield  
darkfield = 0  
  
# 4. Compute the mean pixel intensity value of the image  
mean = reduce(lambda s, t: s + t.get(), img, 0) / img.size()  
  
# 5. Correct the illumination  
corrected = Compute.inFloats(Multiply(Divide(Subtract(img, brightfield),  
                                             Subtract(brightfield, darkfield)), mean))  
  
# 6. ... and show it in ImageJ  
ImgLib.wrap(corrected).show() 
```

## 提取和操纵图像色彩通道：RGBA和HSB
下面的代码是对RGBA色彩空间的操作：
```python
from script.imglib.math import Compute, Subtract, Multiply  
from script.imglib.color import Red, Blue, RGBA  
from script.imglib.algorithm import Gauss, Dither
from script.imglib import ImgLib
from ij import IJ  
  
# Obtain a color image from the ImageJ samples    
clown = ImgLib.wrap(IJ.openImage("https://imagej.nih.gov/ij/images/clown.jpg"))  
    
# Example 1: compose a new image manipulating the color channels of the clown image:    
img = RGBA(Gauss(Red(clown), 10), 40, Multiply(255, Dither(Blue(clown)))).asImage()    
    
ImgLib.wrap(img).show()
```
下面的代码是对HSB色彩空间的操作：
```python
from script.imglib.math import Compute, Add, Subtract  
from script.imglib.color import HSB, Hue, Saturation, Brightness  
from script.imglib import ImgLib  
from ij import IJ  
  
# Obtain an image  
img = ImgLib.wrap(IJ.openImage("https://imagej.nih.gov/ij/images/clown.jpg"))  
  
# Obtain a new clown, whose hue has been shifted by half  
# with the same saturation and brightness of the original  
bluey = Compute.inRGBA(HSB(Add(Hue(img), 0.5), Saturation(img), Brightness(img)))  
  
ImgLib.wrap(bluey).show()  
```
下面的代码是对一个RGB confocal stack进行伽马校正：
```python
# Correct gamma  
from script.imglib.math import Min, Max, Exp, Multiply, Divide, Log  
from script.imglib.color import RGBA, Red, Green, Blue  
from script.imglib import ImgLib
from ij import IJ  
  
gamma = 0.5  
img = ImgLib.wrap(IJ.openImage("/home/qixinbo/temp/flybrain.zip"))  
  
def g(channel, gamma):  
  """ Return a function that, when evaluated, computes the gamma 
        of the given color channel. 
      If 'i' was the pixel value, then this function would do: 
      double v = Math.exp(Math.log(i/255.0) * gamma) * 255.0); 
      if (v < 0) v = 0; 
      if (v >255) v = 255; 
  """  
  return Min(255, Max(0, Multiply(Exp(Multiply(gamma, Log(Divide(channel, 255)))), 255)))  
  
corrected = RGBA(g(Red(img), gamma), g(Green(img), gamma), g(Blue(img), gamma)).asImage()  
  
ImgLib.wrap(corrected).show()
```
## 通过高斯差寻找、计数和显示三维stack中的cell
```python
# Load an image of the Drosophila larval fly brain and segment  
# the 5-micron diameter cells present in the red channel.  
  
from script.imglib.analysis import DoGPeaks  
from script.imglib.color import Red  
from script.imglib.algorithm import Scale2D  
from script.imglib.math import Compute  
from script.imglib import ImgLib  
from ij3d import Image3DUniverse  
from javax.vecmath import Color3f, Point3f  
from ij import IJ  
  
cell_diameter = 5  # in microns  
minPeak = 40 # The minimum intensity for a peak to be considered so.  
imp = IJ.openImage("http://samples.fiji.sc/first-instar-brain.zip")  
  
# Scale the X,Y axis down to isotropy with the Z axis  
cal = imp.getCalibration()  
scale2D = cal.pixelWidth / cal.pixelDepth  
iso = Compute.inFloats(Scale2D(Red(ImgLib.wrap(imp)), scale2D))  
  
# Find peaks by difference of Gaussian  
sigma = (cell_diameter  / cal.pixelWidth) * scale2D  
peaks = DoGPeaks(iso, sigma, sigma * 0.5, minPeak, 1)  
print "Found", len(peaks), "peaks"  
  
# Convert the peaks into points in calibrated image space  
ps = []  
for peak in peaks:  
  p = Point3f(peak)  
  p.scale(cal.pixelWidth * 1/scale2D)  
  ps.append(p)  
  
# Show the peaks as spheres in 3D, along with orthoslices:  
univ = Image3DUniverse(512, 512)  
univ.addIcospheres(ps, Color3f(1, 0, 0), 2, cell_diameter/2, "Cells").setLocked(True)  
univ.addOrthoslice(imp).setLocked(True)  
univ.show()  
```

# ImgLib2：编写通用且高性能的图像处理程序
ImgLib2是一个非常强大的图像处理库：
- 它的介绍在[这篇论文](https://academic.oup.com/bioinformatics/article/28/22/3009/240540)中
- GitHub库在[这里](https://github.com/imglib/imglib2)
- ImageJ官网上关于ImgLib2的[教学例子](https://imagej.net/ImgLib2_Examples)
 
## ImgLib2对于图像的Views
```python
from ij import IJ  
from net.imglib2.img.display.imagej import ImageJFunctions as IL  
from net.imglib2.view import Views  
  
# Load an image (of any dimensions) such as the clown sample image  
imp = IJ.getImage()  
  
# Convert to 8-bit if it isn't yet, using macros  
IJ.run(imp, "8-bit", "")  
  
# Access its pixel data from an ImgLib2 RandomAccessibleInterval  
img = IL.wrapReal(imp)  
  
# View as an infinite image, with a value of zero beyond the image edges  
imgE = Views.extendZero(img)  
  
# Limit the infinite image with an interval twice as large as the original,  
# so that the original image remains at the center.  
# It starts at minus half the image width, and ends at 1.5x the image width.  
minC = [int(-0.5 * img.dimension(i)) for i in range(img.numDimensions())]  
maxC = [int( 1.5 * img.dimension(i)) for i in range(img.numDimensions())]  
imgL = Views.interval(imgE, minC, maxC)  
  
# Visualize the enlarged canvas, so to speak  
imp2 = IL.wrap(imgL, imp.getTitle() + " - enlarged canvas") # an ImagePlus  
imp2.show()
```
对上面的图片的边缘进行填充：
```python
imgE = Views.extendMirrorSingle(img)  
imgL = Views.interval(imgE, minC, maxC)  
  
# Visualize the enlarged canvas, so to speak  
imp2 = IL.wrap(imgL, imp.getTitle() + " - enlarged canvas") # an ImagePlus  
imp2.show()  
```

## ImgLib2的高斯差分检测
```cpp
from ij import IJ  
from ij.gui import PointRoi  
from ij.measure import ResultsTable  
from net.imglib2.img.display.imagej import ImageJFunctions as IL  
from net.imglib2.view import Views  
from net.imglib2.algorithm.dog import DogDetection  
from jarray import zeros  
  
# Load a greyscale single-channel image: the "Embryos" sample image  
imp = IJ.openImage("https://imagej.nih.gov/ij/images/embryos.jpg")  
# Convert it to 8-bit  
IJ.run(imp, "8-bit", "")  
  
# Access its pixel data from an ImgLib2 data structure: a RandomAccessibleInterval  
img = IL.wrapReal(imp)  
  
# View as an infinite image, mirrored at the edges which is ideal for Gaussians  
imgE = Views.extendMirrorSingle(img)  
  
# Parameters for a Difference of Gaussian to detect embryo positions  
calibration = [1.0 for i in range(img.numDimensions())] # no calibration: identity  
sigmaSmaller = 15 # in pixels: a quarter of the radius of an embryo  
sigmaLarger = 30  # pixels: half the radius of an embryo  
extremaType = DogDetection.ExtremaType.MAXIMA  
minPeakValue = 10  
normalizedMinPeakValue = False  
  
# In the differece of gaussian peak detection, the img acts as the interval  
# within which to look for peaks. The processing is done on the infinite imgE.  
dog = DogDetection(imgE, img, calibration, sigmaSmaller, sigmaLarger,  
  extremaType, minPeakValue, normalizedMinPeakValue)  
  
peaks = dog.getPeaks()  
  
# Create a PointRoi from the DoG peaks, for visualization  
roi = PointRoi(0, 0)  
# A temporary array of integers, one per dimension the image has  
p = zeros(img.numDimensions(), 'i')  
# Load every peak as a point in the PointRoi  
for peak in peaks:  
  # Read peak coordinates into an array of integers  
  peak.localize(p)  
  roi.addPoint(imp, p[0], p[1])  
  
imp.setRoi(roi)  
  
# Now, iterate each peak, defining a small interval centered at each peak,  
# and measure the sum of total pixel intensity,  
# and display the results in an ImageJ ResultTable.  
table = ResultsTable()  
  
for peak in peaks:  
  # Read peak coordinates into an array of integers  
  peak.localize(p)  
  # Define limits of the interval around the peak:  
  # (sigmaSmaller is half the radius of the embryo)  
  minC = [p[i] - sigmaSmaller for i in range(img.numDimensions())]  
  maxC = [p[i] + sigmaSmaller for i in range(img.numDimensions())]  
  # View the interval around the peak, as a flat iterable (like an array)  
  fov = Views.interval(img, minC, maxC)  
  # Compute sum of pixel intensity values of the interval  
  # (The t is the Type that mediates access to the pixels, via its get* methods)  
  s = sum(t.getInteger() for t in fov)  
  # Add to results table  
  table.incrementCounter()  
  table.addValue("x", p[0])  
  table.addValue("y", p[1])  
  table.addValue("sum", s)  
  
table.show("Embryo intensities at peaks")  
```
此时会得到一个table。有两种方法来将table中的data存入一个CSV文件中：
第一种是直接在Table窗口中，点击File-Save，这是最简单的方式了。
第二种是使用python内置的csv库，在上面的代码中加入以下代码：
```python
from __future__ import with_statement  
# IMPORTANT: imports from __future__ must go at the top of the file.  
  
#  
# ... same code as above here to obtain the peaks  
#  
  
from operator import add  
import csv  
  
# The minumum and maximum coordinates, for each image dimension,  
# defining an interval within which pixel values will be summed.  
minC = [-sigmaSmaller for i in xrange(img.numDimensions())]  
maxC = [ sigmaSmaller for i in xrange(img.numDimensions())]  
  
def centerAt(p, minC, maxC):  
  """ Translate the minC, maxC coordinate bounds to the peak. """  
  return map(add, p, minC), map(add, p, maxC)  
  
def peakData(peaks, p, minC, maxC):  
  """ A generator function that returns all peaks and their pixel sum, 
      one at a time. """  
  for peak in peaks:  
    peak.localize(p)  
    minCoords, maxCoords = centerAt(p, minC, maxC)  
    fov = Views.interval(img, minCoords, maxCoords)  
    s = sum(t.getInteger() for t in fov)  
    yield p, s  
  
# Save as CSV file  
with open('/tmp/peaks.csv', 'wb') as csvfile:  
  w = csv.writer(csvfile, delimiter=',', quotechar='"''"', quoting=csv.QUOTE_NONNUMERIC) 
  w.writerow(['x', 'y', 'sum']) 
  for p, s in peakData(peaks, p, minC, maxC): 
    w.writerow([p[0], p[1], s]) 
 
# Read the CSV file into an ROI 
roi = PointRoi(0, 0) 
with open('/tmp/peaks.csv', 'r') as csvfile: 
  reader = csv.reader(csvfile, delimiter=',', quotechar='"')  
  header = reader.next() # advance reader by one line  
  for x, y, s in reader:  
    roi.addPoint(imp, float(x), float(y))  
  
imp.show()  
imp.setRoi(roi)  
```
## 使用ImgLib2进行图像变换
以下是使用ImgLib2进行图像变换，如平移、旋转、缩放等。
```python
from net.imglib2.realtransform import RealViews as RV  
from net.imglib2.img.display.imagej import ImageJFunctions as IL  
from net.imglib2.realtransform import Scale  
from net.imglib2.view import Views  
from net.imglib2.interpolation.randomaccess import NLinearInterpolatorFactory  
from ij import IJ  
  
# Load an image (of any dimensions)  
imp = IJ.getImage()  
  
# Access its pixel data as an ImgLib2 RandomAccessibleInterval  
img = IL.wrapReal(imp)  
  
# View as an infinite image, with a value of zero beyond the image edges  
imgE = Views.extendZero(img)  
  
# View the pixel data as a RealRandomAccessible  
# (that is, accessible with sub-pixel precision)  
# by using an interpolator  
imgR = Views.interpolate(imgE, NLinearInterpolatorFactory())  
  
# Obtain a view of the 2D image twice as big  
s = [2.0 for d in range(img.numDimensions())] # as many 2.0 as image dimensions  
bigger = RV.transform(imgR, Scale(s))  
  
# Define the interval we want to see: the original image, enlarged by 2X  
# E.g. from 0 to 2*width, from 0 to 2*height, etc. for every dimension  
minC = [0 for d in range(img.numDimensions())]  
maxC = [int(img.dimension(i) * scale) for i, scale in enumerate(s)]  
imgI = Views.interval(bigger, minC, maxC)  
  
# Visualize the bigger view  
imp2x = IL.wrap(imgI, imp.getTitle() + " - 2X") # an ImagePlus  
imp2x.show()  
```

## 使用ImgLib2旋转图片
```python
from net.imglib2.realtransform import RealViews as RV  
from net.imglib2.realtransform import AffineTransform  
from net.imglib2.img.display.imagej import ImageJFunctions as IL  
from ij import IJ  
from net.imglib2.view import Views  
from net.imglib2.interpolation.randomaccess import NLinearInterpolatorFactory  
from java.awt.geom import AffineTransform as Affine2D  
from java.awt import Rectangle  
from Jama import Matrix  
from math import radians  
  
# Load an image (of any dimensions)  
imp = IJ.getImage()  
  
# Access its pixel data as an ImgLib2 RandomAccessibleInterval  
img = IL.wrapReal(imp)  
  
# View as an infinite image, with value zero beyond the image edges  
imgE = Views.extendZero(img)  
  
# View the pixel data as a RealRandomAccessible  
# (that is, accessible with sub-pixel precision)  
# by using an interpolator  
imgR = Views.interpolate(imgE, NLinearInterpolatorFactory())  
  
# Define a rotation by +30 degrees relative to the image center in the XY axes  
# (not explicitly XY but the first two dimensions)  
# by filling in a rotation matrix with values taken  
# from a java.awt.geom.AffineTransform (aliased as Affine2D)  
# and by filling in the rest of the diagonal with 1.0  
# (for the case where the image has more than 2 dimensions)  
angle = radians(30)  
rot2d = Affine2D.getRotateInstance(  
  angle, img.dimension(0) / 2, img.dimension(1) / 2)  
ndims = img.numDimensions()  
matrix = Matrix(ndims, ndims + 1)  
matrix.set(0, 0, rot2d.getScaleX())  
matrix.set(0, 1, rot2d.getShearX())  
matrix.set(0, ndims, rot2d.getTranslateX())  
matrix.set(1, 0, rot2d.getShearY())  
matrix.set(1, 1, rot2d.getScaleY())  
matrix.set(1, ndims, rot2d.getTranslateY())  
for i in range(2, img.numDimensions()):  
  matrix.set(i, i, 1.0)  
  
print matrix.getArray()  
  
# Define a rotated view of the image  
rotated = RV.transform(imgR, AffineTransform(matrix))  
  
# View the image rotated, without enlarging the canvas  
# so we define the interval as the original image dimensions.  
# (Notice the -1 on the max coordinate: the interval is inclusive)  
minC = [0 for i in range(img.numDimensions())]  
maxC = [img.dimension(i) -1 for i in range(img.numDimensions())]  
imgRot2d = IL.wrap(Views.interval(rotated, minC, maxC),  
  imp.getTitle() + " - rot2d")  
imgRot2d.show()  
  
# View the image rotated, enlarging the interval to fit it.  
# (This is akin to enlarging the canvas.)  
# We compute the bounds of the enlarged canvas by transforming a rectangle,  
# then define the interval min and max coordinates by subtracting  
# and adding as appropriate to exactly capture the complete rotated image.  
# Notice the min coordinates have negative values, as the rotated image  
# has pixels now somewhere to the left and up from the top-left 0,0 origin  
# of coordinates.  
bounds = rot2d.createTransformedShape(  
  Rectangle(img.dimension(0), img.dimension(1))).getBounds()  
minC[0] = (img.dimension(0) - bounds.width) / 2  
minC[1] = (img.dimension(1) - bounds.height) / 2  
maxC[0] += abs(minC[0]) -1 # -1 because its inclusive  
maxC[1] += abs(minC[1]) -1  
imgRot2dFit = IL.wrap(Views.interval(rotated, minC, maxC),  
  imp.getTitle() + " - rot2dFit")  
imgRot2dFit.show()  
```

## 使用ImgLib2处理RGB和ARGB图片
```python
from net.imglib2.converter import Converters  
from net.imglib2.view import Views  
from net.imglib2.img.display.imagej import ImageJFunctions as IL  
from ij import IJ  
  
# # Load an RGB or ARGB image  
imp = IJ.getImage()  
  
# Access its pixel data from an ImgLib2 data structure:  
# a RandomAccessibleInterval<argbtype>  
img = IL.wrapRGBA(imp)  
  
# Convert an ARGB image to a stack of 4 channels: a RandomAccessibleInterval<unsignedbyte>  
# with one more dimension that before.  
# The order of channels in the stack can be changed by changing their indices.  
channels = Converters.argbChannels(img, [0, 1, 2, 3])  
  
# Equivalent to ImageJ's CompositeImage: channels are separate  
impChannels = IL.wrap(channels, imp.getTitle() + " channels")  
impChannels.show()  
  
# Read out a single channel directly  
red = Converters.argbChannel(img, 1)  
  
# Alternatively, pick a view of the red channel in the channels stack.  
# Takes the last dimension, which are the channels,  
# and fixes it to just one: that of the red channel (1) in the stack.  
red = Views.hyperSlice(channels, channels.numDimensions() -1, 1)  
  
impRed = IL.wrap(red, imp.getTitle() + " red channel")  
impRed.show()
``` 

## Image Calculator：在两张或多张图片间进行逐像素操作
ImageJ有一个内置的命令，叫Image Calculator，在Process- Image Calculator下，可以对两张图片进行数学操作，比如从一张图片中减去另一张。
