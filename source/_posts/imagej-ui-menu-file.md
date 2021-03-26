---
title: ImageJ 用户指南 -- 5. 菜单栏之File
tags: [ImageJ]
categories: computer vision
date: 2018-9-3
---

# 本章说明
菜单栏列出了ImageJ的所有命令，它包含八个菜单：
- File：基本的文件操作，包括打开、保存、创建新图片，大多数命令看名字就知道什么意思
- Edit：编辑和绘制操作，以及全局设定
- Image：图像显示，包括图像格式的转化、怎样显示等
- Process：图像处理，包括点操作、过滤器和算术运算
- Analyze：图像分析，统计测量、直方图绘制和其他与图像分析有关的操作
- Plugins：创建、编辑和管理插件，列出了用户安装的所有宏、脚本和插件。
- Window：已打开的窗口的选择和管理
- Help：升级，文档资源和版本信息

# File菜单
## New新建
可以新建的东西有很多：
- Image：可以指定图片的标题、类型、尺寸、初始填充。且如果Slices大于1，则创建了一个stack
- Hyperstack：与Image-Hyperstacks-New Hyperstack相同
- Text Window：创建一个编写宏的文本窗口
- Internal Clipboard：打开ImageJ内部剪贴板中的内容
- System Clipboard：打开系统剪贴板中的内容
- TrakEM2：Fiji中还加入了编写TrakEM2程序
- Script：Fiji中还加入了新建脚本。

## Open打开
可以打开的东西也有很多：
- 常见图片，后缀有TIFF、GIF、JPEG、DICOM、BMP、PGM和FITS格式。也可以通过插件打开额外的后缀的图片
- ImageJ和NIH的图片查询表，后缀是.lut
- 以制表符分割的表格，后缀是.xls和.csv
- 选区，后缀是.roi和.zip
- 文本文件，后缀是.txt、.ijm、.js和.java
- 其他

## Open Next打开下一个
关闭当前图片，打开目录中的下一个图片（如果有的话）。按住Alt打开目录中的前一个图片（如果有的话）。

## Open Samples打开样例
打开ImageJ服务器上的样例图片，可以用来测试宏、脚本、插件等。

## Open Recent打开最近文件
子菜单会显示最近15个打开的文件，可以选择其中一个。

## Import导入
### Image Sequence
打开所选文件夹中的一系列图片作为一个stack。图片可能有不同的尺寸，也可以是任意ImageJ所支持的格式。非图片格式的文件会被忽略。
- Number of Images：指定打开多少张图片
- Starting image：如果设置为n，将会从文件夹中的第n张图片开始导入
- Increment：增量步长，即每隔多少张图片导入
- File Name Contains：填入一个字符串，ImageJ将会仅打开含该字符串的文件
- Enter Pattern：可以使用正则表达式做进一步的过滤
- Scaled Images：设置一个小于100的数会减少内存要求，如填入50会使得所需内存减少$25%$。如果勾选Use Vritual Stack后，该选项会被忽略
- Convert to RGB：允许将RGB和灰度同时存在的图片全部转换为RGB。注意，如果该选项不勾选，且第一张图是8-bit，那么后面所有的图都将转为8-bit。勾选这个选项来避免这种问题。
- Sort Names Numerically：勾选后，将会以数值顺序打开文件，即以1、2、..10的顺序，而不是以1、10、2..的顺序。
- Use Virtual Stack：勾选后，图片将会使用Virtual Stack Opener该插件以只读Virtual Stack的形式打开。这使得太大而难以放入内存的图片的读取成为可能。

### Raw
用于导入ImageJ所不支持的图片文件，需要事先知道关于该特定文件的信息，包括图片大小、与开头数据的偏移量等。
### LUT
打开一个ImageJ或NIH的图片查询表，或者一个原生的表。原生的表必须是768字节大小，且包含256个红色、256个蓝、256个绿。如果事先没有图片打开，那么一个256*32的图片会创建来显示该表。
### Text Image
打开一个制表符分隔的文本文件作为一个32-bit的真实图片。图片的宽度和高度是通过扫描和计算文件的单词数和行数所确定的。对于不大于255的文本文件，使用Image-Type-8-Bit来转换为8-bit图片。在转换前，在Edit-Options-Conversions中不勾选Scale When Converting，从而避免图片被缩放到0-255范围。
### Text File
打开一个文本文件。也可以通过上面的File-Open或拖拽打开。
### URL
通过一个URL来下载和显示图片。
### Results
打开一个ImageJ表格或任意制表符和逗号分隔的文本文件。.csv和.xls文件可直接拖拽打开。
### Stack From List
从一个包含一系列图片文件路径的文本文件或URL中打开stack或virtual stack。文件可以放在不同的文件夹中，但必须是相同的尺寸和类型。
### TIFF Virtual Stack
打开一个TIFF格式的文件作为Virtual Stack。
### AVI
使用内置的AVI reader插件打开一个AVI文件，作为stack或virtual stack。动画速度是从图片帧速率获取的。
### XY Coordinates
导入一个两栏的文本文件，比如通过File-Save As-XY Coordinates所存取的选区。选区可在当前图片中显示，如果当前文件太小，则在新的空白图片中显示。活跃选区的坐标可以通过Edit-Selection-Properties中的List coordinates显示。

## Close
关闭当前活动图片。

## Close All
关闭所有图片。

## Save
将当前活动图片存成TIFF格式。如果仅存储一个所选区域，创建一个选区，然后使用Image-Duplicate。
Save命令与File-Save As-TIFF是相同的。

## Save As
将图片存储为TIFF、GIF、JPEG或原始格式。也能用来存储测量结果、查询表、选区和选区的坐标。

### TIFF
TIFF是唯一一种（除了“raw”原始格式）支持所有ImageJ的数据格式（8-bit、16-bit、32-bit 浮点型和RGB）以及唯一支持空间和密度标定数据的格式。除此以外，选区和Overlay也存储在TIFF文件的header中。
### GIF
将当前活动图片存储成GIF格式。在此之前，首先要将RGB图片通过Image-Type-8-bit Color转换一下格式。Stacks将被存成有动画的GIF。使用Image-Stacks-Tools-Animation Options来设定帧率。
### JPEG
将当前活动图片存储成JPEG格式。通过Edit-Options-Input/Output来设置JPEG的压缩率。
当存成JPEG时，Overlay会被永久嵌入图片中。
### Text Image
将当前活动图片存储成以制表符分隔的文本文件。已标定的和浮点类型的图片是用Analyze-Set Measurements所设定的Decimal places小数位数这样的精度来保存。对于RGB图片，每个像素通过三原色的平均来转成灰度，或者如果Edit-Options-Conversions中的If Weighed RGB to Grayscale Conversion勾选后，通过加权平均来转成灰度。
### Zip
将当前活动图片或stack存成一个压缩的Zip格式的TIFF。
### Raw Data
将当前活动图片或stack存成没有header的原始像素数据。8-bit图片存成unsigned bytes，unsigned的16-bit图片存成unsigned short，signed 16-bit图片存成signed short，32-bit图片存成float，RGB存成每像素3个字节的数据。
### Image Sequence
把一个stack或hyperstack存成一个图片序列。
### AVI
把一个stack或hyperstack存成AVI文件。
### PNG
把当前活动图片存成PNG。
### FITS
把当前活动图片存成FITS。
### LUT
把当前活动图片的查询表存成文件。
### Results
把“Results”窗口的内容存成制表符分隔或逗号分隔的csv文件。
### Selection
把当前选区的边界存到文件中，然后稍后可以使用File-Open再导入。
### XY Coordinates
把当前ROI的XY坐标存入一个两栏、制表符分隔的文本文件。ROI坐标也可以通过Edit-Selection-Properties勾选List coordinates来获得。

## Revert
Revert实际的操作是：不保存而关闭窗口，重新打开图片。

## Page Setup
控制输出的尺寸及其他选项。

## Print
打印当前图片。

## Quit
退出程序。
