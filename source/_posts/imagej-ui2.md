---
title: ImageJ 用户指南--5.菜单栏
tags: [ImageJ]
categories: programming
date: 2018-9-3
---

# 本章说明
菜单栏列出了ImageJ的所有命令，它包含八个菜单：
- File：基本的文件操作，包括打开、保存、创建新图片，大多数命令看名字就知道什么意思
- Edit：编辑和绘制操作，以及全局设定
- Image：图像的转化和修改，包括几何变换
- Process：图片处理，包括点操作、过滤器和算术运算
- Analyze：统计测量、直方图绘制和其他与图像分析有关的操作
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

### 
