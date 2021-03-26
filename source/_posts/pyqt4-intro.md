
---
title: PyQt4教程系列一：基本介绍和入门
tags: [PyQt]
categories: coding
date: 2017-12-2
---
本文是对[TutorialsPoint](https://www.tutorialspoint.com/pyqt/)上的教程的翻译。

# 概述
PyQt是一个GUI控件工具箱，是Qt的Python接口。
PyQt有两个主要版本：PyQt 4.x和PyQt 5.x，两者不兼容，且前者基于Python 2和Python 3，后者仅基于Python 3。

Linux下载安装：
```cpp
sudo apt-get install python-qt4
or 
sudo apt-get install pyqt5-dev-tools
```

PyQt4由以下Modules组成：QtCore、QtGui、QtNetwork、QtXml、QtSvg、QtOpenGL、QtSql。
- QtCore包含非GUI的核心功能，用来处理时间、文件和目录、数据类型、流、URL、MIME类型、线程和进程。
- QtGui包含图形组件及相关类，比如按钮、窗口、状态栏、工具栏、滑块、位图、颜色、字体等。
- QtNetwork包含网络编程的相关类，比如用于TCP/IP和UDP服务端和客户端的编程。
- QtXml包含处理XML文件的类，提供了用于SAX和DOM这些API的实现。
- QtSvg提供了显示SVG文件内容的类。
- QtOpenGL使用OpenGL库来处理2D和3D的图像，将Qt GUI库和OpenGL库无缝链接起来。
- QtSql包含处理数据库的类。

# Hello World
使用PyQt创建"Hello World"的步骤如下：
1. 导入QtGui模块
2. 创建一个应用对象app
3. 创建一个QWidget对象w来创建最顶层的窗口，在上面添加一个QLabel对象b
4. 设置label的标题为“Hello World”
5. 通过setGeometry()方法定义窗口的尺寸和位置
6. 通过app.exec()方法来进入应用对象的主体

源码为：
```cpp
#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui

def window():
   app = QtGui.QApplication(sys.argv)
   w = QtGui.QWidget()
   b = QtGui.QLabel(w)
   b.setText("Hello World!")
   w.setGeometry(100,100,200,50)
   b.move(50,20)
   w.setWindowTitle("PyQt")
   w.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   window()
```

# 信号和槽
与一般按顺序执行的控制台模式的应用程序不同，基于GUI的程序是由事件驱动的。事件events是响应用户动作的函数或方法，比如点击按钮、选择项目、鼠标点击等。用来构建GUI界面的挂件Widgets是这些事件的来源。每个PyQt Widget，都派生自QObject类，用来发射“信号”signals来响应一个或多个事件。信号本身不执行动作，它们连接到“槽”slot上。“槽”是可调用的Python函数。
在PyQt中，信号和槽的连接有多种方式。
最常用的方式是：
```cpp
QtCore.QObject.connect(widget, QtCore.SIGNAL(‘signalname’), slot_function)
```
更方便的方式是当widget发射signal时，调用slot函数：
```cpp
widget.signal.connect(slot_function)
```
以下是两种方式的举例：
```cpp
#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

def window():
   app = QApplication(sys.argv)
   win = QDialog()
   b1 = QPushButton(win)
   b1.setText("Button1")
   b1.move(50,20)
   b1.clicked.connect(b1_clicked)

   b2 = QPushButton(win)
   b2.setText("Button2")
   b2.move(50,50)
   QObject.connect(b2,SIGNAL("clicked()"),b2_clicked)

   win.setGeometry(100,100,200,100)
   win.setWindowTitle("PyQt")
   win.show()
   sys.exit(app.exec_())

def b1_clicked():
   print "Button 1 clicked"

def b2_clicked():
   print "Button 2 clicked"

if __name__ == '__main__':
   window()
```
# 布局管理
一个widget在窗口中的位置可以通过指定以像素为单位的绝对坐标，该坐标是相对于使用setGeometry()方法定义的窗口的尺寸:
```cpp
QWidget.setGeometry(xpos, ypos, width, height)
```
上面这句代码设定了窗口位于显示器的坐标点，以及它的尺寸。
具体的某个widget的位置代码如下：
```cpp
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtGui

def window():
   app = QtGui.QApplication(sys.argv)
   w = QtGui.QWidget()
	
   b = QtGui.QPushButton(w)
   b.setText("Hello World!")
   b.move(50,20)
	
   w.setGeometry(10,10,300,200)
   w.setWindowTitle(“PyQt”)
   w.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   window()
```
但是这种绝对坐标的方式有一些缺点：
1. 当window尺寸改变的时候，该widget的位置却不变
2. 在有不同分辨率的设备上可能显示不统一
3. 当布局需要改变时，该更改会很费劲，因为需要重新设置位置

PyQt的API针对上述问题提供了一些更优雅的管理widget位置的方法：
1. QBoxLayout类：垂直或水平地排列widgets。它的派生类是QVBoxLayout（垂直排列）和QHBoxLayout（水平排列）
2. QGridLayout类：以网格单元的形式排列，包含addWidget()方法。任何Widget都可以通过指定单元的行数和列数来添加
3. QFormLayout类：可以很方便地创建两列的表格，一列（通常是右列）是输入栏，一列（通常是左列）是相应的标签。
