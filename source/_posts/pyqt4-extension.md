---
title: PyQt4教程系列三：扩展功能
tags: [PyQt]
categories: coding 
date: 2017-12-5
---
本文是对[TutorialsPoint](https://www.tutorialspoint.com/pyqt/)上的教程的翻译。

# QDialog
QDialog通常提供一个窗口用来收集用户的响应。它可以设为模态（阻塞其父窗口）和非模态（可以绕过该对话窗口）。
PyQt API也提供了一些配置好的对话框控件，如之前提到的InputDialog、FileDialog、FontDialog。

## 举例
```cpp
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

def window():
   app = QApplication(sys.argv)
   w = QWidget()
   b = QPushButton(w)
   b.setText("Hello World!")
   b.move(50,50)
   b.clicked.connect(showdialog)
   w.setWindowTitle("PyQt Dialog demo")
   w.show()
   sys.exit(app.exec_())
	
def showdialog():
   d = QDialog()
   b1 = QPushButton("ok",d)
   b1.move(50,50)
   d.setWindowTitle("Dialog")
   d.setWindowModality(Qt.ApplicationModal)  # 这里设置模态与否
   d.exec_()
	
if __name__ == '__main__':
   window()
```

# QMessageBox
QMessageBox是一个常用的模态对话框，用来显示一些信息类消息，可以让用户通过点击任意一个标准按钮来响应。每个标准按钮都有一个预定义的标题、角色，并且返回一个预定义的十六进制数字。

## QMessageBox的函数
1. setIcon()：显示预定义的图标
2. setText()：设置主消息的文字
3. setInformativeText()：显示额外信息
4. setDetailText()：显示一个Details按钮
5. setTitle()：显示自定义标题
6. setStandardButtons()：标准按钮及其对应的十六进制数字是QMessageBox.Ok 0x00000400、QMessageBox.Open 0x00002000、QMessageBox.Save 0x00000800、QMessageBox.Cancel 0x00400000、QMessageBox.Close 0x00200000、QMessageBox.Yes 0x00004000、QMessageBox.No 0x00010000、QMessageBox.Abort 0x00040000、QMessageBox.Retry 0x00080000、QMessageBox.Ignore 0x00100000。
7. setDefaultButton()：设置默认按钮。当回车时它会发射clicked信号
8. setEscapeButton()：当按下Escape键时，该按钮会发出clicked信号

## 举例
```cpp
mport sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

def window():
   app = QApplication(sys.argv)
   w = QWidget()
   b = QPushButton(w)
   b.setText("Show message!")

   b.move(50,50)
   b.clicked.connect(showdialog)
   w.setWindowTitle("PyQt Dialog demo")
   w.show()
   sys.exit(app.exec_())
	
def showdialog():
   msg = QMessageBox()
   msg.setIcon(QMessageBox.Information)

   msg.setText("This is a message box")
   msg.setInformativeText("This is additional information")
   msg.setWindowTitle("MessageBox demo")
   msg.setDetailedText("The details are as follows:")
   msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
   msg.buttonClicked.connect(msgbtn)
	
   retval = msg.exec_()
   print "value of pressed message box button:", retval
	
def msgbtn(i):
   print "Button pressed is:",i.text()
	
if __name__ == '__main__': 
   window()
```

# 多文档界面QMdiArea和QMdiSubWindow
一个通常的GUI可能具有多个窗口，之前的标签页和堆栈控件可以允许点击一次激活某个窗口，但是这两种方法有时不适用，因为它们一次只能查看一个窗口。
同时显示多个窗口的方法有：一是把这些窗口创建成独立的，称为SDI (Single Document Interface)，但因为每个窗口都有自己的菜单栏、工具栏等，开销比较大；另外一种方法是MDI (Multiple Document Interface)，比较节省开销，子窗口彼此相对放置在主容器中，这个主容器控件叫做QMdiArea。
QMdiArea通常占据QMainWindow对象的中心地位，在此区域内的子窗口都是QMdiSubWindow类的实例，可以在这些子窗口中设置任意的控件，这些子窗口可以以级联或者瓦片的形式排列。

## QMdiArea和QMdiSubWindow的函数
1. addSubWindow()：添加一个控件作为子窗口
2. removeSubWindow()：删除子窗口
3. setActiveSubWindow()：激活子窗口
4. cascadeSubWindows()：以级联的形式排列子窗口
5. tileSubWindows()：以瓦片的形式排列子窗口
6. closeActiveSubWindow()：关闭活跃的子窗口
7. subWindowList()：返回子窗口列表
8. setWidget()：在子窗口中添加空间

## QMdiArea和QMdiSubWindow的信号
QMdiArea的对象发射subWindowActivated()信号，而QMdiSubWindow的对象发射windowStateChanged()信号。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class MainWindow(QMainWindow):
   count = 0
	
   def __init__(self, parent = None):
      super(MainWindow, self).__init__(parent)
      self.mdi = QMdiArea()
      self.setCentralWidget(self.mdi)
      bar = self.menuBar()
		
      file = bar.addMenu("File")
      file.addAction("New")
      file.addAction("cascade")
      file.addAction("Tiled")
      file.triggered[QAction].connect(self.windowaction)
      self.setWindowTitle("MDI demo")
		
   def windowaction(self, q):
      print "triggered"
		
      if q.text() == "New":
         MainWindow.count = MainWindow.count+1
         sub = QMdiSubWindow()
         sub.setWidget(QTextEdit())
         sub.setWindowTitle("subwindow"+str(MainWindow.count))
         self.mdi.addSubWindow(sub)
         sub.show()
   		
      if q.text() == "cascade":
         self.mdi.cascadeSubWindows()
   		
      if q.text() == "Tiled":
         self.mdi.tileSubWindows()
   		
def main():
   app = QApplication(sys.argv)
   ex = MainWindow()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 拖放QDrag和QMimeData
对于用户来说，能够拖放对象有时是很有用的。
MIME (Multipurpose Internet Mail Extensions) 是描述消息内容类型的因特网标准。 MIME 消息能包含文本、图像、音频、视频以及其他应用程序专用的数据。MIME类型的拖放而进行的文件传输是基于QDrag类。QMimeData将数据及其相应的MIME类型相关联。数据是存于剪贴板中，然后用于拖放过程。

## QMimeData的函数
Tester      |   Getter       |    Setter        |  MIME Types
-           |   -            |    -             |  -
hasText()   |   text()       |    setText()     |  text/plain
hasHtml()   |   html()       |    setHtml()     |  text/html
hasUrls()   |   urls()       |    setUrls()     |  text/uri-list
hasImage()  |   imageData()  |    setImageData()|  image/ *
hasColor()  |   colorData()  |    setColorData()|  application/x-color

很多控件都支持拖放动作。那些允许数据被拖动的控件必须设置setDragEnabled()为true。
另一方面，接收数据的控件也必须响应拖放动作才能顺利存储数据：
1. DragEnterEvent在拖放动作进入目标控件时，向该控件提供一个事件
2. 当拖放动作进行时DragMoveEvent被使用
3. 当拖放动作离开控件时DragLeaveEvent会生成
4. 当松开鼠标时DropEvent会发生。该事件的响应动作会根据条件接收或拒绝。

## 举例
```cpp
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class combo(QComboBox):

   def __init__(self, title, parent):
      super(combo, self).__init__( parent)
	
      self.setAcceptDrops(True)
		
   def dragEnterEvent(self, e):
      print e
		
      if e.mimeData().hasText():
         e.accept()
      else:
         e.ignore()
			
   def dropEvent(self, e):
      self.addItem(e.mimeData().text())
		
class Example(QWidget):

   def __init__(self):
      super(Example, self).__init__()
		
      self.initUI()
		
   def initUI(self):
      lo = QFormLayout()
      lo.addRow(QLabel("Type some text in textbox and drag it into combo box"))
		
      edit = QLineEdit()
      edit.setDragEnabled(True)
      com = combo("Button", self)
      lo.addRow(edit,com)
      self.setLayout(lo)
      self.setWindowTitle('Simple drag & drop')
		
def main():
   app = QApplication(sys.argv)
   ex = Example()
   ex.show()
   app.exec_()
	
if __name__ == '__main__':
   main()
```
