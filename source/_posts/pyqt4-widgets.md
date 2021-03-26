
---
title: PyQt4教程系列二：基本控件
tags: [PyQt]
categories: coding
date: 2017-12-4
---
本文是对[TutorialsPoint](https://www.tutorialspoint.com/pyqt/)上的教程的翻译。

# 标签QLabel
用来显示不可编辑的文字或图片，或GIF动画。也可用作其他控件的占位符。纯文本、超链接或富文本都可以显示在这个Label上。
## QLabel类的函数
1. setAlignment()：对齐文本，参数有Qt.AlignLeft、Qt.AlignRight、Qt.AlignCenter、Qt.AlignJustify
2. setIndent()：设置文本缩进
3. setPixmap()：显示一张图片
4. Text()：显示label的标题
5. setText(): 编写程序来设定标题
6. selectedText()：显示所选文本，其中textInteractionFlag必须设为TextSelectableByMouse
7. setBuddy()：将label与某个输入widget相关联
8. setWordWrap()： Enables or disables wrapping text in the label

## QLabel类的信号
1. linkActivated：如果Label上面的超链接被点击了，那么就打开URL。setOpenExternalLinks必须被设为true。
2. linkHovered：当鼠标悬停在Label上面的超链接时，与该信号相关联的Slot函数将被调用

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

def window():
   app = QApplication(sys.argv)
   win = QWidget() 
	
   l1 = QLabel()
   l2 = QLabel()
   l3 = QLabel()
   l4 = QLabel()
	
   l1.setText("Hello World")
   l4.setText("TutorialsPoint")
   l2.setText("welcome to Python GUI Programming")
	
   l1.setAlignment(Qt.AlignCenter)
   l3.setAlignment(Qt.AlignCenter)
   l4.setAlignment(Qt.AlignRight)
   l3.setPixmap(QPixmap("python.jpg"))
	
   vbox = QVBoxLayout()
   vbox.addWidget(l1)
   vbox.addStretch()
   vbox.addWidget(l2)
   vbox.addStretch()
   vbox.addWidget(l3)
   vbox.addStretch()
   vbox.addWidget(l4)
	
   l1.setOpenExternalLinks(True)
   l4.linkActivated.connect(clicked)
   l2.linkHovered.connect(hovered)
   l1.setTextInteractionFlags(Qt.TextSelectableByMouse)
   win.setLayout(vbox)
	
   win.setWindowTitle("QLabel Demo")
   win.show()
   sys.exit(app.exec_())
	
def hovered():
   print "hovering"
def clicked():
   print "clicked"
	
if __name__ == '__main__':
   window()
```

# 单行输入QLineEdit
最常使用的输入框，可供输入一行文字。输入多行文字时需要使用QTextEdit。

## QLineEdit类的函数
1. setAlignment()：同上
2. clear()：清除内容
3. setEchoMode()：控制输入框中文本的样式，参数有QLineEdit.Normal、QLineEdit.NoEcho、QLineEdit.Password、QLineEdit.PasswordEchoOnEdit
4. setMaxLength()：设置输入的字符最大长度
5. setReadOnly()：使文本框不可编辑
6. setText()：同上
7. text()：取得文本
8. setValidator()：设置生效规则。参数有：QIntValidator(限制输入为整数)、QDoubleValidator(浮点数)、QRegexpValidator(正则表达式)
9. setInputMask()：通过结合符号来设定输入的规范
10. setFont()：设置字体，通过QFont()来设置

## QLineEdit类的信号
1. cursorPositionChanged()：鼠标移动
2. editingFinished()：点击回车或者输入框失去焦点
3. returnPressed()：点击回车
4. selectionChanged()：所选文本变化
5. textChanged()：通过输入或者编程改变了输入框中的文本
6. textEdited()：文本被编辑

## 举例

```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

def window():
   app = QApplication(sys.argv)
   win = QWidget()
	
   e1 = QLineEdit()                               
   e1.setValidator(QIntValidator())                 # e1只接收整数
   e1.setMaxLength(4)                               # 整数最多有四位
   e1.setAlignment(Qt.AlignRight)
   e1.setFont(QFont("Arial",20))
	
   e2 = QLineEdit()
   e2.setValidator(QDoubleValidator(0.99,99.99,2))  # 最多两位小数
	
   flo = QFormLayout()
   flo.addRow("integer validator", e1)
   flo.addRow("Double validator",e2)
	
   e3 = QLineEdit()
   e3.setInputMask('+99_9999_999999')               # 设定输入的格式
   flo.addRow("Input Mask",e3)
	
   e4 = QLineEdit()
   e4.textChanged.connect(textchanged)              # 发射textChanged信号，执行下面的textchanged槽函数
   flo.addRow("Text changed",e4)
	
   e5 = QLineEdit()
   e5.setEchoMode(QLineEdit.Password)               # 设定显示模式为密码模式
   flo.addRow("Password",e5)
	
   e6 = QLineEdit("Hello Python")
   e6.setReadOnly(True)                             # e6只读
   flo.addRow("Read Only",e6)
	
   e5.editingFinished.connect(enterPress)           # e5发射editingFinished信号，执行下面的enterPress槽函数
   win.setLayout(flo)
   win.setWindowTitle("PyQt")
   win.show()
	
   sys.exit(app.exec_())

def textchanged(text):
   print "contents of text box: "+text
	
def enterPress():
   print "edited"

if __name__ == '__main__':
   window()
```

# 按钮QPushButton
显示一个按钮，当点击时可以通过编程来调用一个确定的函数。
## QPushButton类的函数
1. setCheckable()：设为true时识别按钮是按压还是释放
2. toggle()：在可选状态之间切换
3. setIcon()：显示图标
4. setEnabled()：当设为false时，按钮失效，因此点击时不会发射信号
5. isChecked()：返回按钮的布尔状态
6. setDefault()：设置按钮成为默认值
7. setText()：编写程序设置按钮的标题
8. text()：取回标题

## QPushButton类的信号
主要就是clicked信号。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class Form(QDialog):
   def __init__(self, parent=None):
      super(Form, self).__init__(parent)
		
      layout = QVBoxLayout()
      self.b1 = QPushButton("Button1")
      self.b1.setCheckable(True)                               # 识别b1的状态
      self.b1.toggle()                                         # 在状态之间切换
      self.b1.clicked.connect(lambda:self.whichbtn(self.b1))   # 发射clicked信号并连接下面的whichbtn槽函数
      self.b1.clicked.connect(self.btnstate)                   # 发射clicked信号并连接下面的btnstate槽函数
      layout.addWidget(self.b1)
		
      self.b2 = QPushButton()
      self.b2.setIcon(QIcon(QPixmap("python.gif")))            # b2设置图标
      self.b2.clicked.connect(lambda:self.whichbtn(self.b2))
      layout.addWidget(self.b2)
      self.setLayout(layout)

      self.b3 = QPushButton("Disabled")                        # b3不可按
      self.b3.setEnabled(False)
      layout.addWidget(self.b3)
		
      self.b4 = QPushButton("&Default")                        # 名称中加上前缀&，这样就可以使用快捷键Alt+D来点击该按钮
      self.b4.setDefault(True)
      self.b4.clicked.connect(lambda:self.whichbtn(self.b4))
      layout.addWidget(self.b4)
      
      self.setWindowTitle("Button demo")

   def btnstate(self):
      if self.b1.isChecked():
         print "button pressed"
      else:
         print "button released"
			
   def whichbtn(self,b):
      print "clicked button is "+b.text()

def main():
   app = QApplication(sys.argv)
   ex = Form()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```
# 单选框QRadioButton
显示一个带有文字标签的可选按钮，可在表单中选择某一个选项，是单选按钮。该类派生自QAbstractButton。
单选按钮默认是排他的，即一次只能选择一个选项。可以把Radio Button放入QGroupBox或QButtonGroup中以创建更多可选的选项。
## QRadioButton的函数
1. setChecked()：更改单选按钮的状态
2. setText()：设置与按钮相关联的标签
3. text()：取得按钮的标题
4. isChecked()：检查按钮是否被选择

## QRadioButton的信号
默认信号是toggled()，也可以使用从QAbstractButton类中继承的其他信号。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class Radiodemo(QWidget):

   def __init__(self, parent = None):
      super(Radiodemo, self).__init__(parent)
		
      layout = QHBoxLayout()
      self.b1 = QRadioButton("Button1")
      self.b1.setChecked(True)                                # b1默认是勾选的
      self.b1.toggled.connect(lambda:self.btnstate(self.b1))  # 发射toggled信号，连接btnstate槽函数
      layout.addWidget(self.b1)
		
      self.b2 = QRadioButton("Button2")
      self.b2.toggled.connect(lambda:self.btnstate(self.b2))

      layout.addWidget(self.b2)
      self.setLayout(layout)
      self.setWindowTitle("RadioButton demo")
		
   def btnstate(self,b):
	
      if b.text() == "Button1":
         if b.isChecked() == True:
            print b.text()+" is selected"
         else:
            print b.text()+" is deselected"
				
      if b.text() == "Button2":
         if b.isChecked() == True:
            print b.text()+" is selected"
         else:
            print b.text()+" is deselected"
				
def main():

   app = QApplication(sys.argv)
   ex = Radiodemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```
# 复选框QCheckBox
文字标签之前的矩形框，是多选按钮。
多选按钮默认是不排他的，如果想要手动排他，需要将这些复选框放入QButtonGroup中。
## QCheckBox的函数
1. setChecked()：更改按钮状态
2. setText()：设置标签
3. text()：取回标题
4. isChecked()：查看是否被勾选
5. setTriState()：提供无状态变化

## QCheckBox的信号
有toggled()信号。还有stateChanged()信号，每次复选框勾选或清除时，都会发射该信号。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class checkdemo(QWidget):
   def __init__(self, parent = None):
      super(checkdemo, self).__init__(parent)
      
      layout = QHBoxLayout()
      self.b1 = QCheckBox("Button1")
      self.b1.setChecked(True)                                      # b1默认勾选
      self.b1.stateChanged.connect(lambda:self.btnstate(self.b1))   # 发射stateChanged信号
      layout.addWidget(self.b1)
		
      self.b2 = QCheckBox("Button2")
      self.b2.toggled.connect(lambda:self.btnstate(self.b2))

      layout.addWidget(self.b2)
      self.setLayout(layout)
      self.setWindowTitle("checkbox demo")

   def btnstate(self,b):
      if b.text() == "Button1":
         if b.isChecked() == True:
            print b.text()+" is selected"
         else:
            print b.text()+" is deselected"
				
      if b.text() == "Button2":
         if b.isChecked() == True:
            print b.text()+" is selected"
         else:
            print b.text()+" is deselected"
				
def main():

   app = QApplication(sys.argv)
   ex = checkdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

手动排他的形式为：
```cpp
self.bg = QButtonGroup()
self.bg.addButton(self.b1,1)
self.bg.addButton(self.b2,2)
self.bg.buttonClicked[QAbstractButton].connect(self.btngroup)  # 发射buttonClicked信号
def btngroup(self,btn):
   print btn.text()+" is selected"
```

# 下拉列表QComboBox
提供一个下拉列表供选择，这样就可以用很少的屏幕空间来显示当前选择项
## QComboBox的函数
1. addItem()：向集合中添加字符串
2. addItems()：以list的形式添加多个项目
3. Clear()：清除所有项目
4. count()：计算项目总数
5. currentText()：取回当前所选项目的文本
6. itemText()：显示属于特定索引的文本
7. currentIndex()：当前所选项的索引
8. setItemText()：改变特定索引的文本

## QComboBox的信号
1. activated()：用户选择了某项
2. currentIndexChanged()：当前索引被用户或程序改变
3. highlighted()：某项被高亮

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class combodemo(QWidget):
   def __init__(self, parent = None):
      super(combodemo, self).__init__(parent)
      
      layout = QHBoxLayout()
      self.cb = QComboBox()
      self.cb.addItem("C")                                       # 添加一个项目
      self.cb.addItem("C++")
      self.cb.addItems(["Java", "C#", "Python"])                 # 添加多个项目
      self.cb.currentIndexChanged.connect(self.selectionchange)  # 发射currentIndexChanged信号，连接下面的selectionchange槽
		
      layout.addWidget(self.cb)
      self.setLayout(layout)
      self.setWindowTitle("combo box demo")

   def selectionchange(self,i):
      print "Items in the list are :"
		
      for count in range(self.cb.count()):
         print self.cb.itemText(count)
      print "Current index",i,"selection changed ",self.cb.currentText()
		
def main():
   app = QApplication(sys.argv)
   ex = combodemo()
   ex.show()
   sys.exit(app.exec_())

if __name__ == '__main__':
   main()
```
# 调值框QSpinBox
一个显示整数的文本框，右侧可以有上下按钮调节。
默认情况下，整数值从0开始，最大是99，步长为1。如果想用浮点数，需要使用QDoubleSpinBox。

## QSpinBox的函数
1. setMinimum()：设定最小值
2. setMaximum()：设定最大值
3. setRange()：设定最小值、最大值和步长
4. setValue()：编写程序来设定数值
5. Value()：返回当前值
6. singleStep()：设定步长

## QSpinBox的信号
每次点击up/down按钮时，就会发射valueChanged()信号，相应的槽可以通过value()函数获取当前值。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class spindemo(QWidget):
   def __init__(self, parent = None):
      super(spindemo, self).__init__(parent)
      
      layout = QVBoxLayout()
      self.l1 = QLabel("current value:")
      self.l1.setAlignment(Qt.AlignCenter)
      layout.addWidget(self.l1)
      self.sp = QSpinBox()
		
      layout.addWidget(self.sp)
      self.sp.valueChanged.connect(self.valuechange)         # 发射valueChanged信号，与下面的valuechange槽连接
      self.setLayout(layout)
      self.setWindowTitle("SpinBox demo")
		
   def valuechange(self):
      self.l1.setText("current value:"+str(self.sp.value())) # 用QLabel显示

def main():
   app = QApplication(sys.argv)
   ex = spindemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 滑块QSlider
提供一个沟槽，上面有一个滑块可以运动，可以用来控制一个有界值，滑块的位置代表了值的大小。
可以水平或垂直放置：
```cpp
self.sp = QSlider(Qt.Horizontal)
self.sp = QSlider(Qt.Vertical)
```

## QSlider的函数
1. setMinimum()：设置最小值
2. setMaximum()：最大值
3. setSingleStep()：步长
4. setValue()：通过编写程序控制数值
5. value()：取回当前值
6. setTickInterval()：在沟槽上放置刻度的数目
7. setTickPosition()：在沟槽上放置刻度，参数可以是QSlider.NoTicks(没有刻度)、QSlider.TicksBothSides(两边都有刻度线)、QSlider.TicksAbove(在上侧有刻度线)、QSlider.TicksBelow(下侧显示)、QSlider.TicksLeft(左侧)、QSlider.TicksRight(右侧)

## QSlider的信号
1. valueChanged()：滑块的值改变
2. sliderPressed()：用户开始按下滑块
3. sliderMoved()：用户拖动了滑块
4. sliderReleased()：用户释放了滑块

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class sliderdemo(QWidget):
   def __init__(self, parent = None):
      super(sliderdemo, self).__init__(parent)

      layout = QVBoxLayout()
      self.l1 = QLabel("Hello")
      self.l1.setAlignment(Qt.AlignCenter)
      layout.addWidget(self.l1)
		
      self.sl = QSlider(Qt.Horizontal)
      self.sl.setMinimum(10)
      self.sl.setMaximum(30)
      self.sl.setValue(20)
      self.sl.setTickPosition(QSlider.TicksBelow)
      self.sl.setTickInterval(5)
		
      layout.addWidget(self.sl)
      self.sl.valueChanged.connect(self.valuechange)
      self.setLayout(layout)
      self.setWindowTitle("SpinBox demo")

   def valuechange(self):
      size = self.sl.value()
      self.l1.setFont(QFont("Arial",size))
		
def main():
   app = QApplication(sys.argv)
   ex = sliderdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 菜单栏QMenuBar和QMenu
菜单栏是在主窗口QMainWindow的标题栏下面的长条，用来显示QMenu类的对象。QMenu类可以向菜单栏加入widgets，也用来创建上下文菜单和弹出菜单。每个QMenu对象可能包含一个或多个QAction或级联QMenu对象。

## QMenu的函数
1. menuBar()：返回主窗口的QMenubar对象
2. addMenu()：在菜单栏上添加一个新的QMenu对象
3. addAction()：对该QMenu对象添加一个新的动作，可包含文字或图标
4. setEnabled()：设置action的状态为enabled或disabled
5. addSeperator()：在菜单中添加分割线
6. Clear()：清楚菜单栏/菜单中的所有内容
7. setShortcut()：为action添加快捷键
8. setText()：为action添加文本
9. setTitle()：设定QMenu的标题
10. text()：取回与QAction对象相关联的文本
11. title()：取回与QMenu对象相关联的文本

## QMenu的信号
当任一QAction按钮被点击时，QMenu都会发射triggered()信号。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class menudemo(QMainWindow):
   def __init__(self, parent = None):
      super(menudemo, self).__init__(parent)
		
      layout = QHBoxLayout()
      bar = self.menuBar()                                  # 创建菜单栏bar
      file = bar.addMenu("File")                            # 在菜单栏上添加菜单file
      file.addAction("New")                                 # 在菜单上添加动作New
		
      save = QAction("Save",self)
      save.setShortcut("Ctrl+S")                            # 为动作save创建快捷键
      file.addAction(save)                                  # 在菜单file上添加动作save
		
      edit = file.addMenu("Edit")                           # 在菜单file上再添加一个级联菜单edit
      edit.addAction("copy")
      edit.addAction("paste")
		
      quit = QAction("Quit",self) 
      file.addAction(quit)
      file.triggered[QAction].connect(self.processtrigger)  # 发射triggered信号，与processtrigger槽相连接
      self.setLayout(layout)
      self.setWindowTitle("menu demo")
		
   def processtrigger(self,q):
      print q.text()+" is triggered"
		
def main():
   app = QApplication(sys.argv)
   ex = menudemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 工具栏QToolBar
工具栏是一个可以移动的面板，包含文本按钮或者图标按钮或其他小控件。通常把它放在标题栏下面，也可以拖动它让它悬浮。

## QToolBar的函数
1. addAction()：添加文本或图标样式的工具按钮
2. addSeperator()：添加分割线
3. addWidget()：添加其他控件而不是按钮
4. addToolBar()：QMainWindow主窗口新增一个工具栏
5. setMovable()：设置工具栏可移动
6. setOrientation()：设成水平或垂直，参数为Qt.Horizontal或Qt.Vertical

## QToolBar的信号
当点击工具栏中的按钮时，发射ActionTriggered()信号。

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class tooldemo(QMainWindow):
   def __init__(self, parent = None):
      super(tooldemo, self).__init__(parent)
      layout = QVBoxLayout()
      tb = self.addToolBar("File")
		
      new = QAction(QIcon("new.bmp"),"new",self)
      tb.addAction(new)
      open = QAction(QIcon("open.bmp"),"open",self)
      tb.addAction(open)
      save = QAction(QIcon("save.bmp"),"save",self)
      tb.addAction(save)

      tb.actionTriggered[QAction].connect(self.toolbtnpressed)
      self.setLayout(layout)
      self.setWindowTitle("toolbar demo")
		
   def toolbtnpressed(self,a):
      print "pressed tool button is",a.text()
		
def main():
   app = QApplication(sys.argv)
   ex = tooldemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 输入对话框QInputDialog
输入对话框是一个预配置的对话框，有一个文本框和两个按钮OK和Cancel。当用户点击OK或者回车时，父窗口就把输入收集到文本框中。
用户的输入可以是一个数字、字符串或者从列表中选择的项目。

## QInputDialog的函数
1. getInt()：创建一个用于整数的spin box
2. getDouble()：创建一个用于浮点数的spin box
3. getText()：一个简单的行编辑区域用于输入文本
4. getItem()：可以从中选择某项的combo box

## QInputDialog的函数
该对话框实际上是其他widget的整合，所以没有自己的信号

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class inputdialogdemo(QWidget):
   def __init__(self, parent = None):
      super(inputdialogdemo, self).__init__(parent)
		
      layout = QFormLayout()
      self.btn = QPushButton("Choose from list")
      self.btn.clicked.connect(self.getItem)
		
      self.le = QLineEdit()
      layout.addRow(self.btn,self.le)
      self.btn1 = QPushButton("get name")
      self.btn1.clicked.connect(self.gettext)
		
      self.le1 = QLineEdit()
      layout.addRow(self.btn1,self.le1)
      self.btn2 = QPushButton("Enter an integer")
      self.btn2.clicked.connect(self.getint)
		
      self.le2 = QLineEdit()
      layout.addRow(self.btn2,self.le2)
      self.setLayout(layout)
      self.setWindowTitle("Input Dialog demo")
		
   def getItem(self):
      items = ("C", "C++", "Java", "Python")
		
      item, ok = QInputDialog.getItem(self, "select input dialog", 
         "list of languages", items, 0, False)
			
      if ok and item:
         self.le.setText(item)
			
   def gettext(self):
      text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your name:')
		
      if ok:
         self.le1.setText(str(text))
			
   def getint(self):
      num,ok = QInputDialog.getInt(self,"integer input dualog","enter a number")
		
      if ok:
         self.le2.setText(str(num))
			
def main(): 
   app = QApplication(sys.argv)
   ex = inputdialogdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 字体对话框QFontDialog
字体对话框的返回结果是一个QFont对象，可以用于父窗口的字体设置。

## QFontDialog的函数
1. getfont()：显示字体选择对话框
2. setCurrentFont()：设置对话框的默认字体

## QFontDialog的信号
也是没有自己的信号

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class fontdialogdemo(QWidget):
   def __init__(self, parent = None):
      super(fontdialogdemo, self).__init__(parent)
		
      layout = QVBoxLayout()
      self.btn = QPushButton("choose font")
      self.btn.clicked.connect(self.getfont)
		
      layout.addWidget(self.btn)
      self.le = QLabel("Hello")
		
      layout.addWidget(self.le)
      self.setLayout(layout)
      self.setWindowTitle("Font Dialog demo")
		
   def getfont(self):
      font, ok = QFontDialog.getFont()
		
      if ok:
         self.le.setFont(font)
			
def main():
   app = QApplication(sys.argv)
   ex = fontdialogdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 文件对话框QFileDialog
该对话框用于文件选择。

## QFileDialog的函数
1. getOpenFileName()：返回用户所选文件的名字来打开它
2. getSaveFileName()：使用用户所选文件的名字来存储文件
3. setacceptMode()：决定是打开还是保存，参数是QFileDialog.AcceptOpen和QFileDialog.AcceptSave
4. setFileMode()：所选文件的类型，枚举常量有QFileDialog.AnyFile、QFileDialog.ExistingFile、QFileDialog.Directory和QFileDialog.Existingfiles 
5. setFilter()：仅显示有特定扩展名的文件

## QFileDialog的信号
同上，没有自己的信号

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class filedialogdemo(QWidget):
   def __init__(self, parent = None):
      super(filedialogdemo, self).__init__(parent)
		
      layout = QVBoxLayout()
      self.btn = QPushButton("QFileDialog static method demo")
      self.btn.clicked.connect(self.getfile)
		
      layout.addWidget(self.btn)
      self.le = QLabel("Hello")
		
      layout.addWidget(self.le)
      self.btn1 = QPushButton("QFileDialog object")
      self.btn1.clicked.connect(self.getfiles)
      layout.addWidget(self.btn1)
		
      self.contents = QTextEdit()
      layout.addWidget(self.contents)
      self.setLayout(layout)
      self.setWindowTitle("File Dialog demo")
		
   def getfile(self):
      fname = QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Image files (*.jpg *.gif)")
      self.le.setPixmap(QPixmap(fname))
		
   def getfiles(self):
      dlg = QFileDialog()
      dlg.setFileMode(QFileDialog.AnyFile)
      dlg.setFilter("Text files (*.txt)")
      filenames = QStringList()
		
      if dlg.exec_():
         filenames = dlg.selectedFiles()
         f = open(filenames[0], 'r')
			
         with f:
            data = f.read()
            self.contents.setText(data)
				
def main():
   app = QApplication(sys.argv)
   ex = filedialogdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 标签QTabWidget
当需要同时显示很多项目时，可以用标签来归类管理。

## QTabWidget的函数
1. addTab()：添加一个标签以及与之相关联的页面
2. insertTab()：在想要的位置插入一个标签及页面
3. removeTab()：删除给定索引的标签
4. setCurrentIndex()：设置当前可见的页面的索引作为当前操作
5. setCurrentWidget()：使可见页面作为当前
6. setTabBar()：设置标签栏
7. setTabPosition()：设置标签位置，参数有QTabWidget.North(页面上方)、QTabWidget.South(页面下方)、QTabWidget.West(页面左侧)、QTabWidget.East(页面右侧)
8. setTabText()：定义该tab的文本

## QTabWidget的信号
1. currentChanged()：当前页面索引变化
2. tabClosedRequested()：点击了标签上的关闭按钮

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class tabdemo(QTabWidget):
   def __init__(self, parent = None):
      super(tabdemo, self).__init__(parent)
      self.tab1 = QWidget()
      self.tab2 = QWidget()
      self.tab3 = QWidget()
		
      self.addTab(self.tab1,"Tab 1")
      self.addTab(self.tab2,"Tab 2")
      self.addTab(self.tab3,"Tab 3")
      self.tab1UI()
      self.tab2UI()
      self.tab3UI()
      self.setWindowTitle("tab demo")
		
   def tab1UI(self):
      layout = QFormLayout()
      layout.addRow("Name",QLineEdit())
      layout.addRow("Address",QLineEdit())
      self.setTabText(0,"Contact Details")
      self.tab1.setLayout(layout)
		
   def tab2UI(self):
      layout = QFormLayout()
      sex = QHBoxLayout()
      sex.addWidget(QRadioButton("Male"))
      sex.addWidget(QRadioButton("Female"))
      layout.addRow(QLabel("Sex"),sex)
      layout.addRow("Date of Birth",QLineEdit())
      self.setTabText(1,"Personal Details")
      self.tab2.setLayout(layout)
		
   def tab3UI(self):
      layout = QHBoxLayout()
      layout.addWidget(QLabel("subjects")) 
      layout.addWidget(QCheckBox("Physics"))
      layout.addWidget(QCheckBox("Maths"))
      self.setTabText(2,"Education Details")
      self.tab3.setLayout(layout)
		
def main():
   app = QApplication(sys.argv)
   ex = tabdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 堆栈QStackedWidget
QStackedWidget跟QTabWidget类似，也能有效利用空间

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class stackedExample(QWidget):

   def __init__(self):
      super(stackedExample, self).__init__()
      self.leftlist = QListWidget ()
      self.leftlist.insertItem (0, 'Contact' )
      self.leftlist.insertItem (1, 'Personal' )
      self.leftlist.insertItem (2, 'Educational' )
		
      self.stack1 = QWidget()
      self.stack2 = QWidget()
      self.stack3 = QWidget()
		
      self.stack1UI()
      self.stack2UI()
      self.stack3UI()
		
      self.Stack = QStackedWidget (self)
      self.Stack.addWidget (self.stack1)
      self.Stack.addWidget (self.stack2)
      self.Stack.addWidget (self.stack3)
		
      hbox = QHBoxLayout(self)
      hbox.addWidget(self.leftlist)
      hbox.addWidget(self.Stack)

      self.setLayout(hbox)
      self.leftlist.currentRowChanged.connect(self.display)
      self.setGeometry(300, 50, 10,10)
      self.setWindowTitle('StackedWidget demo')
      self.show()
		
   def stack1UI(self):
      layout = QFormLayout()
      layout.addRow("Name",QLineEdit())
      layout.addRow("Address",QLineEdit())
      #self.setTabText(0,"Contact Details")
      self.stack1.setLayout(layout)
		
   def stack2UI(self):
      layout = QFormLayout()
      sex = QHBoxLayout()
      sex.addWidget(QRadioButton("Male"))
      sex.addWidget(QRadioButton("Female"))
      layout.addRow(QLabel("Sex"),sex)
      layout.addRow("Date of Birth",QLineEdit())
		
      self.stack2.setLayout(layout)
		
   def stack3UI(self):
      layout = QHBoxLayout()
      layout.addWidget(QLabel("subjects"))
      layout.addWidget(QCheckBox("Physics"))
      layout.addWidget(QCheckBox("Maths"))
      self.stack3.setLayout(layout)
		
   def display(self,i):
      self.Stack.setCurrentIndex(i)
		
def main():
   app = QApplication(sys.argv)
   ex = stackedExample()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 分割条QSplitter
分割条能够通过拖动子控件之间的边界，使得它们的尺寸动态变化。

## QSplitter的函数
1. addWidget()：对splitter的布局上添加控件
2. indexOf()：返回控件的索引
3. insertWidget()：在指定索引上插入控件
4. setOrientation()：设置布局水平还是垂直，参数有Qt.Horizontal和Qt.Vertical
5. setSizes()：设置每个控件的初始尺寸
6. count()：返回控件的总数

## QSplitter的信号
当分割条被拖动时，发射splitterMoved()信号。

## 举例
```cpp
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class Example(QWidget):

   def __init__(self):
      super(Example, self).__init__()
      self.initUI()
	
   def initUI(self):
	
      hbox = QHBoxLayout(self)
		
      topleft = QFrame()
      topleft.setFrameShape(QFrame.StyledPanel)
      bottom = QFrame()
      bottom.setFrameShape(QFrame.StyledPanel)
		
      splitter1 = QSplitter(Qt.Horizontal)
      textedit = QTextEdit()
      splitter1.addWidget(topleft)
      splitter1.addWidget(textedit)
      splitter1.setSizes([100,200])
		
      splitter2 = QSplitter(Qt.Vertical)
      splitter2.addWidget(splitter1)
      splitter2.addWidget(bottom)
		
      hbox.addWidget(splitter2)
		
      self.setLayout(hbox)
      QApplication.setStyle(QStyleFactory.create('Cleanlooks'))
		
      self.setGeometry(300, 300, 300, 200)
      self.setWindowTitle('QSplitter demo')
      self.show()
		
def main():
   app = QApplication(sys.argv)
   ex = Example()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 可停靠窗口QDock
可停靠窗口是主窗口的一个子窗口，可以悬浮，也可以附着在主窗口的特定位置。

## QDock的函数
1. setWidget()：在可停靠窗口上设置任意Widget
2. setFloating()：如果设为true，那么dock window可悬浮
3. setAllowedAreas()：设置该子窗口可停靠的位置，参数有LeftDockWidgetArea、RightDockWidgetArea、TopDockWidgetArea、BottomDockWidgetArea、NoDockWidgetArea
4. setFeatures()：设置子窗口的特性，参数有DockWidgetClosable、DockWidgetMovable、DockWidgetFloatable、DockWidgetVerticalTitleBar、NoDockWidgetFeatures

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class dockdemo(QMainWindow):
   def __init__(self, parent = None):
      super(dockdemo, self).__init__(parent)
		
      layout = QHBoxLayout()
      bar = self.menuBar()
      file = bar.addMenu("File")
      file.addAction("New")
      file.addAction("save")
      file.addAction("quit")
		
      self.items = QDockWidget("Dockable", self)
      self.listWidget = QListWidget()
      self.listWidget.addItem("item1")
      self.listWidget.addItem("item2")
      self.listWidget.addItem("item3")
		
      self.items.setWidget(self.listWidget)
      self.items.setFloating(False)
      self.setCentralWidget(QTextEdit())
      self.addDockWidget(Qt.RightDockWidgetArea, self.items)
      self.setLayout(layout)
      self.setWindowTitle("Dock demo")
		
def main():
   app = QApplication(sys.argv)
   ex = dockdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 状态栏QStatusBar
状态栏是主窗口下方的水平条，用来显示一个永久信息或者上下文有关的信息。
有三种类型的状态指示子：
1. Temporary：占据大部分的状态栏。例如用来显示工具信息或者菜单信息
2. Normal：占据一部分的状态栏，可能会被temporary的信息覆盖掉，比如文字处理器中显示页数和行号
3. Permanent：从不隐藏。用于重要消息提示。

用QStatusBar()函数调用主窗口的状态栏，再用setStatusBar()激活它：
```cpp
self.statusBar = QStatusBar()
self.setStatusBar(self.statusBar)
```
## QStatusBar的函数
1. addWidget()：在状态栏上添加控件
2. addPermanentWidget()：添加永久控件
3. showMessage()：在特定时间间隔内显示临时信息
4. clearMessage()：清除任意临时信息
5. removeWidget()：删除特定的控件

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class statusdemo(QMainWindow):
   def __init__(self, parent = None):
      super(statusdemo, self).__init__(parent)
		
      bar = self.menuBar()
      file = bar.addMenu("File")
      file.addAction("show")
      file.addAction("add")
      file.addAction("remove")
      file.triggered[QAction].connect(self.processtrigger)
      self.setCentralWidget(QTextEdit())
		
      self.statusBar = QStatusBar()
      self.b = QPushButton("click here")
      self.setWindowTitle("QStatusBar Example")
      self.setStatusBar(self.statusBar)
		
   def processtrigger(self,q):
	
      if (q.text() == "show"):
         self.statusBar.showMessage(q.text()+" is clicked",2000)
			
      if q.text() == "add":
         self.statusBar.addWidget(self.b)
			
      if q.text() == "remove":
         self.statusBar.removeWidget(self.b)
         self.statusBar.show()
			
def main():
   app = QApplication(sys.argv)
   ex = statusdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 列表QListWidget
QListWidget是一个基于item的界面来从一个list中添加或删除item。每个item都是一个QListWidgetItem对象。

## QListWidget的函数
1. addItem()：在列表中增加QListWidgetItem对象或字符串
2. addItems()：添加list中的每一项
3. insertItem()：在特定索引位置插入某项
4. clear()：清除内容
5. setCurrentItem()：通过编写程序来设定当前选择的项目
6. sortItems()：按升序重新排列项目

## QListWidget的信号
1. currentItemChanged()：当前项目改变
2. itemClicked()：当前项目被点击

## 举例
```cpp
from PyQt4.QtGui import *
from PyQt4.QtCore import *

import sys

class myListWidget(QListWidget):

   def Clicked(self,item):
      QMessageBox.information(self, "ListWidget", "You clicked: "+item.text())
		
def main():
   app = QApplication(sys.argv)
   listWidget = myListWidget()
	
   #Resize width and height
   listWidget.resize(300,120)
	
   listWidget.addItem("Item 1"); 
   listWidget.addItem("Item 2");
   listWidget.addItem("Item 3");
   listWidget.addItem("Item 4");
	
   listWidget.setWindowTitle('PyQT QListwidget Demo')
   listWidget.itemClicked.connect(listWidget.Clicked)
   
   listWidget.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 滚动条QScrollBar
一个滚动条有4个区域：两个箭头、滑块、页面控制区。

## QScrollBar的信号
1. valueChanged()：滚动条的值改变
2. sliderMoved()：用户拖动滑块

## 举例
```cpp
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class Example(QWidget):

   def __init__(self):
      super(Example, self).__init__()
      self.initUI()
		
   def initUI(self):
      vbox = QVBoxLayout(self)
      hbox = QHBoxLayout()
      self.l1 = QLabel("Drag scrollbar sliders to change color")
      self.l1.setFont(QFont("Arial",16))
		
      hbox.addWidget(self.l1)
      self.s1 = QScrollBar()
      self.s1.setMaximum(255)
		
      self.s1.sliderMoved.connect(self.sliderval)
      self.s2 = QScrollBar()
      self.s2.setMaximum(255)
      self.s2.sliderMoved.connect(self.sliderval)
		
      self.s3 = QScrollBar()
      self.s3.setMaximum(255)
      self.s3.sliderMoved.connect(self.sliderval)
		
      hbox.addWidget(self.s1)
      hbox.addWidget(self.s2)
      hbox.addWidget(self.s3)
		
      self.setGeometry(300, 300, 300, 200)
      self.setWindowTitle('QSplitter demo')
      self.show()
		
   def sliderval(self):
      print self.s1.value(),self.s2.value(), self.s3.value()
      palette = QPalette()
      c = QColor(self.s1.value(),self.s2.value(), self.s3.value(),255)
      palette.setColor(QPalette.Foreground,c)
      self.l1.setPalette(palette)
		
def main():
   app = QApplication(sys.argv)
   ex = Example()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 日历QCalendarWidget
可以用此控件方便地选择日期。

## QCalendarWidget的函数
1. setDateRange()：设置可选日期的上界和下界
2. setFirstDayOfWeek()：设定第一列是星期几，参数有：Qt.Monday、Qt.Tuesday、...、Qt.Sunday
3. setMinimumDate()：设置日期下界
4. setMaximumDate()：设置日期上界
5. setSelectedDate()：设定一个QDate对象作为所选日期
6. showToday()：显示今天
7. selectedDate()：取得所选日期
8. setGridvisible()：设置日历网格的可见性

## 举例
```cpp
import sys
from PyQt4 import QtGui, QtCore

class Example(QtGui.QWidget):

   def __init__(self):
      super(Example, self).__init__()
      self.initUI()
		
   def initUI(self):
	
      cal = QtGui.QCalendarWidget(self)
      cal.setGridVisible(True)
      cal.move(20, 20)
      cal.clicked[QtCore.QDate].connect(self.showDate)
		
      self.lbl = QtGui.QLabel(self)
      date = cal.selectedDate()
      self.lbl.setText(date.toString())
      self.lbl.move(20, 200)
		
      self.setGeometry(100,100,300,300)
      self.setWindowTitle('Calendar')
      self.show()
		
   def showDate(self, date):
	
      self.lbl.setText(date.toString())
		
def main():

   app = QtGui.QApplication(sys.argv)
   ex = Example()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```
