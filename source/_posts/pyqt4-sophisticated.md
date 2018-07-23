
---
title: PyQt4教程系列四：高阶功能
tags: [PyQt]
categories: programming 
date: 2017-12-6
---
本文是对[TutorialsPoint](https://www.tutorialspoint.com/pyqt/)上的教程的翻译。

# 数据库QSqlDatabase
PyQt API可以和很多基于SQL的数据库进行通信，用的类是QSqlDatabase。
具体使用哪个数据库要使用相应的驱动：
1. QDB2：IBM DB2
2. QIBASE：Borland InterBase Driver
3. QMYSQL：MySQL Driver
4. QOCI：Oracle Call Interface Driver
5. QODBC：ODBC Driver (includes Microsoft SQL Server)
6. QPSQL：PostgreSQL Driver
7. QSQLITE：SQLite version 3 or above
8. QSQLITE2“SQLite version 2

```cpp
db = QtSql.QSqlDatabase.addDatabase('QSQLITE')
```

## QSqlDatabase的函数
1. setDatabaseName()：设置数据库的名称
2. setHostName()：设置安装数据库的主机的名称
3. setUserName()：输入用户名
4. setPassword()：输入密码
5. commit()：提交事务，如果连接成功则返回true
6. rollback()：回滚事务
7. close()：关闭连接

## 举例

先创建一个含有五个记录的数据库：
```cpp
from PyQt4 import QtSql, QtGui

def createDB():
   db = QtSql.QSqlDatabase.addDatabase('QSQLITE')
   db.setDatabaseName('sports.db')
	
   if not db.open():
      QtGui.QMessageBox.critical(None, QtGui.qApp.tr("Cannot open database"),
         QtGui.qApp.tr("Unable to establish a database connection.\n"
            "This example needs SQLite support. Please read "
            "the Qt SQL driver documentation for information "
            "how to build it.\n\n" "Click Cancel to exit."),
         QtGui.QMessageBox.Cancel)
			
      return False
		
   query = QtSql.QSqlQuery()
	
   query.exec_("create table sportsmen(id int primary key, "
      "firstname varchar(20), lastname varchar(20))")
		
   query.exec_("insert into sportsmen values(101, 'Roger', 'Federer')")
   query.exec_("insert into sportsmen values(102, 'Christiano', 'Ronaldo')")
   query.exec_("insert into sportsmen values(103, 'Ussain', 'Bolt')")
   query.exec_("insert into sportsmen values(104, 'Sachin', 'Tendulkar')")
   query.exec_("insert into sportsmen values(105, 'Saina', 'Nehwal')")
   return True
	
if __name__ == '__main__':
   import sys
	
   app = QtGui.QApplication(sys.argv)
   createDB()
```
再实际连接并操作这个数据库：

```cpp
import sys
from PyQt4 import QtCore, QtGui, QtSql

def initializeModel(model):
   model.setTable('sportsmen')
   model.setEditStrategy(QtSql.QSqlTableModel.OnFieldChange)
   model.select()
   model.setHeaderData(0, QtCore.Qt.Horizontal, "ID")
   model.setHeaderData(1, QtCore.Qt.Horizontal, "First name")
   model.setHeaderData(2, QtCore.Qt.Horizontal, "Last name")
	
def createView(title, model):
   view = QtGui.QTableView()
   view.setModel(model)
   view.setWindowTitle(title)
   return view
	
def addrow():
   print model.rowCount()
   ret = model.insertRows(model.rowCount(), 1)
   print ret
	
def findrow(i):
   delrow = i.row()
	
if __name__ == '__main__':

   app = QtGui.QApplication(sys.argv)
   db = QtSql.QSqlDatabase.addDatabase('QSQLITE')
   db.setDatabaseName('sports.db')
   model = QtSql.QSqlTableModel()
   delrow = -1
   initializeModel(model)
	
   view1 = createView("Table Model (View 1)", model)
   view1.clicked.connect(findrow)
	
   dlg = QtGui.QDialog()
   layout = QtGui.QVBoxLayout()
   layout.addWidget(view1)
	
   button = QtGui.QPushButton("Add a row")
   button.clicked.connect(addrow)
   layout.addWidget(button)
	
   btn1 = QtGui.QPushButton("del a row")
   btn1.clicked.connect(lambda: model.removeRow(view1.currentIndex().row()))
   layout.addWidget(btn1)
	
   dlg.setLayout(layout)
   dlg.setWindowTitle("Database Demo")
   dlg.show()
   sys.exit(app.exec_())
```

# 绘图API QPaintDevice和QPainter
PyQt中的所有QWidget类都是QPaintDevice的子类。QPaintDevice是使用QPainter进行绘图的二维空间的抽象。绘图设备的维度是从左上角开始度量。
无论QPainter类的样式更新时，QPaintEvent都会发生。

## QPainter的函数
1. begin()：开始绘图
2. drawArc()：绘制圆弧
3. drawEllipse()：绘制椭圆
4. drawLine()：绘制直线
5. drawPixmap()：从一个图片文件中提取pixmap，然后显示
6. drwaPolygon()：绘制多边形
7. drawRect()：绘制矩形
8. drawText()：显示文字
9. fillRect()：使用QColor参数填充矩形
10. setBrush()：设置画刷样式
11. setPen()：设置画笔的颜色、尺寸和样式

## 画刷样式
### 预定义的QColor样式
Qt.NoBrush           |      No brush pattern
Qt.SolidPattern      |      Uniform color
Qt.Dense1Pattern     |      Extremely dense brush pattern
Qt.HorPattern        |      Horizontal lines
Qt.VerPattern        |      Vertical lines
Qt.CrossPattern      |      Crossing horizontal and vertical lines
Qt.BDiagPattern      |      Backward diagonal lines
Qt.FDiagPattern      |      Forward diagonal lines
Qt.DiagCrossPattern  |      Crossing diagonal lines

### 预定义的QColor对象

Qt.white      |    Qt.black      |  Qt.red
Qt.darkRed    |    Qt.green      |  Qt.darkGreen
Qt.blue       |    Qt.cyan       |  Qt.magenta
Qt.yellow     |    Qt.darkYellow |  Qt.gray

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
      self.text = "hello world"
      self.setGeometry(100,100, 400,300)
      self.setWindowTitle('Draw Demo')
      self.show()
		
   def paintEvent(self, event):
      qp = QPainter()
      qp.begin(self)
      qp.setPen(QColor(Qt.red))
      qp.setFont(QFont('Arial', 20))
		
      qp.drawText(10,50, "hello Python")
      qp.setPen(QColor(Qt.blue))
      qp.drawLine(10,100,100,100)
      qp.drawRect(10,150,150,100)
		
      qp.setPen(QColor(Qt.yellow))
      qp.drawEllipse(100,50,100,50)
      qp.drawPixmap(220,10,QPixmap("python.jpg"))
      qp.fillRect(200,175,150,100,QBrush(Qt.SolidPattern))
      qp.end()
		
def main():
   app = QApplication(sys.argv)
   ex = Example()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
```

# 剪贴板QClipboard
QClipboard提供了对系统剪贴板的访问，可以很简单地在不同应用之间复制和粘贴数据。它的动作跟QDrag类类似，使用类似的数据结构。
QApplication类有一个静态函数clipboard()来访问系统剪贴板。

## QClipboard的函数
1. clear()：清空剪贴板内容
2. setImage()：复制QImage到剪贴板
3. setMimeData()：复制MIME数据到剪贴板
4. setPixmap()：复制Pixmap对象到剪贴板
5. setText()：复制QString到剪贴板
6. text()：从剪贴板中取得文本

## QClipboard的信号
dataChanged()：剪贴板数据变化

# QPixmap
QPixmap和QImage都是图片类，前者专门用于绘图，用作QPaintDevice对象，或者加载到其他控件中，通常是标签或按钮，以及在屏幕上显示；而QImage则是为I/O进行了优化。
可以读取到QPixmap中的图片格式有：
BMP、GIF、JPG、JPEG、PNG、PBM、PGM、PPM、XBM、XPM。

## QPixmap的函数
1. copy()：从一个QRect对象中复制pixmap数据
2. fromImage()：将QImage对象转换成QPixmap
3. grabWidget()：从给定的控件中创建pixmap
4. grabWindow()：在窗口中创建pixmap数据
5. Load()：加载一个图片作为pixmap
6. save()：保存一个QPixmap对象为一个文件
7. toImage()：将一个QPixmap转换成QImage

## 举例
```cpp
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

def window():
   app = QApplication(sys.argv)
   win = QWidget()
   l1 = QLabel()
   l1.setPixmap(QPixmap("python.jpg"))
	
   vbox = QVBoxLayout()
   vbox.addWidget(l1)
   win.setLayout(vbox)
   win.setWindowTitle("QPixmap Demo")
   win.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   window()
```
