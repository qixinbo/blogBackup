---
title: ImagePy解析：5 -- wxPython多线程编程
tags: [ImagePy]
categories: computational material science 
date: 2019-9-1
---

参考文献：
[多线程threading](http://www.liujiangblog.com/course/python/79)
[wxPython Recipes: A Problem - Solution Approach](https://www.amazon.com/wxPython-Recipes-Problem-Solution-Approach/dp/1484232364)
 
本篇是对上面这两个参考文章的摘抄学习，第一篇博客介绍了Python怎样多线程编程，第二个是本书，介绍了wxPython怎样多线程编程，实际是在Python多线程编程的基础上开展的。
ImagePy也是这样的多线程编程思路，所以本篇可以作为理解ImagePy的辅助材料。

# Python多线程threading
在Python3中，通过threading模块提供线程的功能。原来的thread模块已废弃。但是threading模块中有个Thread类（大写的T，类名），是模块中最主要的线程类。
对于Thread类，它的定义如下：
threading.Thread(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None)
- 参数group是预留的，用于将来扩展；
- 参数target是一个可调用对象，在线程启动后执行；
- 参数name是线程的名字。默认值为“Thread-N“，N是一个数字。
- 参数args和kwargs分别表示调用target时的参数列表和关键字参数。
 
Thread类提供了以下方法和属性:
- start()：启动线程，等待CPU调度
- run()：线程被cpu调度后自动执行的方法
- getName()、setName()和name：用于获取和设置线程的名称。
- setDaemon()：设置为后台线程或前台线程（默认是False，前台线程）。如果是后台线程，主线程执行过程中，后台线程也在进行，主线程执行完毕后，后台线程不论成功与否，均停止。如果是前台线程，主线程执行过程中，前台线程也在进行，主线程执行完毕后，等待前台线程执行完成后，程序才停止。
- ident：获取线程的标识符。线程标识符是一个非零整数，只有在调用了start()方法之后该属性才有效，否则它只返回None。
- is_alive()：判断线程是否是激活的（alive）。从调用start()方法启动线程，到run()方法执行完毕或遇到未处理异常而中断这段时间内，线程是激活的。
- isDaemon()方法和daemon属性：是否为守护线程
- join([timeout])：调用该方法将会使主调线程堵塞，直到被调用线程运行结束或超时。参数timeout是一个数值类型，表示超时时间，如果未提供该参数，那么主调线程将一直堵塞到被调线程结束。

## 创建线程
有两种方式来创建线程：一种是继承Thread类，并重写它的run()方法；另一种是在实例化threading.Thread对象的时候，将线程要执行的任务函数作为参数传入线程。

第一种方法：
```python
import threading
class MyThread(threading.Thread):
    def __init__(self, thread_name):
        # 注意：一定要显式的调用父类的初始化函数。
        super(MyThread, self).__init__(name=thread_name)
def run(self):
        print("%s正在运行中......" % self.name)
if __name__ == '__main__':    
    for i in range(10):
        MyThread("thread-" + str(i)).start()
```
第二种方法：
```python
import threading
import time

def show(arg):
    time.sleep(1)
    print('thread '+str(arg)+" running....")

if __name__ == '__main__':
    for i in range(10):
        t = threading.Thread(target=show, args=(i,))
        t.start()
```

## 线程调用
在多线程执行过程中，有一个特点要注意，那就是每个线程各执行各的任务，不等待其它的线程，自顾自的完成自己的任务，Python默认会等待最后一个线程执行完毕后才退出。比如：
```python
import time
import threading

def doWaiting():
    print('start waiting:', time.strftime('%H:%M:%S'))
    time.sleep(3)
    print('stop waiting', time.strftime('%H:%M:%S'))

t = threading.Thread(target=doWaiting)
t.start()
# 确保线程t已经启动
time.sleep(1)
print('start job')
print('end job')
```
上述输出为：
```python
start waiting: 15:27:58
start job
end job
stop waiting 15:28:01
```
上面例子中，主线程没有等待子线程t执行完毕，而是啥都不管，继续往下执行它自己的代码，执行完毕后也没有结束整个程序，而是等待子线程t执行完毕，整个程序才结束。

有时候希望主线程等等子线程，不要“埋头往前跑”。那就使用join()方法，如下所示：
```python
import time
import threading
 
def doWaiting():
    print('start waiting:', time.strftime('%H:%M:%S'))
    time.sleep(3)
    print('stop waiting', time.strftime('%H:%M:%S'))

t = threading.Thread(target=doWaiting)
t.start()
# 确保线程t已经启动
time.sleep(1)
print('start join')
# 将一直堵塞，直到t运行结束。
t.join()
print('end join')
```
输出为：
```python
start waiting: 15:30:21
start join
stop waiting 15:30:24
end join
```

还可以使用setDaemon(True)把所有的子线程都变成主线程的守护线程，当主线程结束后，守护子线程也会随之结束，整个程序也跟着退出。如：

```python
import threading
import time
 
def run():
    print(threading.current_thread().getName(), "开始工作")
    time.sleep(2)       # 子线程停2s
    print("子线程工作完毕")

for i in range(3):
    t = threading.Thread(target=run,)
    t.setDaemon(True)   # 把子线程设置为守护线程，必须在start()之前设置
    t.start()

time.sleep(1)     # 主线程停1秒
print("主线程结束了！")
print(threading.active_count())  # 输出活跃的线程数

```
上述输出为：
```python
Thread-1 开始工作
Thread-2 开始工作
Thread-3 开始工作
主线程结束了！
4
```

# Publish-Subscribe模式

Publish-Subscribe模式，即发布-订阅模式，是计算机科学中用来在一个程序中的不同部分中进行交流的常用模式。基本思想就是先创建一个或多个subscribers，然后它们会监听publisher发送的特定的消息。
wxPython的关于这一模式的实现就是wx.lib.pubsub，但这一API已经deprecated，现在换用PyPubSub，只需稍微把以前的import的wx.lib.pubsub改成pubsub即可。
一个例子是：
```python
import wx
from pubsub import pub

class OtherFrame(wx.Frame):
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, wx.ID_ANY, "Secondary Frame")
        panel = wx.Panel(self)

        msg = "Enter a Message to send to the main frame"
        instructions = wx.StaticText(panel, label=msg)
        self.msgTxt = wx.TextCtrl(panel, value="")

        closeBtn = wx.Button(panel, label="Send and Close")
        closeBtn.Bind(wx.EVT_BUTTON, self.onSendAndClose)
 
        sizer = wx.BoxSizer(wx.VERTICAL)
        flags = wx.ALL|wx.CENTER
        sizer.Add(instructions, 0, flags, 5)
        sizer.Add(self.msgTxt, 0, flags, 5)
        sizer.Add(closeBtn, 0, flags, 5)
        panel.SetSizer(sizer)

    def onSendAndClose(self, event):
        """
        Send a message and close frame
        """
        msg = self.msgTxt.GetValue()
        pub.sendMessage("panelListener", message=msg)
        pub.sendMessage("panelListener", message="test2", arg2="2nd argument!")
        self.Close()
 
class MyPanel(wx.Panel):
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        pub.subscribe(self.myListener, "panelListener")

        btn = wx.Button(self, label="Open Frame")
        btn.Bind(wx.EVT_BUTTON, self.onOpenFrame)

    def myListener(self, message, arg2=None):
        print("Received the following message: " + message)
        if arg2:
            print("Received another arguments: " + str(arg2))

    def onOpenFrame(self, event):
        """
        Opens secondary frame
        """
        frame = OtherFrame()
        frame.Show()


class MyFrame(wx.Frame):
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="New PubSub API Tutorial")
        panel = MyPanel(self)
        self.Show()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()
```

解析如下：
（1）首先在MyPanel类中创建subscriber：
```python
pub.subscribe(self.myListener, "panelListener")
```
myListener函数能够接收一个或多个参数，这里设定至少接收一个参数message，以及另一个可选参数arg2。
然后将MyPanel中的button与onOpenFrame事件绑定，从而可以调用另一个frame。
（2）然后在OtherFrame类中创建publisher：
```python
def onSendAndClose(self, event):
    msg = self.msgTxt.GetValue()
    pub.sendMessage("panelListener", message=msg)
    pub.sendMessage("panelListener", message="test2", arg2="2nd argument!")
    self.Close()
```
可以看出，这里定义了两个publishers，第一个发送一个参数，第二个发送两个参数。有两点需要注意：
一个是subscriber和publisher的事件标识是一致的，这样才能互相接收信号；第二个是在publisher中需要明确写出subscriber的形参名称，否则会报错。
但是，注意，pubsub不是线程安全的，需要配合下面的线程安全的方法进行服用。

# wxPython多线程编程
在wxPython中，有三种线程安全的（thread-safe）的方法：wx.PostEvent、wx.CallAfter和wx.CallLater。据Robin Dunn（wxPython的创建者）所说，wx.CallAfter调用wx.PostEvent来向一个应用对象发送事件，该应用对象将会拥有一个与该事件绑定的事件句柄，然后将根据开发者编写的代码来对事件作出反应；然后根据Mike Driscoll（wxPython Recipes这本书的作者）所理解，wx.CallLater又会调用wx.CallAfter，同时可加上一个特定的时间限制，从而可以实现在等待一定时间后再发送事件。总之，在这三个方法中，wx.CallLater是最抽象的方法，wx.CallAfter次之，wx.PostEvent则是最底层的API。

在wxPython的邮件列表中，推荐最多的就是使用wx.CallAfter和PubSub的组合，因此推荐使用这种方法来实现多线程。

一个例子如下（不适用于wxPython老版本，适用于wxPython 3.0和Phoenix版本，因为PubSub更新了API）：
```python
# wxPython 3.0 and Phoenix
import time
import wx
 
from threading import Thread
from pubsub import pub

class TestThread(Thread):
    """Test Worker Thread Class."""
    def __init__(self):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.start()    # start the thread

    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        for i in range(6):
            time.sleep(10)
            wx.CallAfter(self.postTime, i)
        time.sleep(5)
        wx.CallAfter(pub.sendMessage, "update", msg="Thread finished!")

    def postTime(self, amt):
        """
        Send time to GUI
        """
        amtOfTime = (amt + 1) * 10
        pub.sendMessage("update", msg=amtOfTime)

class MyForm(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")
        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        self.displayLbl = wx.StaticText(panel,
                                        label="Amount of time since thread started goes here")
        self.btn = btn = wx.Button(panel, label="Start Thread")
        btn.Bind(wx.EVT_BUTTON, self.onButton)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.displayLbl, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)

        # create a pubsub receiver
        pub.subscribe(self.updateDisplay, "update")

    def onButton(self, event):
        """
        Runs the thread
        """
        TestThread()
        self.displayLbl.SetLabel("Thread started!")
        btn = event.GetEventObject()
        btn.Disable()

    def updateDisplay(self, msg):
        """
        Receives data from thread and updates the display
        """
        t = msg
        if isinstance(t, int):
            self.displayLbl.SetLabel("Time since thread started: %s seconds" % t)
        else:
            self.displayLbl.SetLabel("%s" % t)
            self.btn.Enable()
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm().Show()
    app.MainLoop()
```
解析如下：
（1）首先在MyForm中创建一个subscriber：
```python
pub.subscribe(self.updateDisplay, "update")

def updateDisplay(self, msg):
    """
    Receives data from thread and updates the display
    """
    t = msg
    if isinstance(t, int):
        self.displayLbl.SetLabel("Time since thread started: %s seconds" % t)
    else:
        self.displayLbl.SetLabel("%s" % t)
        self.btn.Enable()
```
它所调用的updateDisplay事件处理函数接收一个名为msg的参数，接收后则更新面板上的StaticText组件。
（2）通过一个按钮来开启其他线程：
```python
btn.Bind(wx.EVT_BUTTON, self.onButton)

def onButton(self, event):
    TestThread()
    self.displayLbl.SetLabel("Thread started!")
    btn = event.GetEventObject()
    btn.Disable()
```
点击MyForm上的这个按钮后，就会开启其他线程，同时将这个按钮置为不可用。
（3）其他线程的调用：
```python
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        for i in range(6):
            time.sleep(10)
            wx.CallAfter(self.postTime, i)
        time.sleep(5)
        wx.CallAfter(pub.sendMessage, "update", msg="Thread finished!")

    def postTime(self, amt):
        amtOfTime = (amt + 1) * 10
        pub.sendMessage("update", msg=amtOfTime)
```
可以看出，这个新线程在做的事就是进行六次循环，每次等待10秒钟后创建一个发射器，将时间数值发送给接收器，从而更新MyForm面板上的数值。最后，发送一个名为"Thread finished!"的消息给接收器。

