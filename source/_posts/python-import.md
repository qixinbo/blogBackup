---
title: Python3的Import理解
tags: [python]
categories: coding
date: 2019-10-13
---

参考文献：
[Python 的 import 机制](https://loggerhead.me/posts/python-de-import-ji-zhi.html)
[Python 相对导入与绝对导入](http://blog.konghy.cn/2016/07/21/python-import-relative-and-absolute/)
[python: __main__ is not a package](https://blog.csdn.net/junbujianwpl/article/details/79324814)

# Python import的搜索路径
import的搜索路径为：
1. 搜索「内置模块」（built-in module）
2. 搜索 sys.path 中的路径

而sys.path在初始化时，又会按照顺序添加以下路径：
1. foo.py 所在目录（如果是软链接，那么是真正的 foo.py 所在目录）或当前目录；
2. 环境变量 PYTHONPATH中列出的目录（类似环境变量 PATH，由用户定义，默认为空）；
3. site 模块被 import 时添加的路径1（site 会在运行时被自动 import）。

import site 所添加的路径一般是 XXX/site-packages。如果懒得记 sys.path 的初始化过程，可以简单的认为 import 的查找顺序是：
1. 内置模块
2. .py 文件所在目录
3. pip 或 easy_install 安装的包

# 绝对导入和相对导入
绝对导入和相对导入的关系可以类比绝对路径和相对路径。
绝对导入的格式为：
```python
import A.B
或
from A import B
```
相对导入格式为：
```python
from . import B
或
from ..A import B
```
其中，点号.代表当前模块，..代表上层模块，...代表上上层模块，依次类推。
 
# 模块的执行方式
模块的执行可以有两种方式：直接执行和以模块执行，即：
```
python example/foo.py
或
python -m example.foo
```
注意，以模块执行时，一定要有包的概念，即example一定是个包，而foo是这个包下的模块，这样才能顺利执行。

# 包和模块
模块: 一个 .py 文件就是一个模块（module）
包: __init__.py 文件所在目录就是包（package）

# 各种情形测试
## 模块直接导入
即模块所在的目录都不是一个包结构，各个模块都是独立的，比如以下的目录结构：
```python
D:\LEARN\IMPORT_TEST\TEST1
├─pack1
│      modu1.py
└─pack2
       modu2.py
```
modu1.py中的内容为：
```python
import sys
sys.path.append("D:\\learn\\import_test\\TEST1\\pack2")
from modu2 import hello2
hello2()
```
modu2.py中的内容为：
```python
def hello2():
    print("hello, I am module 2")
```
注意在modu1中一定加上sys.path.append那部分内容，即根据上面的描述，一定要让modu1能找到modu2才行，否则就会出现如下错误：
```python
ModuleNotFoundError: No module named 'modu2'
```
此时进入pack1目录下，以直接执行或模块执行的方式都可以顺利输出。

## 包外导入
将上面两个模块所在的目录都变为包结构，即：
```python
D:\LEARN\IMPORT_TEST\TEST2
├─pack1
│      modu1.py
│      __init__.py
└─pack2
       modu2.py
       __init__.py
```
此时也能顺利执行，同时比上面非包结构的多出来一条执行方式，即：
```python
python -m pack1.modu1
```
即以包名+模块名的方式执行。

上面两种情形，即模块与模块、包与包都是相互独立的关系，也就没有相对导入的意义。
如果是在一个包内的不同模块的导入，那么最自然的就是使用相对导入。

## 包内相对导入
```python
D:\LEARN\IMPORT_TEST\Test3
│  __init__.py
│
├─pack1
│      modu1.py
│      __init__.py
│
└─pack2
       modu2.py
       __init__.py
```
此时modu1.py中的内容为：
```python
from ..pack2.modu2 import hello2
hello2()
```
即将sys.path.append去掉，因为是在一个包内相互引用，此时这样写没有意义。
此时正确运行的方式是进入Test3上一层的文件夹，然后：
```python
python -m Test3.pack1.modu1
```
即明确地告诉解释器模块的层次结构。
而如果采用直接运行的方式，比如：
```python
python Test3\pack1\modu1.py
```
就会报如下错误：
```python
ValueError: attempted relative import beyond top-level package
```
这是因为，相对导入使用模块的__name__（这里的name和下面的main都是有两个下划线的，但是网页显示不出来。。）属性来决定模块在包结构中的位置。当__name__属性不包含包信息（i.e. 没有用'.'表示的层次结构，比如'__main__'），则相对导入将模块解析为顶层模块，而不管模块在文件系统中的实际位置。这里模块被直接运行，则它自己为顶层模块，不存在层次结构，所以找不到其他的相对路径。

因此，直接运行带有相对导入的模块是不行的，需要通过模块运行的方式，将包结构明确告诉它才行。

这个原理也适用于下面这种错误，比如将modu2移动到pack1中，即与modu1在同一个目录下，然后将modu1的内容改为这样的相对引用：
```python
from .modu2 import hello2
hello2()
```
此时使用模块执行的方式没有问题，如果还是想尝试直接运行，那么就会出现：
```python
ModuleNotFoundError: No module named '__main__.modu2'; '__main__' is not a package
```
原因就是此时没有包结构，__main__也不是个包。

那么解决方法就是或者使用模块运行的方式运行，或者将它改成下面的绝对导入的方式就可以直接运行。

## 包内绝对导入
那么，如果将modu1.py中的内容改为绝对导入，即：
```python
from Test3.pack2.modu2 import hello2
hello2()
```
此时正确运行方式也是进入Test3上一层文件夹，然后使用模块执行的方式运行：
```python
python -m Test3.pack1.modu1
```

如果此时采用直接运行的方式：
```python
python Test3\pack1\modu1.py
```
那么就会报错：
```python
ModuleNotFoundError: No module named 'Test3'
```
这主要是因为Test3没有被找到，即按照第一部分所说，Test3没有在import的搜索路径中。所以，只要将它加入进去即可，比如：
```python
set PYTHONPATH=D:\learn\import_test\
```
此时再直接运行就没有问题了。
