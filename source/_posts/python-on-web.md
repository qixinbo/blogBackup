---
title: Python爱浏览器，但浏览器不爱它：如何让Python运行在浏览器上
tags: [GUI]
categories: digitalization
date: 2021-11-5
---

# 简介
一直以来，网页浏览器编程所用的编程语言主力就是JavaScript，浏览器就是一个JavaScript的原生解释器。
那么Python能不能直接运行在浏览器上呢，或者说Python能不能作为浏览器开发的编程语言？
本文对这一问题做了详细的调研，结果可以用一句话总结：可以，但很鸡肋。

# 可用方案
调研过程中，发现了很多有趣的解决方案，总结起来可以有两类：
（1）将Python语言编译成JavaScript
即将现成的Python语言翻译成JavaScript，然后在浏览器网页中运行，比如：
[Brython](https://brython.info/index.html)
[Skulpt](https://skulpt.org/)
[Transcrypt](https://www.transcrypt.org/)
（2）在浏览器中内置Python解释器
即将Python解释器放在浏览器中，这样就能直接运行Python代码，比如：
[Pyodide](https://pyodide.org/en/stable/)
[PyPy.js](https://pypyjs.org/)

我觉得这两类各自的重点不一样：
第一种是为了用Python来代替JavaScript，即用Python操作网页DOM（Document Object Model）元素，让不熟悉JS的编程人员也能用Python来做一个简单的动态交互网页；
第二种，比如Pyodide，它的野心很大，即将Python的科学计算生态搬到浏览器中，比如numpy、scipy、matplotlib等科学计算常用的函数库直接放进浏览器中，而不是常见的比如作为浏览器服务器后端来调用。这样实现的效果就是不需要部署服务器，直接在浏览器中做复杂的函数计算，并反映到网页上。
下面详细介绍一下两类实现中的代表：Brython和Pyodide。

# Brython
先看一下用Brython写的Hello World：
```python
<!DOCTYPE html>
<html>
<head>
<meta name="description" content="Hello world demo written in Brython www.brython.info">
<meta name="keywords" content="Python,Brython">
<meta charset="iso-8859-1">
<title>Hello world</title>
<script type="text/javascript" src="/src/brython.js"></script>
<script type="text/python" src="show_source.py"></script>
</head>

// ------------注意看这一块-------------
<body onload="brython(1)">

<script type="text/python">
from browser import document, alert
from browser.widgets.dialog import InfoDialog

def echo(ev):
    InfoDialog("Hello", f"Hello {document['zone'].value} !")

document["test"].bind("click", echo)
</script>
// ------------注意看结束-------------

<p>Your name is : <input id="zone" autocomplete="off">
<button id="test">click !</button>
</body>

</html>
```
从上面片段很显而易见，Brython就是用Python来替代JavaScript的写法，比如获取网页上的元素并改变其值。
如果想快速开发一个动态网页，同时不懂JavaScript，可以使用Brython来写。
但注意Brython没法使用Python的整个生态，只能使用部分标准库，比如sys。

# Pyodide
简言之，Pyodide就是在浏览器中运行Python，且能调用Python的数值计算库。
Pyodide解决的痛点是无法在浏览器中进行科学计算：
（1）一方面，现在越来越多的软件都Web化、浏览器化。而浏览器的通用编程语言是JavaScript，但其没有成熟的数据科学处理库，也缺乏一些数值计算很有用的功能和数据结构。
（2）另一方面，Python具有成熟且活跃的科学计算生态，比如基本上所有函数库都依赖的numpy数据结构，但其无法在浏览器运行。
Pyodide 项目则是通过将现有的 CPython 解释器编译为 WebAssembly 并在浏览器的 JavaScript 环境中运行这个编译出来的二进制文件，这提供了一种在浏览器中运行 Python 的方法。

Pyodide有一些非常炫的功能点：
## 在Python和JavaScript之间进行交互
如果所有Pyodide能做的就只是运行Python代码并写出到标准输出上，它将会增长成为一个不错的很酷的技巧，但是不会成为一个用于实际工作的实用工具。真正的力量源于它与浏览器API以及其它运行在浏览器中的JavaScript库交互的能力。由于我们已经将Python解释器编译为了WebAssembly，它也与JavaScript端具有深度的交互。
Pyodide会在许多Python与JavaScript之间的内建数据类型之间进行隐式转换。其中一些转换时很直接明显的，但如往常一样，那就是很有趣的极端情况。
![data](https://user-images.githubusercontent.com/6218739/140268286-c89f066f-72fc-4cc9-8709-c95fc6d98096.png)

## 访问Web API和DOM
可以通过以下方式获得Web页面上的文档对象模型DOM：
```python
from js import document
```
这会将document对象作为一个代理从JavaScript端导入到Python端。你可以开始从Python中对其调用方法:
```python
document.getElementById("myElement")
```

## 多维数组
在Python中， NumPy 数组是最常用的多维数组的实现。JavaScript具有TypedArrays，其仅含有一个单一的数值类型，但是是一维的，因此需要在其之上构建多维索引。
由于实际上这些数组可能会非常大，我们不想在语言运行时间拷贝它们。那不仅仅会花相当长的时间，而且在内存中同时保留两个拷贝将会加重浏览器所具有的被限制的内存的负担。
幸运的是，我们可以不用拷贝来共享数据。多维数组通常是用少量用于描述值类型和数组形状及内存分布的元数据来实现的。数据本身是从元数据中通过指针访问的另一个内存区域。该内存处于一个叫作“WebAssembly堆”的区域，这带来一个优势，因为其可以从JavaScript和Python中同时访问。我们可以简单地在语言之间拷贝元数据(其本身非常小)，并保持指针指向WebAssembly堆中的数据。
![numpy](https://user-images.githubusercontent.com/6218739/140268781-ab70c08a-1821-4a06-b7de-d65633d54b31.png)

## 实时交互可视化
在浏览器中进行数据科学计算相比于如Jupyter一样在远程内核中进行计算的一大优势就是，交互式可视化不用通过网络来传输数据并重新处理和展示这些数据。这很大程度地减少了延迟—用户移动鼠标的时刻与屏幕更新并显示图案的时刻之间的间隔时间。

要使得其能工作需要上面描述到的所有的技术片段能够很好地协同工作。我们使用matplotlib来看一下用于展示正态分布如何工作的交互性示例。首先，通过Python的Numpy产生随机数据。接下来，Matplotlib接管该数据，并使用内建的软件渲染器来将其绘出。它使用Pyodide对零拷贝共享数组的支持来将像素回馈给JavaScript端，在这里数据最终被渲染为HTML的画布。然后浏览器接管工作，将像素显示到屏幕上。用来支持交互性操作的鼠标和键盘事件通过从Web浏览器到Python的回调函数的调用来处理。
![matplotlib-interacting-with-plots](https://user-images.githubusercontent.com/6218739/140269007-b21d4a82-4def-4581-8631-331f570917e4.gif)

但需要注意的是Pyodide有两个缺点：
（1）包体积巨大，在浏览器中第一次访问内含Pyodide的网页时，会下载相应的python包，最基础的pyodide也有22MB大小，更不用说如果有额外的包，比如matplotlib，会更加巨大，导致长时间加载不出来页面；
（2）对于Python日渐火热的深度学习生态，Pyodide也没法直接利用，毕竟那些函数库会更大。

# 总结
JavaScript作为浏览器的原住民，其在基于网页的应用开发中的地位不可撼动，虽然Python能通过各种方式部分取代它的功能，但目前还很不成熟，开发简易功能尚可，但重度和高阶应用则基本不可能。期待以后的技术发展能将浏览器和Python结合得更加紧密。

# 参考文献
[在浏览器中用Python做数据科学：Pyodide](https://python.freelycode.com/contribution/detail/1567)
[LWN: Pyodide - 浏览器中的Python！](https://jishuin.proginn.com/p/763bfbd5bd1e)
[Pyodide: Bringing the scientific Python stack to the browser](https://hacks.mozilla.org/2019/04/pyodide-bringing-the-scientific-python-stack-to-the-browser/)
[把python装进浏览器，需要几个步骤？](https://www.bilibili.com/video/BV1X541187XK)
[Iodide and Pyodide: Bringing Data Science Computation to the Web Browser - Michael Droettboom](https://www.youtube.com/watch?v=iUqVgykaF-k&t=91s)
