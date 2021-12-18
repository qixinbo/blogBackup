---
title: 开源深度学习计算平台ImJoy解析：2 -- 核心概念
tags: [ImJoy]
categories: computer vision 
date: 2021-11-30
---

这一篇主要介绍ImJoy中的核心概念。
参考文献：
[I2K Workshop Tutorial](https://imjoy.io/docs/#/i2k_tutorial)


# ImJoy插件
ImJoy 提供了一个灵活的框架来开发具有不同类型的 Web 或 Python 编程语言的插件。
![plugins](https://user-images.githubusercontent.com/6218739/143839866-7d945048-055f-4898-b65b-64af66d4d316.png)
有四种类型的插件，其可用于不同的目的：
（1）Web 插件直接在浏览器中运行，支持如下三种类型：
- Window (HTML/CSS/JS)(`type=window`) 插件，用于使用 HTML5/CSS 和 JavaScript 构建丰富的交互式用户界面；
- Web Worker (JS)(type=`web-worker`) 插件，用于使用 JavaScript 或 WebAssembly 执行计算任务；
- Web Python(type=`web-python`) 插件，用于在浏览器中通过 WebAssembly 和 [pyodide](https://github.com/iodide-project/pyodide) 使用 Python 执行计算任务。这样的插件用小蛇🐍图标表示。这处于开发阶段，目前仅支持选定数量的 Python 库。

（2）Native插件在插件引擎中运行，目前支持：
- Native Python(type=`native-python`) 插件，可使用 Python 及其大量库函数来执行繁重计算任务，不过这需要额外安装插件引擎。这些插件用火箭🚀图标表示。

可以通过单击 `+ PLUGINS` 按钮，然后从“创建新插件”下拉菜单中访问上述插件的模板，如图：
![4plugins](https://user-images.githubusercontent.com/6218739/144004594-0ccaac1d-6fb0-4c83-a42d-8dc1c565f823.png)

关于插件具体怎样编写，会在后面博文中具体解析。

# ImJoy API
为了允许基本的用户交互，ImJoy 提供了一组 API（应用程序编程接口）函数，这些函数可以在所有插件类型和支持的编程语言中以相同的方式调用。
例如，与 Javascript 函数 `alert()`等效的 ImJoy API 函数是 `api.alert()`。

可以直接访问 Javascript 插件中的 `api` 对象（使用 type=`window` 或 `web-worker`）：
```js
api.alert("Hello from ImJoy!")
```
在 Python 插件（type=`web-python` 或 `native-python`）中，需要先添加 `from imjoy import api`，然后才能访问 `api` 对象。
```python
# import api object
from imjoy import api
...
# use api object
api.alert("Hello from ImJoy!")
```

可以在 [此处](https://imjoy.io/docs/#/api) 中找到所有 ImJoy API 功能的详细说明。同样，后面会对这些API详细解析。

# 远程过程调用RPC
首先来看一下什么是远程过程调用**Remote Procedure Calls (RPC)**。
洪春涛的[这个知乎回答](https://www.zhihu.com/question/25536695/answer/221638079)非常言简意赅。以下是对该回答的摘引。

## 本地过程调用
RPC就是要像调用本地的函数一样去调远程函数。在研究RPC前，我们先看看本地调用是怎么调的。假设我们要调用函数Multiply来计算`lvalue * rvalue`的结果:
```python
1 int Multiply(int l, int r) {
2    int y = l * r;
3    return y;
4 }
5 
6 int lvalue = 10;
7 int rvalue = 20;
8 int l_times_r = Multiply(lvalue, rvalue);
```
那么在第8行时，我们实际上执行了以下操作：
（1）将lvalue和rvalue的值压栈
（2）进入Multiply函数，取出栈中的值10和20，将其赋予l和r
（3）执行第2行代码，计算`l*r`，并将结果存在y
（4）将y的值压栈，然后从Multiply返回
（5）第8行，从栈中取出返回值200 ，并赋值给`l_times_r`
以上5步就是执行本地调用的过程。（20190116注：以上步骤只是为了说明原理。事实上编译器经常会做优化，对于参数和返回值少的情况会直接将其存放在寄存器，而不需要压栈弹栈的过程，甚至都不需要调用call，而直接做inline操作。仅就原理来说，这5步是没有问题的。）

## 远程过程调用带来的新问题
在远程调用时，我们需要执行的函数体是在远程的机器上的，也就是说，Multiply是在另一个进程中执行的。这就带来了几个新问题：
（1）Call ID映射。我们怎么告诉远程机器我们要调用Multiply，而不是Add或者FooBar呢？在本地调用中，函数体是直接通过函数指针来指定的，我们调用Multiply，编译器就自动帮我们调用它相应的函数指针。但是在远程调用中，函数指针是不行的，因为两个进程的地址空间是完全不一样的。所以，在RPC中，所有的函数都必须有自己的一个ID。这个ID在所有进程中都是唯一确定的。客户端在做远程过程调用时，必须附上这个ID。然后我们还需要在客户端和服务端分别维护一个`{函数 <--> Call ID}`的对应表。两者的表不一定需要完全相同，但相同的函数对应的Call ID必须相同。当客户端需要进行远程调用时，它就查一下这个表，找出相应的Call ID，然后把它传给服务端，服务端也通过查表，来确定客户端需要调用的函数，然后执行相应函数的代码。
（2）序列化和反序列化。客户端怎么把参数值传给远程的函数呢？在本地调用中，我们只需要把参数压到栈里，然后让函数自己去栈里读就行。但是在远程过程调用时，客户端跟服务端是不同的进程，不能通过内存来传递参数。甚至有时候客户端和服务端使用的都不是同一种语言（比如服务端用C++，客户端用Java或者Python）。这时候就需要客户端把参数先转成一个字节流，传给服务端后，再把字节流转成自己能读取的格式。这个过程叫序列化和反序列化。同理，从服务端返回的值也需要序列化反序列化的过程。
（3）网络传输。远程调用往往用在网络上，客户端和服务端是通过网络连接的。所有的数据都需要通过网络传输，因此就需要有一个网络传输层。网络传输层需要把Call ID和序列化后的参数字节流传给服务端，然后再把序列化后的调用结果传回客户端。只要能完成这两者的，都可以作为传输层使用。因此，它所使用的协议其实是不限的，能完成传输就行。尽管大部分RPC框架都使用TCP协议，但其实UDP也可以，而gRPC干脆就用了HTTP2。Java的Netty也属于这层的东西。

## RPC的实现
```python
// Client端 
//    int l_times_r = Call(ServerAddr, Multiply, lvalue, rvalue)
1. 将这个调用映射为Call ID。这里假设用最简单的字符串当Call ID的方法
2. 将Call ID，lvalue和rvalue序列化。可以直接将它们的值以二进制形式打包
3. 把2中得到的数据包发送给ServerAddr，这需要使用网络传输层
4. 等待服务器返回结果
5. 如果服务器调用成功，那么就将结果反序列化，并赋给l_times_r

// Server端
1. 在本地维护一个Call ID到函数指针的映射call_id_map，可以用std::map<std::string, std::function<>>
2. 等待请求
3. 得到一个请求后，将其数据包反序列化，得到Call ID
4. 通过在call_id_map中查找，得到相应的函数指针
5. 将lvalue和rvalue反序列化后，在本地调用Multiply函数，得到结果
6. 将结果序列化后通过网络返回给Client
```
所以要实现一个RPC框架，其实只需要按以上流程实现就基本完成了。

其中：
（1）Call ID映射可以直接使用函数字符串，也可以使用整数ID。映射表一般就是一个哈希表。
（2）序列化反序列化可以自己写，也可以使用Protobuf或者FlatBuffers之类的。
（3）网络传输库可以自己写socket，或者用asio，ZeroMQ，Netty之类。


## ImJoy中的远程过程调用
尽管调用 `alert()` 和 `api.alert()` 会产生相同的结果（都是弹出消息），但要注意的是其底层过程是不同的。当调用`alert()`时，直接从插件启动弹出对话框，而调用`api.alert()`会从ImJoy内核（ImJoy core）中启动弹出对话框。
需要时刻注意的是，ImJoy 是在独立或沙盒环境（即sandboxed iframe、webworker、conda 虚拟环境或 docker 容器）中运行每个插件。简而言之，这意味着默认情况下，函数和变量不会在插件之间或插件与ImJoy内核之间进行共享。
当从插件中调用 ImJoy API 函数时，该函数将在 ImJoy 内核中执行。由于插件运行在不同的环境中，所以ImJoy内核中定义的所有功能都是“远程”功能。相比之下，同一个插件中定义的所有函数都是“本地”的。
因此，调用ImJoy API函数意味着执行远程过程调用。
（ImJoy支持双向RPC，不仅在插件和 ImJoy内核之间，而且在插件之间也是如此。RPC可以在不同编程语言和不同主机之间统一地使用）
比如，当一个在远程服务器上运行的Python插件进行调用`api.alert()`时，弹出对话框则是由用户浏览器中的ImJoy内核（用Javascript实现）来启动的。
RPC允许将任务分发到以不同语言和不同位置运行的不同插件。例如，我们可以使用强大的UI库（例如[D3](https://d3js.org/) 和 [ITK/VTK Viewer](https://kitware.github.io/itk-vtk-viewer/))来构建用户界面，并用[Tensorflow. js](https://www.tensorflow.org/js)中的[Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) 来运行深度学习模型 。对于使用GPU来训练模型这种重型计算任务，可以在本地或远程（例如在GPU 集群或实验室工作站上）的Jupyter笔记本服务器（即ImJoy插件引擎 Plugin Engine）上编写Python 插件来实现。
这篇博文([RPCs, Life and All](http://tomerfiliba.com/blog/RPCs-Life-And-All/)) 解释了用于Python远程过程调用的库([RPyC](https://rpyc.readthedocs.io/en/latest/))背后的想法 ，该库与ImJoy中提供的类似。


# 异步编程
由于 ImJoy API 函数是远程函数，它们的操作与同一插件中定义的本地函数略有不同。更具体地说，远程函数是异步的。
在ImJoy中调用异步函数有一个简化的规则：ImJoy中的所有远程功能都是异步的，可以像其他本地函数一样使用它们，只需在函数调用前添加 `await`。
即应该执行 `await api.alert('hello')` 来调用上面的alert函数。
如果API函数有返回值，例如[`api.prompt`](https://imjoy.io/docs/#/api?id=apiprompt)，应该写成：`result = await api.prompt( 'type a number')`。

但是需要注意的是，如果直接执行 `await api.alert('hello')`，会导致语法错误而不能执行。
要修复错误，需要将代码封装在一个异步函数中：
```js
// Javascript 中的异步/等待示例
async sayHello(){
    await api.alert("Hello from ImJoy!")
}
sayHello()
```
因此，另一个使用 `async/await` 的简单规则是：
在函数中使用`await`时，在函数定义前添加`async`。

再举一个例子，使用另一个 ImJoy API 函数 [`api.prompt`](https://imjoy.io/docs/#/api?id=apiprompt) 在弹出对话框中获取用户的输入，并使用这个API [`api.showMessage`](https://imjoy.io/docs/#/api?id=apishowmessage)来显示消息。
```js
async function choosePokemon(){
    const pokemon = await api.prompt("What is your favorite Pokémon?","Pikachu")
    await api.showMessage("Your have chose "+pokemon +" as your Pokémon.")
}

choosePokemon()
```

## Python的Async/Await
`async/await` 语法在 Python 中类似。例如：
```python
# Python 中的异步/等待示例
async def say_hello():
    await api.alert("Hello from ImJoy!")
```

在 Python 中使用 asyncio 时，一个好的做法是避免直接在主线程中运行繁重的计算，而是可以使用[Executors](https://pymotw.com/3/asyncio/executors.html) (线程和进程）。
还可以通过执行以下操作来使用默认线程执行器：
```python
loop.run_in_executor(None, my_heavy_computation, arg1, arg2...)
```

## Callback、Promise和Async/Await
如前所述，通过 RPC 将任务分配给不同插件的一个优势是可以并行调度和运行任务（通常在 Python、Java 和许多其他编程语言中，还有许多其他技术可以实现并发性，包括多线程和多进程）。异步编程是一种越来越流行的以更具可扩展性的方式实现并发的方式。
其基本思想是，我们不必总是等待一项任务完成，然后才移动到下一项。比如，当我们去一家咖啡店，点一杯卡布奇诺咖啡并获得一张取餐号，在制作咖啡的同时，我们可以拨打电话或阅读报纸。几分钟后，可以通过出示取餐号来获取卡布奇诺咖啡。
异步编程与多线程等其他技术的一大区别在于程序是在一个线程和进程中运行。因此，在 ImJoy 中，异步编程通常用于将任务调度到其他插件，而不是在同一插件内并行运行繁重的计算任务。
`async/await` 并不是进行异步编程的唯一方式，事实上，它在最近几年才变得更加流行。例如， Python 3 之后才引入了它。
关于异步编程，可以后面再详细解析。

## 将 RPC 与 Async/Await 结合使用
另一种理解`await` 和`async` 函数的角度是：
1) 异步函数一旦调用将立即返回；
2) 返回的对象不是实际结果，而是Javascript 中称为`Promise` 或Python中称为`Future` 的特殊对象。直觉上，这就像你点了一杯咖啡后得到的取餐号；
3) 如果将 `await` 应用到 `Promise` 或 `Future` 对象，就会得到实际的结果。
如下两种异步函数是等价的：
```js
async function choosePokemon1(){
    // 直接申请await，我们会得到实际的结果
    const pokemon = await api.prompt("What is your favorite Pokemon?", "Pikachu")
    return pokemon
}
async function choosePokemon2(){
    // 如果不使用 `await`，我们会得到一个对实际结果的承诺promise
    const promise = api.prompt("What is your favorite Pokemon?", "Pikachu"")
    // 要检索实际结果，将 await 应用于 Promise
    const pokemon = await promise
    return pokemon
}
```
虽然上面的例子是用 Javascript 写的，当然也可以在 Python 中做同样的事情。
简单地为所有异步函数应用`await` 将导致顺序执行。要并行运行任务，我们可以在不立即应用 `await` 的情况下调用函数，而是可以先收集所有的 `Promise` 对象，然后一块`await`。
假设我们有 taskA（需要 10 分钟）、taskB（需要 5 分钟）和 taskC（需要 3 分钟），我们想使用从 A 和 B 返回的结果来完成任务 C。以下是不同的实现方式：
（1）在所有函数之前应用 `await`，需要 18(`10+5+3`) 分钟
```js
function doTasks(){
        // 在 A 之后执行任务 B
        const resultA = await doTaskA() // 需要 10 分钟
        const resultB = await doTaskB() // 需要 5 分钟
        return await doTaskC(resultA, resultB) // 需要 3 分钟
}
```
（2）调度这两个任务，然后对两者`await`，需要 13 (`max(10, 5) + 3`) 分钟。
在 Javascript 中，可以使用 `Promise.all` 将两个 promise 合二为一：
```js
function doTasks(){
    // 并行运行任务 A 和 B
    const promiseA = doTaskA()
    const promiseB = doTaskB()
    // 收集结果
    const [resultA, resultB] = await Promise.all([promiseA, promiseB])
    return await doTaskC(resultA, resultB)
}
```
在 Python 中，可以使用 `asyncio.gather` 来收集两个 promise：
```python
import asyncio
async def doTasks():
    # 并行运行任务 A 和 B
    promiseA = doTaskA()
    promiseB = doTaskB()
    # 收集结果
    [resultA, resultB] = await asyncio.gather(promiseA, promiseB)
    return await doTaskC(resultA, resultB)
```

# 外部集成
ImJoy 插件生态系统旨在以两种方式开放：
（1）其他软件工具和网站应该能够轻松使用 ImJoy 及其插件；
（2）其他软件工具应该可以在 ImJoy 中轻松使用，通常是以插件的形式。
一般来说，任何使用ImJoy RPC协议来提供服务功能的软件都可以被视为ImJoy插件。这包括 ImJoy Web 应用程序本身，它可以读取插件文件并生成插件 API。同时，作者还提供了 [imjoy-rpc](https://github.com/imjoy-team/imjoy-rpc) 库，目前支持 Python 和 Javascirpt，供其他软件或 Web 应用程序直接与 ImJoy 内核通信。
目前已经有几个web 应用程序可以在独立模式下运行，也可以作为 ImJoy 插件：
- [ITK/VTK 查看器](https://kitware.github.io/itk-vtk-viewer/docs/imjoy.html) 由 [Matt McCormick](https://github.com/thewtex) 等人撰写。
- [vizarr](https://github.com/hms-dbmi/vizarr) 由 [Trevor Manz](https://github.com/manzt) 等人撰写。
- [Kaibu](https://kaibu.org/#/app) 由 ImJoy 团队提供。
- [ImageJ.JS](https://ij.imjoy.io) 由 ImJoy 团队提供。


例如，[ITK/VTK Viewer](https://kitware.github.io/itk-vtk-viewer/docs/imjoy.html) 是一个开源软件系统，用于医学和科学图像、网格和点集可视化。虽然它可以[作为独立应用程序运行](https://kitware.github.io/itk-vtk-viewer/app/?fileToLoad=https://data.kitware.com/api/v1/file/564a65d58d777f7522dbfb61/ download/data.nrrd)，也可以[作为 ImJoy 插件](https://kitware.github.io/itk-vtk-viewer/docs/imjoy.html)运行 。
可以点击[这个链接](http://imjoy.io/#/app?plugin=https://kitware.github.io/itk-vtk-viewer/app/)进行试用。

[ImageJ.JS](https://ij.imjoy.io)是一个独立的网络应用程序，它以两种方式支持ImJoy：1) 大多数ImJoy插件可以在ImageJ.JS中直接运行； 2) ImageJ.JS可以通过其URL用作ImJoy的插件。
有关更多详细信息，请参阅 [项目存储库](https://github.com/imjoy-team/imagej.js)。

比如，可以在ImageJ.JS的左上角单击ImJoy图标，然后选择加载插件，粘贴插件的Github/Gist URL，即可将自己的插件加载到ImageJ.JS中。
![imagej](https://user-images.githubusercontent.com/6218739/144003370-395d9e87-7469-4f51-9753-4bc7b6e5e00a.png)
