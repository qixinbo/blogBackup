---
title: Electron初探：基于Web的跨平台桌面应用开发
tags: [GUI]
categories: digitalization
date: 2021-11-15
---

# 参考文献
[你能分得清楚 Chromium, V8, Blink, Gecko, WebKit 之间的区别吗？](https://juejin.cn/post/6844904055236460558)
[丝般顺滑的 Electron 跨端开发体验](https://msyfls123.github.io/blog/2020/11/02/%E4%B8%9D%E8%88%AC%E9%A1%BA%E6%BB%91%E7%9A%84Electron%E8%B7%A8%E7%AB%AF%E5%BC%80%E5%8F%91%E4%BD%93%E9%AA%8C/)
[Electron 免费视频教程-用前端技术开发桌面应用](https://jspang.com/detailed?id=62#toc34)
[Electron 快速入门](https://weishuai.gitbooks.io/electron-/content/tutorial/quick-start.html)


# 基础概念
## 引擎
JavaScript引擎的作用是解释和编译JavaScript代码。
而浏览器引擎不仅负责管理网页的布局，同时其包括JavaScript引擎。
当前市场上只有 3 个主要的浏览器引擎：Mozilla 的 Gecko、Google 的 Blink、还有苹果的的 WebKit（Blink 的近亲）。
Blink 是 Google Chrome浏览器及Chromium开源浏览器（可以理解为：Chromium + 集成 Google 产品 = Google Chrome）的渲染引擎，V8 是 Blink 内置的 JavaScript 引擎。具体来说，V8 对 DOM（文档对象模型）一无所知，因为它仅用于处理 JavaScript；而Blink 内置的布局引擎负责处理网页布局和展示。

## 后端
Node.js 就是运行在服务端的 JavaScript，类比Java后端、Python后端等。
因为 Node.js 不需要使用 DOM，所以 Node.js 只使用了 V8 引擎，而没有把整个 Blink 引擎都搬过来用。

## Electron
Electron = Chromium + Node.js + Native API
（1）Chromium : 为Electron提供了强大的UI能力，可以不考虑兼容性的情况下，利用强大的Web生态来开发界面。
（2）Node.js ：让Electron有了底层的操作能力，比如文件的读写，甚至是集成C++等等操作，并可以使用大量开源的npm包来完成开发需求。
（3）Native API ： Native API让Electron有了跨平台和桌面端的原生能力，比如说它有统一的原生界面，窗口、托盘这些。

Electron作用是用Web前端技术来开发桌面应用。

具体原理：
Electron 就是 Chromium（Chrome 内核）、Node.js 和系统原生 API 的结合。它做的事情很简单，整个应用跑在一个 main process（主进程） 上，需要提供 GUI 界面时则创建一个 renderer process（渲染进程）去开启一个 Chromium 里的 BrowserWindow/BrowserView，实际就像是 Chrome 的一个窗口或者 Tab 页一样，而其中展示的既可以是本地网页也可以是线上网页，主进程和渲染进程间通过 IPC 进行通讯，主进程可以自由地调用 Electron 提供的系统 API 以及 Node.js 模块，可以控制其所辖渲染进程的生命周期。

### 主进程
在 Electron 里，运行 package.json 里 main 脚本的进程被称为主进程。在主进程运行的脚本可以以创建 web 页面的形式展示 GUI。

### 渲染进程
由于 Electron 使用 Chromium 来展示页面，所以 Chromium 的多进程结构也被充分利用。每个 Electron 的页面都在运行着自己的进程，这样的进程我们称之为渲染进程。
在一般浏览器中，网页通常会在沙盒环境下运行，并且不允许访问原生资源。然而，Electron 用户拥有在网页中调用 Node.js 的 APIs 的能力，可以与底层操作系统直接交互。

### 主进程与渲染进程的区别
主进程使用 BrowserWindow 实例创建页面。每个 BrowserWindow 实例都在自己的渲染进程里运行页面。当一个 BrowserWindow 实例被销毁后，相应的渲染进程也会被终止。
主进程管理所有页面和与之对应的渲染进程。每个渲染进程都是相互独立的，并且只关心他们自己的页面。
由于在页面里管理原生 GUI 资源是非常危险而且容易造成资源泄露，所以在页面调用 GUI 相关的 APIs 是不被允许的。如果你想在网页里使用 GUI 操作，其对应的渲染进程必须与主进程进行通讯，请求主进程进行相关的 GUI 操作。
在 Electron，我们提供几种方法用于主进程和渲染进程之间的通讯。像 ipcRenderer 和 ipcMain 模块用于发送消息， remote 模块用于 RPC 方式通讯。


# 配置环境
## 安装Electron
可以全局安装：
```python
npm install -g electron
```
或者仅项目安装：新建一个文件夹，然后，
```python
npm install electron --save-dev
```
然后使用以下命令查看是否安装成功：
```python
npx electron -v
或
./node_modules/.bin/electron -v
```

# Electron的Hello World
## 新建index.html文件
在项目的根目录中新建一个index.html文件，相当于UI都写在html中（可以在sublimetext输入html自动生成）：
```python
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hello World</title>
</head>
<body>
    hello World
</body>
</html>
```

## 新建main.js文件
在根目录下新建一个main.js文件，这个就是Electron的主进程文件。
```python
var electron = require('electron')  //引入electron模块

var app = electron.app   // 创建electron引用

var BrowserWindow = electron.BrowserWindow;  //创建窗口引用

var mainWindow = null ;  //声明要打开的主窗口
app.on('ready',()=>{
    mainWindow = new BrowserWindow({width:400,height:400})   //设置打开的窗口大小

    mainWindow.loadFile('index.html')  //加载那个页面

    //监听关闭事件，把主窗口设置为null
    mainWindow.on('closed',()=>{
        mainWindow = null
    })

})
```

## 创建package.json文件
在终端使用命令：
```python
npm init --yes
```
这时候main的值为main.js就正确了。

## 运行
终端下运行：
```python
.\node_modules\.bin\electron .
```

然后结果为：
![hello](https://user-images.githubusercontent.com/6218739/141733207-8ae86b3e-95ab-4c5d-8e49-6f11c5721fd7.png)


试了这个最小例子，感觉使用electron来开发桌面应用的话，既能跨平台，比如Windows、Linux、MacOS，一处水源供全球，还能直接转化成Web应用，即不让用户安装软件，给他一个链接直接访问。
这样就可进可退，一次开发，到处使用，但前提是得熟悉JS开发，这个坑待填。。