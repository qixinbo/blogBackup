---
title: 开源深度学习计算平台ImJoy解析：1 -- 介绍
tags: [ImJoy]
categories: computer vision 
date: 2021-11-28
---

从该博文开始，将会对ImJoy这一开源深度学习计算平台做一详细解析。
# 简介
（这部分是对官方文档（在[这里](https://imjoy.io/docs/#/)）的翻译理解）
ImJoy是一个由插件驱动的混合计算平台，用于部署深度学习应用程序，例如高级图像分析工具。
ImJoy可以运行在跨操作系统的移动和桌面环境中，其中的插件可以运行在浏览器、本地主机、远程和云服务器中。
借助 ImJoy，凭借其灵活的插件系统和可共享的插件 URL，可以非常简单地向最终用户提供深度学习工具，免去了用户自己配置深度学习环境、安装应用程序的繁琐和痛苦。对于开发人员来说，也可以轻松地对自己现有的Python代码添加丰富的交互式 Web 界面，从而让自己的程序更加“触手可及”。
下面是ImJoy的整体架构图：
![arch](https://user-images.githubusercontent.com/6218739/143673530-061125ed-4f2c-4bbd-8cef-b8ea12921c92.png)
可以看出，ImJoy系统非常灵活，体现在以下几个方面：
（1）跨平台获取：因为ImJoy是基于web的，所以只要是有浏览器的地方，ImJoy就可以使用，比如桌面端、移动端调用等；
（2）插件形式灵活：可以使用JavaScript、Python等编程语言；
（3）插件运行环境多样：对于不同量级的插件，可以选择其应用环境，比如一个简单的插件，可以直接在浏览器中运行；如果是一个重型的深度学习应用，可以在本地工作站中运行，也可以在远程服务器或者云服务器中运行。

# ImJoy特点
（1）小巧且灵活的插件驱动的 Web 应用程序
（2）具有离线支持的无服务器渐进式 Web 应用程序（PWA技术）
（3）支持移动设备
（4）基于Web的丰富的交互式用户界面：可以使用任何现有的网页设计库、使用 webGL、Three.js 等以 3D 形式呈现多维数据。
（5）易于使用的工作流组合
（6）用于分组插件的独立工作区
（7）方便的插件原型设计和开发：内置代码编辑器，开发不需要额外的IDE
（8）强大且可扩展的计算后端，可用于浏览器内计算、本地计算和云计算
- 支持 Javascript、原生 Python 和 web Python（即直接在网页中运行Python程序，底层技术是Pyodide）
- 通过异步编程并发插件执行
- 使用 Webassembly 在浏览器中运行 Python 插件
- 浏览器插件与安全沙箱隔离
- 支持Python3 和 Javascript 的async/await语法
- 支持 Python 的 Conda 虚拟环境和 pip 包
- 支持托管在 Github 或 CDN 上的 JavaScript 库
- 通过 GitHub 或 Gist 轻松部署和共享插件
- 将开发者自己的插件仓库部署到 Github
- 原生支持 n 维数组和张量
- 支持 Numpy 的 ndarrays 用于数据交换

ImJoy 大大加快了新工具的开发和传播。开发者可以在 ImJoy 中开发插件，将插件文件部署到 Github，并通过社交网络分享插件 URL。用户可以通过多种方式使用这些插件，比如在手机上单击一下即可调用。
![deploy](https://user-images.githubusercontent.com/6218739/143677962-526c570c-e61e-423e-8092-e78a013ef231.png)

# 依赖库
ImJoy主要使用的开源库有：
- Joy.js（这就是ImJoy的名字由来！）
- Jailed（用于隔离插件）
- Vue.js（主要的前端UI使用 Vue.js 编写）
- vue-grid-layout（用于窗口管理）
- python-socketio（使得插件引擎可以与 ImJoy主程序进行通信）
- pyodide（使用 WebAssembly 启用 web python 模式）
- conda（插件引擎使用 Conda 来管理虚拟环境和包）
- docsify（ImJoy 文档是用 docsify 创建的）

# 发表论文
ImJoy的研究工作也发表在了Nature子刊 Nature Methods上，大佬就是大佬。
文章链接见：[ImJoy: an open-source computational platform for the deep learning era](https://www.nature.com/articles/s41592-019-0627-0)
也可以通过[这个链接](https://rdcu.be/bYbGO)免费获取该论文。

# 快速上手
## 前端界面
可以直接在浏览器中使用ImJoy，网站在[这里](https://imjoy.io/#/app)。
（也可以自己托管ImJoy，即使用GitHub上的[这个仓库](https://github.com/imjoy-team/ImJoy)）
整个应用的前端界面如下：
![UI](https://user-images.githubusercontent.com/6218739/143802507-b85944e6-4a5d-49d6-82d8-fcceb76991a3.png)
包括了插件管理区、工作区、状态栏、工具栏、插件窗口等多个部分。

## 上手体验
官方提供的一个demo是使用一个预训练的神经网络来进行图像识别。
这个插件可以通过在[插件库](https://imjoy.io/repo/)中安装插件Image Recognition来获得，
也可以直接点击[该链接](https://imjoy.io/#/app?plugin=imjoy-team/imjoy-plugins:Image%20Recognition&w=getting-started)来使用。
安装插件后，它将出现在左侧的插件对话框中。然后单击其名称启动插件。这将打开一个窗口并加载训练好的网络。
然后就可以通过上传文件来预测图像中的物体。
注意，如果是在电脑上使用该插件，则是上传电脑中的文件，如果是在手机上使用该插件，则调用摄像头来获取图像。
如下是我在手机上试用的截图：
![test](https://user-images.githubusercontent.com/6218739/143830446-897d525c-c84e-4e85-a689-8d1fcab75b0e.jpg)

通过此例也可以看出，ImJoy提供了一种非常方便地获取最新深度学习技术的方式，能极大地降低技术的应用门槛。
