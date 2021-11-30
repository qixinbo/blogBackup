---
title: 开源深度学习计算平台ImJoy解析：3 -- 插件原理及上手
tags: [ImJoy]
categories: computer vision 
date: 2021-12-1
---

# 参考文献
[I2K Workshop Tutorial](https://imjoy.io/docs/#/i2k_tutorial?id=_2-make-your-first-imjoy-plugin)
[Developing Plugins for ImJoy](https://imjoy.io/docs/#/development)

# 概述
开发ImJoy插件既简单又快速，可直接使用运行在web上的内置的代码编辑器，而不需要额外的 IDE 或编译器。
ImJoy 插件系统的主要功能有：
（1）支持 Python 和 JavaScript
- JavaScript 插件与安全沙箱隔离
- Python 插件在自己的进程中运行
- 使用 `async/await` 语法支持并发 API 调用
- 支持 Python 的虚拟环境和 pip 包
- 支持托管在 GitHub 或 CDN 上的 JavaScript 库

（2）原生支持 n 维数组和张量
- 支持来自 Numpy 或 Numjs 的 ndarrays 进行数据交换
- 支持用于深度学习的 Tensorflow.js 和原生 Tensorflow、PyTorch、MxNet等深度学习库

（3）使用 webGL、Three.js 等3D库渲染多维数据
（4）使用 GitHub 部署自定义的插件

# ImJoy架构
Imjoy 包含**两个主要组件**：
（1）**ImJoy Web App**：这是ImJoy 的核心部分，它是一个web应用，所以可在不同操作系统和设备的浏览器中运行。它提供了一个灵活的插件系统，具有工作流和窗口管理功能。插件可以用不同的编程语言开发，包括 JavaScript 和 Python。插件及其源代码可以组织到工作区中并存储在浏览器数据库中。 Web 插件在 `iframe` 或 `webworker` 中运行，因此开发者原则上可以为每个插件独立使用任何前端框架或 javascript 库。
（2）**Plugin Engine**：这是一个可选项，其用于在 CPython 中运行计算任务，以利用本机硬件（例如 GPU）和Python软件库（例如：numpy、Tensorflow、PyTorch 等）的强大功能。实际上，该插件引擎是一个在后台运行并通过 websocket 连接到 ImJoy Web App 的 Python 包 ([GitHub](https://github.com/imjoy-team/imjoy-engine))。它使用 [conda](https://conda.io/) 来管理软件包（不仅是 Python，还有 C++、Java 等）和虚拟环境。然后开发者可以将具体的`conda` 或 `pypi` 包作为需求添加到插件中，它们可以由插件引擎自动解析。同样，开发人员可以在 Python 插件中使用任何 Python 库甚至非 Python 库。
![full-arch](https://user-images.githubusercontent.com/6218739/144013956-2be625bd-3a30-4c10-be13-c138513d25be.png)


Plugin Engine 通过 websockets 与 ImJoy Web App 连接， 并与一个基于 [socket.io](https://github.com/miguelgrinberg/python-socketio) 定制化的RPC（remote procedure calls）进行通信。

# 什么是ImJoy插件
简而言之，ImJoy插件是一个脚本，它生成一组可以被ImJoy内核或其他插件调用的服务函数（又名插件 API 函数）。
目前有 4 种常见的插件类型：`window`、`web-worker`、`web-python`、`native-python`。
（1）Web 插件直接在浏览器中运行，支持如下三种类型：
- Window (HTML/CSS/JS)(`type=window`) 插件，用于使用 HTML5/CSS 和 JavaScript 构建丰富的交互式用户界面；
- Web Worker (JS)(type=`web-worker`) 插件，用于使用 JavaScript 或 WebAssembly 执行计算任务；
- Web Python(type=`web-python`) 插件，用于在浏览器中通过 WebAssembly 和 [pyodide](https://github.com/iodide-project/pyodide) 使用 Python 执行计算任务。这样的插件用小蛇🐍图标表示。这处于开发阶段，目前仅支持选定数量的 Python 库。

（2）Native插件在插件引擎中运行，目前支持：
- Native Python(type=`native-python`) 插件，可使用 Python 及其大量库函数来执行繁重计算任务，不过这需要额外安装插件引擎。这些插件用火箭🚀图标表示。

## Window插件
Window 插件用于创建一个包含 HTML/CSS 和 JavaScript 的新 Web 界面。其是在`iframe`模式下创建的，将显示为一个窗口。 `<window>` 和 `<style>`块（见以后插件文件格式的详细解析）可用于定义窗口的实际内容。
不同于其他插件会在 ImJoy 启动时加载和初始化，只有在使用`api.createWindow`创建实际插件或用户在菜单中单击时，才会加载`window`插件。在`api.createWindow`执行过程中，`setup`和`run`会被首先调用，并返回一个窗口api对象（包含窗口的所有api函数，包括`setup`、`run`和其他功能（如果已定义））。然后就可以使用这个window api 对象来访问所有的函数，例如通过`win_obj.run({'data': ... })` 来更新窗口的内容。

## Web Worker插件
Web Worker插件用于在另一个线程中执行计算任务，具体途径是使用一个名为 [web worker](https://en.wikipedia.org/wiki/Web_worker) 的新元素。
它没有接口，是在一个新线程中运行，并且在运行过程中不会挂起主线程。它基本上是 JavaScript 实现多线程的一种方式。
由于 Web Worker 旨在执行计算任务，它们无权访问 [html dom](https://www.w3schools.com/whatis/whatis_htmldom.asp)，但是开发者可以使用`ImJoy API`来与ImJoy的图形界面或其他可以触发用户界面更改的插件进行交互。

## Web Python插件
Web Python插件可以完全在浏览器中运行 python 代码和科学库。ImJoy 使用 [pyodide](https://github.com/iodide-project/pyodide/) 来运行 python 插件，它支持通过 WebAssembly 运行带有科学库（包括 numpy、scipy、scikit-learn 等）的 Python3 代码。

## Native Python插件
Native Python插件用于运行原生 Python 代码以完全访问电脑硬件（例如 GPU、NVLINK）和软件（例如 CUDA 和 CUDNN）环境。这需要在使用插件之前安装并启动**Python Plugin Engine**。
与 Web Worker 插件类似，Native Python 插件无法访问 html dom，但可以使用 `ImJoy API` 与ImJoy 的图形界面或其他可以触发用户界面更改的插件进行交互。

## 更多插件类型
插件类型可以通过插件进一步扩展。例如，作者新创建一个新的插件类型来执行Fiji/Scijava脚本，参见 [这篇文章](https://forum.image.sc/t/making-imjoy-plugins-with-fiji-scripts-for-running-remotely/39503)。

## ImJoy App和Plugin Engine与插件的关系
使用 ImJoy App 的推荐方式是通过 [https://imjoy.io](https://imjoy.io)。现代浏览器（例如 Google Chrome）越来越支持运行和使用称为渐进式 Web 应用 (PWA) 的 Web 应用程序的新方法。
例如，在 Chrome 中，用户可以将 ImJoy 安装到 [chrome://apps/](chrome://apps/) 并从 ImJoy App 仪表板启动。一旦安装，ImJoy 就可以在独立的浏览器窗口中运行（没有地址栏）。 ImJoy 的内核部分也支持离线，但插件目前还不支持（作者说后面将支持）。
可以使用ImJoy App运行所有web插件（`web-worker`、`window`、`web-python`），但是，对于本机插件（`native-python`），需要连接到插件引擎在本地或远程运行。
以下是安装插件引擎的两个选项：
如需使用插件引擎运行插件，请下载并安装 Anaconda 或 Miniconda with Python3，然后运行`pip install imjoy`。然后可以通过 `imjoy --jupyter` 命令启动插件引擎。更多详细信息可在 [此处](https://github.com/imjoy-team/imjoy-engine/) 获得。

# ImJoy代码编辑器和开发人员工具
ImJoy提供了一个内置的代码编辑器供编写插件。结合浏览器提供的调试工具（例如：Google Chrome 开发者工具），不需要额外的 IDE 或工具。
可以通过单击插件菜单（插件名称旁边的图标）中的“Edit”来查看和修改任何现有插件的插件代码。
![editor](https://user-images.githubusercontent.com/6218739/144052196-ace51e31-757c-4519-988d-904bb3a1d89a.png)

[Chrome 开发者工具](https://developers.google.com/web/tools/chrome-devtools) 提供了不同的调试HTML/CSS/Javascript、网络等的工具。建议使用它来调试Web插件。例如，可以像正常JavaScript开发那样在JavaScript中使用 `console.log()`、`console.error()`等，然后在浏览器控制台中检查日志和错误。在Python插件中，错误追溯也会转发到浏览器控制台。
![chrome](https://user-images.githubusercontent.com/6218739/144053608-0dd72e3b-752f-4f66-98e7-52ac4ade46c5.png)
除此之外，还可以使用ImJoy API函数，包括 `api.log()`、`api.error()`、`api.alert()`、`api.showMessage()` 等来向ImJoy应用程序显示消息。
特别地，对于Python插件，`print()`只会在启动插件引擎的终端中看到，因此建议开发Python插件时使用这些API函数来辅助debug。

# ImJoy插件文件格式
ImJoy插件通常是一个扩展名为`*.imjoy.html`的文本文件。其中使用HTML/XML标签，例如 `<config>`、`<script>`、`<window>` 来存储代码块。
大多数插件类型至少需要两个代码块：`<config>` 和`<script>`，例如`web-worker`、`web-python` 和`native-python`。对于`window` 插件，代码中需要额外一个`<window>` 块，以及一个可选`<style>` 块用于CSS定义。

对于`<script>`代码块，大多数插件至少会暴露两个特殊函数：`setup`（用于初始化）和`run`（当用户点击插件菜单按钮时调用）。在加载插件时，一个包含所有ImJoy API函数的`api`对象将被传递给插件，然后插件可以构建服务函数并通过调用 `api.export(...)` 函数来注册它们。

比如以下插件中定义了 3 个 API 函数：一个空的 `setup` 函数，一个 `choosePokemon` 函数，以及一个可供调用的 `run` 函数（由 ImJoy内核调用或用户点击插件菜单时）：
```js
class ImJoyPlugin{
    async setup(){
    }
    async choosePokemon(){
        const pokemon = await api.prompt("What is your favorite Pokémon?", "Pikachu")
        await api.showMessage("Your have chose " + pokemon + " as your Pokémon.")
    }
    async run(ctx){
        await this.choosePokemon()
    }
}
api.export(new ImJoyPlugin())
```
关于 ImJoy 插件文件的详细说明可以在这里找到：[插件文件格式](https://imjoy.io/docs/#/development?id=plugin-file-format)。

# ImJoy的Hello World插件
要制作第一个 ImJoy 插件，即ImJoy的Hello World，可以单击`+ PLUGINS`，然后从`+ CREATE A NEW PLUGIN`下拉菜单中选择默认模板`Default template`。
![helloworld](https://user-images.githubusercontent.com/6218739/144052725-44768110-a754-4cce-a9fb-8f70a2bcbfc7.png)
生成的插件代码为：
```js
<docs>
[TODO: write documentation for this plugin.]
</docs>
<config lang="json">
{
  "name": "Untitled Plugin",
  "type": "web-worker",
  "tags": [],
  "ui": "",
  "version": "0.1.0",
  "cover": "",
  "description": "[TODO: describe this plugin with one sentence.]",
  "icon": "extension",
  "inputs": null,
  "outputs": null,
  "api_version": "0.1.8",
  "env": "",
  "permissions": [],
  "requirements": [],
  "dependencies": []
}
</config>
<script lang="javascript">
class ImJoyPlugin {
  async setup() {
    api.log('initialized')
  }
  async run(ctx) {
    api.alert('hello world.')
  }
}
api.export(new ImJoyPlugin())
</script>
```
上述代码的具体语法会在后面具体详述。
在不更改代码的情况下，可以通过单击保存图标来保存它，此时会在插件菜单中添加一个名为“Untitled Plugin”的新条目。
要运行这个插件，可以单击“Untitled Plugin”按钮。此时将看到一个带有“Hello World”的弹出对话框。
![hello](https://user-images.githubusercontent.com/6218739/144055127-ffd19736-bcde-4f58-9c15-e7a976bb82ae.png)

如果是在本地电脑编辑的ImJoy插件文件（扩展名为 `*.imjoy.html`），那么可通过下面操作加载到ImJoy Web App中：
1) 转到 https://imjoy.io/#/app 
2) 将该文件拖放到浏览器中即可。

对于 Python 开发，可以使用 [jupyter notebook 扩展](https://github.com/imjoy-team/imjoy-jupyter-extension)。该部分会在以后详细解析。

# 部署和共享插件
如果你想与他人分享你的插件，可以直接发送插件文件，或者将插件上传到 Github/Gist。建议使用后者，因为它会更大范围地分发插件。
以下步骤可以帮助编写及部署插件：
（1）在 Github 上 [fork imjoy-starter repo](https://github.com/imjoy-team/imjoy-starter/fork)（或者，如果你愿意，可以创建一个空的）。imjoy-starter仓库包含一个[docs 文件夹](https://github.com/imjoy-team/imjoy-starter/tree/master/docs)，开发者可以在 Markdown 中做笔记，它将渲染为像这样的[交互式的网站](https://imjoy-team.github.io/imjoy-starter/)。有关更多信息，请参阅[**此处**](https://docsify.js.org/#/)。可以在Markdown中添加带有一些特殊标记的插件代码，然后就可以看到**Run**和**Edit**按钮。
（2）然后可以将自己的插件命名为，例如，`hello.imjoy.html` 并使用 git 命令将其上传到你所fork的仓库的 `plugins` 文件夹或直接上传到仓库。
（3）然后单击插件文件并复制地址栏中的url，它应该类似于：`https://github.com/<YOUR-GITHUB-USERNAME>/imjoy-starter/blob/master/plugins/hello.imjoy.html`
此路径可用于在ImJoy中安装插件。
（4）单击**Run**打开ImJoy Web App。要安装插件，单击`+PLUGINS`并将URL粘贴到`Install from URL`输入框中，然后按 Enter。
（5）现在可以构建一个URL与他人共享，只需在 `https://imjoy.io/#/app?plugin=` 后面添加 URL 即可，比如这样：`https://imjoy.io/#/app?plugin=https://github.com/<YOUR-GITHUB-USERNAME>/imjoy-starter/blob/master/plugins/hello.imjoy.html`。
如果用户单击这个插件URL，它将直接在ImJoy中打开插件并提示用户安装它。

