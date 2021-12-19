---
title: 开源深度学习计算平台ImJoy解析：8 -- 使用python编写计算插件
tags: [ImJoy]
categories: computer vision 
date: 2021-12-18
---

上一篇着重介绍了如何使用JavaScript库来编写插件的前端UI和后端计算逻辑，这一节会介绍如何将计算后端切换为python语言，即计算逻辑完全使用python编写，充分利用python庞大的计算生态。
使用python开发计算插件有两种：
（1）web-python：即python运行在浏览器中，其原理实际是应用了[Pyodide](https://pyodide.org/en/stable/)这一工具，将python代码编译在浏览器中，但其缺点也很明显，首先是加载速度非常慢，因为第一次运行时需要将所用的python库都下载下来；然后其也无法应用整个python深度学习生态。
（2）native-python：该类型插件会链接一个本地的jupyter插件引擎，可以充分发挥python的最大价值，本篇也将着重介绍该种插件的编写。

# web-python的hello world
先看一个使用web-python编写的hello world例子，它会完全在浏览器中运行Python代码。
注意，当运行以下插件时，会需要一段时间，因为它需要将python库加载到浏览器中：
```python
<docs lang="markdown">
[TODO: write documentation for this plugin.]
</docs>
<config lang="json">
{
  "name": "Untitled Plugin",
  "type": "web-python",
  "version": "0.1.0",
  "description": "[TODO: describe this plugin with one sentence.]",
  "tags": [],
  "ui": "",
  "cover": "",
  "inputs": null,
  "outputs": null,
  "flags": [],
  "icon": "extension",
  "api_version": "0.1.8",
  "env": "",
  "permissions": [],
  "requirements": [],
  "dependencies": []
}
</config>
<script lang="python">
from imjoy import api
class ImJoyPlugin():
    def setup(self):
        api.log('initialized')
    def run(self, ctx):
        api.alert('hello world.')
api.export(ImJoyPlugin())
</script>
```
实测速度非常慢，所以并不推荐使用web-python这种方式来编写插件。

# native-python开发插件
如果想充分利用python的深度学习生态，唯一使用的方式就是native-python这种开发模式。
此模式的使用可以有三种组合方式：
（1）[ImJoy官方部署](https://imjoy.io/#/app)+[MyBinder](https://mybinder.org/)插件引擎；
（2）[ImJoy官方部署](https://imjoy.io/#/app)+本地Jupyter插件引擎；
（3）本地部署+本地Jupyter插件引擎。
第一种因为使用MyBinder这一免费的Jupyter托管方案，其性能会较弱，通常只用于demo用途，因此不推荐；
第二种会使用ImJoy的官方web app来作为应用入口，因此可能会受限于其官网的可连接性，快速开发时推荐使用；
第三种web app和Jupyter都是在本地部署，因此有最大的灵活性。本部分将对第三种的环境搭建做一介绍。

## 本地部署web app
ImJoy的主web app程序也在GitHub上进行了开源，见[这里](https://github.com/imjoy-team/ImJoy)。
（1）clone该仓库：
```python
git clone git@github.com:imjoy-team/ImJoy.git
```
（2）安装依赖包
进入`web`文件夹，然后：
```python
npm install
```
这一步需要安装nodejs，此处不详细介绍，可以移步[这里](https://nodejs.org/zh-cn/)。
（3）编译运行：
有两种编译和运行方式，一种是开发模式：
```python
npm run serve
```
或者生产模式：
```python
npm run build
```
（4）访问app：
上一步运行该app后，就会生成可访问的链接，通常是：
```python
http://localhost:8001
```

## 搭建本地Jupyter插件引擎
搭建本地Jupyter插件引擎有两种方法：
（1）安装Jupyter notebook（通过`pip install jupyter`），然后安装[imjoy-jupyter-extension](https://github.com/imjoy-team/imjoy-jupyter-extension)。
（2）可以通过 `pip install imjoy` 安装这个[ImJoy-Engine](https://github.com/imjoy-team/imjoy-engine)库。
推荐使用后者，因为这样可以对Jupyter服务器做一些对ImJoy有用的设置，并且不需要单独安装imjoy-jupyter-extension。
具体的搭建流程如下：
（1）下载并安装conda环境：推荐使用python3.7版本的Anaconda。
（2）安装引擎：
```python
pip install -U imjoy[jupyter]
```
（3）启动引擎：
```python
imjoy --jupyter
```
然后在终端就会得到形如：
```python
http://localhost:8888/?token=caac2d7f2e8e0...ad871fe
```
的链接。这就是插件引擎的地址。
（4）连接web app
在前面开启的web app页面上`http://localhost:8001/#/app`，点击右上角的小火箭图标，然后点击`Add Jupyter-Engine`，将上面插件引擎的地址填入即可。

# native-python的hello world
```python
<docs lang="markdown">
[TODO: write documentation for this plugin.]
</docs>
<config lang="json">
{
  "name": "Untitled Plugin",
  "type": "native-python",
  "version": "0.1.0",
  "description": "[TODO: describe this plugin with one sentence.]",
  "tags": [],
  "ui": "",
  "cover": "",
  "inputs": null,
  "outputs": null,
  "flags": [],
  "icon": "extension",
  "api_version": "0.1.8",
  "env": "",
  "permissions": [],
  "requirements": [],
  "dependencies": []
}
</config>
<script lang="python">
from imjoy import api

class ImJoyPlugin():
    def setup(self):
        api.log('initialized')
    def run(self, ctx):
        api.alert('hello world.')
api.export(ImJoyPlugin())
</script>
```

# 用python写图像处理插件
这一部分尝试将[构建基于Web的图像分析插件](https://qixinbo.info/2021/12/17/imjoy_7/)这一篇中的opencv.js功能用python版的opencv实现一遍。
在此例中，有**两个插件**：UI插件和compute插件。一般来说，有两种方法可以连接它们：
（1）首先用`api.createWindow(...)`从compute插件实例化UI插件，然后与返回的窗口对象进行交互；
（2）也可以直接启动UI插件，然后通过`api.getPlugin()`来获取compute插件提供的api。
两种方法到底用哪一种取决于应用程序的实际需要，这里推荐第一种方式用于Python插件的编写，因为它可以更轻松地在Jupyter笔记本中调试。

这里的插件是用Python重写计算功能、JavaScript仍然是前端，因此涉及到两种语言对图像格式的转译，需要进行编码和解码以使它们交叉兼容。最简单的方法是将图像编码为“base64”字符串。
因此，整个插件的流程为（本节末尾会给出所有代码，这里是将代码分解）：
（1）从UI插件（即图像查看器）的canvas画布中得到图像的`base64`编码：
```js
const canvas = document.getElementById('canvas-id')

// get `base64` encoded image from a canvas
const base64String = canvas.toDataURL()
```
（2）在UI插件中调用compute插件中的函数，并传递上面的`base64`编码：
UI插件能调用compute插件中的python函数，是通过插件中的`ctx`变量来得到它，形如：
```js
    // the run funciton of the image viewer
    async run(ctx){
        // check if there is a process function passed in
        if(ctx.data && ctx.data.process){
            // show an additional "Process in Python" button
            // and set the call back to use this process function
        }
    }
```
相对应地，在Python插件中就可以执行`await api.createWindow(type="Image Viewer", data={"process": self.process})`来传给JS插件（假设已经在插件中定义了一个名为 `process`的函数）。
在调用`api.createWindow` 时，有两种方法可以引用另一个窗口插件：
（a）将`type`键设置为窗口插件名称，例如如果UI插件名为`My Window Plugin`，就将其设置为`type`。注意，这个名称是从 `<config>` 块中的 `name` 定义中获得的。
（b）如果UI插件是源代码的形式或者由公共服务器提供，可以设置`src`作为插件源代码或者插件URL，比如`name="Kaibu",src="https://kaibu.org/#/app"`。在这种情况下，插件将被动态填充。例如，它允许将窗口插件存储为Python中的字符串，甚至可以根据模板动态生成窗口插件。

另外一个需要注意的是，如果是使用`await api.createWindow(type="Image Viewer", data={"process": self.process})`，此时会发现，如果第二次单击该按钮，它将不再起作用，并且如果转到浏览器控制台，将看到一条错误消息，提示`Callback function can only called once, if you want to call a function for multiple times, please make it as a plugin api function.`。这是因为在第一次调用后从窗口中删除了`process`函数。为了明确地告诉窗口保留`process`函数，可以将一个特殊的键`_rintf`设置为`True`，即把上面的代码改成`data={"process": self.process, "_rintf": True}`。
（3）在python插件中解码`base64`，并读取为numpy类型数组：
```python
import re
import base64
import io
import imageio

def base64_to_image(base64_string, format=None):
    '''This function takes a base64 string as input
    and decode it into an numpy array image
    '''
    base64_string = re.sub("^data:image/.+;base64,", "", base64_string)
    image_file = io.BytesIO(base64.b64decode(base64_string.encode('ascii')))
    return imageio.imread(image_file, format)
```

（4）在python插件中编写图像处理算法：
这里仍然使用的是opencv，不过要用的是它的python版本：
```python
"requirements": ["opencv-python"]

import cv2

def process_image(src):
    dst = cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)
    return dst
```
（5）将numpy数组类型的处理结果编码为`base64`并返回：
```python
def image_to_base64(image_array):
    '''This function takes a numpy image array as input
    and encode it into a base64 string
    '''
    buf = io.BytesIO()
    imageio.imwrite(buf, image_array, "PNG")
    buf.seek(0)
    img_bytes = buf.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('ascii')
    return 'data:image/png;base64,' + base64_string


async def process(self, base64string):
    img = base64_to_image(base64string)
    dst = process_image(img)
    base64dst = image_to_base64(dst)
    return base64dst
```
（6）在JS插件中接收`base64`编码，并在画布中显示为图像：
```js
// draw a `base64` encoded image to the canvas
const drawImage = (canvas, base64Image)=>{
    return new Promise((resolve, reject)=>{
        const img = new Image()
        img.crossOrigin = "anonymous"
        img.onload = function(){
            const ctx = canvas.getContext("2d");
            canvas.width = Math.min(this.width, 512);
            canvas.height= Math.min(this.height, parseInt(512*this.height/this.width), 1024);
            // draw the img into canvas
            ctx.drawImage(this, 0, 0, canvas.width, canvas.height);
            resolve(canvas);
        }
        img.onerror = reject;
        img.src = base64Image;
    })
}
```
整个插件的处理逻辑如上，结果与完全JS作为前端和后端的结果相同，如下图：
![pythonbackend](https://user-images.githubusercontent.com/6218739/146639355-669e3cac-2134-4171-ba21-19884d536abb.png)

完整代码如下：
对于UI插件：
```js
<config lang="json">
{
  "name": "Image Viewer",
  "type": "window",
  "tags": [],
  "ui": "",
  "version": "0.1.0",
  "cover": "",
  "description": "This is a demo plugin for displaying image",
  "icon": "extension",
  "inputs": null,
  "outputs": null,
  "api_version": "0.1.8",
  "env": "",
  "permissions": [],
  "requirements": [
    "https://cdn.jsdelivr.net/npm/bulma@0.9.1/css/bulma.min.css", 
    "https://use.fontawesome.com/releases/v5.14.0/js/all.js"],
  "dependencies": []
}
</config>

<script lang="javascript">
const drawImage = (canvas, base64Image)=>{
    return new Promise((resolve, reject)=>{
        const img = new Image()
        img.crossOrigin = "anonymous"
        img.onload = function(){
            const ctx = canvas.getContext("2d");
            canvas.width = Math.min(this.width, 512);
            canvas.height= Math.min(this.height, parseInt(512*this.height/this.width), 1024);
            // draw the img into canvas
            ctx.drawImage(this, 0, 0, canvas.width, canvas.height);
            resolve(canvas);
        }
        img.onerror = reject;
        img.src = base64Image;
    })
}
const readImageFile = (file)=>{
    return new Promise((resolve, reject)=>{
        const U = window.URL || window.webkitURL;
        if(U.createObjectURL){
            resolve(U.createObjectURL(file))
        }
        else{
            const fr = new FileReader();
            fr.onload = function(e) {
                resolve(e.target.result)
            };
            fr.onerror = reject
            fr.readAsDataURL(file);
        }
    })
}

class ImJoyPlugin{
    async setup(){
        const fileInput = document.getElementById("file-input");
        const canvas = document.getElementById("input-canvas");
        const outputcanvas = document.getElementById("output-canvas");
        fileInput.addEventListener("change", async ()=>{
        const img = await readImageFile(fileInput.files[0]);
        await drawImage(canvas, img);
        }, true);
        await api.log("plugin initialized")
        const selectButton = document.getElementById("select-button");
        selectButton.addEventListener("click", async ()=>{
        fileInput.click()
        }, true);
    }
    async run(ctx){
        if(ctx.data && ctx.data.process){
            const canvas = document.getElementById("input-canvas");
            const outputcanvas = document.getElementById("output-canvas");
            const btn = document.getElementById('process-button')
            btn.disabled = false;
            btn.addEventListener("click", async ()=>{
                const base64String = canvas.toDataURL()
                const base64dst = await ctx.data.process(base64String)
                await drawImage(outputcanvas, base64dst)
            }, true);
        }
    }
}
api.export(new ImJoyPlugin())
</script>

<window>
    <div>
        <input  id="file-input" accept="image/*" capture="camera" type="file"/>
        <nav class="panel">
        <p class="panel-heading">
            <i class="fas fa-eye" aria-hidden="true"></i> My Image Viewer with Python backend
        </p>
        <div class="panel-block">
            <button id="select-button" class="button is-link is-outlined is-fullwidth">
            Open an image
            </button>
            <button id="process-button" disabled class="button is-link is-outlined is-fullwidth">
            RGB to Gray
            </button>            
        </div>
        <div class="panel-block">
            <canvas id="input-canvas" style="width: 100%; object-fit: cover;"></canvas>
            <canvas id="output-canvas" style="width: 100%; object-fit: cover;"></canvas>
        </div> 
        <div class="panel-block">
            <button id="predict-button" class="button is-link is-outlined is-fullwidth">
            Predict
            </button>        
        </div>
    </div>
</window>

<style>
#file-input{
    display: none;
}
h1{
    color: pink;
}
</style>
```
对于compute插件：
```python
<config lang="json">
{
 "type": "native-python",
 "name": "my-python-plugin",
 "id": "9l3fewe7l",
 "namespace": "9l3fewe7l",
 "lang": "python",
 "window_id": "code_9l3fewe7l",
 "api_version": "0.1.8",
 "description": "[TODO: describe this plugin with one sentence.]",
 "tags": [],
 "version": "0.1.0",
 "ui": "",
 "cover": "",
 "icon": "extension",
 "inputs": null,
 "outputs": null,
 "env": "",
 "permissions": [],
 "requirements": ["opencv-python"],
 "dependencies": []
}
</config>

<script lang="python">
from imjoy import api
import re
import base64
import io
import imageio
import cv2

def image_to_base64(image_array):
    '''This function takes a numpy image array as input
    and encode it into a base64 string
    '''
    buf = io.BytesIO()
    imageio.imwrite(buf, image_array, "PNG")
    buf.seek(0)
    img_bytes = buf.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('ascii')
    return 'data:image/png;base64,' + base64_string

def base64_to_image(base64_string, format=None):
    '''This function takes a base64 string as input
    and decode it into an numpy array image
    '''
    base64_string = re.sub("^data:image/.+;base64,", "", base64_string)
    image_file = io.BytesIO(base64.b64decode(base64_string.encode('ascii')))
    return imageio.imread(image_file, format)

def process_image(src):
    dst = cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)
    return dst

class ImJoyPlugin():
    async def setup(self):
        pass
    async def process(self, base64string):
        img = base64_to_image(base64string)
        dst = process_image(img)
        base64dst = image_to_base64(dst)
        return base64dst
    async def run(self, ctx):
        await api.createWindow(
            type="Image Viewer", 
            data={
                "process": self.process,
                "_rintf": True})

api.export(ImJoyPlugin())
</script>
```
在调试上述插件时，因为涉及到了`base64`的编码和解码，我频繁用到了如下debug方法，推荐尝试：
（1）使用`api.log(base64string)`将base64编码结果显示在控制台中；
（2）使用这个网站[Base64 to Image](https://base64.guru/converter/decode/image)将base64编码可视化，以查看结果正不正确。

# 使用python深度学习库
如上，我们使用了opencv-python进行了简单的图像处理，验证了native python插件的可行性。
而除了opencv-python，python背后还有着更为广阔的深度学习生态，如tensorflow、pytorch、mxnet、paddlepaddle等深度学习框架，以及这些框架可调用的GPU资源，因此可以说整个python计算生态都可以被ImJoy的native python插件所调用，这就提供了非常广阔的应用空间。
该部分不再介绍native python怎样调用python深度学习库，而是在后面的具体应用中详细解析。
