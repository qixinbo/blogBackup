---
title: 开源深度学习计算平台ImJoy解析：4 -- 插件文件格式
tags: [ImJoy]
categories: computer vision 
date: 2021-12-4
---

# 参考文献
[Developing Plugins for ImJoy](https://imjoy.io/docs/#/development)

# 概览
ImJoy插件文件本质上是包括一系列自定义块的html文件（受`.vue` 格式启发）。
如下是一个插件文件的典型组成。需要注意的是，这些块的顺序无关紧要，因此可以将块打乱。
```html
<config lang="json">
   ** 该代码块以Json格式定义插件的属性**
</config>

《script lang="javascript">
   ** 该代码块以JavaScript 或 Python 格式编写插件功能**
</script>

<window lang="html">
   ** 该代码块以HTML 格式编写界面**（适用于window类型插件）
</window>

<style lang="css">
   ** 该代码块以CSS 格式定义界面样式**（适用于window类型插件）
</style>

<docs lang="markdown">
   ** 该代码块以Markdown 格式编写插件文档 **
</docs>

<attachment name="XXXXX">
   ** 该代码块用于存储文本数据**
</attachment>
```
其中，只有`<config>`和`《script>`这两个代码块是必须的，其他块都是可选的。


# config块
使用 [JSON](https://www.json.org/) 或 [YAML](https://yaml.org/) 格式的字段来定义插件的通用属性。
`json`格式的配置为（注意JSON本身不支持注释，所以下面#号以后的内容只是为了这里方便说明，实际要去除；一些需要详细解释的字段会在下面单独说明）：
```html
<config lang="json">
{
  "name": "Untitled Plugin", # 插件名称。它必须是唯一的，以避免与其他插件冲突
  "type": "web-worker", # 插件类型，选项有：`web-worker`、`window`、`native-python` 和`web-python`
  "ui": "image processing", # 配置在插件下方显示的内容，详细用法将在下面详述
  "url": "", # 当前文件的路径，用户从插件存储库安装插件时需使用该url来下载该插件
  "cover": "", # 插件封面图片的url，详细用法将在下面详述
  "labels": [], # 定义一个“labels”列表来分类插件以允许基于语义标签进行搜索或过滤。
  "authors": [], # 作者姓名列表。
  "license": "", # 插件所采用的许可证
  "repository": "", # 插件项目存储库的url路径
  "website": "", # 插件项目网站的url
  "tags": [], # 插件所支持的配置标签，详细用法将在下面详述
  "version": "0.1.0", # 指定插件版本
  "api_version": "0.1.6", # 指定插件所基于的 ImJoy 的 api 版本。
  "description": "A plugin for image processing.", # 插件的简短描述
  "flags": ["functional"], # 用来控制插件行为，详细用法将在下面详述
  "icon": "extension", # 定义插件菜单的图标，详细用法将在下面详述
  "inputs": null, # 定义用来触发插件的输入条件，比如拖进来一个文件，详细用法将在下面详述
  "outputs": null, # 定义插件的输出，详细用法将在下面详述
  "cmd": "python", # 仅适用于python插件，用于运行插件的命令。默认情况下，它将使用`python`运行。根据安装的不同，它也可能是“python3”或“python27”等。
  "env": null, # 仅适用于python插件，虚拟环境或docker镜像命令，用于创建插件运行的环境。具体用法见另一篇博客的“指定依赖”一节
  "permissions": [], # 指定权限，详细用法将在下面详述
  "requirements": [], # 定义插件依赖，具体用法见另一篇博客的“指定依赖”一节
  "dependencies": [], # 指定当前插件依赖的其他ImJoy插件，详细用法将在下面详述
  "default": {}, # 仅适用于window插件，定义窗口的默认值，详细用法将在下面详述
  "base_frame": null, # 仅适用于window插件，定义在该窗口插件中内嵌的外部html的url路径，详细用法将在下面详述
  "runnable": true # 定义插件是否可以通过点击插件菜单来执行，详细用法将在下面详述
}
</config>
```
而`yaml`格式的配置为:
```html
<config lang="yaml">
name: Untitled Plugin
type: web-worker
tags: []
ui: image processing
cover: ''
version: 0.1.0
api_version: 0.1.6
description: A plugin for image processing.
icon: extension
inputs: 
outputs: 
env: 
permissions: []
requirements: []
dependencies: []
</config>
```

## cover字段
`cover`字段用来指定插件封面图片的url，它将显示在图像安装对话框中，以及插件文档的顶部。 
示例：`"cover":"https://imjoy.io/static/img/imjoy-card-plain.png"`。
建议封面图片的纵横比为 16:9。 它可以托管在 GitHub 存储库中，此时应该使用图像的绝对路径URL。
可以使用多个图像，将 `cover` 设置为一个数组即可：`"cover": ["url_to_image1", "url_to_image2", "url_to_image3"]`。

## tags字段
`tags`字段用来定义插件所支持的配置标签。
（注意：这里的`tags`不是用于分类或分组目的，分类或分组可以使用`labels`字段实现。）
这些定义的标签为插件执行提供了灵活的可配置的模式，比如可以配置插件是在 CPU 或 GPU 上运行。
以Unet Segmentation这个插件为例，如下是安装该插件时的界面及其源代码：
![unet-tags](https://user-images.githubusercontent.com/6218739/144185494-b3db5ddc-de2d-4d3b-a9fe-6e8549e062a5.png)
可以看出，该插件在tags字段中定义了`CPU`、`GPU`、`Windows-CPU`、`Windows-GPU`四个标签，那么相应地，就可以在后面的环境配置和依赖库解析时，对其进行不同的要求：
```js
"env": 
{"CPU": ["conda config --add channels conda-forge", "conda create -n tf-cpu python=3.6"],
"GPU": ["conda config --add channels conda-forge", "conda create -n tf-gpu python=3.6"],
"Windows-CPU": ["conda config --add channels conda-forge", "conda create -n tf-cpu python=3.6"],
"Windows-GPU": ["conda config --add channels conda-forge", "conda create -n tf-gpu python=3.6"]
},
"requirements": 
{"CPU": ["repo:https://github.com/zhixuhao/unet", "pip:h5py scikit-image keras==2.1.4 numpy==1.18.0 tensorflow==1.15.4"],
"GPU": ["repo:https://github.com/zhixuhao/unet", "pip:h5py scikit-image keras==2.1.4 numpy==1.18.0 tensorflow-gpu==1.15.4"],
"Windows-CPU": ["repo:https://github.com/zhixuhao/unet", "pip:h5py scikit-image keras==2.1.4 numpy==1.18.0 tensorflow==1.15.4"],
"Windows-GPU": ["repo:https://github.com/zhixuhao/unet", "pip:h5py scikit-image keras==2.1.4 numpy==1.18.0 tensorflow-gpu==1.15.4"]
},
```
如果插件定义了tags，它们将出现在代码编辑器的顶部以及安装过程中。如果使用`url`分发插件，还可以具体指定插件将使用哪个标签安装。
在`<config>`代码块中，以下字段可以被tags所配置：`env`、`requirements`、`dependencies`、`icon`、`ui`、`type`、`flags`、`cover`。
`<script>`代码块也可以被tags配置，此时必须把`tags`字段作为属性放到`<script>`中，同时注意原来的`lang`属性还要保留。比如：开发者可能想在插件的稳定版和开发版之间自由切换。那么可以定义这样的标签： `"tags": ["stable", "dev"]`，同时定义两个脚本块：`<script lang="python" tag="stable">` 和 `<script lang="python "tag="dev">`。在开发和测试插件时，ImJoy 编辑器会识别插件有多个标签，此时选择其中一个标签，那么加载插件时就会加载此标签下的相应代码。

## ui字段
`ui`字段是为了配置在插件下方显示的内容。
以HPA Classification插件为例，安装完成后，在ImJoy的左侧插件区，点击插件右侧的箭头，会看到`ui`字段定义的内容：
![hpa-ui](https://user-images.githubusercontent.com/6218739/144191940-48d0ab2b-5c8c-4607-9543-b05daf28a1ee.png)

以下是`ui`字段的一个详细用法示例：
![ui-full](https://user-images.githubusercontent.com/6218739/144377959-51a13531-9ce3-4752-8c13-7d841bcec51d.png)
包含了各种输入框的用法，以及同一个ui的三种写法：`option1`、`option11`、`option12`。
对于每个元素，都定义了一个唯一的`id`，然后可以使用 `ctx.config.id`来访问插件中此元素的值。

## flags字段
`flags`字段定义一个标志数组，用来控制插件的行为。
一个重要的flag是 `functional`：`functional`标志表示插件暴露的所有api函数都是[纯函数](https://segmentfault.com/a/1190000039807327)。这意味着它们的输出将仅取决于输入。同时纯函数保证在调用任何插件 api 函数之后对插件没有副作用。这意味着应该避免修改插件函数中的全局变量（不过一个例外是`setup()` 函数）。
使一个插件具有`functional`标志，可以使调试更容易，重要的是其他插件或工作流调用`functional`插件时能严格重现其行为。`functional`插件对于 ImJoy 在将来执行并行化和批处理时至关重要。
作者还没有编写真正的测试来验证插件是否是`functional`，所以当前需要确定自己的插件仅包含纯函数时才添加 `functional` 标志。
此外，`flags`还支持运行时控制。这些标志允许用户界面和插件引擎如何处理 ImJoy 实例：
- `single-instance`（仅适用于 python 插件）：在此标志下，Python引擎只会运行一个插件进程，即使从不同的浏览器或工作区调用了该插件。在这种情况下，不同的 ImJoy 实例将共享相同的插件进程。当插件需要独占有限的资源（例如 GPU）时，这尤其有用。
- `allow-detach`（仅适用于 python 插件）：在此标志下，允许插件进程与用户界面分离。这意味着当用户界面断开或关闭时，插件不会被杀死。然而，为了重新连接到这个进程，需要从同一个浏览器和相同的工作区重新连接，或添加 `single-instance` 标志。比如：如果想制作一个插件，它可以在没有用户界面的情况下在后台运行，那就对该插件赋予`"flags": ["single-instance", "allow-detach"]`。那么重新启动时，用户界面将自动重新连接到此进程。需要注意的是，如果多个 ImJoy 实例附加到此插件进程，每个实例都会调用 `setup()` 函数。这可能会导致冲突，因此建议 (1) 将与接口相关的代码保留在 `setup()` 中，例如`api.register()`; (2)将只想每个进程运行一次的代码移动到插件类的`__init__` 函数中。

## icon字段
定义插件菜单中使用的图标。可以选择以下格式：
（1）在[https://material.io/tools/icons/](https://material.io/tools/icons/)找到图标，直接使用指定的名称即可；
![icon](https://user-images.githubusercontent.com/6218739/144605127-ece26296-0933-4534-a798-cba6c4f0ace5.png)
实测使用该网站上的图标名称时，不能使用带空格的名称，以及全部要用小写。
（2）可以直接复制粘贴emoji符号，例如从[这里](https://getemoji.com/)。
![emoji](https://user-images.githubusercontent.com/6218739/144605559-f3a9798b-23e1-4427-8185-3de397873d6c.png)
（3）指定 JPEG、PNG 或 GIF 格式图像的 URL，推荐大小：64x64。
![gif-icon](https://user-images.githubusercontent.com/6218739/144605914-ada14870-30c1-45f8-b563-db5439a4710f.png)
（4）如果设置为`null`或`""`，它会默认使用第一条的material icon的`extension`图标。

## inputs字段
定义用于触发插件的输入条件，包括文件输入或数据匹配模式。例如，当用户将某个后缀的文件拖放到工作区时，相应的插件被激活。
基本上可以使用标准的 json 模式（http://json-schema.org/）来验证输入数据对象。例如，要定义插件使用 png 文件，可以指定 `"inputs": {"type": "object", "properties": {"name": {"type": "string", "maxLength" : 100}}, "required": ["name"]}`。在后台，ImJoy使用 [ajv](https://github.com/epoberezkin/ajv) 库来验证对象。为了简化模式的使用，还使用以下关键字扩展了标准的 json 模式：
（1）`file`：对于文件对象，使用 [mime types](https://en.wikipedia.org/wiki/Media_type) 或文件扩展名。可以使用以下关键字之一：
- `mime`：常见的 mime 类型字符串（或列表）。例如`{"file": {"mime": "image/png"}}`。还可以指定一个 mime 类型列表：`"mime": ["image/png", "image/jpeg", "image/tiff"]`。
- `ext`：文件扩展名字符串（或列表）。例如`{"file": {"ext": "png"}}`，或者可以指定一个扩展名列表：`{"file": {"ext": ["png", "jpg", " jpeg"]}}`。

（2）`ndarray`：例如`{"ndarray": {"shape": [10, 10], "dtype": "uint8"}}`。
- `shape`：数组的形状。需要为每个维度指定一个数字，或者使用 `null` 表示该维度的任何大小。例如：`{"type": "ndarray", "shape": [10, 10]}`，或者如果第一维可以是任意大小：`{"type": "ndarray", "shape": [null，10]}`
- `dtype`：数组的数据类型。例如：`{"ndarray": {"dtype": "uint8"}}` 或者如果同时支持 `uint8` 和 `float32`：`{"ndarray": {"dtype": ["uint8", " float32"]}}`。支持的 `dtype` 是：`["int8", "int16", "int32", "uint8", "uint16", "uint32", "float32", "float64", "array"]`。
- `ndim`：维数。例如：` {"ndarray": {"ndim": 2}}` 或者如果同时支持 2D 和 3D：` {"ndarray": {"ndim": [2, 3]}}`

（3）对于通过 http 或其他协议暴露的远程 url，`{"type": "url", "extension": "zarr"}`。
另外需注意，如果在模式中使用正则表达式来验证字符串，可能需要设置 `maxLength`，否则它会非常慢，甚至在验证长字符串时可能会崩溃。例如，如果想匹配一个包含以`.tiff`结尾的`file_name`的对象，那么可以这样设置：`{"properties": {"file_name": {"type": "string","pattern": ".*\\.tiff$", "maxLength": 1024}}}`。

## outputs字段
使用 json-schema 语法 (http://json-schema.org/) 定义输出。
格式与`inputs`相同。

## permissions字段
对于 `window` 插件，可以声明以下权限：
- 摄像头camera
- 音乐数字接口midi
- 地理位置geolocation
- 麦克风microphone
- 加密媒体encryted-media
- 全屏full-screen
- 支付请求payment-request

例如，如果某个`window`插件需要网络摄像头的访问权限，需要添加以下权限：
```js
 “permissions”：[“camera”]，
```
注意，摄像头和麦克风等设备只有在 ImJoy 为 `https` 时才能工作，这意味着如果是从https://imjoy.io 运行这个插件，那么是可以工作的，但如果使用的是自己托管的以`http`协议访问的ImJoy服务器，那么它就不会运行。解决方法是使用隧道服务，例如使用 [Telebit](https://telebit.cloud) 或 [ngrok](https://ngrok.com) 将 `http` url 转换为 `https`。

## dependencies字段
该字段指定当前插件依赖的其他ImJoy插件。这些所依赖的插件将在安装过程中自动安装。
要定义依赖项，请使用以下格式：
1) 对于没有tag的依赖，使用 `REPOSITORY:PLUGIN_NAME` 或 `PLUGIN_URL`，例如：`imjoy-team/imjoy-plugins:Image Window`； 
2) 对于带有指定tag的依赖，使用`REPOSITORY:PLUGIN_NAME@TAG` 或`PLUGIN_URL@TAG`，例如：`imjoy-team/imjoy-plugins:Unet Segmentation@GPU`。在这种情况下，标签“GPU”用于指定托管在 GitHub 存储库`imjoy-team/imjoy-plugins`（https://github.com/imjoy-team/）上的名为`Unet Segmentation`的插件。如果插件没有托管在GitHub上或者该GitHub仓库不是ImJoy插件仓库的标准格式（即在仓库的根目录中没有定义 `manifest.imjoy.json` 文件），则可以直接使用 url，例如：`https://github.com/imjoy-team/imjoy-demo-plugins/blob/master/repository/3dDemos.imjoy.html`（标签可以用`@TAG`添加）。

## default字段
仅适用于`window`插件，用于定义一个对象的默认值。
例如，可以通过设置 `"defaults": {"w": 10, "h": 7}` 来指定默认窗口大小。
或者，可以使用 `"defaults": {"fullscreen": true}` 默认使窗口处于全屏模式。
要使窗口默认处于独立模式（全屏并与工作区分离），可以设置 `"defaults": {"standalone": true}`。
如果要将窗口显示为对话框，设置 `"defaults": {"as_dialog": true}`。

## base_frame字段
仅适用于window插件，定义在该窗口插件中内嵌的外部html的url路径。
虽然可以在`base_frame`字段中使用任何其他网站的url，但是为了Imjoy内核可以与该html进行通信，它需要满足以下条件：
（1）该网站需要允许嵌入，不过这并不总是能够有效，因为它们可能有严格的 [内容安全策略](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)的限制，通常是通过`X-Content-Security-Policy`的header，或页面中的 `<meta>` 元素来实现该限制。要解决这个问题，如果你可以控制该站点，需要将 `*.imjoy.io` 添加到header中。
（2）在`base_frame`里面，需要开启`imjoy-rpc`协议。这可以按照 [imjoy-core](https://github.com/imjoy-team/ImJoy-core) 仓库中的说明轻松搞定。参考“ImJoy RPC library to your website”这一部分进行操作，基本上就是需要导入`imjoy-loader`，并加载imjoy RPC库，然后导出想公开给其他ImJoy插件的api。
完成上述操作后，就可以将第三方网站集成为一个ImJoy插件。

## runnable
定义插件是否可以通过点击插件菜单来执行（默认情况下，所有插件都是`runnable`）。
对于不单独运行的辅助插件，（例如，`native-python` 插件可以被`window` 插件调用，不一定由用户直接执行），设置`"runnable": false` 会向下移动插件到插件菜单的底部，并使其不可点击。

# docs块
在该块中，使用 Markdown 语言来编写插件文档。
Markdown语言的介绍见[这里](https://guides.github.com/features/mastering-markdown/)。
注意，如果在文档中提供的链接将在另一个选项卡中打开，则 ImJoy 实例将继续运行。

# window块
在该块中，使用HTML 代码来编写窗口的显示内容。
ImJoy 使用 vue.js 来解析插件文件，它强制要求仅有根元素存在于模板中。这意味着在`<window>`块中必须使用一个`div`来包装所有节点：
```html
<window>
  <div>
    <p> line 1</p>
    <p> line 2</p>
  </div>
</window>
```
如下则不可以：
```html
<window>
  <p> line 1</p>
  <p> line 2</p>
</window>
```

# style块
在该块中，使用CSS 代码来编写窗口显示内容的样式。

以MNIST CNN插件为例，给出以上`<window>`块和`style`块的一个说明：
![html-css](https://user-images.githubusercontent.com/6218739/144701434-77ef0f5c-f952-4bdf-aec7-c5c7209ae7cb.png)

# script块
该块中包含实际的插件代码。
插件可以用 JavaScript 或 Python 编写，一个最小的插件需要实现两个函数：`setup()` 和 `run()`。有一个例外是那种辅助插件（用 `"runnable": false` 指定），它不需要 `run()` 函数。
（1）`setup()` 函数：在插件第一次加载和初始化时执行它。
（2）`run()` 函数：每次执行插件时都会调用。执行时，一个带有上下文（名为“ctx”）的对象object（Javascript插件）或字典dictionary（Python插件）将被传递到函数中。返回的结果将显示为一个新窗口或传递给工作流中的下一个 `op`。更多内容请参见另一篇博客的 [运行时插件](development?id=plugin-during-runtime) 部分。
（3）可选：`resume()` 函数：仅适用于带有 `allow-detach` 标志的可分离 `native-python` 插件。当ImJoy 重新连接到正在运行的插件进程时，`resume()` 将被调用（而不是 `setup()`） 。
（4）可选：`update()` 函数：将在操作的任何设置更改时调用。
（5）可选：`exit()`函数：当插件被杀死时，函数`exit` 将被调用。

`<script>` 块的 `lang` 属性用于指定使用的编程语言：
（1）对于 Javascript，使用 `<script lang="javascript"> ... </script>`
（2）对于 Python，使用 `<script lang="python"> ... </script>`
对于 Javascript 插件，还支持 ES 模块，要启用它，将 `type="module"` 添加到`<script>`标签中。例如：`<script type="module" lang="javascript">...</script>`。
`<script>` 也支持 `tags`，有关信息参考上面的`tags`字段的解析。