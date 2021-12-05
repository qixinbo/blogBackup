---
title: 开源深度学习计算平台ImJoy解析：5 -- 插件开发流程
tags: [ImJoy]
categories: computer vision 
date: 2021-12-5
---

# 参考文献
[Developing Plugins for ImJoy](https://imjoy.io/docs/#/development)

[上一篇文章](https://qixinbo.info/2021/12/04/imjoy_4/)中介绍了插件的文件格式，这一篇介绍如何进行实际的插件开发。

# 指定依赖
对于一个插件，其往往不是单一的功能实现，往往需要其他软件库的配合。
插件的依赖在其文件的`config` 块的`requirements`字段中进行指定。
根据不同的插件类型，可以指定不一样的依赖。

## Web Worker 和 Window 插件
对于这两类插件，可以通过一个JavaScript库的url数组来指定依赖。这些库会被`importScripts`方法导入。
例如，要指定最新的 [plotly.js](https://plot.ly/javascript/) 库，可以这样写：
```json
"requirements": ["https://cdn.plot.ly/plotly-latest.min.js"]
```

特别地，对于window插件，还可以指定CSS库的url，这些需要以`.css` 结尾，否则需要在url 后添加前缀`css:`。
例如，要使用[W3.CSS框架](https://www.w3schools.com/w3css/)，可以这样指定：
```json
"requirements": ["https://www.w3schools.com/w3css/4/w3.css"]
```
如果url不以`.css`结尾，则需要在其前面加上`css:`，例如：
```json
"requirements": ["css:https://fonts.googleapis.com/icon?family=Material+Icons"]
```
ImJoy在这个[GitHub仓库](https://github.com/imjoy-team/static.imjoy.io)中托管常用的库。可以使用简单的url来引用在`docs`文件夹中的所有文件：`https://static.imjoy.io` + `RelativePathInDocs`。
例如，在文件夹`static.imjoy.io/docs/js/`中的文件`FileSaver.js`可以这样引用：
```json
"requirements": ["https://static.imjoy.io/js/FileSaver.js"]
```
如果url不以`.js`结尾，则需要在其前面加上`js:`，例如：
```json
"requirements": ["js:https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.2"]
```
对于离线访问的场景，Javascript和CSS文件在被添加到`requirements`后会被自动缓存。
还可以使用 `cache:` 前缀将其他资源（例如图像、字体文件）添加到离线缓存中，例如：
```json
"requirements": ["cache:https://use.fontawesome.com/releases/v5.8.2/webfonts/fa-solid-900.woff2"]
```
需要注意的是，离线缓存过程不会跟踪所依赖的资源，需要手动将它们添加为依赖后才能缓存。
在`script`标签中导入 ES 模块时（使用 `type="module"`），建议在 `requirements` 中明确添加 es 模块的 URL，并在 URL 前加上 `cache:` 前缀。这将允许 ImJoy 缓存动态脚本以供离线使用。
## Web Python插件
可以使用所需 Python 模块的字符串列表来指定依赖。例如，
```json
"requirements": ["numpy", "matplotlib"]
```
默认情况下，这些包是从ImJoy开发者在[该Github仓库](https://github.com/imjoy-team/static.imjoy.io/tree/master/docs/pyodide)上的静态托管加载的。特别地，对于 `scipy`，需要包含一个绝对路径的url：
`"requirements": ["https://alpha.iodide.app/pyodide-0.10.0/scipy.js"]`。
如果想导入额外的 js 文件，需要在 javascript url 之前使用 `js:` 前缀。
注意，`web-python`插件是基于[pyodide](https://github.com/iodide-project/pyodide/)的，目前该技术只支持有限数量的 python 模块。

## Native Python插件
也是通过一个字符串列表来指定依赖，然后添加一个前缀用于指定依赖的类型：`conda:`、`pip:`和`repo:`。
一般语法是`"requirements": ["prefix:requirementToInstall"]`。
下表列出所有支持的依赖、ImJoy实际执行的命令，以及相应的例子。

| Prefix  | Command| Example |
| ------- | -------| ---------------- |
| `conda` | `conda install -y`| `"conda:scipy==1.0"`|
| `pip`   | `pip install` | `"pip:scipy==1.0"`|
| `repo`  | `git clone` (new repo) <br> `git pull` (existing repo) | `"repo:https://github.com/userName/myRepo"` |
| `cmd` | Any other command | `"cmd:pip install -r myRepo/requirements.txt"` |

一些注意点：
（1）如果没有使用前缀，那么依赖将被视为`pip`库。
   ```json
   "requirements": ["numpy", "scipy==1.0"]
   ```
（2）如果通过`env`字段定义了一个虚拟环境，那么所有`pip`和`conda` 软件包将安装到此环境中。更多的信息将在下面的“虚拟环境”一节进行介绍。
（3）可以直接在一个前缀后以一个字符串的形式列出多个依赖：
   ```json
   "requirements": ["conda:numpy scipy==1.0"]
   ```
或在一个列表中单独指定前缀：
   ```json
   "requirements": ["conda:numpy", "conda:scipy==1.0"]
   ```
（4）不同的依赖类型可以合并为一个列表：
   ```json
    "requirements": ["conda:numpy", "pip:scipy==1.0", "repo:https://github.com/userName/myRepo"]
    ```
### 从GitHub安装Python包
如果想引用的python模块有一个可用的 `setup.py` 文件，则可以直接从 Github 使用它，而无需上传到 Python Package Index ([PyPI](https://pypi.org))。方法是按照下面的 PyPI 格式提供github仓库的URL。[这个链接](https://packaging.python.org/tutorials/packaging-projects/)对这个方法有详细解释。
通用语法如下所示，参数以“{}”表示：
 ```json
 "requirements": ["pip:git+https://github.com/{username}/{reponame}@{tagname}#egg={reponame}"]
 ```
语法 `"pip:git+https..."` 被 ImJoy 翻译成命令 `pip install git+https...`，该命令允许直接从Git安装python包，即[pip install from GIT](https://pip.pypa.io/en/stable/topics/vcs-support/)。
必须指定以下参数：
（1）`username`：GitHub 帐户的名称。
（2）`reponame`：GitHub 存储库的名称。
（3）`tagname`：可以加上标签。这个参数可以是一个commit的hash tag，一个[Git tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging)，或[GitHub release](https://help.github.com/articles/creating-releases/)。该参数可以精确地选择仓库的版本。
（4）`eggname`：这通常是仓库的名称。这个参数会告诉pip进行依赖项检查。
需要注意的是，一旦安装了软件包，它将不会被升级，除非指定一个新标签。有关的完整说明，请参阅 [pip 文档](https://pip.pypa.io/en/latest/reference/pip_install/#git)。
测试起见的话，可以在终端使用`pip`命令加上上面指定的参数：
```bash
pip install git+https://github.com/{username}/{reponame}@{tagname}#egg={reponame}
```

##插件依赖的典型场景
以下描述了添加依赖时的典型场景。

### pip库
想添加的python 模块在 pip 仓库（`pip.pypa.io`）中，此时非常简单，将该python模块的 pip 名称添加到 `requirements` 中即可，也可以添加版本号。
例如，要添加 `scipy`的1.0 版本，可以指定
```json
"requirements": ["pip:scipy==1.0"]
```
### 带有 `setup.py` 的仓库
当python包中存在`setup.py` 时，`pip` 命令可以从 GitHub 仓库安装包及其依赖项。
例如：帐户 `myUserName`在GitHub上托管了`myRepo`仓库，其最新的 Git 标签是“v0.1.1”。然后可以这样添加：
```json
"requirements": "pip:git+https://github.com/myUserName/myRepo@v0.1.1#egg=myRepo"
```

### 带有 `requirements.txt` 的仓库
文件`requirements.txt`中列出了某个python库所依赖的所有软件包及其版本。有关更多详细信息，请参阅 [此处](https://pip.pypa.io/en/stable/user_guide/#requirements-files)。
下面这个例子给出了如何引入这种带`requirements.txt`的python库。
比如，帐户 `myUserName`有一个GitHub 仓库 `myRepo` ，将此仓库添加到插件工作区，并安装依赖：
 ```json
 "requirements": ["repo:https://github.com/myUserName/myRepo", "cmd: pip install -r myRepo/requirements.txt"]
 ```
然后，在自己要编写的 Python 插件中，可以将上述路径添加到 Python 系统路径中，这样就可以导入想要的库：
```python
sys.path.insert(0, './myRepo')
from  ... import ...
```
### 带有 `environment.yml` 的仓库
yaml 文件 `environment.yml` 定义了一个带有 conda 和 pip 依赖关系的虚拟环境。详细的文件格式描述可以在[这里](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)找到。
仍然是上面的例子：帐户 `myUserName`有一个GitHub 仓库 `myRepo`，可以这样添加：
```json
"requirements": ["repo:https://github.com/myUserName/myRepo"]
```
并在`env`字段中安装虚拟环境：
```json
"env": ["conda env create -f myRepo/environment.yml"]
```
### 托管在Dropbox上的仓库
还可以在 dropbox 上托管代码（以及数据）并通过https 请求来安装它。不过国内用户基本用不到这种场景。

## 虚拟环境
默认情况下，来自ImJoy的Python插件将在默认的conda环境中执行。然而，这些插件也可以具有特定的虚拟conda环境，这提供了一种隔离插件的方法。这样一来，就可以使用不同版本的Python运行它们，或者使用不同的conda或pip包。
建议每个插件及不同的标签使用独自的虚拟环境以保证稳定性。进一步地，建议为python、pip和conda包指定完整版本号(X.X.X)。通过指定完整版本，conda将尝试跨虚拟环境重用相同版本（和python版本）的包，从而减少所需的磁盘空间。
例如，以下两个环境将重用指定的 scipy 包，但不会重用 numpy 包：
```
conda create -n test_env1 python=3.6.8 scipy=1.1.0 numpy=1.16.1
conda create -n test_env2 python=3.6.8 scipy=1.1.0 numpy=1.15.4
```
为了在特定的 conda 环境中运行插件，可以通过在插件的 `<config>` 部分设置 `env` 字段来指定它。
`env` 可以是字符串或数组。在一行中需要执行多个命令时，需要使用 `&&` 或 `||`（注意操作系统的差别）。如果有多个相互独立的命令，可以使用一个数组来存储这些命令。例如：
`"env": ["export CUDA_VISIBLE_DEVICES=1", "conda create -n XXXXX python=3.7"]`。
还可以直接从“environment.yml”文件创建环境，例如
`"env": "conda env create -f ANNA-PALM/environment.yml"`。
有关更多信息，请参阅专用的 [conda 帮助页面](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)。

# 插件开发
配置好插件的属性及依赖后，最重要的部分是在 `<script>` 块中编写实际的插件代码。
ImJoy的插件系统建立在远程过程调用（RPC）之上，使用一套编码和解码方案在插件之间传输数据和功能。ImJoy开发者开发出了一组 API 函数（称为`ImJoy APIs`），以便插件与主应用程序以及插件彼此之间进行交互。
任何 ImJoy 插件都可以通过名为`api`的预定义对象访问`ImJoy APIs`。[ImJoy API 函数](https://imjoy.io/docs/#/api) 这一部分展示了这些API可用的完整列表（稍后的博客会详细介绍）。
## 导出插件函数
每个插件都需要导出一个带有一组函数的插件对象，这些函数将被注册为插件 API 函数（也称为`Plugin APIs`）。这是通过在插件代码末尾调用`api.export`来完成的。一旦导出，这些函数就可以在 ImJoy 中注册，并且可以由用户从 ImJoy 用户界面或其他插件调用。重要的是，`setup` 和 `run` 是两个必须被定义和导出的插件 API 函数。
## 回调函数
未导出为`Plugin API` 函数的插件函数，可以作为对象object发送给另一个插件或 ImJoy主程序。这些函数将被视为一个“回调”函数并且只能被调用一次。
一个典型的例子是通知函数，它可以告诉调用它的插件”计算已完成“。这样的功能必须只调用一次。
## 插件菜单
大多数插件都可以从插件菜单访问，默认情况下，当用户单击插件菜单中的插件名称时，将调用 `run` 函数。在插件（或`op`，下面会详述）的执行过程中，它可以从它的第一个参数（通常命名为`ctx`）中访问不同的字段：
（1）`ctx.config`：GUI 中使用 `ui` 字段定义的数值（来自插件文件的`<config>` 块或来自单独的操作`api.register`，更多信息如下）。例如，如果你在插件`<config>`中定义了一个ui字符串（例如`"ui": "option {id: 'opt1', type: 'number'}"`），你可以通过`ctx.config.opt1`访问它。
（2）`ctx.data`：它存储来自当前活动窗口的数据和运行插件的状态。请注意，仅当数据与插件或 op 定义的 `inputs`字段的 json 模板语法匹配时，才会将其传递给插件。比如拖拽进来后缀为png的文件时，它会匹配在`inputs`字段定义了png格式的插件，然后该文件就传给该插件进行处理。
（3）`ctx._variables`：当插件在工作流中执行时，工作流中设置的变量将作为 `ctx._variables` 传递。如果用户使用诸如“Set [number]”之类的操作，它将被设置为实际变量值。

结果可以直接被返回，它们将显示为通用结果窗口。如果想对结果定义特定的窗口类型，可以返回一个至少有两个字段 `type` 和 `data` 的对象。ImJoy 将使用 `type` 来查找用于渲染存储在 `data` 中的结果的窗口。
在下面的示例中，来自 url 的图像将显示在图像窗口中或传递给工作流中的下一个操作。
```javascript
   return { "type": "imjoy/image", "data": {"src": "https://imjoy.io/static/img/imjoy-icon.png"} }
```
ImJoy 使用 postMessage 在插件之间交换数据。这意味着对于 JavaScript插件，对象在传输过程中被复制。如果交换的数据特别大，那么创建的对象就不应该直接传输。要启用这种模式，可以在返回对象时将 `_transfer=true` 添加到对象中。在上面例子中，可以在返回 `ctx` 时设置 `ctx._transfer = true`。然而，它会只加速 `ArrayBuffers` 或 `ArrayBufferViews` 的传输（以及ndarrays 由 Python 生成），传输完毕后，将无法访问它。
(注意：在Python中，`ctx`的数据类型是字典，ImJoy添加了允许点表示法的接口，就像在JavaScript中一样。如果愿意，也可以在两种语言中使用`[]`访问字典或对象。）

## 插件操作符（操作）
可以使用 Plugin API 为插件定义独立的运算符（或 `ops`）（有关详细信息可参考api.register）。这些`ops` 的定义方式与`<config>`块类似：可以定义一个接口来设置参数，它有一个专门的运行功能。当按下插件列表中的向下箭头按钮时，会显示这些不同的操作`ops`。

## 数据交换
### 加载/保存文件
在 ImJoy 中可以以不同方式访问存储在文件系统上的文件。
一方面，在没有插件引擎的情况下，若想访问本地文件，用户需要打开或拖动文件导入ImJoy，对于结果文件则只能通过触发下载来保存。
另一方面，插件引擎拥有对本地文件系统的完全访问权限，native python 插件可以直接从本地文件系统读取和写入。
![data-access](https://user-images.githubusercontent.com/6218739/144711988-32c6ef87-129a-4942-a5b8-2a0bb1185085.png)

因此，ImJoy提供了几种不同的方法来处理插件的加载/保存文件：
（1）如果Plugin Engine正在运行，那么对于所有类型插件，有 3 个api函数可以访问本地文件系统：`api.showFileDialog`、`api.getFileUrl`、`api.requestUploadUrl`。特别地，对于在插件引擎上运行的Python插件，可以直接使用标准的 python 文件操作来将文件加载或写入文件系统。
（2）如果Plugin Engine未运行，JavaScript或Web Python 插件访问文件的唯一途径是要求用户将文件或文件夹直接拖到 ImJoy 工作区中。这将呈现一个包含文件或文件夹内容的窗口。然后这些数据可以被插件访问并进行处理。要将结果导出为文件，可以使用`api.exportFile` 函数来触发下载。
与通过插件引擎运行的插件相比，在浏览器中运行的web插件无法直接访问本地文件系统，因此它们提供比 `native-python` 插件更高的安全性。

###通过函数调用来交换数据
对于运行时生成的少量数据（例如：数组、对象等），可以将数据对象作为参数传递给API函数（或者从API函数返回数据对象），从而将它们发送到另一个插件。
例如，可以将包含小型numpy数组、字符串、字节的数据（<10MB）从远程插件引擎中运行的`native-python`插件直接发送到在浏览器中运行的`window`插件。
为了快速显示小图像，还可以将其保存为 png 格式并将其编码为 base64 字符串，然后可以使用标准的 HTML `《img>` 标签直接显示。
```python
with open("output.png", "rb") as f:
    data = f.read()
    result = base64.b64encode(data).decode('ascii')
    imgurl = 'data:image/png;base64,' + result
    api.createWindow(name='unet prediction', type='imjoy/image', w=7, h=7, data= {"src": imgurl})
```

###通过窗口传递数据
ImJoy中显示的窗口可以包含数据，例如图像，因此您可以将窗口用作将数据从一个插件传递到另一个插件的一种方式。
选择此类窗口并执行插件时（通过单击插件菜单），所包含的数据将被传送到该插件。
重要的是，如果窗口中包含的数据对象匹配某插件的 `<config>` 中 `inputs` 定义的 json 模式，ImJoy 就会将窗口中包含的数据传递给该插件。然后插件可以通过访问 `ctx.data` 来处理数据（通常在 `run` 函数内）。
ImJoy 原生支持 Numpy 数组和TensorFlow张量的转换和传输。插件开发人员可以直接使用这些数据类型并在插件之间进行交换，而不用管它们是在 Python 还是 JavaScript 中。
###处理大数据量
对于大数据量，一种方式是以较小的块chunks来发送大文件，但这不是最佳的，并且可能会阻碍引擎和ImJoy应用程序之间的正常通信。建议将数据存储在磁盘上（例如在工作区目录中），然后使用`api.getFileUrl` 生成访问文件的 url。然后可以将生成的 url 发送到 Web 应用程序，并通过下载链接或使用 JavaScript 库（如“axios”）进行访问。很多库如 Three.js、Vega 等都可以直接通过 url 加载文件。

## 工作流管理
ImJoy在 `ctx` 中提供了额外的字段，允许跟踪、维护和重建整个分析工作流程。
（1）`ctx._op`：给出正在执行的操作的名称。当一个插件注册了多个 op 并且没有为 op 指定回调函数时，`run` 函数将被调用，可以使用 `ctx._op` 来确定正在执行哪个 op。
（2）`ctx._source_op`：给出启动当前动作的op的名称。
（3）`ctx._workflow_id`：当插件在工作流中执行时，它的 id 将被传递到这里。当插件从插件菜单执行时，ImJoy 将尝试重用当前活动窗口中的工作流 id，如果没有窗口处于活动状态，则会分配一个新的工作流 id。所有具有相同 `_workflow_id` 的数据窗口在管道或计算图中虚拟连接。重要的是，通过将 `_workflow_id` 与 `_op` 和 `_source_op` 结合起来，`_workflow_id`、`_variables`、`_op` 和`_source_op` 可用于实现插件之间的交互，这意味着如果用户在结果窗口中更改了状态，下游工作流将自动更新。

## Native Python插件的执行标志
可以使用 `<config>` 块中的 `flags` 字段来控制 Python 插件进程的执行。接下来，将解释插件引擎上运行的 Python 进程如何与 ImJoy 界面交互。
![python-process](https://user-images.githubusercontent.com/6218739/144712604-1b9ba517-ea26-424f-ac87-9a5ffb2e98f1.png)
上图中的几个概念：
- Interface：ImJoy 的web界面。可以在多个浏览器窗口（即多个界面）上运行 ImJoy。
- Plugin Engine：在后台运行以执行来自不同 Python 插件的 Python 代码。
- Python Plugin：包含 Python 代码的插件。一些插件可能有`tags`来进一步指定它们的执行细节。
- Python Process：在插件引擎上运行的特定 Python 插件。进程可以在任务管理器上看到。
- Workspace：已安装的 ImJoy 插件的集合。对于带有`tags` 的插件，用户选择合适的标签。工作区中的每个 Python 插件都有自己的进程。每个工作区都有一个唯一的名称。
- ImJoy instance：是在一个 ImJoy 界面中运行的工作区。

下面将介绍控制python插件执行的三个执行标志：
- 默认（没有设置任何标志）：每个 ImJoy 实例在插件引擎上都有自己的进程。如果关闭该界面，则会终止该进程。
- `single-instance`：该标志将只允许一个进程为整个插件引擎运行。对于具有相同名称和标签的插件，那么`single-instance` 意味着它们访问相同的进程。
- `allow-detach`： 该标志表示进程在其 ImJoy 实例关闭时不会被终止。例如，这允许在后台执行长时间的计算任务，这些任务不需要额外的用户反馈并自动终止。还可用于保护长时间的计算任务免受浏览器不稳定的影响。如果希望能够附加到分离的进程，可以从相同的浏览器和工作区重新连接，或者结合使用 `single-instance` 标志，这样尽管从不同的浏览器和工作区连接，仍然能连接到之前的进程。

当 ImJoy 尝试重新连接之前分离的插件进程时，如果在插件类中定义了`resume()`，那么`resume()`将被调用，否则像往常一样调用`setup()`。请注意，当存在`resume`时，在重新附加期间不会调用 `setup`。
这是 `flags` 选项在 `<config>` 块中的样子：
```json
<config lang="json">

  ...
  "flags": ["single-instance", "allow-detach"],
  ...

</config>
```
`flags` 也可以使用 `tags` 进行配置，例如：
```json
<config lang="json">

  ...
  "tags": ["Single", "Multi"],
  "flags": {"Single": ["single-instance", "allow-detach"], "Multi": []},
  ...

</config>
```
上面的 `<config>` 块将创建一个带有两个标签（`Single` 和 `Multi`）的插件。在安装期间，用户可以选择希望使用哪种运行时行为（插件进程的单个实例（`Single`），或者在同一个工作区打开多个 ImJoy 界面对应的多插件进程（`Multi`））。

# 构建用户界面
ImJoy的一个重要部分是提供一种灵活的方式与用户交互，以丰富的交互方式指定输入信息或生成输出。
ImJoy自带有一组基本元素，例如表单和进度条，它们提供了一种与用户交互的方式。
还可以使用自定义窗口构建更先进和强大的用户界面，开发人员可以利用基于Web的UI库来生成控件、交互式图表或呈现 3D 视图。
## 基本的用户输入和输出
获取用户输入的最简单方法是使用 ImJoy 生成的表单。
可以在 `<config>` 块中定义一个 `gui` 字符串，它将在插件菜单下呈现为表单。如果插件使用了多个插件操作op（将在API一文中解析），还可以为每个插件操作单独提供`gui`字符串。
当用户通过插件菜单运行插件或 op 时，`gui` 字符串中定义的所有字段值将被包装为 `config` 对象并传递到 `run` 函数中。然后可以通过 `ctx.config.FIELD_ID` 访问它们。
对于其他类型的输入，可以使用其他ImJoy API，例如弹出一个文件对话框`api.showFileDialog`。
为了显示结果或向用户提供反馈，ImJoy 提供了多个 API 函数来显示结果，例如使用`api.alert()`或`api.showMessage()`显示消息，使用`api.log()`或`api.error()`记录消息或错误，以及`api.showProgress`表示进度、`api.showStatus`更新ImJoy状态等。这些API函数将在API一节中详细解析。

## 自定义窗口的用户输入和输出
对于更灵活的用户界面，开发者可以制作专门的窗口插件。由于是基于`iframe`，所以大部分前端（HTML/CSS/JS）框架都可以在`window`插件中使用。此外，这样的接口可以与另一个插件通信，例如执行实际分析的 Python 插件。
使用`window`插件模板可以轻松创建这样的插件。除了其他插件之外，`window`插件还有两个额外的代码块：`<window>` 和`<style>`。用户可以将前端代码添加到相应的代码块中。创建后，这个新的`window`插件将用作模板来创建新的窗口实例。
ImJoy提供了两个API函数，用于从`window`插件创建窗口`api.createWindow`或显示对话框`api.showDialog`。
除此之外，对于常用的窗口类型，ImJoy 支持一组内部窗口类型，详见`api.createWindow`这个API。

## 更多示例插件
请前往[Demos](https://imjoy.io/docs/#/demos)。

# 托管和部署插件
这一部分提供了有关如何托管或部署 ImJoy 插件的详细信息。
这包括从存储单个文件到设置开发者自己的 ImJoy 插件库。然后插件可以直接作为文件分发或使用专用的 url 语法从而允许自动安装。
ImJoy 插件的默认方式及推荐方式都是部署在 GitHub 上（或作为单个文件或在插件仓库中），然后使用插件 url 分发。
这里推荐 GitHub，因为它提供稳定性和版本控制，保证了重现性和可追溯性。
## 托管单个插件文件
这是开发过程中的典型案例。
插件代码可以托管在网络上，例如GitHub、Gist 或 Dropbox。

## 自定义的ImJoy插件库
可以轻松地为现有 GitHub 项目创建 ImJoy 插件仓库。
可以在 [此处](https://github.com/imjoy-team/imjoy-project-template) 找到模板项目。
然后将 ImJoy 插件保存在一个专用文件夹中，并添加一个清单文件 `manifest.imjoy.json` 到 GitHub 根文件夹。
此清单指定了仓库中有哪些插件，以及在哪里可以找到它们。该文件的架构如下所示，完整的模板可以在[这里](https://github.com/imjoy-team/imjoy-project-template/blob/master/manifest.imjoy.json)找到。
```json
{
 "name": "NAME OF THE REPOSITORY",
 "description": "DESCRIBE THE REPOSITORY",
 "version": "0.1.0",
 "uri_root": "",
 "plugins": [
   //copy and paste the <config> block of your plugin here
 ]
}
```
然后就可以自动或手动更新此清单：
（1）对于自动更新，ImJoy提供了一个[node脚本](https://github.com/imjoy-team/imjoy-project-template/blob/master/update_manifest.js)。此脚本需要执行node.js。然后在包含`manifest.imjoy.json`文件的根文件夹中使用命令 `node update_manifest.js` 运行它，它会自动搜索ImJoy插件及生成清单。请注意，当第一次使用此nodejs脚本时，必须手动更改插件仓库的名称`name`。对于后续更新，该名称将保持不变。
（2）对于手动更新，按照下列步骤操作：
（2.1）将所有插件文件放在 GitHub 仓库中的一个文件夹中。例如一个名为[imjoy-plugins](https://github.com/imjoy-team/imjoy-project-template/tree/master/imjoy-plugins)的文件夹。
（2.2）修改`manifest.imjoy.json`。对于每个插件：
（2.2.1）从插件代码中复制并粘贴`<config>`块的内容到 `manifest.imjoy.json` 中的 `plugins` 块。
（2.2.2）添加一个名为`"uri"`的字段，并将值设置为插件的实际文件名，包括 GitHub 仓库中的相对路径。例如，对于一个名为“untitledPlugin.imjoy.html”的插件文件，设定`"uri": "imjoy-plugins/untitledPlugin.imjoy.html"`。如果你将插件的名字设为与插件文件的名字相同，则可以跳过此步骤。
在ImJoy中，可以以一个简单的url形式`http://imjoy.io/#/app?repo=GITHUB_USER_NAME/REPO_NAME`呈现仓库中所有插件的列表，其中`GITHUB_USER_NAME`是用户名，`REPO_NAME`是包含 ImJoy 插件的GitHub仓库的名称。然后用户就可以通过此列表来安装插件。有关如何生成此 url 的更多详细信息，并查看如何可以安装特定的插件，可以参阅下面的专门部分。
## 官方ImJoy插件库
`ImJoy.io` 上显示的 ImJoy 插件库通过[GitHub](https://github.com/imjoy-team/imjoy-plugins)部署。
为了将开发者自己的插件部署到[该插件库](https://github.com/imjoy-team/imjoy-plugins)，可以fork该库，添加插件并发送pull request。PR被接受后，用户将能够从官方的插件仓库安装这个插件。

# 分发插件
要分发自己开发的插件，有两个主要选项。
（1）可以创建一个完整的url地址。单击时，ImJoy 将自动打开并安装插件。此链接可直接通过电子邮件或社交网络分享。在下面将详细说明如何创建此链接以及支持哪些选项。
（2）可以直接发送插件文件（扩展名`*.imjoy.html`）。这个文件可以被拖入ImJoy工作区，此时它会被自动识别为插件。

## 使用自定义库分发插件
如果开发的插件依赖于非标准库和模块，那么开发者必须随着插件一块提供这些库。可以将这些库和模块上传到 GitHub 仓库、GitHub Gist 或其他数据共享平台（例如 Dropbox）并将它们链接到插件代码中。
（1）对于JavaScript插件，需要创建一个 Gist 或 GitHub。将插件（以`.imjoy.html` 结尾）文件与其他 JavaScript 文件一起上传。在插件文件中，可以将 url 添加到插件 `requirements` 中。但是由于 GitHub 限制，不能直接使用 GitHub url，不过可以使用 [combinatronics.com](https://combinatronics.com/) 进行转换。
（2）对于Python插件，建议将这些非标准库打包为 pip 模块，并放在GitHub上。

## 存储在 Dropbox 上的代码/数据的分发
此示例描述了如何部署和分发存储在 Dropbox 上的 Python 插件。
这允许共享私有项目。
（1）将code或data以 zip 文件的形式存储在 Dropbox 上。这允许通过替换 zip 文件来替换代码/数据（请参阅下面的注释）。
（2）将ImJoy 插件文件 (`.imjoy.html`) 使用私有或公开的gist托管。
假设 Python 代码位于存储在 Dropbox 上的 Zip 存档`testcode.zip`中，并且可通过链接“DROPBOXLINK/testcode.zip”获得。然后，可以将以下代码片段放在插件的 `setup()` 函数中以使其可用。该片段执行以下步骤：
（1）执行 [http 请求](http://docs.python-requests.org)。注意此请求中的`dl=1`选项。默认情况下，此值设置为 0。
（2）使用返回的请求对象在本地生成zip文件，解压，最后删除。
（3）将本地路径添加到系统路径中。
```python
import sys
import os
import requests
import zipfile

url = 'https://DROPBOXLINK/testcode.zip?dl=1'
r = requests.get(url, allow_redirects=True)

# download the zip file
name_zip = os.path.join('.','testcode.zip')
open(name_zip, 'wb').write(r.content)

# extract to the current folder (i.e. workspace)
with zipfile.ZipFile(name_zip, 'r') as f:
    f.extractall('./')
os.remove(name_zip)

# If you want to import your python modules, append the folder to sys.path
sys.path.append(os.path.join('.','testcode'))
```
一些注意点：
（1）代码本地存储在`USER_HOME/ImJoyWorkspace/WORKSPACENAME/testcode`中，其中WORKSPACENAME为当前ImJoy工作空间的名称。可以在提供的 URL 中自动设置工作区以分发插件。
（2）更新zip存档时，不要删除旧的，用新版本替换它。这保证了相同的链接是有效的。
（3）该代码每次都会安装当前版本的ZIP插件。

## 生成用于共享的插件url
分发插件的最简单方法是创建一个url，它可以通过电子邮件或社交网络共享。
基本格式是`http://imjoy.io/#/app?plugin=PLUGIN_URI`。注意要用实际plugin URI（统一资源标识符）来替换`PLUGIN_URI`。例如：[https://imjoy.io/#/app?plugin=https://github.com/imjoy-team/imjoy-plugins/blob/master/repository/imageWindow.imjoy.html](https://imjoy.io/#/app?plugin=https://github.com/imjoy-team/imjoy-plugins/blob/master/repository/imageWindow.imjoy.html)。当用户点击此链接时，将显示一个插件安装对话框，提示安装指定的插件，用户只需单击“安装”即可确认。
此 url 支持一些额外参数来控制插件加载方式。这些参数将在下面的专门部分中进行描述。
有两种类型的URI，具体取决于插件的部署方式：
（1）如果插件部署在`ImJoy Plugin Repository`（如上所述），就可以使用格式为“GITHUB_USER_NAME/REPO_NAME:PLUGIN_NAME”的简短URI。比如可以用`imjoy-team/imjoy-project-template:Untitled Plugin`来表示托管在 https://github.com/oeway/DRFNS-Lite 上的插件。
还可以通过在 `PLUGIN_NAME` 后添加 `@TAG` 来指定plugin的标签。例如：`oeway/DRFNS-Lite:DRFNS-Lite@GPU`。
如果还想指定一个git commit hashtag来将插件固定为某些提交，可以在`REPO_NAME`之后添加`@COMMIT_HASHTAG`。例如：`oeway/DRFNS-Lite@4063b24:DRFNS-Lite`，其中`4063b24`是[4063b24f01eab459718ba87678dd5c5db1e1eda1](https://github.com/oeway/DRFNS-Lite/tree/4063b24f01eab459718ba87678dd5c5db1e1eda1)的短格式。
（2）还可以使用url指向任何托管的插件网站，包括开发者自己的项目站点、博客、GitHub、Gist或Dropbox。注意，插件文件需要以`.imjoy.html` 结尾。下面将介绍如何为不同的托管平台获取此 url：
（2.1）GitHub上的文件，只需复制文件链接即可。例如：`https://github.com/imjoy-team/imjoy-plugins/blob/master/repository/imageRecognition.imjoy.html`。
（2.2）对于Gist或其他Git提供者（如GitLab），如果Gist中只有一个文件，可以直接使用Gist链接（从浏览器地址栏复制）或获取插件`raw`文件的链接。对于具有多个文件的 Gist，需要为要使用的插件文件指定 `raw` 链接。要创建 Gist `raw` 链接：
（2.2.1）在自己的 GitHub 账户上访问Gist[https://gist.github.com/](https://gist.github.com/)
（2.2.2）创建新的 Gist，指定插件名称后跟 `.imjoy.html`，然后复制并粘贴插件代码。
（2.2.3）创建公共或私有 Gist。
（2.2.4）可以从`Raw`按钮获得Gist的链接（这链接到文件的未处理版本）。该链接如下所示：`https://gist.githubusercontent.com/oeway/aad257cd9aaab448766c6dc287cb8614/raw/909d0a86e45a9640c0e108adea5ecd7e78b81301/chartJSDemo.imjoy.html`。
（2.2.5）需要注意，当更新文件时，此url会更改。
（2.3）对于Dropbox，需要修改可共享的url如下：
（2.3.1）将`dl=0`替换为`dl=1`；
（2.3.2）将 `https://www.dropbox.com/` 替换为 `https://dl.dropboxusercontent.com/`。

要指定插件标签，只需在 `.imjoy.html` 后附加 `@TAG`。例如：
`https://raw.githubusercontent.com/oeway/DRFNS-Lite/master/DRFNS-Lite.imjoy.html@GPU`。
可以在 ImJoy 中测试插件 url 是否有效：将其粘贴到 `+ PLUGINS` 对话框中，（`Install from URL`）并按`Enter`。如果一切正常，应该能够查看使用插件呈现的卡片，然后可以单击`INSTALL`。

##支持的url参数
可以使用自定义参数来构建 ImJoy url，以便于安装。这些 url 参数可以在 `https://imjoy.io/#/app?` 之后使用，使用 `PARAM=VALUE` 语法。
这些参数相互独立，多个参数可以通过`&`连接。例如我们要指定`par1=99`和`par2=hello`，相应的 url 将是 `https://imjoy.io/#/app?par1=99&par2=hello`。
目前支持以下 url 参数：
（1）`plugin` 或 `p`：在插件管理对话框中显示指定的插件。如果存在具有相同名称和版本的插件，则不会显示该对话框。如果需要，添加 `upgrade=1` 以强制显示插件对话框。例如：`https://imjoy.io/#/app?p=imjoy-team/imjoy-demo-plugins:alert&upgrade=1`。
（2）`workspace` 或 `w`：工作区的名称。 ImJoy 将切换到指定的工作区（如果存在），或创建一个新的工作区。例如，`https://imjoy.io/#/app?workspace=test`
（3）`engine` 或 `e`：定义插件引擎的url。例如：`http://imjoy.io/#/app?engine=http://127.0.0.1:9527`。注意，如果想通过http（而非https）连接到远程机器，则只能使用 `http://imjoy.io` 而不是 `https://imjoy.io`。如果在某些浏览器（例如Firefox）中使用localhost，则此限制也存在。为避免这种情况，需要使用 `http://127.0.0.1:9527` 而不是 `http://localhost:9527`，因为大多数浏览器会认为 `127.0.0.1` 是安全连接，而`localhost`不是。但是，有一个例外，在 Safari 上，使用 `127.0.0.1` 不起作用，因为[此限制](https://bugs.webkit.org/show_bug.cgi?id=171934)，请使用Firefox或Chrome。
（4）`token` 或 `t`：定义连接令牌。例如：`http://imjoy.io/#/app?token=2760239c-c0a7-4a53-a01e-d6da48b949bc`
（5）`repo` 或 `r`：指定指向 ImJoy 插件仓库的清单文件。这可以是一个完整的 repo 链接，例如 `repo=https://github.com/imjoy-team/imjoy-plugins`或简化的 GitHub 链接 `repo=imjoy-team/imjoy-plugins`。如果从非GitHub网站（例如GitLab）托管插件库，就使用指向 `manifest.imjoy.json` 文件的 `raw` 链接。
（6）`start` 或 `s`：定义一个启动插件名称，它会在 ImJoy web 应用程序加载后自动启动。所有 url 参数将作为 `ctx.config` 传递给插件到 `run(ctx)` 函数。这允许添加自定义参数并在 `run(ctx)` 中使用它们。例如，插件可以使用`load=URL` 自动加载图像，并使用`width=1024&height=2048` 设置图像的宽度和高度。例如，将`123`作为`ctx.data.x`传递给插件的`run`函数：`https://imjoy.io/#/app?x=123&start=AwesomePlugin`。如果正在启动一个`window`插件，还可以将`standalone`或`fullscreen`设置为`1`以使窗口脱离工作区或处于全屏模式。例如：`https://imjoy.io/#/app?x=123&start=AwesomeWindowPlugin&fullscreen=1`。
（7）`load` 或 `l`：定义一个用于发起 http GET 请求的 URL，这个参数应该只在定义了一个带有 `start` 或 `s` 的启动插件时使用。从 URL 获取的数据将作为 `ctx.data.loaded` 传递给启动插件 `run(ctx)` 函数。
（8）`expose`：当 imjoy 嵌入到 iframe 中时，该参数指定是否应该将其 API 暴露给外部上下文（默认情况下不会暴露）。要启用，可以将 `expose=1` 添加到 URL 查询。

## 添加ImJoy徽章
如果开发者在项目中使用 ImJoy，建议将ImJoy徽章之一添加到项目仓库（例如在 Github 上）或网站。ImJoy有两个官方徽章：![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)和![powered by ImJoy](https://imjoy.io/static/badge/powered-by-imjoy-badge.svg)。
对于存储 ImJoy 插件的仓库，可以使用![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)徽章。
Markdown：
```
[![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?plugin=<YOUR PLUGIN URL>)
```
reStructuredText：
```
.. image:: https://imjoy.io/static/badge/launch-imjoy-badge.svg
 :target: https://imjoy.io/#/app?plugin=<YOUR PLUGIN URL>
```

对于其他情况，例如，如果只是想感谢ImJoy，则可以使用![powered by ImJoy](https://imjoy.io/static/badge/powered-by-imjoy-badge.svg)。
Markdown:
```
[![powered by ImJoy](https://imjoy.io/static/badge/powered-by-imjoy-badge.svg)](https://imjoy.io/)
```

reStructuredText:
```
.. image:: https://imjoy.io/static/badge/powered-by-imjoy-badge.svg
 :target: https://imjoy.io/
```
