---
title: 开源深度学习计算平台ImJoy解析：6 -- API
tags: [ImJoy]
categories: computer vision 
date: 2021-12-11
---

# 参考文献
[ImJoy API](https://imjoy.io/docs/#/api)

# 简介
ImJoy的每个插件都在自己的类似沙箱的容器环境中运行（如JavaScript插件的Web Worker 或iframe、Python插件的进程process）。这样可以避免其他插件的干扰并使ImJoy应用程序更加安全。
插件与ImJoy主程序或插件与插件之间的交互是通过一组API函数（`ImJoy API`）进行的。所有插件都可以访问到一个名为`api`的特殊对象。在Javascript中，`api` 是一个可以直接使用的全局对象。在Python中，可以通过调用 `from imjoy import api` 来导入它。有了这个对象，插件可以做很多事情，比如显示一个对话框、将结果发送到主应用程序，或调用另一个插件的参数和数据等。

# 异步编程
为了使交互更加高效和并发，对于这些API函数，ImJoy使用一种称为[异步编程](http://cs.brown.edu/courses/cs168/s12/handouts/async.pdf)的编程模式。
所有ImJoy API函数都是异步的。这意味着当一个`ImJoy API`函数被调用时，ImJoy不会阻止原程序的执行，而是会立即返回一个名为 [Promise(JS)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/ Promise) 或 [Future (Python)](https://docs.python.org/3/library/asyncio-future.html)的对象。开发者可以决定等待实际结果返回或设置一个回调函数以在进程终止后再取回结果。例如，如果弹出一个对话框要求用户输入，在许多编程语言（同步编程）中，代码执行将被阻塞，直到用户关闭对话框。但是，对于一个异步程序，尽管用户没有关闭对话框，它会返回一个`promise`对象，然后继续执行。
由于每个API调用都是异步和非阻塞的，一个给定的插件可以调用多个其他插件来同时执行任务，而无需使用多线程或多进程等技术。
对于Python和JavaScript插件，ImJoy 都支持两种异步编程风格来访问这些异步函数：`async/await` 和 `callback` 风格。几个重要的注意点是：
- 对于JavaScript和Python 3，推荐`async/await`
- `callback` 样式可用于 JavaScript、Python 2 和 Python 3。
- 注意，不能同时使用这两种风格。
- 对于`async/await` 风格，可以使用 `try catch` (JavaScript) 或 `try except` (Python) 语法来捕获错误，但对于 `callback` 风格，则不能使用它们来捕获错误。

在下面的API函数列表中，提供了 `async` 风格的示例。对于 Python 2，也可以轻松地转换为相应的`callback`风格。

## async/await风格
Javascript 和 Python 3+ 插件原生支持并推荐使用这种风格。
使用 `async` 关键字声明函数。在异步之前添加`await`函数等待结果返回。这基本上就是无需设置回调函数即可进行同步编程。
下面是一个名为`XXXXX`的api函数的简单示例。注意，在函数中使用await时，要在函数定义前添加async。对于 Python3，不要忘记`import asyncio`。
JavaScript语言的实现为：
```javascript
 class ImJoyPlugin(){
  async setup(){
  }
  async run(ctx){
    try{
      result = await api.XXXXX()
      console.log(result)
    }
    catch(e){
      console.error(e)
    }
  }
}
```
Python语言的实现为：
```python
import asyncio
from imjoy import api

class ImJoyPlugin():
    async def setup(self):
        pass

    async def run(self, ctx):
        try:
            result = await api.XXXXX()
            print(result)
        except Exception as e:
            print(e)
 ```

## Callback风格
对于 Python 2 或 Web Python，不支持 `asyncio`，因此此时需要使用`callback`风格。
调用异步函数并使用 `.then(callback_func)` 设置其回调。对于 JavaScript 插件，将返回原生 JavaScript `Promise`（[更多关于 Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)。对于 Python 插件，它将返回`Promise`的一个简化的Python实现。
下面是名为“XXXXX”的 api 函数示例：
```javascript
class ImJoyPlugin(){
  setup(){
  }

  run(ctx){
      api.XXXXX().then(this.callback)

      // optionally, you can catch error
      const error_callback(error){
        console.error(error)
      }
      api.XXXXX().then(this.callback).catch(error_callback)
  }

  callback(result){
     console.log(result)
  }
}
```
或
```python
from imjoy import api
class ImJoyPlugin():
    def setup(self):
        pass

    def run(self, ctx):
        api.XXXXX().then(self.callback)

        # this function will be called when we get results from api.XXXXX()
        def result_callback(result):
            print(result)

        # optionally, you can catch an error for the call
        def error_callback(error):
            print(error)

        api.XXXXX().then(result_callback).catch(error_callback)
```

# 输入参数
调用API函数时，大多数函数接收一个对象object（对于JavaScript）或字典dictionary（对于Python）作为它的第一个参数。
以下函数调用对JavaScript和Python都适用：
```javascript
// JavaScript or Python
await api.XXXXX({"option1": 3, "option2": 'hi'})
```
以下调用则仅适用于 Python：
```python
# Python
await api.XXXXX(option1=3, option2='hi')
```

# 净化的HTML和CSS 
出于安全原因，ImJoy使用[DOMPurify](https://github.com/cure53/DOMPurify) 来净化ImJoy主界面中使用的HTML和CSS，主要是为了防止跨站脚本攻击XSS，包括所有的markdown、`<config>`块和`api.register`中的`ui`字符串，以及`api.alert`、`api.confirm`和`api.prompt`中显示的内容。 另请注意，`window`插件中显示的内容没有这些限制。


# API函数
对于每个API函数，下面提供了一个简短的代码片段来说明如何使用它的功能。这些例子是JavaScript语言编写的，但在Python中可以以类似的方式调用。

## api.alert
```javascript
await api.alert(message)
```
参数：
对于纯文本：
* `message`：String类型。其包含要显示的消息。消息中可以使用HTML标签，但仅限于一组受限的标签和 css，如上面[净化的HTML和CSS]一节所述。

对于HTML：
* `message`：Object类型。它包含以下字段：
   - content：包含要显示的问题。消息中可以使用HTML标签，但仅限于一组受限的标签和 css，如上面[净化的HTML和CSS]一节所述。
   - title：对话框的标题

举例：
```javascript
await api.alert('hello world')
```

## api.prompt
```javascript
const answer = await api.prompt(question, default_answer)
```
显示要求用户输入的提示。
参数：
对于纯文本：
* question：String。包含要显示的问题。消息中可以使用 HTML 标签，但仅限于一组受限的标签和 css，如上面[净化的HTML和CSS]一节所述。

对于 HTML：
* question：Object。它包含以下字段：
  - content：包含要显示的问题。消息中可以使用 HTML 标签，但仅限于一组受限的标签和 css，如上面[净化的HTML和CSS]一节所述。
  - placeholder：问题的默认答案
  - title：对话框的标题

* default_answer（可选）：String。包含问题的默认答案。

返回值：
* answer：布尔值Boolean。来自用户的回答。

例子：
```javascript
const answer = await api.prompt('What is your name?')
```

## api.confirm
```javascript
const confirmation = await api.confirm(question)
```
向用户显示确认消息。
参数：
对于纯文本：
* question：String。包含要显示的问题。消息中可以使用 HTML 标签，但仅限于一组受限的标签和 css，如上面[净化的HTML和CSS]一节所述。

对于 HTML：
* question：Object。它包含以下字段：
  - content：包含要显示的问题。消息中可以使用 HTML 标签，但仅限于一组受限的标签和 css，如上面[净化的HTML和CSS]一节所述。
  - title：对话框的标题

返回值：
* confirmation：布尔值。对或错。

例子：
```javascript
const confirmation = await api.confirm('Do you want to delete these files?')
if(confirmation){
  delete_file()
}
else{
  console.log('User cancelled file deletion.')
}
```

## api.call
```javascript
result = await api.call(plugin_name, plugin_op, arg1, arg2 ...)
```
从另一个插件调用函数。
还可以传输数据。参数的数量必须与被调用函数的参数所需的数目相匹配。
如果想频繁调用其他插件的功能，推荐使用`api.getPlugin`。

参数：
* plugin_name：String。被调用插件的名称。
* plugin_op：String。插件的函数的名称 (op)。
* args（可选）：可以传输的任何受支持的原始数据类型。

返回值：
* result：插件的函数返回的结果（如果有）。

例子：
调用插件`PluginX`中定义的函数`funcX`，并传入参数`1`：
```javascript
await api.call("PluginX", "funcX", 1)
```

## api.createWindow
```javascript
win = await api.createWindow(config)
```

在 ImJoy 工作区中创建一个新窗口。
一旦创建了一个窗口，它将返回一个带有相应`window`插件API的对象，该对象可用于更新窗口，比如可以使用 `win.run({"data": ...})` 更新数据字段。
注意1：调用关闭窗口的函数。如果窗口关闭并且尝试调用它的函数，将会得到一个错误，解决这个问题的一种方法是使用 `try ... catch `(JavaScript) 或 `try: ... except: ...`(Python) 语句来捕获错误。
注意2：`api.createWindow`和`api.getPlugin`之间的区别。两个函数都可用于获取包含`window`插件api的对象。但是，只有通过`api.createWindow`获取的对象才能用于更新现有窗口。相比之下，`api.getPlugin`返回的对象每次使用都会创建一个新窗口。这种行为差异可以通过ImJoy处理不同插件类型的行为来解释。当ImJoy加载一个不是`window`插件的插件时，它会启动独立的python进程、webworker或 iframe。这确保了只有一个插件实例正在运行。相比之下，`window` 插件将仅用来注册并创建代理插件，此时没有实际实例开启，除非用户点击插件菜单或插件被另一个插件所调用。工作区中的每个窗口都是 `window` 插件的一个新实例。当`api.getPlugin` 被调用时，它会返回代理插件的api，例如`proxy = await api.getPlugin({'name': 'Image Window'})`)。每次`run`函数执行后，将创建一个新窗口。例如，如果运行`proxy.run({data: ...})`10次，将创建10个窗口。要取得使用`api.createWindow`（或`api.showDialog`）创建的窗口，可以使用`api.getWindow`。当使用`api.createWindow`时，它会返回一个`window`插件的实例，例如`win=await api.createWindow({'name': 'new window', 'type': 'Image Window', 'data': {...}})`)。如果运行 `win.run({'data': ...})` 10 次，同一个窗口实例将被更新10次。运行 `win.close()` 将关闭窗口。

参数：
* config：String或Object。创建窗口的选项。当它是一个字符串时，它会被转换成一个对象。转换按照以下规则进行： 1) 如果字符串包含多行，比如URL或插件URI，则将其视为窗口插件源（参见下面的 `src` 键）； 2) 否则将被视为`window`插件类型（参见下面的“type”键）。
它包含以下字段：
   - `name`：String。指定新窗口的名称。
   - `type`：String。指定窗口类型。这可以是`window`插件名称、ImJoy内部窗口类型或`external`。以下内部窗口类型是支持的：
      - `imjoy/generic`。将显示 `data` 中的所有对象。
      - `imjoy/image`。显示图像。需要`data.src` 指向图像位置。
      - `imjoy/image-compare`。显示进行比较的两个图像。图像作为 `data.first` 和 `data.second` 传递。请参阅下面的示例。
      - `imjoy/panel`。渲染`<config>` 块中的`ui`字段。 
      - `imjoy/markdown`。渲染`data.source`中提供的Markdown文本。
      - `imjoy/plugin-editor`。打开源代码编辑器。 `data.id`是一个唯一的字符串（最好是随机的），用来指定窗口的id，`data.code` 包含源代码。
   
   如果从外部url创建窗口，例如托管在Github页面上的Web应用程序。在这种情况下，需要使用 `src` 键指定url。有关如何从外部Web应用程序支持ImJoy的详细信息，请参见 [此处](https://github.com/imjoy-team/imjoy-core/)。
   如果外部网页加载了ImJoy 插件api，就可以像普通的 ImJoy 插件一样与外部网站进行交互。但是，如果外部网页不支持ImJoy，则需要设置`passive=true`来告诉ImJoy该窗口将没有插件api。
   - `src`：String，指定窗口插件的源代码、窗口插件源代码的 url 或支持 `imjoy-rpc` 的网络应用程序的 url。如果 url 以 `.imjoy.html` 结尾、 或是`gist` url 或是github源代码页面 url，则该 url 将被视为源代码。创建窗口时可以传递源代码这就使得如下场景成为可能，例如将窗口插件的源代码存储在 Python 插件中，并在需要时实例化它。
     - tag：String。如果插件支持多个`tags`，则与`src` 一起使用来指定插件的标签。
     - namespace：String。与 `src` 一起使用来指定插件的命名空间。
     - passive: Boolean，仅在指定了 `src` 时使用。标记插件是否为被动网页（没有暴露ImJoy api）。默认值为 `false`。
     - w：Integer。网格的窗口列宽（1 列 = 30 像素）。
     - h：Integer。网格的窗口行高（1 行 = 30 像素）。
     - config：Object对象（JavaScript）或字典（Python）。
     - data：Object对象 (JavaScript) 或字典 (Python)。包含要传输到窗口的数据。

返回值：
* win：Object对象。创建的窗口的API对象。可以存储在插件类中（Python的`self`，或JavaScript的`this`）供以后使用，例如更新窗口。注意：如果`type="external"`和`passive=true`，则窗口不会暴露api，此时`win`为空。

例子：
用 JavaScript 创建一个简单的窗口：
```javascript
win = await api.createWindow({name: 'new window', type: 'Image Window', w:7, h:7, data: {image: ...}, config: {}})
```

用python实现的话，Python 3可以使用 `async/await` 风格：
```python
win = await api.createWindow(name='new window', type='Image Window', w=7, h=7, data={image: ...}, config={})
```
Python 2可以使用`callback` 风格：
```python
def window_callback(win):
    self.win = win
    print('window created')

api.createWindow({name: 'new window', type: 'Image Window', w:7, h:7, data: {image: ...}, config: {}}).then(window_callback)
```
然后使用返回的对象更新窗口，或者使用`win.on('close', callback)`来设置关闭窗口时的回调函数。类似地，`win.on('resize', callback)` 可用于设置窗口大小改变时的回调函数。
要关闭创建的窗口，则调用 `win.close()`。
要滚动ImJoy工作区中的窗口，调用“win.focus()”。
在`window`插件中，可以使用`this.close`、`this.on`、`this.emit`、`this.focus`、`this.resize`。
```javascript
// win is the object returned from api.createWindow
await win.run({'data': {'image': ...}})

// set `on-close` callback
win.on('close', ()=>{
  console.log('closing window.')
})
```
创建一个包含两个图像和一个比较滑块的窗口。
```python
  api.createWindow({
    name: 'test compare',
    type: 'imjoy/image-compare',
    data: {
      first: '//placehold.it/350x150/0288D1/FFFFFF',
      second: '//placehold.it/350x150/E8117F/FFFFFF'
    }
  })
```
调用`win`对象：
```python
# win 是 api.createWindow 返回的对象
# 作为字典传递
await win.run({'data': {'image': ...}})

# 或命名参数
await win.run(data={'image': ...})

# 设置`on-close`回调
def close_callback():
    print('closing window.')

win.on('close', close_callback)
```


## api.error
```javascript
api.error(message)
```
记录当前插件的错误消息，该消息存储在其日志历史记录中。
插件名称旁边的红色图标表明有错误存在。点击此图标将打开一个显示日志历史记录的窗口。
与 `console.error` 或 `print` 类似，`api.error` 可以接受多个参数，这些参数将通过空格连接。
参数：
* message：String。要记录的错误消息。

例子：
```javascript
api.error('Error occurred during processing.')
```

## api.echo
```javascript
api.echo(obj)
```
返回与传入对象相同值的函数，用于测试目的。
这对于测试编码/解码对象很有用。

参数：
* obj：任意值。

例子：
```javascript
ret = await api.echo('hi')
console.log(ret) // should get 'hi'
```

## api.registerCodec
```javascript
api.registerCodec(config)
```
注册用于发送和接收远程对象的自定义编解码器。
注意：`web-python` 插件尚不支持该api。
参数：
* config：Object对象（JavaScript）或字典（Python）。编解码器的选项。它包含以下字段：
  - name：String。编解码器名称
  - type：Class。用于匹配对象进行编码的类对象。在Javascript中，`instanceof` 将用于匹配类型。在Python中将使用 `isinstance()`，这也意味着在 Python 中，`type` 可以是类的元组。
  - encoder：Function。`encoder`函数将一个对象作为输入，需要返回要表示的对象/字典。注意，只能在要表示的对象中使用原始类型加上数组/列表和对象/字典。默认情况下，如果返回的对象不包含`_rtype`键，则编解码器`name`将用作 `_rtype`。还可以指定不同的`_rtype`名称，以允许不同类型之间的转换。
  - decoder：Function。 `decoder` 函数将编码对象转换为实际对象。仅当对象的`_rtype`与编解码器的`name`匹配时才会调用它。

例子：
```javascript
class Cat{
  constructor(name, color, age, clean){
    this.name = name
    this.color = color
    this.age = age
    this.clean = clean
  }
}

api.registerCodec({
    'name': 'cat', 
    'type': Cat, 
    'encoder': (obj)=>{
        // convert the Cat instance as a dictionary with all the properties
        return {name: obj.name, color: obj.color, age: obj.age, clean: obj.clean}
    },
    'decoder': (encoded_obj)=>{
        // recover the Cat instance
        return new Cat(encoded_obj.name, encoded_obj.color, encoded_obj.age, encoded_obj.clean)
    }
})

class Plugin {
    async setup(){
    }
    async run(){
        const dirtyCat = new Cat('boboshu', 'mixed', 0.67, false)
        // assuming we have a shower plugin
        const showerPlugin = await api.getPlugin({'name': 'catShower'})
        // now pass a cat into the shower plugin, and we should get a clean cat, the name should be the same
        // note that the other plugin is running in another sandboxed iframe or in Python
        // because we have the cat codec registered, we can send the Cat object to the other plugin
        // Also notice that the other plugin should also define custom encoding decoding following the same representation
        const cleanCat = await showerPlugin.wash(dirtyCat)
        if(cleanCat.clean) api.alert(cleanCat.name + ' is clean.')
    }
};
api.export(new Plugin())
```

## api.disposeObject
```javascript
api.disposeObject(obj)
```
从其对象存储中删除远程对象，以便垃圾收集器可以回收它。
当不再需要远程对象时调用此函数很重要，否则，它将导致内存泄漏，因为对象将保留在其对象存储中。
参数：
* obj：对象。要删除的远程对象。

## api.export
将插件定义的函数导出为`Plugin API`。每个imjoy插件都应该导出其插件api函数，除非 `<config>` 下的 `passive` 键设置为 `true`。
`Plugin API` 可以导出为插件类或包含所有api功能的对象/字典：
（1）JavaScript类
```javascript
class ImJoyPlugin(){
  async setup(){
  }
  async run(ctx){
  }
}

api.export(new ImJoyPlugin())
```
（2）JavaScript函数
```javascript
function setup(){

}

function run(){

}

api.export({setup: setup, run: run})
```
（3）Python类
```python
class ImJoyPlugin():
    def setup(self):
        pass

    def run(self, ctx):
        pass

api.export(ImJoyPlugin())
```

（4）Python函数
```python
def setup():
    pass

def run(ctx):
    pass

api.export({'setup': setup, 'run': run})
```

该API调用对于每个ImJoy插件都是强制性的（通常作为插件脚本的最后一行）。
`ImJoyPlugin` 实例的每个成员都将导出为 `Plugin API`，这意味着这些导出的函数可以被ImJoy主程序或其他插件调用，其他插件可使用`api.run` 或`api.call` 来调用插件的功能。
注意：只能导出具有原始类型的函数和变量（数字、字符串、布尔值）。如果变量或函数的名称以`_`开头，则表示它是内部变量或函数，不会被导出。
还有需要注意，在JavaScript中，`new` 关键字是创建一个类的实例，而在Python中没有`new`关键字。

## api.init
```javascript
api.init(config)
```
使用最小的插件接口和配置进行初始化。当不想导出任何插件api时，这可以用作 `api.export` 的快捷方式。
以Python为例：
```python
api.init(config)
```
等价于：
```
def setup():
    pass

api.export({"setup": setup}, config)
```

参数：
* config：Object，可选。插件的配置，包括所有配置字段，比如`name`、`type`等（完整列表可以在[这里](https://imjoy.io/docs/#/development?id=ltconfiggt-block)） .

## api.exportFile
```javascript
api.exportFile(file, name)
```
触发从浏览器下载文件。
参数：
* file：文件、Blob 或字符串。要下载的文件对象。如果传递了一个字符串，它将被包装为一个文本文件。
返回值：
* name：String。文件名。

例子：
```javascript
var blob = new Blob(["Hello, world!"], {type: "text/plain;charset=utf-8"});
api.exportFile(blob, 'hello.txt')
```

## api.getAttachment
```javascript
content = await api.getAttachment(att_name)
```
获得存储在插件文件的`<attachment>`块中的数据。
可以在`<attachment>`块中存储任何文本数据，例如base64编码的图像、代码和 json。
参数：
* att_name：String。附件的标识符。
返回值：
* content：存储在`<attachment>`块中的文本内容。

例子：
```html
<attachment name="att_name"></attachment>
```
```javascript
content = await api.getAttachment(att_name)
```

## api.getConfig
```javascript
config_value = await api.getConfig(config_name)
```
获得插件的配置。
注1：使用 `api.setConfig` 保存时，数字会转换为字符串。在使用它们之前必须将它们转换回数字（在JavaScript中使用 `parseInt()` 或 `parseFloat()`，在Python中使用 `int()` 或 `float()`）。
注2：也可以通过在字段名后加上`_`来访问`<config>`块中定义的字段，例如，如果想读取`<config>` 块，可以使用 `plugin_name = await api.getConfig('_name')`。
参数：
* param_name：String。参数名称。
返回值：
* param：String。返回的参数值。注意，数字也将作为字符串返回。

例子：
```javascript
sigma = await api.getConfig('sigma')
```

## api.installPlugin
```javascript
plugin = await api.installPlugin(config)
```
通过传递插件URI或源代码来安装插件，插件源代码将保存到浏览器数据库（在当前工作区中）。
参数：
* config：Object。配置对象。目前，可以传递以下配置：
  - `src`：String。要安装的插件的源代码、URI。
  - `tag`：String，可选。如果插件有多个标签，则选择插件标签。
  - `namespace`：String，可选。插件的命名空间。

例子：
```javascript
await api.installPlugin({uri: "https://raw.githubusercontent.com/imjoy-team/imjoy-core/master/src/plugins/webWorkerTemplate.imjoy.html"})
```

## api.uninstallPlugin
```javascript
plugin = await api.uninstallPlugin(config)
```
卸载已安装的插件。
参数：
* config：Object。配置对象。目前，可以传递以下配置：
  - `name`：String。要卸载的插件的名称。
  - `namespace`：String，插件的命名空间（注意，目前尚不支持该参数，但后面会支持）。

例子：
```javascript
await api.uninstallPlugin({name: "MyAwesomePlugin"})
```

## api.getPlugin
```javascript
plugin = await api.getPlugin(config)
```
通过id或名称获取已加载插件的API对象。插件必须已经加载到工作区中。
注1：如果插件被终止并且尝试调用其函数，则将收到错误消息。对此的一种解决方案是使用`try ... catch`(JavaScript) 或`try: ... except: ...`(Python)语句来捕获错误。
注2：关于`api.getPlugin`和`api.call`，如果想不断访问另一个插件的不同功能，最好使用`api.getPlugin`来获取该插件的所有API，然后可以通过返回的对象访问它们。如果只偶尔访问另一个插件中的API函数，则也可以使用`api.call`。

参数：
* config：String或Object。如果是一个字符串String，那么其应该是插件的名称，否则，可以传递一个包含键`id`或`name`的对象。
目前，可以传递以下配置：
  - name：String。插件名称。
  - id：String。插件的id。

返回值：
* plugin：对象。可用于访问插件 API 函数的对象。

例子：
获取插件`PluginX`的API，并访问其功能：
```javascript
pluginX = await api.getPlugin("PluginX")
result = await pluginX.run()

// Assuming that PluginX defined an API function `funcX`, you can access it with:
await pluginX.funcX()
```

## api.loadPlugin
```javascript
plugin = await api.getPlugin(config)
```
从源代码或URL加载插件，然后返回插件API对象。
参数：
* config：字符串或对象。获取插件的配置。如果它是一个字符串，那么根据字符串的内容，它将被转换为一个配置对象。转换按照以下规则进行： 1) 如果字符串包含多行，是URL或插件URI，则将其视为插件源（请参阅下面的 `src` 键）； 2) 否则将被视为插件名称并返回与`api.getPlugin` 相同的结果。目前，可以传递以下配置：
  - src：字符串。插件的URL或源代码，在这种情况下，它将即时实例化。通过传递源代码，它可以灵活地将一个或多个插件源代码嵌入到另一个插件中。例如，一个Python插件可以动态填充一个HTML格式的`window`插件。
  - tag：字符串，可选。如果插件支持多个`tags`，则指定插件的标签，仅在`src`为插件源代码时使用。
  - namespace：字符串，可选。指定插件的命名空间，仅在`src`为插件源代码时使用。
  - engine_mode：字符串，可选。 仅适用于通过插件引擎运行的插件。选择默认引擎模式，它可以是 `auto` 或引擎 URL（例如：`https://mybinder.org`）。

返回值：
* plugin：Object对象。可用于访问插件API函数的对象。

例子：
```javascript
const pokemonChooser = await api.loadPlugin({src: "https://gist.github.com/oeway/3c2e1ee72c79a6aafd9d6e3b473f0bbf"})
const result = await pokemonChooser.choosePokemon()
```

## api.getServices
```javascript
services = await api.getServices(config)
```
通过指定插件服务的`name`、`type`、`id` 等，获取插件服务列表（使用`api.registerService()` 注册）。
参数：
* config：Object对象。它是一个查询对象，由几个字段（至少一个）组成：
  - `id`：字符串。服务的ID（如果匹配，将返回一个包含一个元素的列表）。
  - `name`：字符串。服务的名称。
  - `type`：字符串。服务类型。
  - 在服务api中定义的任何其他键。

返回值：
* services：Object对象。服务api或对象的列表。

例子：
获取所有使用 `type="@model"` 注册的插件服务：
```javascript
const models = await api.getServices({type: "@model"})

console.log(models)
```

## api.getWindow
```javascript
w = await api.getWindow(config)
```
通过窗口的`id`、`window_id`、`name`或`type`获取现有窗口。
参数：
* config：字符串或对象。它可以是一个窗口名称字符串，也可以是一个由几个字段（至少一个）组成的对象：
  - `name`：字符串。窗口名称。
  - `type`：字符串。窗口的类型。
  - `window_id`：字符串。窗口的id。
  - `plugin_id`：字符串。附加到窗口的插件实例的id。

返回值
* w：对象。可用于访问窗口API函数的窗口对象。

例子：
获取现有的[Kaibu](https://kaibu.org) 窗口并访问它。
```javascript
await createWindow({name: "My Kaibu Window", src: "https://kaibu.org"})

w = await api.getWindow("My Kaibu Window")
await w.open("https://imjoy.io/static/img/imjoy-icon.png")
```

## api.getEngine
```javascript
engine = await api.getEngine(engine_url)
```
获取插件引擎的API对象。
参数：
* engine_url：字符串。插件引擎的URL。

返回值：
* engine：对象。可用于访问引擎API函数的引擎对象。

例子：
获取引擎的API（url = `https://127.0.0.1:2957`），并访问其功能：
```javascript
engine = await api.getEngine("https://127.0.0.1:2957")
await engine.disconnect()
```

## api.getEngineFactory
```javascript
engine_factory = await api.getEngineFactory(engine_factory_name)
```
获取插件引擎工厂的API对象。
参数：
* engine_factory_name：字符串。插件引擎工厂的名称。
返回值：
* engine_factory：对象。一个插件引擎工厂对象，可用于访问引擎 API 函数。

例子：
获取插件引擎工厂的API（name = `ImJoy-Engine`），并访问其功能：
```javascript
engine_factory = await api.getEngineFactory("ImJoy-Engine")
await engine_factory.addEngine(config)
```

## api.getFileManager
```javascript
file_manager = await api.getFileManager(file_manager_url)
```
获取文件管理器的API对象。
注意：自从`api_version > 0.1.6`，`api.getFileUrl` 和`api.requestUploadUrl` 已弃用，替换方案是先使用`api.getFileManager` 获取文件管理器，然后从返回的文件管理器对象中访问`getFileUrl` 和` requestUploadUrl`。
参数：
* file_manager_url：字符串。文件管理器的 URL。

返回值：
* file_manager：对象。可用于访问文件管理器 API 函数的文件管理器对象。

例子：
获取文件管理器的API（url = `https://127.0.0.1:2957`），并访问其功能：
```javascript
file_manager = await api.getFileManager("https://127.0.0.1:2957")
await file_manager.listFiles()
```
获取下载文件地址：
```javascript
file_manager = await api.getFileManager("https://127.0.0.1:2957")
await file_manager.getFileUrl({'path': './data/output.png'})
```
如果要获取当前文件的文件URL：
```
from imjoy import api

class ImJoyPlugin():
    def setup(self):
        api.log('initialized')

    async def run(self, ctx):
        file_manager = await api.getFileManager(api.config.file_manager)
        url = await file_manager.getFileUrl({"path": './screenshot-imjoy-notebook.png'})
        await api.alert(url)

api.export(ImJoyPlugin())
```

用于上传的文件URL请求：
```javascript
file_manager = await api.getFileManager("https://127.0.0.1:2957")
await file_manager.requestUploadUrl({'path': './data/input.png'})
```

## api.log
```javascript
api.log(message)
```
记录当前插件的状态消息，该消息存储在其日志历史记录中。
插件名旁边的灰色图标表明该状态的存在，点击此图标将打开一个窗口，显示历史记录的消息。
状态消息可以是字符串或图像。后者可用于创建自动日志，例如，记录神经网络的训练。
与 `console.log` 或 `print` 类似，`api.log` 可以接受多个参数，这些参数将通过空格连接。

参数：
* message：字符串。要记录的消息

例子:
创建一个简单的文本消息：
```javascript
api.log('Processing data ...')
```
记录一个图像文件。
```javascript
api.log({type: 'image', value: 'https://imjoy.io/static/img/imjoy-icon.png' })
```

## api.progress
```javascript
api.progress(progress)
```
更新当前插件的进度条。
此进度条将显示在插件菜单本身中。使用`api.showProgress` 为ImJoy状态栏生成进度条。
参数：
* progress：浮点数或整数。进度百分比。整数的允许范围为 0 到 100，浮点数的范围为 0 到 1。

例子：
```javascript
api.progress(85)
```

## api.registerService
```javascript
const service_id = await api.registerService(config)
```
注册一个插件服务。
参数：
* config：对象（JavaScript）或字典（Python）。它必须至少包含一个 `type` 键和一个 `name` 键。其他键取决于相应的类型定义（见下文）。

返回值：
* service_id：字符串。该服务的id可用于获取或取消注册服务。

### 内置插件服务
operator service (type=`operator`) 是一种内置服务，用于扩展插件菜单以执行特定任务。
对于服务的`config`，允许几个字段：
  - `name`：字符串。操作`op`名称。
  - `ui`：对象（JavaScript）或字典（Python）。渲染界面。与 `<config>` 中的 `ui` 字段具有相同的定义规则。
  - `run`：函数，可选。指定当`op`运行时所执行的 `Plugin API` 函数。注意，它必须是使用 `api.export` 导出的插件类的成员或函数。如果未指定，将执行插件的 `run` 函数。
  - `update`：字符串，可选。指定当`ui`字段中的任何选项发生更改时将运行的`Plugin API`函数。
  - `inputs`：对象，可选。 定义此`op`的输入，格式为[JSON Schema](https://json-schema.org/)。
  - `outputs`：对象，可选。定义此`op`的输出，格式为[JSON Schema](https://json-schema.org/)。

（另外参阅上面的[输入参数]一节了解如何设置参数。）
`op`可以有自己的 GUI，由 `ui` 字符串定义。默认情况下，插件的所有操作都会调用插件的`run`函数。可以在`run`函数中使用 `ctx.config.type` 来区分调用了哪个操作。
如果想动态改变界面，可以运行`api.registerService`多次覆盖以前的版本。 `api.registerService` 也可以用于覆盖 `<config>` 中定义的插件的默认`ui`字符串，只需将插件名称设置为`op`名称（或不设置名称）。

例子：
注册一个新的插件操作符：
```javascript
// JavaScript
await api.registerService({
     "type": "operator",
     "name": "LUT",
     "ui": [{
        "apply LUT": {
            id: 'lut',
            type: 'choose',
            options: ['hot', 'rainbow'],
            placeholder: 'hot'
          }
      }],
      "run": this.apply_lut,
      "update": this.update_lut
});

```
### 开发者贡献的插件服务
可以使用以下服务类型的插件服务：
 * type=`@transformation`：[scikit-learn兼容数据集转换](https://scikit-learn.org/stable/data_transforms.html)
   - `transform(data)`: 将此转换模型应用于未见过的数据
   - `fit(data)`：从数据中学习模型参数
   - `fit_transform(data)`：同时建模和转换训练数据
 * type=`@model`: keras 兼容模型服务
   - `predict(data)`：对数据进行预测（见[这里](https://keras.io/api/models/model_training_apis/#predict-method)）
   - `fit(data)`：在数据上训练模型（参见 [此处](https://keras.io/api/models/model_training_apis/#fit-method)）
 * 还可以贡献自己的服务类型

### 定义一个新的插件服务类型
如果内置和开发者贡献的服务类型都不能满足要求，还可以定义一个新的插件服务类型。最简单的方法是从定义自定义类型开始，例如：`api.registerService({"type": "my-awesome-service", ...})`。可以使用它来开发和测试服务。一旦服务类型定义稳定可用了，可以在服务类型名称中添加一个 `@` 并将该类型提交到 imjoy-core 存储库。
以下是提交类型定义的步骤：
1. fork[imjoy-core repo](https://github.com/imjoy-team/imjoy-core)
2. 编辑[本页](https://github.com/imjoy-team/imjoy-core/blob/master/docs/api.md) 将新类型添加到上面的列表中，并提供详细说明
3. 在 [serviceSpec.js](https://github.com/imjoy-team/imjoy-core/blob/master/src/serviceSpec.js) 文件中定义模板。
4. 向[imjoy-core repo](https://github.com/imjoy-team/imjoy-core)提交PR。

## api.run
```javascript
await api.run(plugin_name)
```
通过指定其名称运行另一个插件。
也可以传递`ctx`到这个插件中以传输数据。
参数：
* plugin_name：字符串。插件名称。

例子：
调用一个插件：
```python
await api.run("Python Demo Plugin")
```

下面是两个插件并发执行的例子，其中两个插件是同时执行，但ImJoy一个接一个地等待结果。
```python
# Python
p1 = api.run("name of plugin 1")
p2 = api.run("name of plugin 2")

result1 = await p1
result2 = await p2
```
这也可以通过Python中的 `asyncio.gather` 来实现：
```python
p1 = api.run("name of plugin 1")
p2 = api.run("name of plugin 2")
result1, result2 = await asyncio.gather(p1, p2)
```
如果用JavaScript写上面的例子，可以使用
```javascript
const p1 = api.run("name of plugin 1")
const p2 = api.run("name of plugin 2")
const [result1, result2] = [await p1, await p2]
```
而两个插件顺序执行的例子如下：
```python
result1 = await api.run("name of plugin 1")
result2 = await api.run("name of plugin 2")
```

## api.setConfig
```javascript
api.setConfig(config_name, config_value)
```
将插件数据存储在其配置中。
当ImJoy重新启动时，可以获得这些值。此功能非常适合存储和重新加载设置。但是，该函数旨在存储少量数据，而不是大型对象。当前的实现是使用`localStorage` 进行存储。大多数浏览器只允许ImJoy自身及所有插件一共使用5MB数据存储。
要删除一个参数，将其值设置为 `null` (JavaScript) 或 `None` (Python)。
参数：
* config_name：字符串。参数名称。不要使用以`_`开头、且后跟`<config>` 块的任何字段名称这样的名字。
* config_value：数字或字符串。既不是对象/数组（JS）也不是字典/列表（Python）。请注意，数字被存储为字符串。

例子：
```javascript
api.setConfig('sigma', 928)
```

## api.showDialog
```javascript
answer = await api.showDialog(config)
```
将窗口或自定义 GUI 显示为对话框。
类似于`api.createWindow`，可以传递一个对象`{"type": "WINDOW_PLUGIN_NAME", "name": "new dialog", "config": {...}, "data": {... }}`。这会将窗口插件实例显示为对话框。该对话框可以通过`win.close()`以编程方式关闭，也可以由用户使用关闭按钮关闭。
对于带有joy ui的简单对话框，可以传递`{"type": "joy", "name": "new dialog", "config": {...}, "data": {...}} `。对话框的回答将存储在返回的对象中，可以使用指定的`id`获得。当考虑用户按下`cancel`的情况，可以使用`try catch`（JavaScript）或`try except`（Python）语法。
参数：
* config：对象 (JavaScript) 或字典 (Python)。定义对话框。包含以下字段：
    - `name`：字符串。对话框的标题。
    - `type`：字符串。对话框的类型（使用`window`插件名称或`joy`）。如果`type="joy"`，则需要提供`ui`，它的定义与`<config>` 中的`ui` 字段相同。否则，需要为 `api.createWindow`提供 `config` 和 `data`。

返回值：
* answer。对象 (JavaScript) 或字典 (Python)。通过字段`answer[id]`包含对话框的答案。

例子：
```javascript
result = await api.showDialog({
   "name": "This is a dialog",
   "ui": "Hey, please select a value for sigma: {id:'sigma', type:'choose', options:['1', '3'], placeholder: '1'}.",
})
```

## api.showFileDialog
```javascript
ret = await api.showFileDialog(config)
```
显示一个文件对话框来选择文件或目录。
该函数将返回一个`promise`，可以从中获取文件路径的字符串。
根据插件引擎的实现，ImJoy将尝试通过`api.config.file_manager` 选择插件引擎指定的文件管理器。
ImJoy主程序和插件引擎的文件处理是不同的。
注意，JavaScript插件的文件路径作为url返回，而对于Python插件，它将是绝对文件路径。对于JavaScript插件，需要url格式来打开文件 。可以使用`uri_type` 选项（见下文）来改变此行为。例如，对于JavaScript插件也可以获取绝对路径。但是，不能使用此路径打开文件JavaScript，但可以将其传递给另一个Python插件进行处理。

参数：
* config：对象 (JavaScript) 或字典 (Python)。显示文件对话框的选项。它包含以下字段：
- type：字符串。支持的文件对话框模式：
    - `file`（默认）：选择一个或多个文件；
    - `directory`：选择一个或多个目录。对于Python插件，如果不指定类型，文件或目录都可以选择。
- title：字符串。对话框的标题。
- root：字符串。显示对话框的初始路径。注意：对于Windows上的Python插件，可能希望使用 `r"xxxxxx"` 语法将路径字符串定义为原始字符串，因为有可能遇到无法识别普通字符串的路径问题。
- mode：字符串。文件选择模式。默认情况下，用户可以选择单个或多个文件（按下 `shift` 键）
    - `single`：只能选择单个文件或目录。
    - `multiple`：选择多个文件或目录，并以数组或列表的形式返回。
    - `single|multiple`（默认）：允许单选和多选。
- file_manager：字符串。通过url指定文件管理器，例如在`native-python`插件中，可以通过`api.config.file_manager`获取文件管理器的URL。

返回值：
* selected：对象数组 (JavaScript) 或字典 (Python)。它可以包含 0 到多个选定的文件/目录。如果返回的数组为空，则表示用户没有选择任何文件/目录。数组中的文件项通常包含（取决于不同的文件管理器实现）：
  - path：字符串。文件路径。
  - url：字符串。文件的URL。
  - 其他字段。

例子：
以下示例将显示指定的文件名或用户取消或插件引擎未运行的消息。
```javascript
const selected = await api.showFileDialog()
if(selected.length>0){
  await api.alert("Selected file " + selected[0].url)
}
else{
  await api.alert("User cancelled file selection.")
}
```

## api.showMessage
```javascript
api.showMessage(message,duration)
```
更新ImJoy状态栏上的状态文本并显示一个带有相应消息的快速弹出栏。
如果未指定持续时间，则快速弹出栏将显示 10 秒。
参数：
* message：字符串。要显示的消息。
* duration（可选）：整数。显示消息的持续时间（以秒为单位）。

例子：
```javascript
api.showMessage('Processing...', 5)
```

## api.showProgress
```javascript
api.showProgress(progress)
```
更新Imjoy GUI 的进度条。
参数：
* progress：浮点数或整数。进度百分比。整数的允许范围为 0 到 100，浮点数的范围为 0 到 1。

例子：
```javascript
api.showProgress(85)
```

## api.showSnackbar
```javascript
api.showSnackbar(message, duration)
```
显示一个带有消息的快速弹出栏，并在特定时间段内消失。

参数：
* message：字符串。要显示的消息。
* duration：整数。将显示以秒为单位的持续时间消息。

例子:
```javascript
api.showSnackbar('processing...', 5)
```

## api.showStatus
```javascript
api.showStatus(status)
```
更新Imjoy GUI上的状态文本。
参数:
* status：字符串。要显示的消息。

例子：
```javascript
await api.showStatus('processing...')
```

## api.TAG
这是一个常量，是用户在安装过程中选择的当前标签。

## api.unregisterService
```javascript
await api.unregisterService(config)
```
取消注册插件服务。
参数：
* config：对象。它必须包含插件服务的`id`。

例子：
```javascript
const sid = await api.registerService({type: 'my-service', my_data: 123})

await api.unregisterService({id: sid})
```

## api.utils.*
```javascript
await api.utils.UTILITY_NAME()
```
调用效用函数。
目前所有插件都支持的功能是：
 * `api.utils.$forceUpdate()`：手动刷新 GUI。
 * `api.utils.openUrl(url)`：在新的浏览器选项卡中打开一个 `url`。
 * `api.utils.sleep(duration)`：以秒为单位休眠指定的`duration`。注意对于Python插件，请用 `time.sleep`。
 * `api.utils.showOpenFilePicker`：仅适用于 Chrome 86+，弹出一个对话框，用于使用 [文件系统访问](https://web.dev/file-system-access/) API打开文件
 * `api.utils.showSaveFilePicker`：仅适用于 Chrome 86+，弹出一个对话框用于使用 [文件系统访问](https://web.dev/file-system-access/) API保存文件
 * `api.utils.showDirectoryPicker`: 仅适用于 Chrome 86+，弹出一个对话框用于选择具有[文件系统访问](https://web.dev/file-system-access/) API的目录

## api.config
配置信息包括：
 * `workspace`：当前工作区。
 * `engine`：当前插件引擎的 URL，仅适用于原生 python 插件。
 * `file_manager`：当前插件引擎注册的文件管理器的 URL，仅适用于原生 python 插件。

## api.WORKSPACE
**已弃用！** 使用 `api.config.workspace` 代替
当前工作区的名称。
## api.ENGINE_URL
**已弃用！** 使用 `api.config.engine` 代替
**仅适用于原生 python 插件**
当前插件引擎的 URL。

## api.FILE_MANAGER_URL
**已弃用！** 使用 `api.config.file_manager` 代替
**仅适用于原生 python 插件**
当前插件引擎注册的文件管理器的 URL。

# 内部插件
除了默认的 ImJoy api，还提供了一组内部支持的插件，这些插件可以直接使用。只有当另一个插件通过 `api.getPlugin(...)` 请求插件时，才会加载这些插件。
以下是这些内部插件及其 api 功能的列表。

## BrowserFS
要使用 `BrowserFS` 插件，需要先调用：
在Javascript中`const bfs = await api.getPlugin('BrowserFS')` ，或在Python中`bfs = await api.getPlugin('BrowserFS')`。
然后，可以使用 [Node JS 文件系统 API](https://nodejs.org/api/fs.html) 访问浏览器内的文件系统（例如：`bfs.readFile('/tmp/temp.txt', 'utf-8')`)。更多底层实现详见[BrowserFS](https://github.com/jvilk/BrowserFS)，ImJoy默认文件系统支持以下节点：
* `/tmp`: `InMemory`，数据保存在浏览器内存中，ImJoy关闭时清除。
* `/home`：`IndexedDB`，数据存储在浏览器IndexedDB数据库中，可以作为持久化存储。

例子：
JavaScript写法：
```javascript
async function test_browser_fs(){
  const bfs_plugin = await api.getPlugin('BrowserFS')
  const bfs = bfs_plugin.fs

  bfs.writeFile('/tmp/temp.txt', 'hello world', function(err, data){
      if (err) {
          console.log(err);
          return
      }
      console.log("Successfully Written to File.");
      bfs.readFile('/tmp/temp.txt', 'utf8', function (err, data) {
          if (err) {
              console.log(err);
              return
          }
          console.log('Read from file', data)
      });
  });
}
```

Python写法：
```python
async def test_browser_fs():
  bfs_plugin = await api.getPlugin('BrowserFS')
  bfs = bfs_plugin.fs

  def read(err, data=None):
      if err:
          print(err)
          return

      def cb(err, data=None):
          if err:
              print(err)
              return
          api.log(data)
      bfs.readFile('/tmp/temp.txt', 'utf8', cb)

  bfs.writeFile('/tmp/temp.txt', 'hello world', read)

```

在 JavaScript 中逐块读取大文件：
```javascript
function generate_random_data(size){
    var chars = 'abcdefghijklmnopqrstuvwxyz'.split('');
    var len = chars.length;
    var random_data = [];

    while (size--) {
        random_data.push(chars[Math.random()*len | 0]);
    }

    return random_data.join('');
}

function fsRead(fd, buffer, offset, chunkSize, bytesRead) {
  return new Promise((resolve, reject) => {
    fs.read(fd, buffer, offset, chunkSize, bytesRead, (err, bytesRead,
      read_buffer) => {
      if (err) {
        console.log('err : ' + err);
        reject(err)
        return
      }
      const bytes = read_buffer.slice(0, bytesRead)
      resolve(bytes)
    });
  })
}

bfs.writeFile('/tmp/test.txt', generate_random_data(100000), function(err){
if (err){
    console.error(err);
}
bfs.open('/tmp/test.txt', 'r', function(err, fd) {
    bfs.fstat(fd, async function(err, stats) {
      if(err){
          console.error(err)
          return
      }
      var bufferSize = stats.size,
          chunkSize = 512,
          buffer = new Uint8Array(new ArrayBuffer(chunkSize)),
          bytesRead = 0;
      try{
        while (bytesRead < bufferSize) {
            if ((bytesRead + chunkSize) > bufferSize) {
                chunkSize = (bufferSize - bytesRead);
            }
            const bytes = await fsRead(fd, buffer, 0, chunkSize, bytesRead)
            console.log(bytes)
            bytesRead += chunkSize;
        }
        console.log("Finished reading.")
      }
      catch(e){
        console.error(e)
      }
      finally{
        bfs.close(fd);
      }

    });
  });
})
```
