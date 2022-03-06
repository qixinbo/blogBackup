---
title: 开源深度学习计算平台ImJoy解析：9 -- 集成
tags: [ImJoy]
categories: computer vision 
date: 2022-2-12
---

# 简介
ImJoy设计的初衷就包括可以被集成使用。这意味着它可以根据需求以多种方式集成到一个项目中。
这里的“集成”有两个意思：
（1）将ImJoy的核心core库集成在自己的网站或web应用中，从而可以调用ImJoy生态中的各种插件；
（2）将ImJoy的运行时rpc库集成到自己的web应用中，从而将自己的web应用转换为ImJoy的一个插件，即被别人可调用和访问。

# 安装
集成ImJoy（无论是imjoy-core还是imjoy-rpc）主要有三种方式：
（1）在页面上以 CDN 包的形式导入。
（2）下载 JavaScript 文件并自行托管。
（3）使用 npm 安装它。
下面详细说明。
## CDN
可以直接使用托管在CDN上的JS库（对于imjoy-core库和imjoy-rpc库都是引用这个js文件）：
```js
<script src="https://lib.imjoy.io/imjoy-loader.js"></script>
```
可以看出，imjoy的核心库都在[lib.imjoy.io](https://lib.imjoy.io)上进行了托管，可以直接使用，这里放的是最新的版本。
另外，在[jsdelivr](https://www.jsdelivr.com/?query=imjoy)上也有托管。

## 下载并自托管
如果你想避免使用构建工具，但又无法在生产环境使用CDN，那么可以下载相关js文件并自行托管在自己的服务器上。然后可以通过script标签引入，与使用 CDN 的方法类似。
可以在[这个GitHub仓库](https://github.com/imjoy-team/lib.imjoy.io)里直接下载那些js文件。

### 自己打包
这些打包好的链接库的源码在[imjoy-core这个仓库](https://github.com/imjoy-team/imjoy-core)里，如果想对它的源码进行修改，可以克隆下来这个仓库，然后再自己打包，即：
```js
git clone https://github.com/imjoy-team/imjoy-core.git
cd imjoy-core
npm run install

# test
npm run test

# build
npm run build
```
上面的命令对于linux和mac系统是适用的，如果是windows系统，可以直接：
```js
npx webpack
```
进行打包。

## npm
在用 imjoy 构建大型应用时推荐使用 npm 安装。npm能很好地和webpack等打包器配合使用。
对于imjoy-core库：
```js
npm install imjoy-core
```
对于imjoy-rpc库：
```js
npm install imjoy-rpc
```

# 集成Imjoy Core
## 直接引用
### 在线引用
新建一个core-example.html的文件（这个文件在[这里](https://github.com/imjoy-team/imjoy-core/blob/master/src/core-example.html)），写入以下内容：
```js
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>ImJoy Core Example</title>
  </head>

  <body>
    <h1>ImJoy Core Example</h1>
    <script src="https://lib.imjoy.io/imjoy-loader.js"></script>
    <script>
      loadImJoyCore().then(imjoyCore => {
        const imjoy = new imjoyCore.ImJoy({
          imjoy_api: {},
          //imjoy core config
        });
        imjoy.start({ workspace: null }).then(() => {
          alert("ImJoy Core started successfully!");
        });
      });
    </script>
  </body>
</html>
```
双击该页面会在浏览器中运行它，效果是呈现一个弹出框表明成功启动了ImJoy。

### 本地托管引用
上面例子是在线引用[https://lib.imjoy.io/imjoy-loader.js](https://lib.imjoy.io/imjoy-loader.js)这个文件。
对于本地托管的文件，可以将上述页面的对应部分改为如下：
```js
    <script src="/imjoy-loader.js"></script>
    <script>
      loadImJoyCore(
        {
          base_url: "/",
        }
```
即引用本地路径下的文件。
但这样改写后并不能直接运行该html文件，因为它现在是本地文件，需要将其放置在一个服务器上以http传输才行。
最简单的一个运行方式是在本地启用一个测试服务器，毕竟此处只是为了测试这个文件的运行。
如果使用python，可以使用如下命令：
```python
python -m http.server
```
这样就可以在这个本地web服务器上查看该目录下的内容。
具体原理见[如何设置一个本地测试服务器？](https://developer.mozilla.org/zh-CN/docs/Learn/Common_questions/set_up_a_local_testing_server)

python的这个测试服务器功能非常弱，到了具体部署应用时，可以使用nginx来构建web服务器从而顺利找到这些文件。
[Windows环境利用nginx搭建web服务器](https://blog.csdn.net/vfsdfdsf/article/details/89354541)

## npm包引用
如上所述，如果要开发大型程序，还是推荐使用npm包的形式进行。
```js
<script>
import * as imjoyCore from 'imjoy-core'
const imjoy = new imjoyCore.ImJoy({
    imjoy_api: {},
    //imjoy config
});

imjoy.start({workspace: 'default'}).then(async ()=>{
    await imjoy.api.alert("hello world");
  }
)
</script>
```
不过上面代码现在还没法直接运行。
因为ImJoy是基于浏览器的，它需要获取浏览器的window对象（即包含DOM文档的窗口），所以它得运行在前端，不能作为后端的包使用。
因此首先不能使用node.js来运行包含ImJoy的程序，与此同时，npm包这种编程引用方式还需要被浏览器认识，因此此时需要webpack这一类的打包工具来进行开发。
这里可以使用原生的webpack来打包开发，不过webpack的配置挺复杂的，这里推荐直接使用Vue.js来开发，这样既能方便地构建前端页面，也能借助Vue-cli脚手架来很方便的打包和部署（参考教程可以看下面的推荐链接）。

如果选择Vue.js进行开发，那么上面的代码可以直接放在某个Vue组件中，然后再通过：
```js
npm run serve
```
进行热部署即可运行查看。

# 集成Imjoy RPC
## 直接引用
### 在线引用
新建一个plugin-example.html的文件（这个demo文件在[这里](https://github.com/imjoy-team/imjoy-core/blob/master/src/plugin-example.html)），写入以下内容：
```html
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>ImJoy Plugin Example</title>
  </head>

  <body>
    <h1>ImJoy Plugin Example</h1>
    <script src="https://lib.imjoy.io/imjoy-loader.js"></script>
    <script>
      loadImJoyRPC().then(async imjoyRPC => {
        const api = await imjoyRPC.setupRPC({ name: "My Awesome App" });
        function setup() {
          api.alert("ImJoy RPC initialized. Hello!");
        }
        // define your api which can be called by other plugins in ImJoy
        function my_api_func() {}
        // Importantly, you need to call `api.export(...)` in order to expose the api for your web application
        api.export({ setup: setup, my_api_func: my_api_func });
      });
    </script>
  </body>
</html>
```
但该文件并不能像上面的core-example.html那样可以直接双击运行，会报如下错误：
```html
Uncaught (in promise) Error: imjoy-rpc should only run inside an iframe or a webworker.
```
这是因为注入了ImJoy RPC运行时的web应用必须在ImJoy中作为一个插件使用。`这样做的目的是隔离运行环境，从而让插件支持任意web框架。
`（来自ImJoy作者的交流指点，thanks）。
那么，为了让上面这个文件正常运行，需要进行如下操作：
（1）将它托管成为一个web app，得到它的使用链接。这里可以使用任意托管服务，比如上面的测试服务器、nginx服务器，以及Github pages等；
（2）在任意一个其他的ImJoy插件中，使用该web app。

为了简化演示，采用最简单的方法：
对于第一步，直接使用python的测试服务器；
对于第二步，修改上面的core-example.html文件为：
```html
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>ImJoy Core Example</title>
  </head>

  <body>
    <h1>ImJoy Core Example</h1>
    <script src="/imjoy-loader.js"></script>
    <script>
      loadImJoyCore().then(imjoyCore => {
        console.log(imjoyCore);
        const imjoy = new imjoyCore.ImJoy({
          imjoy_api: {},
          //imjoy core config
        });

        imjoy.start({ workspace: null }).then(async () => {
          alert("ImJoy Core started successfully!");
          const win = await imjoy.api.createWindow({
          type: 'My Window',
          src: '/plugin-example.html',
          data: { }
          })
        });

        imjoy.event_bus.on("add_window", w => {
          const container = document.createElement('div');
          container.id = w.window_id; // <--- this is important
          container.style.backgroundColor = '#ececec';
          container.style.height = "300px";
          container.style.width = "100%";
          // Here we simply add to the body
          // but in reality, you can embed it into your UI
          document.body.appendChild(container)
        });

      });
    </script>
  </body>
</html>
```
现在，运行效果如下：
![rpc-demo](https://user-images.githubusercontent.com/6218739/155644211-416c4dc6-2b49-4810-a1d5-c590a854e7b8.png)
即生成了一个窗口，里面运行了那个demo app。

### 本地托管引用
跟上面imjoy-core库的引用方式相同。

## npm包引用
引用方式也是类似于imjoy-core库：
```js
import { imjoyRPC } from 'imjoy-rpc';

imjoyRPC.setupRPC({name: 'My Awesome App'}).then((api)=>{
 // call api.export to expose your plugin api
})
```
然后也是在其他插件中使用该app才可以：
```js
// as a new window
const win = await api.createWindow({
    type: 'My Awesome App',
    src: 'https://my-awesome-app.com/',
    data: { }
})

// or, as a dialog
const win = await api.showDialog({
    type: 'My Awesome App',
    src: 'https://my-awesome-app.com/',
    data: { }
})

// further interaction can be performed via `win` object
```

# 直接引用js的API
上面两种直接引用js文件的方式，其提供了三个主要的函数供调用：
（1）loadImJoyCore：加载ImJoy核心库，使得可以调用ImJoy的各种API
（2）loadImJoyRPC：加载ImJoy的rpc库，使得插件可以通信，注意仅在iframe里才能使用这个函数
（3）loadImJoyBasicApp：一个简易但功能完善的ImJoy最小化app。

对于这三个加载函数，可以选择性地传入以下配置对象：
- version: 指定imjoy-core或imjoy-rpc的版本
- api_version: 仅适用于imjoy-rpc，限定RPC的api版本
- debug: 加载imjoy-core未压缩过的包含调试信息的版本，开发阶段使用
- base_url: 自定义加载这些库的路径（上面已演示过，可以设置这个参数来使用本地托管的库）

对于loadImJoyBasicApp，还有其他额外的选项：
- process_url_query: 布尔值，是否处理url请求
- show_window_title: 布尔值，是否显示窗口标题
- show_progress_bar: 布尔值，是否显示进度条
- show_empty_window: 布尔值，是否显示空白窗口
- hide_about_imjoy: 布尔值，是否隐藏“关于”菜单
- menu_style: Object, 菜单样式
- window_style: Object, 窗口样式
- main_container: String, 定义主容器的id
- menu_container: String, 定义菜单容器的id
- window_manager_container: String, 定义窗口管理器的id
- imjoy_api: Object, 重载一些ImJoy API函数的实现


这里是使用loadImJoyBasicApp编写的一个[轻量化的ImJoy app](https://imjoy.io/lite)，源代码在[这里](https://github.com/imjoy-team/ImJoy/blob/master/web/public/lite.html)。

## 自动切换core和plugin
对于一个既可以作为插件plugin，又可以使用imjoy core的web app，ImJoy提供了一个函数来检测当前app，从而在两种模式间自动切换：
```js
// check if it's inside an iframe
if(window.self !== window.top){
    loadImJoyRPC().then((imjoyRPC)=>{
        
    })
}
else {
    loadImJoyCore().then((imjoyCore)=>{

    })
}
```

# 参考资料
上面的技术需要用到的web知识：
（1）JavaScript语法：
[ES6 模块](https://www.runoob.com/w3cnote/es6-module.html)
[JavaScript 异步编程](https://www.runoob.com/js/js-async.html)
[Javascript异步编程的4种方法](https://www.ruanyifeng.com/blog/2012/12/asynchronous%EF%BC%BFjavascript.html)
[Promise](https://www.liaoxuefeng.com/wiki/1022910821149312/1023024413276544)
[使用Promise](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Guide/Using_promises)
[async和await:让异步编程更简单](https://developer.mozilla.org/zh-CN/docs/Learn/JavaScript/Asynchronous/Async_await)
[async函数](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Statements/async_function)
[await](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Operators/await)
（2）Webpack打包：
[Webpack 教程](https://www.jiangruitao.com/webpack/)
（3）Node.js开发：
[es6的import是不是也可以导入css和scss？](https://segmentfault.com/q/1010000022020967)
[vue main.js中引入node_modules中的文件为什么路径不用写node_modules?](https://www.zhihu.com/question/358026810)
[package.json 中 你还不清楚的 browser，module，main 字段优先级](https://github.com/SunshowerC/blog/issues/8)
（4）Vue.js开发：
视频教程推荐：
[尚硅谷Vue2.0+Vue3.0全套教程丨vuejs从入门到精通](https://www.bilibili.com/video/BV1Zy4y1K7SH?from=search&seid=396170412927101372)
文字版的教程推荐：
[Vue.js菜鸟教程](https://www.runoob.com/vue2/vue-tutorial.html)
[Vue.js官方教程](https://cn.vuejs.org/v2/guide/index.html)
