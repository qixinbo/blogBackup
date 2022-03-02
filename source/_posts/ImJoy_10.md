---
title: 开源深度学习计算平台ImJoy解析：10 -- ImJoy主站之入口文件
tags: [ImJoy]
categories: computer vision 
date: 2022-2-27
---

# 简介
前面讲了ImJoy的core库和rpc库作为函数库如何被集成调用，而基于这两个核心库有一个能开箱即用的完整网站，即[ImJoy主站](https://imjoy.io)，使用它就可以无需了解上面的技术细节，直接加载各种函数插件即可（但也因为这样，你无法更改网站样貌，如果没有定制开发的需求，则直接使用该网站即可）。

从该文章开始，将尝试对ImJoy主站进行详细分析，看官方是怎样应用ImJoy的core和rpc库的。

# 代码结构
ImJoy主站是用vue.js前端框架写的，整个目录结构由vue脚手架vue-cli 4生成，所以首先要分析清楚vue脚手架生成的代码结构是怎样的，可以参考[这篇教程](https://blog.csdn.net/weixin_43734095/article/details/106990775)
截取其中的代码分析图：
![vue-cli](https://user-images.githubusercontent.com/6218739/155942379-aeaeb37b-f46f-4145-b488-4ef8ea82c446.png)

理清楚代码结构后，从哪里开始分析呢？

# 程序入口
`src/main.js`是程序执行的入口文件，所以最好是顺着代码的执行过程一步步分析。
```js
import Vue from "vue";
```
引入Vue函数，后面将实例化它，作为整个程度的总调度，即：
```js
import App from "./App";

new Vue({
  el: "#imjoy-app",
  router,
  data: {
    store: store,
    router: router,
  },
  template: "<App/>",
  components: {
    App,
  },
});
```
可以看出，整个Vue实例绑定的是ID名为`imjoy-app`的html元素，而该元素是位于`public\index.html`文件中：
```html
<div id="imjoy-app"></div>
```
即：`src/main.js`是程序执行的入口文件，`public\index.html`是网页显示的入口文件，至于这两个文件作为入口是vue的脚手架在后台指定好的。知道这两个入口文件后，就可以往后按图索骥般研究。
从上述代码还可以看出，该Vue实例管理了App这个根组件，由该根组件再统一去管理其他所有组件，即App根组件显示到`imjoy-app`这个html元素上。
新版的Vue脚手架对于vue2的实例化这块是这样书写的：
```js
import Vue from 'vue'
import App from './App.vue'

new Vue({
  render: h => h(App),
}).$mount('#app')
```
同样很简单地指明了Vue实例与App组件、app元素之间的关系。

# 路由
在`src/main.js`里第二行就是路由的设置。
Vue.js 路由允许我们通过不同的 URL 访问不同的内容。
```js
// 引入路由
import router from "./router";
```
路由的实现涉及很多文件，比如在如上`router.js`文件中定义路由的路径和组件（即key和value）的映射关系、在模板html文件中定义哪些元素触发路由跳转（如router-link）以及匹配到路由后组件在哪显示（router-view）、注册路由等。
推荐[这篇教程](https://www.runoob.com/vue2/vue-routing.html)。
Imjoy定义了多个路由，如`/`、`/app`、`/about`等。

# 前端UI
## UI组件库
在`src/main.js`里第三行就是UI库的引入。
```js
import VueMaterial from "vue-material";
import "vue-material/dist/vue-material.min.css";
import "vue-material/dist/theme/default.css";
Vue.use(VueMaterial);
```
ImJoy前端组件使用的是[Vue Material](https://github.com/vuematerial/vue-material)这个组件库，其风格是Google开发的Material Design这种设计语言，即原生Android操作系统上的设计风格。
上述代码是使用了全局引入的方式。其组件的具体使用方式在[这里](https://www.creative-tim.com/vuematerial/components/app)。

## 栅格布局
接着引入了vue-grid-layout：
```js
import VueGridLayout from "vue-grid-layout";

Vue.component("grid-layout", VueGridLayout.GridLayout);
Vue.component("grid-item", VueGridLayout.GridItem);
```
vue-grid-layout是一个可拖拽、可调整大小的栅格布局系统，用于拖拽调整ImJoy各个程序运行窗口的显示位置等。 

## 自定义组件
ImJoy写了很多的自定义的组件。
组件（Component）是 Vue.js 最强大的功能之一。组件可以扩展 HTML 元素，封装可重用的代码。组件系统让我们可以用独立可复用的小组件来构建大型应用。
组件的相关知识推荐[这篇教程](https://www.runoob.com/vue2/vue-component.html)。
在`src/main.js`中ImJoy就引入了它写的很多组件，并进行了全局注册：
```js
// Imjoy组件，即https://imjoy.io/#/app这个链接所展示的页面。
import Imjoy from "@/components/Imjoy";
// About组件，即http://localhost:8001/#/about所展示的页面
import About from "@/components/About";
// Whiteboard组件，即ImJoy中间的展示区，其在上面的Imjoy组件中被使用
import Whiteboard from "@/components/Whiteboard";
// PluginList组件，即安装插件时从云端搜索并下载插件的窗口，在Imjoy组件中被使用
import PluginList from "@/components/PluginList";
// PluginEditor组件，即代码编辑器，出现在Imjoy组件中查看插件代码时，以及PluginList组件中同样查看插件代码时。
import PluginEditor from "@/components/PluginEditor";
// PluginIcon组件，即插件的图标，出现在插件列表最右侧，如果插件自定义了图标，则显示该图标；否则显示默认的extension图标
import PluginIcon from "@/components/PluginIcon";
// FileItem组件，即文件列表
import FileItem from "@/components/FileItem";
// FileDialog组件，即文件对话框
import FileDialog from "@/components/FileDialog";
// Window组件，即代码编辑和程序运行窗口，在Imjoy组件中被使用
import Window from "@/components/Window";
// EngineControlPanel组件，即ImJoy app右上角的小火箭图标菜单所对应的组件
import EngineControlPanel from "@/components/EngineControlPanel";

// 全局注册以上组件
Vue.component("imjoy", Imjoy);
Vue.component("about", About);
Vue.component("whiteboard", Whiteboard);
Vue.component("plugin-list", PluginList);
Vue.component("plugin-editor", PluginEditor);
Vue.component("plugin-icon", PluginIcon);
Vue.component("file-item", FileItem);
Vue.component("file-dialog", FileDialog);
Vue.component("window", Window);
Vue.component("engine-control-panel", EngineControlPanel);
```
还有一些组件没有被全局注册，只是被部分实例使用，如Home组件（即[imjoy首页](https://imjoy.io/#/)展示的页面）。

## 模态框插件
ImJoy引入了[vue-js-modal模态框插件](https://github.com/euvl/vue-js-modal)：
```js
import vmodal from "vue-js-modal";
Vue.use(vmodal);
```
它的使用教程见[这里](https://euvl.github.io/vue-js-modal/Intro.html#static-modals)。

# 事件总线
接着看`src/main.js`，接下来引入了事件总线库：
```js
import store from "./store.js";

在store.js中：
import Minibus from "minibus";

const event_bus = Minibus.create();
export default {
  event_bus: event_bus,
};
```
可以看出Imjoy引入了[minibus库](https://github.com/axelpale/minibus)，实现在一个地方触发（发布）事件，然后通过事件中心通知所有订阅者（订阅）。
on发布订阅、once发布订阅(触发一次)、emit通知执行(触发事件)、off取消订阅。