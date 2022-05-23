---
title: 开源深度学习计算平台ImJoy解析：13 -- Kaibu应用
tags: [ImJoy]
categories: computer vision 
date: 2022-5-22
---

# 简介
ImJoy有一个很有用的插件或独立应用，叫做[Kaibu](https://kaibu.org/)，它可以展示并标注普通的位图、矢量图及vtk、stl等3D格式的数据。
比如如下展示：
![kaibu](https://user-images.githubusercontent.com/6218739/158537518-a3583f3e-4b06-494b-928a-3904f6b29fb2.png)
其就是位图（png格式）、矢量图（json格式）、3D模型（stl格式）的一个叠加。

Kaibu主要用了两个JS库，一个是[OpenLayers](https://openlayers.org/)，一个是[ITK-VTK](https://kitware.github.io/itk-vtk-viewer/docs/)，前者用于展示矢量图形、普通位图等数据，且对地图的展示异常强大，后者用于展示在医疗及科学计算中常用的3D图像、网格、点集等。

这一篇主要分析一下Kaibu的运行机理。

从script标签开始，看其运行过程。

# 导入组件
```js
import * as layerComponents from "@/components/layers";
import * as widgetComponents from "@/components/widgets";
```
导入自定义的各种layers和widgets。
这里的layers即是Kaibu能承载的数据类型，其概念就是“图层”的含义，即不同类型的图层叠加起来，就能展示复杂的图像。具体有2D图像层ImageLayer、矢量层VectorLayer和3D模型层ItkVtkLayer。
这里的widgets即是Kaibu自定义的各种控件，比如树形控件TreeWidget、表单控件FormWidget、列表控件ListWidget等。

```js
const components = {};
const layerTypes = {};

for (let c in layerComponents) {
  components[layerComponents[c].name] = layerComponents[c];
  layerTypes[layerComponents[c].type] = layerComponents[c];
}

console.log('components = ', components)
console.log('layerTypes = ', layerTypes)
```
将不同的layers根据name和type分别提取到两个变量中，即components存储了以layer名称为键的图层组件，layerTypes中存储了以layer类型为键的图层组件。

```js
const widgetTypes = {};
for (let c in widgetComponents) {
  components[widgetComponents[c].name] = widgetComponents[c];
  widgetTypes[widgetComponents[c].type] = widgetComponents[c];
}
console.log('components = ', components)
console.log('widgetTypes = ', widgetTypes)
```
将不同的widgets根据name和type分别提取到两个变量中，即之前的components中又接收了以widgets名称为键的控件组件，widgetTypes则存储了以widget类型为键的控件组件。

# 挂载生命周期钩子
Kaibu定义了很多计算属性和方法，可以先从mounted挂载这一生命周期钩子看其怎么调用的。
```js
  mounted() {
    this.init();
    this.sortableOptions.layer_configs = this.layer_configs;
    window.addEventListener("resize", this.updateSize);
    window.dispatchEvent(new Event("resize"));
    this.openSidebar(true);
  },
```
可以看出，挂载后第一步是初始化，它也是最重要的一步。下面对这一步进行详细的解析。

## 创建Map
因为Kaibu用的是OpenLayers进行渲染，因此首先创建一个OpenLayers的Map实例对象是至关重要的，即它作为画布。
```js
      const extent = [0, 0, 1024, 968];
      const projection = new Projection({
        code: "xkcd-image",
        units: "pixels",
        extent: extent
      });
      const map = new Map({
        interactions: defaults({
          altShiftDragRotate: false,
          pinchRotate: false
        }),
        target: "map",  // 这里就是与template里id为map的div标签进行绑定
        layers: [],
        view: new View({
          projection: projection,
          center: getCenter(extent),
          zoom: 2,
          maxZoom: 8
        })
      });
      this.$store.commit("setMap", map);
```
这样新建了一个Map实例对象map，并且传给了vuex状态管理器。


## 添加图层事件
接下来就是添加具体的图层。
```js
        await this.addLayer({
          type: "vector",
          name: "shape vectors",
          data:
            "https://gist.githubusercontent.com/oeway/7c62128939a7f9b1701e2bbd72b809dc/raw/example_shape_vectors.json",
          predefined_tags: ["nuclei", "cell"],
          only_predefined_tags: true,
          single_tag_mode: false
        });
```
如上就是增加Vector图层的代码。它调用了addLayer这个方法，明显地，它接收上面的配置对象作为参数：
```js
    addLayer(config) {
      return new Promise((resolve, reject) => {
        config.id = randId(); // 对配置对象添加id属性
        this.$store.dispatch("addLayer", config); // 触发vuex共享状态
        config._add_layer_promise = { resolve, reject };
      });
    },
```
可以看出，它首先对上面接收的配置对象config添加了一个随机的id，然后触发vuex共享状态的addLayer这个action（Vuex通过Vue的插件系统将store实例从根组件中“注入”到所有的子组件里。且子组件能通过`this.$store`访问到）。
详细看一下共享状态的这些事件：
```js
export const store = new Vuex.Store({
  state: {
    layers: {},
    widgets: {},
    layer_configs: [],
    currentLayer: null,
    currentLayerWidget: null,
    standaloneWidgets: {},
    map: null
  },
  actions: {
    addLayer(context, config) {
      context.commit("addLayer", config);
      Vue.nextTick(() => {
        if (config.init) {
          config
            .init()
            .then(layer => {
              if (!layer) {
                if (config._add_layer_promise) {
                  config._add_layer_promise.reject(
                    "Failed to create layer for " + config.name
                  );
                }
              }
              layer.config = config;
              layer.setVisible(config.visible);
              layer.getLayerAPI = layer.getLayerAPI || function() {};
              context.commit("initialized", layer);
              context.commit("setCurrentLayer", layer.config);
              context.commit("sortLayers");
              if (config._add_layer_promise) {
                config._add_layer_promise.resolve(layer);
                delete config._add_layer_promise;
              }
            })
            .catch(e => {
              if (config._add_layer_promise) {
                config._add_layer_promise.reject(e);
                delete config._add_layer_promise;
              } else {
                console.error(e);
              }
            });
        } else {
          debugger;
        }
      });
    }
  },
  mutations: {
    addLayer(state, config) {
      if (config.visible === undefined) config.visible = true;
      if (typeof config.index === "number") {
        const index = config.index;
        state.layer_configs.splice(index, 0, config);
        delete config.index;
      } else state.layer_configs.push(config);
    },
```
即，上面通过dispatch触发addLayer这个action后，在该action里又通过commit触发了addLayer这个mutation。在这个mutation中，会将配置对象config推入共享状态state的`layer_configs`属性中。
接下来就是非常tricky的一个操作。
先看store的action的这段：
```js
    addLayer(context, config) {
      context.commit("addLayer", config);
      Vue.nextTick(() => {
        if (config.init) {
          config
            .init()
```
如上所述，这个addLayer会通过commit触发mutations中的addLayer，从而对state的`layer_configs`属性进行更新。注意，此时的config还是非常普通的配置对象，就是上面的：
```js
        {
          type: "vector",
          name: "shape vectors",
          data:
            "https://gist.githubusercontent.com/oeway/7c62128939a7f9b1701e2bbd72b809dc/raw/example_shape_vectors.json",
          predefined_tags: ["nuclei", "cell"],
          only_predefined_tags: true,
          single_tag_mode: false
        }
```
但是，接下来`Vue.nextTick()`一执行后，该config就会根据添加的layer的不同，变成特有的config，如下所示：
![config](https://user-images.githubusercontent.com/6218739/163119092-4e401f8a-aac1-44cb-8912-6bafb11115dc.png)

这是为什么呢？
经过一番仔细的定位，发现奥妙如下，非常巧妙（同时好难懂。。。）。
前面已经说了，在mutations中会对`state.layer_configs`进行更新，而在Kaibu的主组件ImageViewer里有该状态的映射，因此组件的`layer_configs`也会有更新：
```js
    ...mapState({
      layers: state => state.layers,
      layer_configs: state => state.layer_configs,
      standaloneWidgets: state => state.standaloneWidgets,
      currentLayer: state => state.currentLayer,
      currentLayerWidget: state => state.currentLayerWidget,
      map: state => state.map
    })
```
而该计算属性的更新会引起组件的模板中如下部分的变化：
```js
            <b-menu-list label="Properties">
              <component
                v-for="layer in layer_configs"
                v-show="currentLayer === layer"
                @update-extent="updateExtent"
                :ref="'layer_' + layer.id"
                :key="layer.id"
                :is="layerTypes[layer.type]"
                @loading="loading = $event"
                :selected="layer.selected"
                :visible="layer.visible"
                :map="map"
                :config="layer"
              />
            </b-menu-list>
```
这个地方用到了Vue的component标签（标签，注意不是组件），它可以动态绑定组件，即根据不同的数据来显示不同的组件。
其通过`is`的值来确定哪个组件被渲染。
一个教程见[这里](https://www.cnblogs.com/yjiangling/p/12794933.html)。
具体到这个例子中，因为传入的config的type键的值是`vector`，所以它会找到VectorLayer这个组件（即该文最开始的layerTypes中存储了以layer类型为键的图层组件）。
然后将`layer_configs`中遍历的layer（实际就是上面的config）作为config参数（注意这里是config参数标识）从ImageViewer父组件中传给VectorLayer这个子组件（子组件中会通过props属性进行接收）。换句话说，子组件中的config参数就是父组件传入的layer变量，也就是之前的普通的config对象（因为这个config对象被放入`layer_configs`中，在遍历时被称为layer）。
但是这个地方有一个不合常规的做法，即父组件和子组件通过props传递参数，常规是单向数据流，即子组件不能修改父组件的值。这里明显违背了这一原则，而且Vue没有报错。
这是因为config是个对象，而不是诸如字符串、数字等非引用类型数据，所以它不会报错，但也造成了理解的困难。
下面这两篇解析很好：
[ Vue 之在子组件中改变父组件的状态 ](https://nekolr.github.io/2020/04/26/Vue%20%E4%B9%8B%E5%9C%A8%E5%AD%90%E7%BB%84%E4%BB%B6%E4%B8%AD%E6%94%B9%E5%8F%98%E7%88%B6%E7%BB%84%E4%BB%B6%E7%9A%84%E7%8A%B6%E6%80%81/)
[vue2.0中，子组件修改父组件传递过来得props，该怎么解决？](https://segmentfault.com/q/1010000008525755)
可以说是，这里正好应用了这一特性，使得可以特定的子组件可以对config有特定的改变。即在子组件中会对这个config属性进行针对性地加工，比如将该组件地`init`方法传给它：
```js
    this.config.init = this.init;
```

上面那部分模板更新后，意味着DOM更新了，此时Vue地nextTick执行，那么这时候的config就是“被特定子组件修饰过”的config。

## 初始化图层
前面已说过，config被特定组件修饰后，就有了该组件或该图层的特性，下面就是调用特定组件的初始化方法：
```js
      Vue.nextTick(() => {
        if (config.init) {
          config
            .init()
            .then(layer => {
...
```js
即`config.init()`方法调用。
以上面的VectorLayer这个组件为例，看一下它的初始化方法：
```js
    async init() {
      this.layer = await this.setupLayer();
      this.map.addLayer(this.layer);
      if (this.config.select_enable && !this.config.draw_enable) {
        this.enableSelectInteraction();
      }
      this.updateDrawInteraction();
      this.$forceUpdate();
      return this.layer;
    },
```
其他组件也大同小异，就是在该组件中进行一系列的操作，最终是返回`this.layer`这样的特定的对象。

## 更新共享状态
特定组件初始化成功后，就会接着调用`then`方法：
```js
      Vue.nextTick(() => {
        if (config.init) {
          config
            .init()
            .then(layer => {
              console.log('layer = ', layer)
              layer.config = config;
              layer.setVisible(config.visible);
              layer.getLayerAPI = layer.getLayerAPI || function() {};
              context.commit("initialized", layer);
              context.commit("setCurrentLayer", layer.config);
              context.commit("sortLayers");
              if (config._add_layer_promise) {
                config._add_layer_promise.resolve(layer);
                delete config._add_layer_promise;
              }
            })
```
可以看出，会对上一步返回的layer对象进行一些加工后，再依次调用共享状态store中的一些mutations对共享状态进行更新：
```js
    initialized(state, layer) {
      state.layers[layer.config.id] = layer;
      layer.setZIndex(state.layer_configs.length - 1);
    },

    setCurrentLayer(state, layer) {
      layer = state.layer_configs.filter(l => l.id === layer.id)[0];
      if (state.currentLayer === layer) return;
      if (state.currentLayer) {
        state.currentLayer.selected = false;
      }
      state.currentLayer = layer;
      layer.selected = true;
      state.currentLayerWidget = null;
      for (let k of Object.keys(state.widgets)) {
        if (
          state.currentLayer.name &&
          state.widgets[k].attach_to == state.currentLayer.name
        ) {
          state.currentLayerWidget = state.widgets[k];
          break;
        }
      }
    },

    sortLayers(state) {
      for (let i = 0; i < state.layer_configs.length; i++) {
        if (state.layers[state.layer_configs[i].id])
          state.layers[state.layer_configs[i].id].setZIndex(i);
        else {
          console.warn("Layer not ready", state.layer_configs[i]);
        }
      }
    },
```
主要是对state中的属性进行更新，比如`layers`信息、`currentLayer`信息等，更新后的共享状态state如下：
![state](https://user-images.githubusercontent.com/6218739/163129381-2a30e9dd-2c30-4ff0-a4a8-75495236656c.png)


# 组件
前面已说到，kaibu有三个非常重要的组件，用于承载不同的图层数据，分别是2D图像层ImageLayer、矢量层VectorLayer和3D模型层ItkVtkLayer。
2D图像层使用的是OpenLayers的`static image`，矢量层使用的是OpenLayers的`Vector`，对于3D模型层，则是使用的[ITk/Vtk Viewer](https://kitware.github.io/itk-vtk-viewer/docs/)。

## ImageLayer
```js
<template>
  <div class="image-layer">
    <!-- 通过组件的layer属性切换是否显示 -->
    <section v-if="layer">
      <b-field label="opacity">
        <!-- 该组件在模板中只有一个滑动条 -->
        <b-slider
          v-model="config.opacity"
          @input="updateOpacity"
          :min="0"
          :max="1.0"
          :step="0.1"
        ></b-slider>
      </b-field>
      <!-- <b-field v-if="config.climit" label="contrast limit">
        <b-slider v-model="config.climit" :min="1" :max="255" :step="0.5" ticks>
        </b-slider>
      </b-field> -->
    </section>
  </div>
</template>

<script>
// 从openlayers中导入必要的包
import { Map } from "ol";
import Static from "ol/source/ImageStatic";
import ImageLayer from "ol/layer/Image";
import Projection from "ol/proj/Projection";

//将File对象转为base64编码的对象
function file2base64(file) {
  return new Promise((resolve, reject) => {
    var reader = new FileReader();
    reader.onload = event => {
      resolve(url2base64(event.target.result));
    };
    reader.onerror = err => {
      reject(err);
    };
    reader.readAsDataURL(file);
  });
}

// 将图片url转为base64编码的对象
async function url2base64(url) {
  return new Promise((resolve, reject) => {
    var img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = function() {
      var canvas = document.createElement("canvas");
      var ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      resolve({
        url: canvas.toDataURL("image/png"),
        w: img.width,
        h: img.height
      });
    };
    img.onerror = function(e) {
      reject("image load error:" + String(e));
    };
    img.src = url;
  });
}

// 将无符号的整型数组转化为base64编码的对象
function array2rgba(imageArr, ch, w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const canvas_img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const canvas_img_data = canvas_img.data;
  const count = w * h;

  const raw = new Uint8Array(imageArr.buffer);
  if (imageArr instanceof Uint8Array) {
    if (ch === 1) {
      for (let i = 0; i < count; i++) {
        canvas_img_data[i * 4] = raw[i];
        canvas_img_data[i * 4 + 1] = raw[i];
        canvas_img_data[i * 4 + 2] = raw[i];
        canvas_img_data[i * 4 + 3] = 255;
      }
    } else if (ch === 2) {
      for (let i = 0; i < count; i++) {
        canvas_img_data[i * 4] = raw[i * 2];
        canvas_img_data[i * 4 + 1] = raw[i * 2 + 1];
        canvas_img_data[i * 4 + 2] = 0;
        canvas_img_data[i * 4 + 3] = 255;
      }
    } else if (ch === 3) {
      for (let i = 0; i < count; i++) {
        canvas_img_data[i * 4] = raw[i * 3];
        canvas_img_data[i * 4 + 1] = raw[i * 3 + 1];
        canvas_img_data[i * 4 + 2] = raw[i * 3 + 2];
        canvas_img_data[i * 4 + 3] = 255;
      }
    } else if (ch === 4) {
      for (let i = 0; i < count; i++) {
        canvas_img_data[i * 4] = raw[i * 3];
        canvas_img_data[i * 4 + 1] = raw[i * 3 + 1];
        canvas_img_data[i * 4 + 2] = raw[i * 3 + 2];
        canvas_img_data[i * 4 + 3] = raw[i * 4 + 3];
      }
    }
  } else {
    throw "unsupported array type";
  }
  ctx.putImageData(canvas_img, 0, 0);
  return {
    url: canvas.toDataURL("image/png"),
    w: w,
    h: h
  };
}

export default {
  name: "image-layer",
  type: "2d-image",
  show: false,
  // 从父组件中接收参数
  props: {
    // map也是从父组件中传过来的存放于vuex中的map属性
    map: {
      type: Map,
      default: null
    },
    selected: {
      type: Boolean,
      default: false
    },
    visible: {
      type: Boolean,
      default: false
    },
    // 这里的config会修改父组件中的config
    config: {
      type: Object,
      default: function() {
        return {};
      }
    }
  },
  data() {
    return {
      layer: null
    };
  },
  watch: {
    visible: function(newVal) {
      this.layer.setVisible(newVal);
    }
  },
  mounted() {
    this.config.climit = [4, 50];
    this.config.opacity = 1.0;
    this.config.init = this.init;
  },
  beforeDestroy() {
    if (this.layer) {
      this.map.removeLayer(this.layer);
    }
  },
  created() {},
  methods: {
    // init初始化函数非常重要，它在vuex状态管理中被调用
    async init() {
      this.layer = await this.setupLayer();
      this.map.addLayer(this.layer);
      this.$forceUpdate();
      return this.layer;
    },
    updateOpacity() {
      if (this.layer) this.layer.setOpacity(this.config.opacity);
    },
    selectLayer() {},

    // 这里就是对this.layer进行加工的详细过程了
    async setupLayer() {
      let imgObj;
      const data = this.config.data;
      // 根据data的类型，进行不同的转换
      // 如果data是个字符串，比如是个图片路径
      if (typeof data === "string") {
        // 调用url2base64函数将其转化成base64编码格式
        imgObj = await url2base64(this.config.data);
      } else if (data instanceof File) { // 如果是File对象实例，则调用file2base64
        imgObj = await file2base64(this.config.data);
      } else if ( // 如果不是上面两种格式，就应该是无符号的整型数组及一些尺寸属性
        data &&
        data.imageType &&
        data.size &&
        data.imageType.componentType &&
        data.data
      ) {
        if (data.imageType.componentType !== "uint8_t") {
          throw `Unsupported data type: ${data.imageType.componentType}`;
        }

        if (data.imageType.components < 1 && data.imageType.components > 4) {
          throw `Unsupported components number: ${data.imageType.components}`;
        }

        if (data.imageType.dimension !== 2) {
          throw `Dimension must be 2`;
        }

        if (data.imageType.pixelType !== 1) {
          throw `Pixel type must be 1`;
        }
        imgObj = array2rgba(
          data.data,
          data.imageType.components,
          data.size[0],
          data.size[1]
        );
      } else {
        imgObj = {
          url: "https://images.proteinatlas.org/19661/221_G2_1_red_green.jpg",
          w: 2048,
          h: 2048
        };
      }
      const extent = [0, 0, imgObj.w, imgObj.h];
      // Map总是需要一个projection，这里只是想把图像坐标系映射到地图坐标系中，所以直接使用以像素为单位的图像内容来创建projection
      const projection = new Projection({
        code: "image",
        units: "pixels",
        extent: extent
      });
      // 创建一个static对象来作为下面ImageLayer的source
      const image_source = new Static({
        url: imgObj.url,
        projection: projection,
        imageExtent: extent
      });
      // 新建一个ImageLayer图层
      const image_layer = new ImageLayer({
        source: image_source
      });
      // 向父组件发射消息，使其能够监听到自定义事件，目前看主要是影响了显示中心位置
      this.$emit("update-extent", { id: this.config.id, extent: extent });
      return image_layer;
    }
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style></style>
```


# API
## 作为Imjoy插件时的API
Kaibu作为ImJoy的插件使用时，有许多api可以调用，如
（1）`view_image`可以增加一个2D图像、itk/vtk 3D模型等，
（2）`add_shapes`可以增加一个矢量图层、
（3）`add_widgets`可以增加按钮、文件树等控件。
具体的api文档见[这里](https://kaibu.org/docs/#/api?id=kaibu-api)，而相应源代码见[这里](https://github.com/imjoy-team/kaibu/blob/master/src/imjoyAPI.js)。
其实细细研究源码可知，这些api是对原来的imageviewer这个组件中的`AddLayer`、`addWidget`等方法的二次封装。
所以如果不作为ImJoy插件使用时，可以直接看imageviewer这个组件中的这些方法。

## 原生API
我们就以上面添加的矢量图层这个例子看看kaibu的原生API。
```js
        // 将通过AddLayer添加的矢量图层接收到一个变量中
        this.shape_layer = await this.addLayer({
          type: "vector",
          name: "shape vectors",
          data:
            "https://gist.githubusercontent.com/oeway/7c62128939a7f9b1701e2bbd72b809dc/raw/example_shape_vectors.json",
          predefined_tags: ["nuclei", "cell"],
          only_predefined_tags: true,
          single_tag_mode: false
        });
```
此时`this.shape_layer`上会有`getLayerAPI`这个方法，里面有若干适用于矢量层的API，比如`add_feature`、`add_features`、`get_features`等。
具体api可以见[这里](https://github.com/imjoy-team/kaibu/blob/master/src/components/layers/VectorLayer.vue)，搜`getLayerAPI`函数。
除了上述API，该`this.shape_layer`也会天然地拥有openlayers库中对于它所赋予的函数，详见[这里](https://openlayers.org/en/latest/apidoc/module-ol_layer_Vector-VectorLayer.html)。
