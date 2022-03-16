---
title: 开源深度学习计算平台ImJoy解析：12 -- 基于web的绘图库OpenLayers
tags: [ImJoy]
categories: computer vision 
date: 2022-3-16
---

# 简介
ImJoy有一个很有用的插件或独立应用，叫做[Kaibu](https://kaibu.org/)，它可以展示普通的位图、矢量图及vtk、stl等3D格式的数据。
比如如下展示：
![kaibu](https://user-images.githubusercontent.com/6218739/158537518-a3583f3e-4b06-494b-928a-3904f6b29fb2.png)
其就是位图（png格式）、矢量图（json格式）、3D模型（stl格式）的一个叠加。

Kaibu主要用了两个JS库，一个是[OpenLayers](https://openlayers.org/)，一个是[ITK-VTK](https://kitware.github.io/itk-vtk-viewer/docs/)，前者用于展示矢量图形、普通位图等数据，且对地图的展示异常强大，后者用于展示在医疗及科学计算中常用的3D图像、网格、点集等。

这一篇主要介绍OpenLayers的相关知识。

# 配置环境
从[OpenLayers workshop releases](https://github.com/openlayers/workshop/releases)里下载最新的资料包。
安装依赖：
```js
npm install
```
启动：
```js
npm start
```
这会启动一个开发服务器。可以通过`http://localhost:1234`查看一个“欢迎”的弹出窗口，以及`http://localhost:1234/doc/`查看说明文档。

# 开发入门
这一部分会通过OpenLayers map来创建一个简单的web页面。
在OpenLayers中，一个map是在web页面中被渲染的一系列“层”layers的集合。OpenLayers支持很多种layers：
（1）针对平铺光栅切片数据的Tile layer；
（2）针对位图图像的Image layer；
（3）针对矢量数据的Vector layer；
（4）针对平铺矢量切片数据的Vector tile layer。
除了这些layers，一个map还可以通过一系列的控制（即在map上面的UI元素）和交互（即与map进行交互反馈的部件）来进行配置。
为了创建一个map，需要通过HTML中的元素来创建（如一个`<div>`元素），以及一些样式来指定合适的尺寸。

## HTML页面
将项目根目录中的`index.html`里的内容替换为如下代码（注释写在了代码中）：
```js
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>OpenLayers</title>
    <style>
      /*引入OpenLayers的样式*/
      @import "node_modules/ol/ol.css";
    </style>
    <style>
      /*该部分样式使得map容器完全充满整个页面*/
      html, body, #map-container {
        margin: 0;
        height: 100%;
        width: 100%;
        font-family: sans-serif;
      }
    </style>
  </head>
  <body>
    <!-- 该div标签是map的渲染容器 -->
    <div id="map-container"></div>
    <!-- 引入相关的js代码 -->
    <script src="./main.js" type="module"></script>
  </body>
</html>
```

## 具体应用
将项目根目录中的`main.js`里的内容替换为如下代码：
```js
// 从OpenLayers中导入必要的模块
import OSM from 'ol/source/OSM';
import TileLayer from 'ol/layer/Tile';
import {Map, View} from 'ol';
import {fromLonLat} from 'ol/proj';

// 创建一个Map对象
new Map({
  // 目标是HTML中的那个div元素
  target: 'map-container',
  layers: [
    // 具体的layer是使用了Tile Layer
    new TileLayer({
      source: new OSM(),
    }),
  ],
  // view定义了初始的中心点和缩放比例
  view: new View({
    // 中心点的指定是通过fromLonLat函数获取地理坐标
    center: fromLonLat([0, 0]),
    zoom: 2,
  }),
});
```

## 效果
此时打开`http://localhost:1234`，会看到世界地图：
![basic](https://user-images.githubusercontent.com/6218739/158340705-a8876d2b-035e-4764-9c77-229e722be721.png)

# 矢量数据
在这一部分，将会创建一个可以操作矢量数据的编辑器，使得用户可以导入数据、绘制形状、修改已有形状及导出结果等。
本部分会使用[GeoJSON](https://geojson.org/)数据，不过OpenLayers支持其他大量的矢量数据格式。
## 渲染GeoJSON
在开发编辑功能之前，先看一下基本的对矢量数据的渲染功能。
在项目的data路径下有一个名为`countries.json`的GeoJSON文件，这里将加载该数据并在地图上渲染出来。
首先，编辑一下刚才的`index.html`，这里新加一行控制背景颜色的代码：
```js
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>OpenLayers</title>
    <style>
      @import "node_modules/ol/ol.css";
    </style>
    <style>
      html, body, #map-container {
        margin: 0;
        height: 100%;
        width: 100%;
        font-family: sans-serif;
        /*新加了这一行来控制背景颜色*/
        background-color: #04041b;
      }
    </style>
  </head>
  <body>
    <div id="map-container"></div>
    <script src="./main.js" type="module"></script>
  </body>
</html>
```
然后将`main.js`中的内容替换为如下代码：
```js
// 导入GeoJSON包来读写该格式的数据
import GeoJSON from 'ol/format/GeoJSON';
import Map from 'ol/Map';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';

new Map({
  target: 'map-container',
  layers: [
    // layer使用的是处理和渲染矢量数据的VectorLayer
    new VectorLayer({
      // VectorSource用来获取GeoJSON数据，并管理空间索引
      source: new VectorSource({
        format: new GeoJSON(),
        // 导入data目录下的JSON文件
        url: './data/countries.json',
      }),
    }),
  ],
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});
```
效果如下：
![json](https://user-images.githubusercontent.com/6218739/158343566-6f7fa033-d8ec-4741-b50f-5e65ce122485.png)

因为我们会重载这个页面很多次，目前代码下每次重载页面都会回到初始的view方式，即初始的中心点和缩放大小。如果能每次重载都能保持map在相同的位置就能节省很多人力。
此时可以借助`ol-hashed`包实现，修改代码如下：
```js
import GeoJSON from 'ol/format/GeoJSON';
import Map from 'ol/Map';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
// 导入ol-hashed包
import sync from 'ol-hashed';

// 将Map对象分配到一个变量上
const map = new Map({
  target: 'map-container',
  layers: [
    new VectorLayer({
      source: new VectorSource({
        format: new GeoJSON(),
        url: './data/countries.json',
      }),
    }),
  ],
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

// 将上面的变量传递给sync函数
sync(map);
```
此时，你会发现，将地图移动和缩放到某一特定程度后，下次重新载入代码仍然保持该视角不变。

## 拖放
对于要实现的编辑器，想要允许用户能够导入自己的数据进行编辑。为此，这里将添加`DragAndDrop`功能。
跟以前一样，这里仍只处理GeoJSON这种数据，不过该交互也支持其他类型的数据格式。
修改`main.js`为：
```js
import GeoJSON from 'ol/format/GeoJSON';
import Map from 'ol/Map';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
import sync from 'ol-hashed';
// 导入DragAndDrop包
import DragAndDrop from 'ol/interaction/DragAndDrop';

// 定义一个Map对象，只指定它的目标和视图
const map = new Map({
  target: 'map-container',
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

sync(map);

// 创建数据源VectorSource，但是里面没有任何数据
const source = new VectorSource();

// 创建VectorLayer，里面的source传入上面定义的空的source
const layer = new VectorLayer({
  source: source,
});
// 将layer添加到map中
map.addLayer(layer);

// 对map添加拖放交互
map.addInteraction(
  new DragAndDrop({
    // 将拖放动作作用在Vector Source上
    source: source,
    // 指定GeoJSON格式
    formatConstructors: [GeoJSON],
  })
);
```
此时就能将GeoJSON文件拖放到该页面上，从而进行渲染。

## 修改特征
现在可以将数据拖放到编辑器中，下面是添加“修改”功能。
实现方式是使用`Modify`交互。
修改`main.js`为：
```js
import DragAndDrop from 'ol/interaction/DragAndDrop';
import GeoJSON from 'ol/format/GeoJSON';
import Map from 'ol/Map';
// 导入Modify包
import Modify from 'ol/interaction/Modify';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
import sync from 'ol-hashed';

const map = new Map({
  target: 'map-container',
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

sync(map);

const source = new VectorSource();

const layer = new VectorLayer({
  source: source,
});
map.addLayer(layer);

map.addInteraction(
  new DragAndDrop({
    source: source,
    formatConstructors: [GeoJSON],
  })
);

// 在map上添加Modify交互，并配置交互对象
map.addInteraction(
  new Modify({
    source: source,
  })
);
```
此时就可以拖动顶点来修改特征。也可以使用`Alt+Click`来删除顶点。

## 绘制特征
接下来添加`Draw`交互来使得用户可以绘制新的特征，并添加到数据中。
```js
import DragAndDrop from 'ol/interaction/DragAndDrop';
// 导入Draw包
import Draw from 'ol/interaction/Draw';
import GeoJSON from 'ol/format/GeoJSON';
import Map from 'ol/Map';
import Modify from 'ol/interaction/Modify';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
import sync from 'ol-hashed';

const map = new Map({
  target: 'map-container',
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

sync(map);

const source = new VectorSource();

const layer = new VectorLayer({
  source: source,
});
map.addLayer(layer);

map.addInteraction(
  new DragAndDrop({
    source: source,
    formatConstructors: [GeoJSON],
  })
);

map.addInteraction(
  new Modify({
    source: source,
  })
);

// 在map上添加Draw交互
map.addInteraction(
  new Draw({
    // 指定绘制形状，该值可以是任意的GeoJSON的几何形状
    type: 'Polygon',
    // 配置交互对象
    source: source,
  })
);
```

## 自动吸附
上面的绘制功能添加后，可以发现，当绘制图形时，很难沿着之前的图形进行精确绘制。
此时可以添加`snap`功能，当鼠标移动到某个像素一定范围内时，就能自动吸附到该像素，从而完成精确绘制。
```js
import DragAndDrop from 'ol/interaction/DragAndDrop';
import Draw from 'ol/interaction/Draw';
import GeoJSON from 'ol/format/GeoJSON';
import GeometryType from 'ol/geom/GeometryType';
import Map from 'ol/Map';
import Modify from 'ol/interaction/Modify';
// 添加Snap包
import Snap from 'ol/interaction/Snap';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
import sync from 'ol-hashed';

const map = new Map({
  target: 'map-container',
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

sync(map);

const source = new VectorSource();

const layer = new VectorLayer({
  source: source,
});
map.addLayer(layer);

map.addInteraction(
  new DragAndDrop({
    source: source,
    formatConstructors: [GeoJSON],
  })
);

map.addInteraction(
  new Modify({
    source: source,
  })
);

map.addInteraction(
  new Draw({
    source: source,
    type: GeometryType.POLYGON,
  })
);

// 在map上添加Snap交互
map.addInteraction(
  new Snap({
    // 配置作用对象
    source: source,
  })
);
```

## 下载特征
当上传数据，且对其编辑后，希望能下载特征。
为了能实现这个功能，这里将特征数据序列化为GeoJSON数据，然后创建一个带`download`属性的`<a>`元素，这样就能触发浏览器的文件保存对话框。
同时，在map上添加一个按钮，可以使得用户清除现在的特征，重新绘制。
修改`index.html`为：
```js
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>OpenLayers</title>
    <style>
      @import "node_modules/ol/ol.css";
    </style>
    <style>
      html, body, #map-container {
        margin: 0;
        height: 100%;
        width: 100%;
        font-family: sans-serif;
        background-color: #04041b;
      }
      /*对id为tools的div进行样式设定*/
      #tools {
        position: absolute;
        top: 1rem;
        right: 1rem;
      }
      /*对id为tools中的两个后代a元素设定样式*/
      /*css语法可以参见这里：*/
      /*https://www.runoob.com/css/css-combinators.html*/
      #tools a {
        display: inline-block;
        padding: 0.5rem;
        background: white;
        cursor: pointer;
      }
      /*! [tools] */
    </style>
  </head>
  <body>
    <div id="map-container"></div>
    <script src="./main.js" type="module"></script>
    <!-- 新增一个div元素，里面包含了两个a元素 -->
    <div id="tools">
      <a id="clear">Clear</a>
      <a id="download" download="features.json">Download</a>
    </div>
  </body>
</html>
```
修改`main.js`为：
```js
import GeoJSON from 'ol/format/GeoJSON';
import GeometryType from 'ol/geom/GeometryType';
import Map from 'ol/Map';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
import sync from 'ol-hashed';
import {DragAndDrop, Draw, Modify, Snap} from 'ol/interaction';

const map = new Map({
  target: 'map-container',
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

sync(map);

const source = new VectorSource();

const layer = new VectorLayer({
  source: source,
});
map.addLayer(layer);

map.addInteraction(
  new DragAndDrop({
    source: source,
    formatConstructors: [GeoJSON],
  })
);

map.addInteraction(
  new Modify({
    source: source,
  })
);

map.addInteraction(
  new Draw({
    source: source,
    type: GeometryType.POLYGON,
  })
);

map.addInteraction(
  new Snap({
    source: source,
  })
);

// 实现清除功能
// 首先通过DOM选取clear按钮
const clear = document.getElementById('clear');
// 对该按钮添加鼠标事件
clear.addEventListener('click', function () {
  source.clear();
});

// 实现下载功能
// 这里序列化数据为GeoJSON格式
const format = new GeoJSON({featureProjection: 'EPSG:3857'});
// 通过DOM获取download按钮
const download = document.getElementById('download');

// 因为这里是期望随时都能下载最新的数据，所以将数据获取及序列化的工作绑定在source的change事件上
// 即，只要source改变，download按钮所能获得的数据就是最新的source
source.on('change', function () {
  // 获得特征
  const features = source.getFeatures();
  // 序列化特征
  const json = format.writeFeatures(features);
  // 这里将原json字符串转换成URI组成部分，将附加到下载按钮的href中
  download.href =
    'data:application/json;charset=utf-8,' + encodeURIComponent(json);
});
```
效果如下：
![download](https://user-images.githubusercontent.com/6218739/158392776-41d856f4-1c85-43e2-a129-380ab1131356.png)

## 配置绘图样式
前面的编辑功能都是使用了默认样式，这里增加更多的属性来使得编辑功能更加强大，比如设置画笔宽度、设置填充颜色等。
### 静态样式
如果单纯想将样式都调成一个模样，那么可以直接简单地将样式固定即可，如下面代码：
```js
const layer = new VectorLayer({
  source: source,
  style: new Style({
    fill: new Fill({
      color: 'red'
    }),
    stroke: new Stroke({
      color: 'white'
    })
  })
});
```
即都填充成红色，笔画都是白色。
### 动态样式
更多情况下，动态样式使用得更多，即按照一定的规则自动设置样式。
如下面：
```js
constlayer = newVectorLayer({
  source: source,
  style: function(feature, resolution) {
    constname = feature.get('name').toUpperCase();
    returnname < "N"? style1 : style2; // assuming these are created elsewhere}
});
```
就是根据feature的name来设置样式，如果是`A-M`，就用style1，如果是`N-Z`，则使用style2。
所以设定好规则非常重要。
下面将展示如何根据几何区域设定样式。
修改`main.js`为：
```js
import DragAndDrop from 'ol/interaction/DragAndDrop';
import Draw from 'ol/interaction/Draw';
import GeoJSON from 'ol/format/GeoJSON';
import GeometryType from 'ol/geom/GeometryType';
import Map from 'ol/Map';
import Modify from 'ol/interaction/Modify';
import Snap from 'ol/interaction/Snap';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import View from 'ol/View';
import sync from 'ol-hashed';
// 导入必要的样式库
import {Fill, Stroke, Style} from 'ol/style';
// 导入colormap包
import colormap from 'colormap';
// 从OpenLayers导入getArea包
import {getArea} from 'ol/sphere';

// --- 根据面积计算颜色：开始 ----
const min = 1e8; // the smallest area
const max = 2e13; // the biggest area
const steps = 50;
const ramp = colormap({
  colormap: 'blackbody',
  nshades: steps,
});

function clamp(value, low, high) {
  return Math.max(low, Math.min(value, high));
}

function getColor(feature) {
  const area = getArea(feature.getGeometry());
  const f = Math.pow(clamp((area - min) / (max - min), 0, 1), 1 / 2);
  const index = Math.round(f * (steps - 1));
  return ramp[index];
}
// --- 根据面积计算颜色：结束 ----

const map = new Map({
  target: 'map-container',
  view: new View({
    center: [0, 0],
    zoom: 2,
  }),
});

sync(map);

const source = new VectorSource();

// 添加样式
const layer = new VectorLayer({
  source: source,
  style: function (feature) {
    return new Style({
      fill: new Fill({
        color: getColor(feature),
      }),
      stroke: new Stroke({
        color: 'rgba(255,255,255,0.8)',
      }),
    });
  },
});

map.addLayer(layer);

map.addInteraction(
  new DragAndDrop({
    source: source,
    formatConstructors: [GeoJSON],
  })
);

map.addInteraction(
  new Modify({
    source: source,
  })
);

map.addInteraction(
  new Draw({
    source: source,
    type: GeometryType.POLYGON,
  })
);

map.addInteraction(
  new Snap({
    source: source,
  })
);

const clear = document.getElementById('clear');
clear.addEventListener('click', function () {
  source.clear();
});

const format = new GeoJSON({featureProjection: 'EPSG:3857'});
const download = document.getElementById('download');
source.on('change', function () {
  const features = source.getFeatures();
  const json = format.writeFeatures(features);
  download.href = 'data:text/json;charset=utf-8,' + json;
});
```
效果如下：
![style](https://user-images.githubusercontent.com/6218739/158398065-704634ef-a6dc-4d17-a4a6-a87edc51c8f1.png)

# 移动端地图和数据集成
这一部分将创建一个移动端的地图来展示用户的GPS位置和朝向。该项目的目的是为了展示怎样将OpenLayers与浏览器的API及第三方工具进行集成。
具体地，仅使用几行代码即可调用浏览器的关于地理位置的API，从而得到GPS位置，以及使用[kompas](https://www.npmjs.com/package/kompas)库通过设备的陀螺仪获得朝向。然后，通过使用Vector Layer，就能很轻易地在地图上显示结果。

因为这一部分需要移动端的配合，不再具体分析。



更多用法留坑待填。
