---
title: 物联网应用的低代码开发工具Node-RED上手及案例
tags: [IoT]
categories: coding 
date: 2022-6-19
---

# 简介
[Node-RED](https://nodered.org/)是一种低代码的将各种硬件设备、API和在线服务连接起来的编程工具，比如将各种传感器（如树莓派上的传感器）、工业控制器（如`MQTT`、`OPC-UA`协议）、在线API（比如获取天气情况、发送邮件等）等以图形化拖拽编程的形式进行高效组态。因此，`Node-RED`广泛地应用在物联网流程和程序开发中。
# 安装和运行
`Node-RED`可以运行在多种设备上，比如本地计算机运行、设备上运行（如树莓派、Arduino和Android等），以及在云端运行（比如IBM云、AWS云和微软Azure云等）。
具体可以查看[这里](https://nodered.org/docs/getting-started/)。
下面将以本地计算机运行为例，同时最简单的就是使用`docker`方式进行安装、运行。
`docker`安装运行只需一条命令即可：
```sh
docker run -it -p 1880:1880 -v node_red_data:/data --name mynodered nodered/node-red
```
对该命令具体解释一下：
```sh
docker run              - 利用镜像创建容器，会先检查本地是否存在指定的镜像，或不存在就从公有仓库下载
-it                     - 附加一个交互式的终端，以查看输出内容
-p 1880:1880            - 指定端口映射
-v node_red_data:/data  - 挂载容器数据卷，实现容器数据的持久化，前者是主机目录，后者是容器内目录
--name mynodered        - 自定义一个容器名
nodered/node-red        - 该容器所基于的镜像文件
```
启动命令后就可以在浏览器中通过`http://{host-ip}:1880`这样的地址访问`Node-RED`，比如`http://127.0.0.1:1880/`。

## 后台运行
此时命令运行在终端，如果对运行情况满意，可以通过`Ctrl-p` `Ctrl-q`命令将其转入守护模式，此时容器就会在后台运行，

## 前台运行
如果想重新在终端中查看运行日志，则重新附加到终端运行，即：
```sh
docker attach mynodered
```
## 停止和重启
停止容器，使用：
```sh
docker stop mynodered
```
如果想重启容器（比如计算机重启后或Docker daemon守护进程重启后），则：
```sh
docker start mynodered
```

## 更多
更多关于`docker`安装和运行`Node-RED`，可以查看[这里](https://nodered.org/docs/getting-started/docker)，里面详述了数据持久化、升级、其他设备上的安装方式等。

# 创建第一个flow
这部分将创建`Node-RED`的第一个`flow`，来入门学习`Node-RED`。
## 访问编辑器
当在后台运行了`Node-RED`后，就可以在浏览器中打开编辑器。
如果浏览器与`Node-RED`运行在同一台电脑上，那么就可以通过`http://localhost:1880`打开编辑器。
如果在另一台电脑上打开浏览器，则使用具体ip地址访问即可，即：`http://<ip-address>:1880`。

## 添加inject节点
`inject`节点可以对一个`flow`发射消息，比如在这个节点上点击它上面的左侧小按钮，或者在多个`injects`之间设置时间间隔。
从左侧面板中拖动`inject`节点到中间工作区中。

## 添加debug节点
`debug`节点能够将信息显示在`Debug`边栏中。默认情况下，它会只显示消息的`payload`（即承载的内容），也有可能显示整个消息对象。
从左侧面板中拖动`debug`节点到中间工作区中。

## 连线
在`inject`节点的输出拖动到`debug`节点的输入上，两者之间的连线即表明两者连接起来了。

## 部署
经过如上步骤，这两个节点仅存在于编辑器中，还必须部署到服务器上。
点击右上角的`Deploy`按钮。
打开`debug`边栏（即右上角那个小虫子图标），然后点击`inject`节点上的左侧小按钮，就可以在边栏中看到有数字出现。默认情况下，`inject`节点将`1970年1月1日`依赖的毫秒数作为它的`payload`。

## 添加function节点
`function`节点可以将消息传入一个`JavaScript`函数中。
选择现在已有的连线，然后删除它。
从左侧面板中选择`function`节点，放到`inject`节点和`debug`节点中间，然后将它们三个顺序连接起来。
双击`function`节点，打开编辑对话框，填入如下内容：
```js
// Create a Date object from the payload
var date = new Date(msg.payload);
// Change the payload to be a formatted Date string
msg.payload = date.toString();
// Return the message so it can be sent on
return msg;
```
点击`done`按钮关闭该对话框，然后点击`deploy`按钮。
此时再次点击`inject`节点，就会发现边栏中的消息已经变成了带格式的可读的时间戳形式。

## 总结
这个`flow`清晰地展示了如何创建`flow`的基本过程。

# 创建第二个flow
第二个`flow`基于上面的第一个`flow`，它可以从外部导入数据，并进行操作。
## 添加inject节点
在第一个`flow`中，`inject`节点是用来在鼠标点击时触发一个`flow`。
在第二个`flow`中，`inject`节点会配置成在一定间隔下触发一个`flow`。
从左侧面板中拖动`inject`节点到中间工作区中，然后双击打开编辑对话框，在`Repeat`下选择`interval`，然后设置成`every 5 seconds`。点击`done`保存该设置。
## 添加http request节点
`http request`节点可以用来获得一个web页面。
从左侧面板中拖动`http request`节点到中间工作区中，编辑`URL`一项为：`https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.csv`。该链接是从美国地质调查局网站上获得的上一个月的地震信息，是一个`csv`文件。
## 添加csv节点
添加`csv`节点，在`Input`中勾选`First row contains column names`。
## 添加debug节点
添加`debug`节点作为输出。
## 连线
将上述节点顺序连接。
## 添加switch节点
添加`switch`节点，双击打开编辑框，依次设置如下属性：
将`Property`设置为`msg.payload.mag`，判断条件设置为`>=`，判断对象类型设置为`number`（从下拉列表选择），判断值设置为`7`。
从`csv`节点的输出中引出第二条连线到`switch`节点上。
## 添加change节点
添加`change`节点，将`switch`节点的输出连接到该节点上。
双击`change`节点，设置`msg.payload`到字符串`PANIC!`。
## 添加debug节点
再次添加`debug`节点，将`change`节点的输出连接到它的输入上。
## 部署
点击`Deploy`进行部署。
可以发现，每隔`5s`它就会发一次请求，然后得到如下内容：
```js
msg.payload : Object
{ time: "2022-05-19T10:13:31.625Z", latitude: -54.132, longitude: 159.0545, depth: 10, mag: 6.9 … }
```
可以点击该消息左侧的小箭头查看详情。
如果地震的烈度大于`7`，还会收到如下消息：
```js
msg.payload : string[6]
"PANIC!"
```
## 总结
该`flow`演示了每隔一段时间自动触发从某个`url`获取数据的操作。同时对数据进行解析和判断。

# 新增节点
参考教程见[这里](https://nodered.17coding.net/docs/getting-started/adding-nodes)。
`Node-RED`本身只包含了一些核心的基本节点，但还有大量来自于`Node-RED`项目和广大社区的其他节点可以使用。
可以在[Node-RED代码库](https://flows.nodered.org/)或[npm仓库](https://www.npmjs.com/search?q=keywords:node-red)中寻找所需要的节点。
## 使用编辑器
从`0.15`版开始，可以直接通过编辑器安装节点。具体做法是，在右上角的菜单中选择“管理面板”`Manage Palette`，然后在出现的面板中选择`Install`标签，这样就可以搜索安装新的节点，并启用或禁用已有节点了。
## 通过npm安装
可以通过`npm`将节点安装到保存用户数据的本地目录中(默认为`$HOME/.node-red)`：
```js
cd $HOME/.node-red
npm install <npm-package-name>
```
也可以安装到全局目录中:
```js
sudo npm install -g <npm-package-name>
```
但是这样需要重启`Node-RED`，以便它能够获取到新的节点。
## 复制节点文件
将`.js`和`.html`文件复制到保存用户数据目录中的`nodes`子目录，也是安装节点的一个可行方式。如果这些节点中存在一些`npm`依赖项，那么也必须将其安装到用户数据目录中。但建议只是在开发阶段使用这种方式。

# MQTT例子
参考教程[1](https://www.emqx.com/zh/blog/using-node-red-to-process-mqtt-data)、[2](https://www.v5w.com/iot/node-red/127.html)、[3](https://www.youtube.com/watch?v=XDrwgMSQrEY)。
除`HTTP`、`WebScoket`等一些基础的网络服务应用节点外，`Node-RED`还提供对于`MQTT`协议的连接支持。目前同时提供了一个`MQTT`的订阅节点和`MQTT`的发布节点，订阅节点用于数据的输入，而发布节点可以用于数据的输出。
关于`MQTT`，[这里](https://siot.readthedocs.io/zh_CN/latest/1.about/02_mqtt.html)有一个很好的介绍。摘录一个架构图如下：
![mqtt](https://user-images.githubusercontent.com/6218739/174246248-dce70e2e-1986-47a8-a92a-1f2743c91efc.png)

## 安装MQTT消息服务器
### 安装EMQX
开源免费的`MQTT`消息服务器有很多，如[EMQX](https://github.com/emqx/emqx)、[Mosquitto](https://github.com/eclipse/mosquitto)、[Mosca](https://github.com/moscajs/mosca)等。
这里选择`EMQX`，使用`docker`安装：
```sh
docker run -d --name emqx -p 1883:1883 -p 8081:8081 -p 8083:8083 -p 8883:8883 -p 8084:8084 -p 18083:18083 emqx/emqx
```

## 配置Node-RED
### 整体架构
`Node-RED`中的`flow`结构配置为：
![flow](https://user-images.githubusercontent.com/6218739/174427721-ad0b02c6-144d-4042-b209-c95f6b0f97f6.png)
上方是发布节点`mqtt out`，用于发布数据；下方是订阅节点`mqtt in`，是用于订阅数据。
### 发布消息
对触发消息的节点`inject`进行配置（这是为了模拟一个简单的传感器的开关）：
![sensor1](https://user-images.githubusercontent.com/6218739/174256774-dc145287-371d-4f9e-a3a4-98b70f36a7d6.png)
对发布数据的节点`mqtt out`进行配置：
最开始配置时，`Server`一项需要新建，可按如下新建（注意`IP`地址设置正确，因为这里是`docker`安装的消息服务器，所以不能用`localhost`或者`127.0.0.1`这样的本地地址）：
![broker](https://user-images.githubusercontent.com/6218739/174427243-23b6211b-29f2-49e8-bec8-4ca2995aa4d6.png)
然后`mqtt out`的配置都使用如下的默认即可（因为它的`topic`和`payload`都是通过前面的`inject`节点传输过来）
![mqttout](https://user-images.githubusercontent.com/6218739/174427394-cc56d998-fbf6-4dc2-8bca-fcbd4d4144bb.png)
以上就是相当于建立了一个`mqtt`的发布服务，点击`inject`后就会对`EMQX`消息服务器发送指定主题的消息。
### 订阅消息
然后再配置`mqtt in`节点用来订阅消息，如下：
![mqttin](https://user-images.githubusercontent.com/6218739/174427603-ca2a14ef-fd15-4725-b7ea-8e4e445d1ddc.png)
它的`Server`默认与`mqtt out`的相同，以及`Topic`设置为`sensors/#`，其中`#`时一个通配符。

### 运行
点击`inject`后，就会发现在`Debug`边栏中出现：
```js
sensors/sensor1 : msg.payload : string[4]
"Open"
```
即，通过`mqtt out`客户端发送了主题为`sensors/sensor1`、内容为`Open`的消息给`EMQX`消息服务器，该服务器又将消息发送给了订阅了该类主题的`mqtt in`客户端。

### 升级
再把整个`flow`架构升级一下，如图：
![newflow](https://user-images.githubusercontent.com/6218739/174429564-a46b42f9-a03c-4e38-a158-4b7b381379e4.png)
即新增了一个`inject`节点，其配置为：
![close](https://user-images.githubusercontent.com/6218739/174429673-ce52d8ce-b4ab-4c0c-9f62-8a6ba244cae1.png)
当点击它时，`Debug`边栏中会出现`Close`消息。
这样就模拟了一个简单的传感器开关的效果。

### 其他
在上述例子中，`mqtt`的发送和订阅消息都是使用了`Node-RED`中的节点，更为普遍的一个情况是其他客户端或服务发送了相关的`topic`及其`payload`，然后在`Node-RED`中进行订阅（反之亦然），总之是订阅和发送的客户端不在一个地方。
一个尝试可以使用上面的`EMQ`开源的`MQTT X`软件，它是一款优雅的跨平台 MQTT 5.0 桌面客户端，支持 macOS, Linux, Windows。
`MQTT X` 的`UI`采用了聊天界面形式，简化了页面操作逻辑，用户可以快速创建连接，允许保存多个客户端，方便用户快速测试`MQTT/MQTTS`连接，及`MQTT`消息的订阅和发布。
下载地址在[这里](https://mqttx.app/zh)，安装后再与`EMQX`消息服务器进行连接。然后发送或者订阅`sensors/#`相关的主题，就可以实现与`Node-RED`中的`flow`的联动。


# 时序数据的可视化
这一部分介绍对时序数据序列进行采集，然后存储进入时序数据库中，并使用看板进行可视化。使用到的技术和软件分别为：通过`MQTT`来生成时序数据序列、通过`Node-RED`采集该时序数据序列、通过时序数据库`InfluxDB`进行存储、以及最后通过`Grafana`进行数据可视化。

主要参考教程在[这里](https://www.bilibili.com/video/BV1uW4y1C7jt)。

## 时序数据库InfluxDB
使用`docker`进行安装和部署，见[教程](https://zhuanlan.zhihu.com/p/459383022)：
```sh
docker run -d --name myinfluxdb --restart always -p 8086:8086 -v influxdb2:/var/lib/influxdb2 influxdb
```
其中的命令的含义为：
```sh
-d  启动后在后台运行，不打印日志
--name 容器名  给容器命名，方便管理，也可以不写此参数
--restart always  如果容器死掉，就自动将其拉起，也可以不写此参数
-v 宿主机路径:容器内路径  将容器内指定路径挂载出来到宿主机中，这里是把数据库本地存储的目录挂出来，保证容器销毁以后数据还在
-p 宿主机端口:容器内端口  将宿主机端口与容器内端口进行映射，influxdb默认的容器内端口是8086，容器外端口可以根据需要自己调整
```

如果下载很慢，可以先配置一下`docker`的国内镜像源加速，教程见[这里](https://yeasy.gitbook.io/docker_practice/install/mirror)：
在`/etc/docker/daemon.json`中写入如下内容（如果文件不存在请新建该文件）：
```sh
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://mirror.baidubce.com"
  ]
}
```
注意，一定要保证该文件符合json规范，否则Docker将不能启动。
之后重新启动服务：
```sh
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

`InfluxDB`自带一个图形化的管理后台，可以通过`http://127.0.0.1:8086`登录。
进入以后，配置一下初始用户（比如`root`)、密码、组织、`buckets`（比如`test`）。
进行如上的初始配置后，另一个非常重要的是`API Token`，它可以规定对数据库的操作权限，比如可读、可写等。

## Node-RED配置
首先安装一个名为`node-red-contrib-influxdb`的节点。
我们这里依然复用上面`mqtt`的例子，但是要做一定的改动。整体架构图如下：
![newflow](https://user-images.githubusercontent.com/6218739/174447253-efa5ead0-5bac-4e9b-8b0d-8d62a6efcdfd.png)
具体改动为：
（1）两个`inject`节点的`payload`的类型由原先的`string`都改为`number`，即原来的`Open`改为`1`、`Close`改为`0`，以数值形式存入时序数据库中，同时触发条件分别改为`每5秒触发一次`和`每3秒触发一次`，这样形成一个周期性的类似锯齿波形状的时间序列数据；
（2）`mqtt in`节点的`Output`属性更改为`a parsed JSON object`，如下图：
![newmqttin](https://user-images.githubusercontent.com/6218739/174447513-e936e93f-4ec2-420b-af10-50ecdba68ef7.png)
这样才能将正确的数据传输给时序数据库。
（3）如总架构图所述，新增一个节点`influxdb out`节点，这样才能将`Node-RED`与`InfluxDB`连接起来。对节点第一次配置时，需要在里面新建一个`Server`，具体配置如下图（仍然注意`IP`地址，不能使用类似`localhost`这样的，因为`InfluxDB`是通过`docker`部署的）：
![influxdbserver](https://user-images.githubusercontent.com/6218739/174447659-69abc04c-269d-4941-a0b2-99971a60e3be.png)
配置好`Server`后，然后再对具体连接的数据库进行配置，如下图所示：
![dataconnect](https://user-images.githubusercontent.com/6218739/174447756-cbf82931-77c6-44a1-9f2e-d370141f2c46.png)
其中`Organization`和`Bucket`就是之前通过后台创建的。`Meansurement`一项可以任意填写，其会在`test`这个数据库中新建此表。

部署后，就可以通过`InfluxDB`的管理后台看到数据的输入，如下图（注意箭头的地方）：
![influxdata](https://user-images.githubusercontent.com/6218739/174448258-e84dd2a1-9737-4be1-ba1c-8445753c4569.png)
这个看板只是`InfluxDB`后台的一个简单的可视化，没法用于真正的前端可视化（比如嵌入到网页中、免登录查看等）。因此还需要下面专业的`Grafana`进行可视化。


## Grafana可视化看板
通过`docker`安装和部署：
```sh
docker run -d --name=grafana -p 3000:3000 grafana/grafana-enterprise
```
通过`http://127.0.0.1:3000/`登录`Grafana`后台，默认的用户名和密码都是`admin`。
然后与具体的数据库相连接，通过`Add data source`，选择`InfluxDB`，连接它的方式与之前`Node-RED`中的类似，但是注意选择`Query Language`为`Flux`，然后不勾选`Basic auth`，才能出现`orgnization`、`token`、`Default Bucket`等选项。
然后再`Create new dashboard`，如下图：
![grafana](https://user-images.githubusercontent.com/6218739/174448938-5d219710-a860-447e-8fb4-fe1c74d0cce3.png)
这里就考验查询语句的功力了。对`Grafana`有需求的话可以仔细研究一下。