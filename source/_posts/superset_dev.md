---
title: 顶级开源商业智能BI开发软件Superset————开发篇
tags: [Visualization]
categories: digitalization
date: 2021-9-6
---

[上一篇](https://qixinbo.info/2021/08/28/superset/)介绍了怎样搭建和运行superset，这一篇着重于怎样对superset进行特殊配置和二次开发。

# 更新并重新编译前端代码
前面已经说到，使用pip安装的是已经编译好的superset，无法修改前端源码，所以这里如果想对前端源码做改动，需要使用docker方式安装。
参考资料：
[superset/CONTRIBUTING.md](https://github.com/apache/superset/blob/master/CONTRIBUTING.md)

## 使用docker安装superset
这一部分可以参考[入门篇](https://qixinbo.info/2021/08/28/superset/)的docker安装部分。
但是需要万分注意的是，第一步拉取的docker镜像需要是dev分支，比如latest-dev，而不能是不带dev的分支，这里具体原理不清楚，但是我尝试时选择不带dev的分支，编译好多次都编译不成功。
```python
docker pull apache/superset:latest-dev
```

## 安装nodejs和npm
有两种方式，一种是使用nvm来管理：
（这个地方遇到一个坑，第一次使用了root账户安装了nvm，结果后面在使用npm时报“拒绝权限”错误，换用普通用户安装nvm后解决了）
```python
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.0/install.sh | bash
cd superset-frontend
nvm install --lts
nvm use --lts

npm install -g npm@7
```
另一种是手动安装nodejs和npm：
```python
wget https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz    // 下载，如果下载慢，就去nodejs.cn上找国内链接
tar xf  node-v10.9.0-linux-x64.tar.xz       // 解压
cd node-v10.9.0-linux-x64/                  // 进入解压目录
./bin/node -v                               // 执行node命令 查看版本

## 解压文件的 bin 目录底下包含了 node、npm 等命令，我们可以使用 ln 命令来设置软连接：
ln -s /usr/software/nodejs/bin/npm   /usr/local/bin/ 
ln -s /usr/software/nodejs/bin/node   /usr/local/bin/

# 设置npm镜像为国内的淘宝源
# 不过有时会显示该源里没有一些库，此时可以再切回官方源
# npm config set registry https://registry.npmjs.org/
npm config set registry http://registry.npm.taobao.org
```

## 安装相关依赖
```python
# 确保安装的时npm 7
npm install -g npm@7

# 进入前端文件夹
cd superset-frontend
# 从package-lock.json安装依赖
npm ci
```

## 编译资源文件
```python
npm run build
```
编译时遇到了如下问题：
```python
Error: EPIPE: broken pipe
```
多试几次。

## 需要注意的问题
（1）再强调一遍，拉取docker镜像时选择dev分支。
（2）superset有很多种编译的方式，上面用到的是编译成生产环境所需的资源，还有其他，比如：
```python
npm run build: the production assets, CSS/JSS minified and optimized
npm run dev-server: local development assets, with sourcemaps and hot refresh support
npm run build-instrumented: instrumented application code for collecting code coverage from Cypress tests
npm run build-dev: build assets in development mode.
npm run dev: built dev assets in watch mode, will automatically rebuild when a file changes
```
然而只尝试成功了第一个。。
（3）编译完后，要注意重启docker容器、强制刷新（不适用缓存）页面来使得修改生效。


# 开启Prophet时间序列预测算法
[Facebook 在2017年开源了一个叫fbprophet的时间序列预测的算法该算法支持自定义季节和节假日，解决了像春节、618和双十一这种周期性节假日的指标预测难题](https://www.modb.pro/db/50442)。prophet不仅可以处理时间序列存在一些异常值的情况，也可以处理部分缺失值的情形，还能够几乎全自动地预测时间序列未来的走势。而且Prophet包提供了直观易调的参数，即使是对缺乏模型知识的人来说，也可以据此对各种商业问题做出有意义的预测。
新版Superset（比如1.3版本）中对Time series这类图表（包括Line Chart、Area Chart、Bar Chart、Scatter plot）已经支持了prophet的调用，但是默认prophet包是不自动安装的，所以需要另外将prophet安装一下即可：
```python
pip install pystan==2.19.1.1
pip install prophet
```
然后在上面这些图表的explore中的Predictive Analytics进行开启。


# 配置PostgreSQL远程访问
（1）修改postgresql.conf
确保数据库可以接受来自任意IP的连接：
```python
listen_addresses = '*'
```
（2）修改pg_hba.conf
默认pg只允许本机通过密码认证登录，修改为以下内容后即可以对任意IP访问进行密码验证：
```python
host  all  all 0.0.0.0/0 md5
```
（3）配置防火墙端口
在防火墙的入站规则里添加一条规则，使外部能够访问数据库端口。
（如果通过路由器的话，还要在路由器中设置一下端口规则）
（4）重启PostgreSQL服务
在windows的services中重启服务。

# 权限控制
Superset初始化权限之后，会创建5个角色，分别为Admin、Alpha、Gamma、sql_lab以及Public（现在又新增了一个granter角色）。Admin，Alpha和Gamma角色，分配了很多的菜单/视图权限，如果手工去修改，改错的可能性很大，加之Superset并没有说明每一项权限的完整文档，所以不建议去修改这些角色的定义。灵活使用预置的角色，可以快速满足业务上安全控制需求。
角色权限介绍：
（1）Admin：拥有所有权限，包括授予或取消其他用户的权限，以及更改其他用户的切片和仪表板。
（2）Alpha：能访问所有数据源，增加或者更改数据源，但不能更改其他用户权限。
（3）Gamma：只能使用来自数据源的数据，这些数据源是通过另一个补充角色授予他们访问权限的。只能查看由他们有权访问的数据源生成的切片和仪表板，无法更改或添加数据源。还要注意，当Gamma用户查看仪表板和切片列表视图时，他们将只看到他们有权访问的对象。
（4）sql_lab：能访问SQL Lab菜单。请注意，虽然默认情况下管理员用户可以访问所有数据库，但Alpha和Gamma用户都需要根据每个数据库授予访问权限。
（5）Public：默认没有任何权限。允许已注销的用户访问某些Superset功能是可能的。

## 公开看板
目前分享看板时都是需要登录某个用户，而对权限控制进行更改后就能使得匿名用户或未登录用户也能正常查看看板。
首先在superset的配置文件config.py中更改Public角色的权限：
```python
PUBLIC_ROLE_LIKE = "Gamma"
```
即向Public角色授予与Gamma角色相同的权限。
然后：
```python
superset init
```
重新初始化角色和权限，使上述更改生效。
此时在后台可以看到public角色拥有了与Gamma相同的权限集。
然后对public权限手动新增如下权限：
```python
all datasource access on all_datasource_access
```
才能正常公开查看看板。

参考文献：
[Security](https://superset.apache.org/docs/security)
[Superset权限使用场景](https://cloud.tencent.com/developer/article/1031496)

# 多个数据源
## 在多个数据源中查询
默认superset只能在一张表中进行数据可视化。
可以通过superset的SQL Lab工具箱进行连接查询（JOIN关键字）、合并查询（UNION关键字）和多表查询（注意这种多表查询又称笛卡尔查询，使用时要非常小心，因为结果集是目标表的行数乘积）等。（但是仍然需要所有的tables都在同一个数据库的同一个schema中）
然后将查询到的结果explore存储为虚拟virtual数据集。该数据集能和physical的数据集一样地进行explore。
这个地方需要注意的是需要对database进行额外设置，编辑database，然后勾选“Allow DML”。
（在SQL Lab调试时注意查看报错信息）

可参考preset上的教程：
[https://docs.preset.io/docs/sql-editor](https://docs.preset.io/docs/sql-editor)
[https://docs.preset.io/docs/table-joins](https://docs.preset.io/docs/table-joins)

## 合并图表
如果想在一张图表内制作多张图表，比如将拥有相同x坐标轴的折线图和柱状图放在一块，可以使用Mixed Time-Series。

# 过滤器
可以通过添加过滤器Filter Box对数据进行筛选，比如对特定时间、特定数值、特定字段等。
方式就是从图表中选择Filter Box，并对其进行配置即可。
配置时，会对过滤器指定具体的Column。默认情况下，过滤器会作用在有该Column的所有数据源上（即使是来自于多种数据库、多张数据表，只要该Column相同就可被过滤。当然某个数据集的该Column需要设定为filterable）。
默认情况下过滤器也会作用在dashboard中的所有图表上，可以配置哪些图表使用这些过滤器，有两种方式设置：
一种是图形化设置，在Edit Dashboard的Set Filter Mapping中就可以对具体哪个过滤器作用在哪个图表上进行配置，推荐这种方式，简单方便；
另一种是通过参数设置，在dashboard的JSON元数据配置中：
```python
{
    "filter_immune_slices": [324, 65, 92],
    "filter_immune_slice_fields": {
        "177": ["country_name", "__time_range"],
        "32": ["__time_range"]
    },
    "timed_refresh_immune_slices": [324]
}
```
那些数字就是图表在该看板中的slice id，这些id可以通过Export Dashboard（回到总的dashboards页面）形成的json文件中查看。

# 实时数据更新
Superset可以设置以多长时间刷新看板，在"Set Auto Refresh Interval"中，默认有10s、30s、1min等，此处最小间隔为10s，最大间隔为24hours。
虽然从该入口处可选择的时间间隔种类有限，但可以在此处随意选择一个间隔，然后再在该看板的JSON元数据中进行更加细致的设置：
```python
{
  "timed_refresh_immune_slices": [86, 89, 165, 166, 172, 191],
  "refresh_frequency": 1
}
```
比如上面将刷新频率设置为1s，同时设置哪些slices可以不刷新（默认是所有图表都刷新）。
但是需要注意的是，目前版本的superset（1.3.0）中每刷新一次，都会在界面上显示：
```python
This dashboard is currently force refreshing ....
```
这样的弹窗，非常影响观感。
目前github上也有一个issue提到了该问题，开发人员正在修改该部分的设计：
[Make possible to disable the annoying "This dashboard is currently force refreshing" message #13242](https://github.com/apache/superset/issues/13242)
如果等不到新的版本，可以手动修改如下地方的源码：
[superset/superset-frontend/src/dashboard/components/Header/index.jsx](https://github.com/apache/superset/blob/master/superset-frontend/src/dashboard/components/Header/index.jsx)

以下是Youtube上一个演讲，使用Kafaka消息队列+Druid数据存储+Superset可视化的方案：
[Interactive real-time dashboards on data streams using Kafka, Druid, and Superset](https://www.youtube.com/watch?v=HOk7WtxBMzM)

# 查看stl模型文件
stl文件是一种常用的描述三维物体的文件格式。superset没法直接展示这种文件格式，不过这里可以采用iframe的方式嵌入外部的stl阅读器来实现。
找了一圈后，发现viewstl这个网站提供优雅的渲染stl文件的功能，同时能免费嵌入其他网页中。网址见：
[View 3D STL files directly in your browser - no software installation is required](https://www.viewstl.com/)
具体的嵌入功能见：
[https://www.viewstl.com/embed/](https://www.viewstl.com/embed/)
进行一番配置后，复制其产生的代码即可，比如：
```python
<iframe id="vs_iframe" src="https://www.viewstl.com/?embedded" style="border:0;margin:0;width:100%;height:100%;"></iframe>
```
其中的stl文件可以由本地手动选择、本地服务托管、外部文件加载等多种方式。
本地服务托管就是本地建一个服务器放置stl文件，然后把内网地址作为参数传入即可，这样可以解决有时文件不能传到外网上这种问题。
外部文件加载需要提供文件URL地址，测试了几个网盘，比如google drive、百度网盘等，都有这样那样的问题无法解析，这里推荐使用阿里云的OSS存储服务。


# deck.gl
deck.gl是由uber开发并开源出来的基于WebGL的大数据可视化框架。它具有提供不同类型可视化图层、GPU渲染的高性能，React和Mapbox GL集成展示地理信息数据（GPS）等特点。

# Mapbox
Mapbox是一个开源的地图制作系统，superset也与其进行了良好的集成，只需一个token，就可在superset中进行地图相关的操作。
获取token只需在Mapbox官网上注册一个账号，然后在superset的配置文件中设置环境变量即可（或者自己export或set该环境变量）：
```python
MAPBOX_API_KEY = "your token"
```

# 内置地图
superset也内置了两种地图：
（1）World Map：可以显示各个国家相关数据，国家代码可以有四种形式，比如Full name、code International Olympics Committee、code ISO 3166-1 alpha-2和code ISO 3166-1 alpha-3；
（2）Country Map：可以显示某个具体国家的省市的相关数据，具体省市的代码需要遵循ISO 3166-2标准。

# CSS模板
superset可以通过CSS方便地改变整个dashboard的样式。
CSS的语法可以通过如下教程快速上手：
[菜鸟CSS教程](https://www.runoob.com/css/css-tutorial.html)
具体改变哪个元素或哪个类的样式，可以通过浏览器的Inspect检查功能，定位到想改变的元素上，然后通过临时修改其css查看效果，再在superset中添加css模板配置。

# Markdown
Markdown组件可以允许添加Markdown语法的资源，以及html格式的资源。
比如可以加入图片、超链接、文本等。

# 最大化看板
可以通过"Enter Fullscreen"来使看板最大化，但这种方式仍然会存在看板的标题栏。
通过在看板链接中配置standalone这个参数等于2，则可以将标题栏也给去掉，即：
````python
http://localhost:5000/superset/dashboard/al-lca/?standalone=2
```
