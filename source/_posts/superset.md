---
title: 顶级开源商业智能BI开发软件Superset————入门篇
tags: [Visualization]
categories: digitalization
date: 2021-8-28
---

# 介绍
Apache Superset是一个现代的、企业级的商业智能（Business Intelligence）网络应用程序，它使得用户可以使用无代码可视化构建器和SQL编辑器来轻松探索和可视化自己的数据。
其最初由Airbnb开源，后来由Apache进行孵化，并且于今年（2021年）1 月 21 日宣布毕业并成为 Apache 软件基金会（ASF）的顶级项目（Top-Level Project），截止到现在（2021年8月25日）已经在GitHub上收获了超过4万颗star。
官网地址在[这里](https://superset.apache.org/)。
示例看板在[这里](https://superset.apache.org/gallery)。
有一句评价非常中肯：[对开发人员最大的吸引力在于：支持的数据源足够多，界面足够花里胡哨！](https://xie.infoq.cn/article/ff7e60ae303e0de531f0b4bf5)。

# 安装
有多种方式安装superset，比如使用docker、使用pip安装等方式。
使用docker安装是最简单的一种方式，因为它已经将相关依赖都做成了一个镜像，同时其包含了github上的源码，有最大的自由度可供开发。
使用pip安装也较为方便，但是pip包本质上是一个已经编译好的包，没法修改源码，尤其是没法修改前端ui相关的代码。
下面会介绍两种安装方式，这两种方式都试验过，能正常运行。因为我对docker命令不太熟悉，后面的操作（比如配置数据库等）都是基于pip安装方式。
## docker安装
### 安装docker软件
可以参考此处的[教程](https://www.runoob.com/docker/windows-docker-install.html)。
### 拉取superset镜像
```python
docker pull apache/superset
```
### 使用镜像
（1）开启一个superset实例：
```python
docker run -d -p 8080:8088 --name superset apache/superset
```
（2）初始化实例：
```python
# 配置管理员账号
docker exec -it superset superset fab create-admin \
               --username admin \
               --firstname Superset \
               --lastname Admin \
               --email admin@superset.com \
               --password admin

# 迁移数据库
docker exec -it superset superset db upgrade

# 加载实例
docker exec -it superset superset load_examples

# 初始化
docker exec -it superset superset init
```
（3）登录：
在浏览器中的地址为：
```python
http://localhost:8080/login/ 
```

## pip安装
### 创建虚拟环境
使用virtualenv或Conda。
使用虚拟环境主要是为了安装环境的独立性，防止里面的库的版本混乱。这一步不详述了。

### 安装必要的包
大部分的包都能自动下载，但是下面这两个有可能会在自动安装时出现错误，导致整个安装出错（我在windows平台上安装时遇到了这两个问题）。
建议是自动安装，如果出错，再手动安装一下看看是不是这两个出现的问题。
（1）安装Sasl:
下载Sasl的wheel文件:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#sasl
然后：
```python
pip install
```
（2）安装python-geohash package:
下载wheel包，然后pip install。
https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-geohash

### 安装Superset
```python
# 最好确保一下superset是最新版
# 第一次安装时1.1.0版本有个注释层的bug
# 更新到1.3.0版本后就好了
pip install apache-superset
```

### 初始化数据库
```python
superset db upgrade
```

### 配置superset
```python
# 将Flask默认的app设置为superset，这样flask就能找到它
export FLASK_APP=superset # 在windows上就是set命令
# 创建管理员账户
superset fab create-admin
# 加载一些示例看板
superset load_examples
# 初始化superset
superset init
```

### 启动superset
```python
superset run -p 8088 --with-threads --reload --debugger
```


# 连接数据库
Superset本身不提供数据库，其需要连接已有的数据库来作为数据存储的容器。
Superset支持各种数据库，包括MySQL，Presto，Hive，Postgres，Dremio，Snowflake，Teradata和其他数PB级的。由于Superset后端是用Python编写的，因此本质上是Python后端的Flask应用程序……在Python中，所有数据库都有很多驱动程序支持。
这里我们选用PostgreSQL数据库作为后端。

## 安装PostgreSQL
可以通过下面的链接进行下载安装：
[PostgreSQL Database Download](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)
里面自带了pgAdmin图形管理工具来操作PostgreSQL数据库。

## 创建数据库
安装好pgAdmin后，再通过它来手动创建一个自己的数据库，用于后续存储数据。
具体可以参考如下教程：
[PostgreSQL 创建数据库](https://www.runoob.com/postgresql/postgresql-create-database.html)
特别注意的是该数据库的用户名username、密码password、主机地址host（本机就是localhost）、端口号port（默认是5432）和名称database。

初次创建后该数据库就直接跑起来后，但后面电脑关机后，有可能出现明明信息都正确，但是启动不起来的问题，比如出现下面这个问题：
```python
Is the server running on host "localhost" (::1) and accepting TCP/IP connections on port 5432?
```
这是因为后台的数据库服务没有启动。解决方法是在windows的Services中找到postgresql-x64-13这个服务，然后启动它。

## 安装数据库驱动
首先需要安装一个额外的库：
```python
pip install psycopg2
```

## 连接
在上面的启动的superset的web页面中，选择添加一个数据库，然后根据PostgreSQL的连接语法与前面创建的数据库进行连接，语法格式为：
```python
postgresql://{username}:{password}@{host}:{port}/{database}
```
然后点击“测试连接”，连接成功后即表明可以正确添加该数据库。

## 数据集
有了底层数据库，还需要提取里面的数据。
对于本来就存在数据的数据库，可以在superset的“数据集”中进行添加选择，按照提示进行相关操作就行。
对于初次创建的数据库，里面是空的，没有任何的数据。此时可以通过上传csv文件进行添加，这样既在superset中添加了数据集，也在底层PostgreSQL数据库中添加了数据。
开启上传csv功能需要首先在数据库中进行设置，在superset的某个数据库的Extra/扩展选项卡中勾选“Allow Data Upload”/“允许数据上传”。
然后再在“数据”菜单中选择“上传CSV文件”。

额外福利：如果手头没有可玩的数据，可以通过下面三个链接获取一些示例数据（第三个时superset教程中的示例数据）：
[https://github.com/plotly/datasets](https://github.com/plotly/datasets)
[https://github.com/fivethirtyeight/data](https://github.com/fivethirtyeight/data)
[https://github.com/apache-superset/examples-data](https://github.com/apache-superset/examples-data)

导入数据集后，可以对数据集的属性进行配置，比如哪一列是时间条件、是否可被过滤等。
需要注意的是superset对数据集加了一个语义层semantic layer，它存储了两种类型的计算数据：
（1）虚拟指标：对应Metrics这一标签页，可以编写不同列之间的聚合SQL查询，然后使得结果作为“列”来使用。这里可以使用并且鼓励使用SQL的聚合函数，如COUNT、SUM等；
（2）虚拟计算列：对应Calculated Columns这一标签页，可以对某一特定的列编写SQL语句来定制它的行为。在这里不能使用SQL的聚合函数。 

# 可视化数据
Superset有两种探索数据的方式：
（1）Explore：零代码可视化编辑器，只需选择数据集，选定相应图表，配置一些外观属性，然后就可以创建可视化图表；只需点击相应的数据集，就可以进入Explore模式；Save Chart时可以选择添加到新看板或者某一个已存在的看板。
（2）SQL Lab：SQL工具箱，可以提供强大的SQL语言编辑功能，用于清洗、联合和准备数据，可以用于下一步的Explore流程。

superset的官方教程中给出了一个详细地Explore模式的使用教程，其使用的示例数据来自以下链接：
[flights](https://github.com/apache-superset/examples-data/blob/master/tutorial_flights.csv)
强烈建议根据官方教程一步步走一遍，教程在[这里](https://superset.apache.org/docs/creating-charts-dashboards/exploring-data)。
这里列举一下自己跑教程时踩的一些坑：
（1）上传CSV文件时，一定要在“Parse Dates”解析日期那里手动填上“Travel Date”，否则如果不明确指定时间的话，在数据库里存的该项的数据类型是Text，无法进行后面的时间序列的计算。
（2）在添加“指标Metrics”时，保存的指标指的是“编辑数据集”时的“指标Metrics”那个选项卡的指标。
（3）配置“分组Group by”时，选择“Time”这一项，就会自动使用之前在Time那块定义的时间列、时间粒度等。
（4）添加“注释层annotation layer”那一块时，注意使用最新版的superset，已测试1.1版本会有bug，1.3版本已经修复该bug，见该[PR](https://github.com/apache/superset/pull/13969)。
（5）在Advanced Analystics一项中，有对时间序列数据的更强大的操作，比如求平均、时间平移、使用python函数重新采样等操作。

# 其他
superset开发团队基于开源的superset推出了SaaS云服务Preset，可以使得用户在无需安装任何软件的情况下直接使用superset。
Preset除了提供开箱即用的superset，其官网上的教程也比superset官网上的要详细很多，所以可以参考preset的文档来学习superset，如下：
[https://docs.preset.io/](https://docs.preset.io/)
