---
title: 使用Heroku云平台免费托管web应用
tags: [Docker, Cloud]
categories: coding 
date: 2022-6-12
---

# 简介
当开发了一款`web`应用后，需要使用服务器将其托管以供用户进行访问。服务器的选择多种多样，可以选择自己搭建服务器，也可以选择如今大火的各种云服务器。
虽说现在云服务器的价格已经非常亲民（尤其是对于新用户的各种优惠政策），但毕竟还是需要真金白银的付出，尤其是考虑自己开发的应用可能只是为了大家尝试的情形下，此时可以选择一些提供免费部署的云平台。
[`Heroku`](https://www.heroku.com/)是一个非常优秀的`PaaS`(`Platform as a Service`)云平台，它有如下几个优点：
（1）自动软件部署：当软件的代码变动后，通过`git`进行代码追踪后，就可以自动触发软件的部署交付；
（2）无需关心背后的基础设施：因为`Heroku`是一个`PaaS`平台，而不是`IaaS`(`Infrastructure as a Service`)平台，所以它屏蔽了很多的细节，比如使用的操作系统、运行环境等。用户写好配置文件后，自动搭建应用程序所需的环境；
（3）免费额度：对于小型应用来说，免费版已足够，但是有不少限制，比如应用在`30`分钟内无访问的话会自动休眠，再次有访问时会被唤醒，可能会有`10`多秒的延迟；每个月限制在`550`小时的免费运行时长；提供给用户的`Postgres`数据库存储的数据不能超过`10000`行等。

# 注册
登录[heroku网站](https://signup.heroku.com/)注册一个账号。

# 下载并安装Heroku工具
注意，需要电脑上提前安装git。
## 下载
Heroku工具的[下载地址](https://devcenter.heroku.com/articles/heroku-cli)。
## 验证
安装后进行验证：
```python
heroku --version
```
## 登录
登录账号：
```python
heroku login
```

# 创建或使用已有git仓库
在项目中创建git仓库，或使用已有的git仓库：
```python
cd my-project/
git init
```
# 创建Heroku App
## 新建或连接Heroku App
在命令行中新建Heroku App：
```python
heroku create my-project-test
```
或者在Heroku网站上已创建了App，此时连接即可：
```python
heroku git:remote -a my-project-test
```

# 操作git仓库
对于项目文件的更新，使用常规的git命令即可：
```python
git add .
git commit -am "make it better"
```


# 部署到Heroku
```python
git push heroku master
```
如果涉及到`master`分支更名为`main`分支，可以查看[该教程](https://help.heroku.com/O0EXQZTA/how-do-i-switch-branches-from-master-to-main)。

## Vue项目部署
特别地，对于`Vue.js`开发的项目，要注意以下几点：
（[教程1](https://cli.vuejs.org/guide/deployment.html#heroku), [教程2](https://medium.com/unalai/%E8%AA%8D%E8%AD%98-heroku-%E5%AD%B8%E7%BF%92%E5%B0%87-vue-%E5%B0%88%E6%A1%88%E9%83%A8%E7%BD%B2%E8%87%B3heroku-4f5d8bd9b8e2)）
### 文件打包
```js
npm run build
```
### git加入dist路径
在`.gitignore`文件中删除`/dist`条目，使得该路径可以被`git`监控。
### 新增文件
新建`static.json`文件：
```js
{
  "root": "dist",
  "clean_urls": true,
  "routes": {
    "/**": "index.html"
  }
}
```
注意将上述改动都通过`git add`和`git commit`提交。

### 新增Heroku指令
```js
heroku buildpacks:add heroku/nodejs
heroku buildpacks:add https://github.com/heroku/heroku-buildpack-static
```
### 部署
```js
git push heroku master
```

## Python项目部署
### 添加python版本指定文件
新建`runtime.txt`文件，然后在里面写上`python`版本，如：
```python
python-3.7.13
```
不过`python`版本与`heroku`的版本需要对应，具体查看[这里的说明](https://devcenter.heroku.com/articles/python-support)。

### 添加依赖需求文件
```python
pip freeze > requirements.txt
```

### 添加Procfile
新建`Procfile`文件，里面包含的是Web应用服务器启动时执行的命令，比如：
```python
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
```
一篇很好的教程见[这里](https://towardsdatascience.com/how-to-deploy-your-fastapi-app-on-heroku-for-free-8d4271a4ab9)。

### 问题汇总
（1）遇到`Your account has reached its concurrent build limit`这个错误，可以：
```python
heroku restart
```
参考[这个问题](https://stackoverflow.com/questions/47028871/heroku-your-account-has-reached-its-concurrent-build-limit)。
（2）项目不能太大，免费版实测压缩后最大是`500M`。
（3）如果用到`opencv`，需要这样配置：
首先添加指令：
```python
heroku buildpacks:add --index 1 heroku-community/apt
```
然后新建`Aptfile`，内容为：
```python
libsm6
libxrender1
libfontconfig1
libice6
```
参考[该教程](https://medium.com/analytics-vidhya/deploying-your-opencv-flask-web-application-on-heroku-c23efcceb1e8)。

# 启动App
在`Heroku`网站上的该`App`的`settings`中找到该`App`的网站`url`。
或者通过以下命令：
```js
heroku open
```
即可直接打开App。
以后即可通过该网址访问该`App`。