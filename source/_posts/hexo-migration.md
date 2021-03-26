---
title:  hexo博客在不同电脑间迁移记录
tags: [blog]
categories: coding
date: 2021-3-25
---

# 简介
该博客是基于hexo搭建的，部署在github pages里，用netlify加速。之前一直用自己的笔记本写博客，现在需要换用另一台电脑，因此需要在新电脑上将环境重新搭建一遍，顺便对hexo及其next主题进行升级。

# 安装Node.js
下载地址见[这里](https://nodejs.org/en/download/)。
然后正常安装。
安装完成后，输入node -v和npm -v，如果出现版本号，那么就安装成功了。
# 安装git
下载地址见[这里](https://git-scm.com/download/win)。
然后正常安装，只不过最后一步添加路径时选择Use Git from the Windows Command Prompt，这样我们就可以直接在命令提示符里打开git了。
安装完成后在命令提示符中输入git --version验证是否安装成功。
# 安装hexo
新建一个文件夹，如Blog，然后安装hexo：
```python
npm i hexo-cli -g
```
安装完成后输入hexo -v验证是否安装成功。
还要安装用于hexo的git部署插件：
```python
npm install hexo-deployer-git --save
```
# 初始化博客
```python
hexo init
```
# 测试
```python
hexo g
hexo s
```
# 安装next主题
```python
git clone https://github.com/theme-next/hexo-theme-next themes/next
```
# 迁移
离开该目录，然后将备份在github上的博客仓库下载下来：
```python
git clone https://github.com/qixinbo/blogBackup.git
```
复制该blogBackup文件下的以下文件及文件夹到Blog文件夹下，并覆盖原始文件：
```python
_config.yml文件
theme/next下的_config.yml文件
source文件夹
.git文件夹
将next/source/images下的wechat_reward复制到相应位置
```
注意，由于next主题时有大版本更新，原有的配置可能不适用于新版本，此时直接覆盖可能会出错，解决方法只有对照两个配置文件，然后手工更改配置。

# 重新备份
在Blog文件夹下重新git备份（因为.git文件夹已经自带了配置信息，这里无需再次配置）：
```python
git add .
git commit -m "migration"
git push
```
# 重新部署
```python
hexo d -g
```
如果出现实际效果与本地不符，可以尝试清理缓存：
```python
hexo clean
```

# 其他可能问题
## 添加rss
首先安装必要插件：
```python
npm install hexo-generator-feed --save
```
然后在next主题配置文件中添加：
```python
#订阅RSS
feed:
  type: atom
  path: atom.xml
  limit: false
```
并且增加RSS字段：
```python
follow_me:
  RSS: /atom.xml || fa fa-rss
```

## 首页自动生成摘要
首先安装必要插件：
```python
npm install --save hexo-auto-excerpt
```
然后在next的主题配置文件中添加：
```python
auto_excerpt:
  enable: true
  length: 150
```

## 修改dns服务器
将hexo博客部署在netlify上，可以充分利用netlify的cdn加速。这里需要将dns解析服务器由原来的dnspod改为netlify。
首先需要在netlify上对域名开启netlify的dns服务，这一步在网页上可以很方便的操作。
然后在godaddy（这是域名服务商）上设置nameservers：
原来是dnspod家的：
```python
F1G1NS1.DNSPOD.NET
F1G1NS2.DNSPOD.NET
```
现在改为netlify家的：
```python
dns1.p04.nsone.net
dns2.p04.nsone.net
dns3.p04.nsone.net
dns4.p04.nsone.net
```
改完后无论是境内还是境外，速度飞起~~

# 参考文献
[超详细Hexo+Github博客搭建小白教程](https://zhuanlan.zhihu.com/p/35668237)
[GitHub+Hexo 搭建个人网站详细教程](https://zhuanlan.zhihu.com/p/26625249)
[为Hexo添加RSS订阅](https://hasaik.com/posts/19c94341.html)
[给 Hexo 中的 Next 主题添加 RSS 功能](https://suyin-blog.club/2020/2M3YWE7/)
[hexo博文摘要生成方案](https://ninesix.cc/post/hexo-yilia-auto-excerpt.html)

