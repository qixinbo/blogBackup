---
title:  hexo博客在不同电脑间迁移记录
tags: [blog]
categories: coding
date: 2021-3-25
---

# 说明
在最近一次在不同电脑间迁移时，使用了[`Vercel`](https://vercel.com/)这个站点托管工具，有如下几个优点：
- 完全不需要像下面那样在本地搭建开发环境，只需`git clone`源码到本地即可
- 对源码的更改会自动触发其对站点的部署
- 对大陆的访问友好，速度快

具体的使用教程可以参考[该博文](https://www.zywvvd.com/notes/hexo/website/31-vercel/vercel-deploy/)。

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

## LaTeX公式渲染
首先卸载原有的渲染器，然后安装下面的其中一个（实测pandoc不行，而应该安装kramed）：
```python
npm uninstall hexo-renderer-marked
npm install hexo-renderer-pandoc # 或者 hexo-renderer-kramed
```
然后在next主题的配置文件中，进行配置：
```python
# Math Formulas Render Support
math:
  # Default (true) will load mathjax / katex script on demand.
  # That is it only render those page which has `mathjax: true` in Front-matter.
  # If you set it to false, it will load mathjax / katex srcipt EVERY PAGE.
  per_page: false # 这个per_page非常重要，如果为True，那么就只渲染带`mathjax: true`的页面，设为False则渲染所有页面，所以可能速度会慢。

  # hexo-renderer-pandoc (or hexo-renderer-kramed) required for full MathJax support.
  mathjax:
    enable: true
    cdn: //cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML
    # See: https://mhchem.github.io/MathJax-mhchem/
    mhchem: false

  # hexo-renderer-markdown-it-plus (or hexo-renderer-markdown-it with markdown-it-katex plugin) required for full Katex support.
  katex:
    enable: false
    cdn: //cdn.jsdelivr.net/npm/katex@0/dist/katex.min.css
    # See: https://github.com/KaTeX/KaTeX/tree/master/contrib/copy-tex
    copy_tex: false
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

## 连接不上github
在执行
```python
hexo d -g
```
出现了一次连接不上github的问题，但在该终端下直接：
```python
git push
```
却是可以的，这可能跟wifi有关。。也可能跟代理有关。
反正是换了一个wifi，关闭代理，就神奇地好了。。
[这篇博客](https://java4all.cn/2020/01/24/github%E7%9A%84%E4%B8%80%E4%B8%AA%E5%9D%91/)也遇到了这个问题。

# 参考文献
[超详细Hexo+Github博客搭建小白教程](https://zhuanlan.zhihu.com/p/35668237)
[GitHub+Hexo 搭建个人网站详细教程](https://zhuanlan.zhihu.com/p/26625249)
[为Hexo添加RSS订阅](https://hasaik.com/posts/19c94341.html)
[给 Hexo 中的 Next 主题添加 RSS 功能](https://suyin-blog.club/2020/2M3YWE7/)
[hexo博文摘要生成方案](https://ninesix.cc/post/hexo-yilia-auto-excerpt.html)

