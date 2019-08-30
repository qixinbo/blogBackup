---
title: Git配置单机器多账号配置
tags: [git,linux]
categories: linux
date: 2019-8-30
---

参考资料：
[Windows下配置多个git账号的SSH Key](https://www.jianshu.com/p/d195394f7d2e)
[Git的多账号如何处理？](https://gist.github.com/suziewong/4378434)
[同一台电脑配置多个git账号](https://github.com/jawil/notes/issues/2)

有以下两种场景需要进行区分。
# 多个账号+同一邮箱
对于 Git 而言，邮箱是识别用户的唯一手段。因此如果在不同的代码托管服务商（GitHub、GitLab或Bitbucket）中使用同一邮箱作为账号，此时不需要担心密钥的问题，因为这些网站push pull 认证的唯一性是邮箱。此时只需生成一个通用的私钥和公钥对即可：

```shell
ssh-keygen -t rsa -C "simba@gmail.com"
```
此时会在用户目录的.ssh/ 下生成两个文件，id_rsa 是私钥，id_rsa.pub 是公钥，然后登陆服务器（如GitHub），将公钥中的内容添加进去即可。

# 多个账号+不同邮箱
此时原理上就是对 SSH 协议配置 config 文件，对不同的域名采用不同的认证密钥。
## git config 介绍
Git有一个工具被称为git config，它允许你获得和设置配置变量；这些变量可以控制Git的外观和操作的各个方面。这些变量可以被存储在三个不同的位置：
（1）/etc/gitconfig 文件：包含了适用于系统所有用户和所有库的值。如果你传递参数选项’--system’ 给 git config，它将明确的读和写这个文件。
（2）~/.gitconfig 文件 ：具体到你的用户。你可以通过传递 ‘--global’ 选项使Git 读或写这个特定的文件。
（3）当前项目Git目录的 config 文件 (也就是 .git/config) ：无论你当前在用的库是什么，特定指向该单一的库。每个级别重写前一个级别的值。因此，在 .git/config 中的值覆盖了在/etc/gitconfig中的同一个值，可以通过传递‘--local’选项使Git 读或写这个特定的文件。

## 为不同邮箱账号生成不同的公钥和私钥
```shell
ssh-keygen -t rsa -f ~/.ssh/id_rsa.github -C "simba@example.com"
```
一定注意加上-f选项明确指定私钥文件名称，防止覆盖默认生成的id_rsa文件；或者此时不加，但在下面的生成过程中仔细写上。
然后将公钥内容上传到特定的服务器上。

## 配置config文件
在.ssh/目录下的config文件（如果没有，则新建）中，写入：
```shell
Host github.com
    HostName github.com
    IdentityFile C:\Users\xxx\.ssh\id_rsa.github
    PreferredAuthentications publickey
```
其中每项的意义为：
```shell
Host    　    #　主机别名
HostName　    #　服务器真实地址
IdentityFile　#　私钥文件路径
PreferredAuthentications　#　认证方式
User　        #　用户名
```
每个不同的账号都配置一个这样的ssh-key。

## 为具体的git项目设置账号信息
首先需要先git clone下来具体的git项目（或者git init新建项目），然后在项目下：
```shell
git config  user.email "xxxx@xx.com"
git config  user.name "xxx"
```
