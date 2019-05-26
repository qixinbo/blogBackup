---
title: 命令行终端设置代理上网（pac文件）
tags: [Proxy]
categories: programming
date: 2019-5-26
---

==== 2019.5.26更新：增加socks代理 ===
一般我们使用SSR来代理，所以这里增加这种方式的设置。
首先，还是需要先设置好SSR，这部分不详述。
然后，终端配置的命令是：
```cpp
export ALL_PROXY=socks5://127.0.0.1:1080
```

有时这样设置好了以后，通过pip下载时又报如下错误：
```python
Missing dependencies for SOCKS support
```
此时，可以：
```python
pip install pysocks
```

# 事件起因
之前公司给了一个pac文件，可以实现对国外网站的访问，具体加载该文件的过程也非常简单，即：
- 在IE浏览器中通过「Internet选项 -> 连接 -> 局域网设置 -> 使用自动配置脚本」，在地址栏中填写PAC文件的URI，这个 URI 可以是本地资源路径(file:///)，也可以是网络资源路径(http:// )。
- 在Chrome中则是「chrome://settings/ -> 显示高级设置 -> 更改代理服务器设置」。

但问题来了：这样的设置可以在浏览器中实现科学上网，对于命令行终端仍然是老样子。于是，需要对终端进行一番配置才行。

# 事件经过

## 方式1：直接设置代理IP地址
从网上搜索“Windows 终端 代理”，很容易就找到很多教程，搜索结果如下：
[Google "Windows 终端 代理"](https://www.google.com/search?newwindow=1&rlz=1C1GCEV_enCN824US824&ei=lVLRXOfQDon5wAKc0KKIAw&q=windows+%E7%BB%88%E7%AB%AF+%E4%BB%A3%E7%90%86&oq=windows+%E7%BB%88%E7%AB%AF+%E4%BB%A3%E7%90%86&gs_l=psy-ab.12...0.0..68252...0.0..0.0.0.......0......gws-wiz._htvS52bUVQ)

大家给出的最常用的方法就是如下：
```python
set http_proxy=http://127.0.0.1:1189
set https_proxy=http://127.0.0.1:1189
```
即通过对如上两个环境变量设置代理IP地址和端口号来实现代理。
这种方式大家都给出了，说明大家都通过它成功了，但我这里没有IP地址和端口号，只有pac文件。。于是此法对我不适用（实际是适用的，只是我是网络小白，没意识到。。），还得继续探索。

## 方式2：继承IE的代理配置
一番搜索后，发现知乎上有一个回答，见如下链接：
[如何对windows的command(命令行)设置代理IP？ - 潘小潘的回答 - 知乎](https://www.zhihu.com/question/23059121/answer/130382105)
答主给出的解决方法是：
> 先设置ie代理IE -> Settings -> Internet options -> connections -> LAN,
> 然后以管理员身份打开命令行执行如下命令：
> netsh winhttp import proxy source=ie
> 然后会提示如下信息：
> 当前的 WinHTTP 代理服务器设置:
> 代理服务器: http://xxx.sss.com:8080
> 绕过列表 : <-loopback>;*.ddd.com
> =======如何取消代理=========
> netsh winhttp reset proxy

并且答主在评论里也说到：
> 这个主要是解决cmd使用ie 中设置的代理。尤其是使用pac文件设置的自动代理

但为什么我照做了，仍然不行！！说是“no proxy”。。。
所以这个方法对我也不适用，不过这里也放上了，没准正在看该文的你是个有缘人。。

## 方式3（正解）：解析pac文件
在搜索的过程中，发现了一片对pac文件的讲解，发现它就是个JavaScript函数！！正好前几天学了一下JavaScript，说明此时我是有缘人。。有缘的讲解链接如下：
[如何使用PAC文件“科学上网”](https://exp-team.github.io/blog/2017/01/13/tool/using-pac/)
可以看出，pac文件就是为了让代理能够有意识地判断何种情况去怎样代理，它的主要三个归处有：

> DIRECT 直接联机而不透过 Proxy
> PROXY host:port 使用指定的 Proxy 伺服机
> SOCKS host:port 使用指定的 Socks 伺服机

哈哈，这里就出现了代理IP地址和端口号！接下来就很简单了，就是找到公司给的pac文件中PROXY后面的那个host和port（不过也要仔细排查，因为有很多if语句，那么该if语句后面的PROXY就不是想要的那个通用的proxy），然后再用方式1进行设置即可！
按说按上面的来就行了，但这里还很有必要地再加一个小节，因为有个巨坑。。

## 方式3 Plus （更完美的正解）
我第一次按方式3设置后，还是没生效。。。:( :(
为啥呢，答案见下面链接：
[给 Windows 的终端配置代理](https://zcdll.github.io/2018/01/27/proxy-on-windows-terminal/)
具体为：
> cmd 中用 set http_proxy 设置
> Git Bash 中用 export http_proxy 设置
> PowerShell 中按照这样设置
```python
# NOTE: registry keys for IE 8, may vary for other versions
$regPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'

function Clear-Proxy
{
    Set-ItemProperty -Path $regPath -Name ProxyEnable -Value 0
    Set-ItemProperty -Path $regPath -Name ProxyServer -Value ''
    Set-ItemProperty -Path $regPath -Name ProxyOverride -Value ''

    [Environment]::SetEnvironmentVariable('http_proxy', $null, 'User')
    [Environment]::SetEnvironmentVariable('https_proxy', $null, 'User')
}

function Set-Proxy
{
    $proxy = 'http://example.com'

    Set-ItemProperty -Path $regPath -Name ProxyEnable -Value 1
    Set-ItemProperty -Path $regPath -Name ProxyServer -Value $proxy
    Set-ItemProperty -Path $regPath -Name ProxyOverride -Value '<local>'

    [Environment]::SetEnvironmentVariable('http_proxy', $proxy, 'User')
    [Environment]::SetEnvironmentVariable('https_proxy', $proxy, 'User')
}
```

我是用的方式1中的set命令，但我用的终端是Cygwin，虽然Cygwin认识了set，但就是没生效。于是我灵机一动换成了Windows官方的终端cmd。。。
Love and Peace!
