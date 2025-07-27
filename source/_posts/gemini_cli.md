---
title: Google Gemini CLI配置与试用
tags: [Google]
categories: Vibe Coding
date: 2025-7-27
---

[Gemini CLI](https://cloud.google.com/gemini/docs/codeassist/gemini-cli?hl=zh-cn)是Google推出的基于命令行的AI编程助手，特点在于它是免费的。

# 配置
## 代理配置
首先要保证命令行终端是可以访问Google的。
以使用Clash代理为例，找到Clash代理软件中的端口号，然后设置：
```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=$http_proxy
```
注意将这两个配置写入`.zshrc`或者`.bashrc`文件中，保证新开终端能默认执行。

使用如下命令测试是否能连通Google：
```bash
curl -vvk google.com
```

## 账号配置
需要配置Google Cloud Project账号信息。


## 

# 安装
运行命令：
```bash
npm install -g @google/gemini-cli
```

# 运行
在终端运行：
```bash
gemini
```


# 参考文件
[终端配置代理](https://weilining.github.io/294.html)

