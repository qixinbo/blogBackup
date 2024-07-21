---
title: 使用LazyVim将Neovim打造成强大IDE
tags: [IDE]
categories: coding
date: 2024-7-21
---

# 介绍
之前写过[一篇使用NvChad来配置Neovim的博客](https://qixinbo.info/2022/08/11/neovim_nvchad/)，今天试试使用[LazyVim](https://github.com/LazyVim/LazyVim)来将Neovim打造成IDE。

# 前置条件
LazyVim需要的前置条件（软件及其版本号）要满足要求，可参见[官方文档](https://www.lazyvim.org/)。
## Neovim
```sh
brew install neovim
```
如果遇到问题，可以参考[这一篇](https://rumosky.com/archives/517/)。

# 安装LazyVim Starter
## 备份当前配置
```sh
# required
mv ~/.config/nvim{,.bak}

# optional but recommended
mv ~/.local/share/nvim{,.bak}
mv ~/.local/state/nvim{,.bak}
mv ~/.cache/nvim{,.bak}
```
## 克隆starter库
```sh
git clone https://github.com/LazyVim/starter ~/.config/nvim
```

## 删除.git文件夹
```sh
rm -rf ~/.config/nvim/.git
```
这样就能添加到自己的repo中。

## 启动
```sh
nvim
```
启动后会自动安装插件。

