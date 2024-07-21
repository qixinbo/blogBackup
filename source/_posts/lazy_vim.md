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

# 启动
```sh
nvim
```
启动后会自动安装插件。
这个地方要注意，插件的安装是`lazy.vim`这个插件管理的，即`lazy.vim`是插件管理器，而`LazyVim`可以说是neovim的“发行版”，两者的作者是同一个人。

# 常用快捷键
默认的`<leader>`键是`<space>`，默认`<localleader>`键是`\`。

- `<leader>l`：打开Lazy Plugin Manager，即`lazy.vim`，该悬浮窗内的快捷键是大写字母，因此需要`Shift`键配合。最常用的快捷键是`S`，即`Sync`，它是`install`、`clean`和`update`的组合技，效果就是能保证插件版本与配置中指定的版本精确一致。
- `s`：进入`flash`模式快速搜索文本，底层是使用`flash.vim`插件实现，非常快速地将鼠标移动到想要到的地方。
- `f`：也是查找模式，只查找此时光标后的内容，且光标直接跳到第一个目标处，多次按`f`则会继续下一个，按`F`则上一个。
- `w`和`e`：按单词移动
- `c-d`和`c-u`：向下、向上滚动半屏
- `c-b`和`c-f`：向下、向上滚动整屏
- `<leader><leader>`: 当前目录下的文件名搜索，只有小写字母时搜索结果是大小写不敏感的，但一旦输入了大写字母，则大小写敏感。该部分使用的是`telescope`插件，也可以配合`s`快捷键使用，此时会显示每个文件的索引，直接输入索引就能定位该文件
- `<leader>e`：打开导航树，然后可以使用`h`折叠、`l`展开、`j`向下、`k`向上、`d`删除、`a`新增文件或文件夹（末尾用`\`）、`r`重命名、`x`剪切、`y`复制、`p`粘帖，导航树使用的是`neo-tree`插件



# 启用额外插件
LazyVim除了预装了很多插件，还有一些默认不启用的插件，称为`Lazy Extras`。可以在启动页按`x`进入该插件库。

