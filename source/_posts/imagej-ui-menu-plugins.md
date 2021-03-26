---
title: ImageJ 用户指南 -- 10. 菜单栏之Plugins
tags: [ImageJ]
categories: computer vision
date: 2018-9-10
---

# 本章说明
这里详解Plugins菜单的功能。
Plugins菜单反映了ImageJ/plugins文件夹（至多两个子文件夹）的层级结构，因此可以创建子菜单（即子文件夹）来保持该菜单的简洁性，比如将EPS\_Writer.class移动到ImageJ/Plugins/Input/PDF文件夹就可以实现将EPS Writer插件移入Plugins-Input-PDF子菜单下。
另外，勾选Edit-Options-Misc中的Move isolated plugins，就可以将仅有一个命令的插件移入Plugins-Miscellaneous菜单中。
# Plugins
## Macros
该菜单包含了安装、运行、录制宏等命令。在文件StartupMacros.txt中包含的宏会在ImageJ启动时自动加载。ImageJ被设计成一次仅能安装一个集合的宏，因此，通过Install加载的最后一个集合的宏总会替换上一次的宏。
### Install
安装宏。
### Run
加载宏并运行，而不在Editor中打开。为了运行一个宏，同时查看它的代码，使用File-Open，然后在编辑器里点Macros-Run Macro。
### Startup Macro
打开ImageJ/macros/StartupMacro.txt文件。
### Record
打开ImageJ的命令录制器。为了创建一个宏，先打开录制器，然后使用一个或多个命令，然后点击Create。当录制器打开时，使用的每一个菜单命令都将产生一个run函数。

## Shortcuts
快捷键相关的操作。
### List Shortcuts
该命令显示快捷键列表。在command一列中用星号开头的快捷键是用Create Shortcuts创建的，而用^号开头的表明是通过所安装的macro创建，其会覆盖掉ImageJ的默认热键。
### Create Shortcuts
为ImageJ的菜单命令指定一个快捷键。
### Install Plugins
在用户指定的子菜单下安装一个插件。如果一个插件有showAbout()函数，那么它会自动添加到Help-About Plugins子菜单下。
注意，新版的ImageJ将Install Plugins单独提到Plugins这个一级菜单下了。
### Remove
删除通过Create Shortcuts添加的命令。

## Utilities
### Control Panel
该命令用一个遗传树的结构来显示ImageJ的菜单。点击一个叶子节点来启动对应的命令。双击一个主干节点（文件夹图标）会展开或收起它。点击和拖拽一个主干节点可以在另外一个窗口中显示它的子节点。
### Find Commands
无需浏览所有菜单而直接找到一个命令的最快捷的方式。
快捷键是“L”。
### Search
查找包含某个特定字符串的宏、脚本、插件源代码等。
### Monitor Events
通过使用IJEventListener、CommandListener、ImageLister界面，可以监视前景色和背景色的变化、工具切换、日志窗口、命令执行、图形窗口的打开、关闭和升级等。
### Monitor Memory
显示内存使用情况。
### Capture Screen
将电脑的当前屏幕截屏，显示成一个RGB图片。
### Capture Image
将当前显示的图片保存进一个RGB图片，所见即所得。
### ImageJ Properties
显示ImageJ的属性，如Java版本、OS名字和版本、文件路径、屏幕尺寸等信息。
### Threads
显示当前运行的线程和优先级。
### Benchmark
在当前图片上运行62种图像处理操作，然后在状态栏上显示运行时间。
### Reset
使用该命令解锁一个锁定的图片、释放剪贴板所使用的内存和undo的缓存。

## New
打开一个编辑窗口，用来编辑和运行宏、脚本和插件。
### Macro
打开一个空白的编辑器窗口。
### Macro Tool
打开一个创建圆形选区的宏demo。
### JavaScript
打开一个名为Script.js的空白的编辑器窗口。
### Plugin
打开一个使用PlugIn接口的原型插件。该类型的插件打开、捕捉和差生图片。使用Ctrl+R来编译和运行。注意插件的名字应该包含至少一个下划线。
### Plugin Filter
打开一个使用PlugInFilter接口的原型插件。该类型的插件处理当前图片。
### Plugin Frame
打开一个使用PlugInFrame类的原型插件。该类型的插件显示一个包含控制体（如按钮和滑块）的窗口。
### Plugin Tool
打开一个使用PlugInTool的原型插件，该插件用于与画布交互。
### Text Window
打开一个特定尺寸的文本窗口，用于宏的写入。
### Table
打开一个类似于Results Table的空白table，用于宏的写入。

## Compile and Run
编译和运行一个插件。如果一个文件的名字后缀是.class，则运行该插件。

