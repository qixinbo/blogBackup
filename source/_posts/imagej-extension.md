---
title: ImageJ 用户指南 -- 3. 扩展：宏、插件和脚本
tags: [ImageJ]
categories: computer vision
date: 2018-9-2
---

# 本章说明
ImageJ的功能可以通过宏、插件和脚本三种形式进行扩展：
- 宏：宏是最简单的执行一系列ImageJ命令的方式。ImageJ的宏语言是一种类Java的语言，包含了一系列的控制体、算子和内置函数，可以用来调用内置命令和其他宏。宏的代码存储在以.txt和.ijm为扩展名的文本文件中。
- 插件：比宏更强大、更灵活、也更快，但也更难写和调试。ImageJ的大多数的菜单命令其实是插件。插件是用Java语言编写，后缀是.java源文件，然后编译成.class文件。
- 脚本：ImageJ使用Mozilla Rhino解释器来运行JavaScripts脚本。类似于插件，脚本也对所有的ImageJ API和Java API有访问权限，但是不需要编译。另一方面，脚本不如宏语言简单，与ImageJ的集成不那么紧密。Fiji也支持其他语言写成的脚本。

# 宏
宏是一个自动执行一系列ImageJ命令的简单程序。创建宏的最简单的方法是录制一系列的命令：Plugins-Macros-Record。
宏存成一个.txt或.ijm后缀的文本文件，然后通过Plugins-Macros加载。
关于宏编程的教程有：
- [The ImageJ Macro Language](http://imagej.net/docs/macro_reference_guide.pdf)
- [The Built-in Macro Functions webpage](http://imagej.net/developer/macro/functions.html)
- [Tutorials on the Fiji webpage](http://imagej.net/Introduction_into_Macro_Programming)
- [How-tos and tutorials on the ImageJ Documentation Portal](http://imagejdocu.tudor.lu/)

# 脚本
原生ImageJ脚本是用JavaScript语言写成。
资源有：
- [The ImageJ web site, with growing documentation](http://imagej.net/developer/javascript.html)
- [Tutorials on the Fiji webpage](http://imagej.net/JavaScript_Scripting)
- [Online scripts repository](http://imagej.net/macros/js/)

Fiji则支持其他语言，比如BeanShell、Clojure、Python和Ruby。
资源有：
- [Jython Scripting](https://imagej.net/Jython_Scripting)
- [Jython Scripting Examples](http://imagej.net/Jython_Scripting_Examples)
- [The extensive tutorial on scripting Fiji with Jython by Albert Cardona](http://www.ini.uzh.ch/~acardona/fiji-tutorial/)
- [Dedicated tutorials on the Fiji webpage](http://fiji.sc/wiki/index.php/Scripting_comparisons)

# 插件
插件是用Java写成。
资源有：
- [Developer Resources Page on the ImageJ website](http://imagej.net/developer/index.html)
- [Dedicated tutorials on Fiji’s webpage](http://fiji.sc/wiki/index.php/Introduction_into_Developing_Plugins)
- [Dedicated tutorials on the ImageJ Documentation Portal](http://imagejdocu.tudor.lu/)
- [Dedicated tutorials on the ImageJDev webpage](http://imagej.net/IDEs)

# 命令行运行ImageJ
可以在命令行运行ImageJ，教程有：
- [Running ImageJ in headless mode](http://imagejdocu.tudor.lu/doku.php?id=faq:technical:how_do_i_run_imagej_without_a_graphics_environment_headless)
- [Using Cluster for Image Processing with IJ](http://cmci.embl.de/documents/100922imagej_cluster)
