---
title: gdb调试c++程序知识点
tags:
  - gdb
  - c++
categories: coding
date: 2016-03-29 10:41:28
---

持续更新中~~~

# 条件断点
为断点设置条件表达式，只有条件满足时才激活断点
如：
break 670 if sum == 3

转自[这里](http://lesca.me/archives/gdb-breakpoints-command-list-watchpoint.html)

# 出现"value optimized out"错误
gdb调试程序的时候打印变量值会出现value optimized out情况,可以在gcc编译的时候加上-O0参数项,意思是不进行编译优化,调试的时候就会顺畅了,运行流程不会跳来跳去的,发布项目的时候记得不要在使用-O0参数项,gcc默认编译或加上-O2优化编译会提高程序运行速度. 

转自[这里](http://dsl000522.blog.sohu.com/180439264.html)。
