---
title: Latex中文排版——XeLatex的用法
tags: [Latex]
categories: coding 
date: 2016-1-13
---
本文参考了以下三个网址：
http://linux-wiki.cn/wiki/zh-hans/LaTeX%E4%B8%AD%E6%96%87%E6%8E%92%E7%89%88%EF%BC%88%E4%BD%BF%E7%94%A8XeTeX%EF%BC%89
http://electronic-blue.wikidot.com/doc:xetex
http://blog.jqian.net/post/xelatex.html

Latex中文排版相对于原生英文来说有些麻烦，虽然使用CJK这个包可以解决，但设置较为繁琐，如需要自己编译生成中文字体集（从而可用字体受限）等，而XeLatex宏包因为原生支持系统字体，则将中文与英文完全等价，将两者的隔阂完全消除，因此也就无所谓“中文”排版这一特定说法了。

# XeLatex安装
安装Texlive时自带。

# 准备字体
由于Linux系统下的中文字体较少，可以复制Windows或Adobe的字体。XeLatex可以直接使用系统字体，只需把字体复制到指定位置即可，无需自己生成字体文件。
比如复制Windows中的宋体、黑体等：
sudo mkdir /usr/share/fonts/win
sudo cp /media/Win系统盘挂载点/Windows/Fonts/{SIM,sim}* /usr/share/fonts/win/
此外，有些免费的字体，可以直接下载使用。
如下载文泉驿的微米黑和正黑：
sudo apt-get install xfonts-wqy ttf-wqy-microhei ttf-wqy-zenhei
更新字体缓存：
fc-cache
为了更新整个系统下的用户的字体缓存，建议使用root执行：
sudo fc-cache -f -s -v
查看可用的字体：
fc-list
或只查看中文的字体：
fc-list :lang=zh

# 使用方法
## 单独设置字体：
编写Latex文件：
\documentclass{article}
 
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
\font\ming="华文楷体" at 10pt
 
\begin{document}
\ming 中文测试
\end{document}

然后使用XeLatex编译即可。其中\XeTexlinebreaklocale指定XeLatex以中文的方式断行，因为一般英文只会在空白处断行，而中文除了避头避尾以外可以断在任何地方。\XeTexlinebreakskip则是让XeLatex在字符间加入0pt~1pt的弹性间距，这样才能排出左右相齐的文档。

## 使用fontspec设定字型
上述是单独设置字体，而要设定全文使用的字型、或是使用某些字型的特殊功能（如连字符ligature）时，使用fontspec宏包是比较方便的做法。
\documentclass{article}
 
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
 
\usepackage{fontspec}
\setmainfont{STKaiti}
 
\begin{document}
中文测试
\end{document}
fontspec常用的指令有：
\setmainfont 设定预设字型（衬线字型），也是使用\rmfamily命令时会选用的字型。
\setsansfont  设定无衬线字型sans-serif，也是使用\sffamily命令时会选用的字型。
\setmonofont 设定等宽字型，也是使用\\ttfamily命令时会选用的字型。
\newfontfamily 定义新的字型。
\documentclass{article}
 
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
 
\usepackage{fontspec}
\setmainfont{STKaiti}
\setsansfont{SimHei}
\newfontfamily \yaoti {FZYaoTi}
 
\begin{document}
这些字体使用楷体\\
{\sffamily 这些字体使用黑体} \\
{\yaoti 这些字体使用方正姚体}
\end{document}
注意：这里改用英文名称（虽然\setmainfont可以接受中文字型名称，但当与其他命令使用的字型重复时会出现问题，所以还是使用英文名保险）。
