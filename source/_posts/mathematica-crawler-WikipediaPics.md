---
title: 用Mathematica抓取历年Wikipedia年度照片 
tags: [mathematica]
categories: programming
date: 2016-5-23
---

今天看到科学网上一位老师介绍Wikipedia年度照片，点[这里](http://blog.sciencenet.cn/blog-274385-979054.html)，瞬间就感觉以后桌面背景不用愁了。爬的方法跟之前爬美国队长剧照的方法相同，直接上源码：

```cpp
input = Import["https://commons.wikimedia.org/wiki/Commons:Picture_of_the_Year","Source"];
data1 = StringCases[input, "<div style=\"margin" ~~ Shortest[__] ~~ "</div>"];
data2 = StringCases[data1, "\"https:" ~~ Shortest[__] ~~ "jpg\""];
data3 = Flatten[data2];
data4 = StringReplace[data3, "\"" -> ""];
data5 = StringReplace[data4,"jpg/" ~~ Shortest[__] ~~ "px" -> "jpg/1280px"];
Export["~/Public/" <> StringSplit[#, "/"][[-1]], Import[#]] & /@ data5;
```
图集欣赏被忽略，因为七牛云的临时域名被收回了。。。
