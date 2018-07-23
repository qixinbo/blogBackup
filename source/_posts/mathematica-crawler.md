---
title: 用Mathematica抓取《美国队长3》的剧照 
tags: [mathematica]
categories: programming
date: 2016-5-4
---

今下午实在不想改论文，学着做了一个Mathematica爬虫，用来爬取即将上映(后天5月6号)的<美国队长3>的剧照。
参考文献：
[看这里](http://shuli.xianyungu.com/software-download-mathematica-quick-start-application-handouts-videos/)

# 第一步：导入网页源文件
首先得找到<美队3>的照片网站吧，这里用的是经典大牌电影网站——时光网，链接是[here](http://movie.mtime.com/209122/posters_and_images/)，然后将网页的源文件导入Mathmatica中：
```cpp
input = Import["http://movie.mtime.com/209122/posters_and_images/posters/hot.html", "Source"];
```
注意这里导入的元素是Source，即原始的源文件，没有经过任何转化。还可以用XMLObject，见[这个教程](http://www.kylen314.com/archives/1647)，但是我感觉没有Source好使。

# 第二步：分析源文件
这一步是分析出来图片在网站的哪一部分。可以通过两种途径，一个是可视化操作，即用浏览器的检查功能，比如火狐的“查看元素”和Chrome的“检查”，通过点击不同的地方，就能在原网页中高亮该部分;另一种是直接分析，即在源文件中直接查看寻找。一般是两者结合着用，比如先用方法一找到该元素的标记，然后在源文件中直接找到该标记。但是此例中找到的网页元素在“查看源代码”中找不到，难道是因为启用了JS的某项特技？搞不懂为啥，所以还是按部就班地在源文件中查找，于是发现了这样一行：
```cpp
var imageList = \
[{"stagepicture":[{"officialstageimage":[{"id":7141431,"title":"官方剧照 \
#01","type":6,"subType":6001,"status":1,"img_ \
220":"http://img31.mtime.cn/pi/2016/03/11/133802.60497903_ \
220X220.jpg","img_ \
1000":"http://img31.mtime.cn/pi/2016/03/11/133802.60497903_ \
1000X1000.jpg","width":4785,"height":3190,"fileSize":5236,"enterTime":\
"2015-12-10","enterNickName":"旁聽生","description":"","commentCount":0,\
"imgDetailUrl":
```
这里面就暴露了图片的地址。
找到地址后，复制出来实际验证一下，这时可以发现这里每张图片都对应了两种尺寸，一种是220x220的缩略图形式，一种是1000x1000的高清无码大图。两个都下下来有些浪费空间，后面选哪个下那很清楚了。

# 第三步：提取图片地址
上一步虽然找到了图片地址，但是它埋没在了整个源文件中，所以还需要通过Mathematica的字符串操作将它规矩地提取出来。
```cpp
data1 = StringCases[input, "\"img_1000\"" ~~ Shortest[__] ~~ "jpg"]
```
得到的结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mathematica-crawler-1.png)

继续利用StringCases进一步提炼：
```cpp
data2 = StringCases[data1, "http" ~~ Shortest[__] ~~ "jpg"];
data3 = Flatten[data2];
```
得到的结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mathematica-crawler-2.png)

# 第四步：批量导出图片
将图片批量保存到本地，这一步参考了[这篇博文](http://www.kylen314.com/archives/1647)。
```cpp
Export["~/Public/" <> StringSplit[#, "/"][[-1]], Import[#]] & /@ data3;
```
结果为：
![](http://7xrm8i.com1.z0.glb.clouddn.com/mathematica-crawler-3.png)

Over!

