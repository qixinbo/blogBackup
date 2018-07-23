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
以下是图集欣赏，具体图片介绍见[官网](https://commons.wikimedia.org/wiki/Commons:Picture_of_the_Year):
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-%25D0%259D%25D1%2596%25D0%25B6%25D0%25BD%25D0%25B8%25D0%25B9_%25D1%2580%25D0%25B0%25D0%25BD%25D0%25BA%25D0%25BE%25D0%25B2%25D0%25B8%25D0%25B9_%25D1%2581%25D0%25B2%25D1%2596%25D1%2582%25D0%25BB%25D0%25BE.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-A_butterfly_feeding_on_the_tears_of_a_turtle_in_Ecuador.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Biandintz_eta_zaldiak_-_modified2.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Broadway_tower_edit.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Cyanocitta-cristata-004.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Eichh%25C3%25B6rnchen_D%25C3%25BCsseldorf_Hofgarten_edit.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Elakala_Waterfalls_Swirling_Pool_Mossy_Rocks.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Fire_breathing_2_Luc_Viatour.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Gl%25C3%25BChlampe_explodiert.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Gl%25C3%25BChwendel_brennt_durch.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Hoverflies_mating_midair.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Lake_Bondhus_Norway_2862.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Lanzarote_5_Luc_Viatour.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Laser_Towards_Milky_Ways_Centre.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Locomotives-Roundhouse2.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Magnificent_CME_Erupts_on_the_Sun_-_August_31.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Mostar_Old_Town_Panorama_2007.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-New_York_City_at_night_HDR.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Pair_of_Merops_apiaster_feeding.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Penguin_in_Antarctica_jumping_out_of_the_water.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Polarlicht_2.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Russian_honor_guard_at_Tomb_of_the_Unknown_Soldier%252C_Alexander_Garden_welcomes_Michael_G._Mullen_2009-06-26_2.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-SQM_GE_289A_Boxcab_Carmelita_-_Reverso.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Sarychev_Peak.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Sikh_pilgrim_at_the_Golden_Temple_%2528Harmandir_Sahib%2529_in_Amritsar%252C_India.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Swallow_flying_drinking.jpg)
![](http://7xrm8i.com1.z0.glb.clouddn.com/wikipedia1280px-Tracy_Caldwell_Dyson_in_Cupola_ISS.jpg)
