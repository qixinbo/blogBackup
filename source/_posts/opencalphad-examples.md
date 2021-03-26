---
title: 开源计算相图软件OpenCalphad四个算例
tags: [opencalphad]
categories: simulation
date: 2016-3-5
---
# 纯铁在不同条件下的平衡态计算
```cpp
r t steel1   //read TDB steel的缩写，读取steel1.TDB文件
fe  //选择Fe元素
l data  //list data的缩写，列出所有的热力学数据
set c t=1000 p=1e5 n=1 //set condition的缩写，设置温度K，压力Pa和摩尔数mol
c e  //calculate equilibrium的缩写，计算当前条件下的系统平衡态
l,,,,  //list的缩写，列出平衡计算的结果
l sh  //list short的缩写，以简化的形式列出结果
set c b //set condition Bz的缩写，将条件设置成质量
set c n=none //去除摩尔数的条件
c e  //开始平衡计算
l,,,,  //列出计算结果
set c t=2000 //将温度修改为2000K
c e  // 开始平衡计算
l,,,,  //列出计算结果
set c h  //设置焓的条件
l c //list condition的缩写，列出条件（注意此时自由度为-1）
set c t=none  //去除温度条件（注意此时自由度为0）
c e //开始平衡计算
l,,,,  //列出计算结果
set c N=1 //上步计算中同时包含了质量B和焓H，这里将条件重新设置成摩尔数
set c B=none  //取出质量条件
set c h  //设置焓的条件
50000  //输入焓的具体数值
c e  //开始平衡计算
l,,,,  //列出计算结果
```
# 含6个元素的高速钢的平衡态计算
```cpp
r t steel1
set c t=1200 p=1e5 n=1 x(c)=.01 x(cr)=.05, x(mo)=.05 x(si)=.003 x(v)=.01
c e
l ,,,,
c tran  //calculate transition的缩写，通过去除某个条件，计算某个相的出现或消失
liq //指定液相为要研究的相
1 //1代表温度（这个视提示而定），表示去除温度条件
list,,,,,
set c x(fcc,c)  //设置fcc相的成分
.02
set c x(c)=none
c e
l,,,,
set ph liq   //set phase liquid的缩写，设置液相的属性
status fix 0  //设置液相的状态固定为0以求解新的熔点
set c t=none
c e
l,,,,
set c H
set c N=none
c e
l,,,,,
set c H=50000
c e
l,,,,
```
# 高速钢的相组分图、成分及热容计算
```cpp
r t steel1
set c t=1200 p=1e5 n=1 w(c)=.009 w(cr)=.045, w(mo)=.1 w(si)=.001 w(v)=.009
@$ Enter a composition set for the MC carbide (FCC)
amend phase fcc comp_set y MC , //设置fcc结构的MC碳化物的成分
NONE
<.1
NONE
<.1
NONE
>.5
<.2
amend phase fcc default  //设置fcc的默认成分，使之是奥氏体
<.2
NONE
<.2
<.1
<.2
<.2
>.5
amend phase hcp comp_set y M2C , //设置hcp结构的M2C碳化物的成分
NONE
NONE
NONE
NONE
NONE
>.5
<.2
c e
l r 1  //list results 1的缩写，列出结果，选择模式1，即以摩尔分数格式输出，按相的多少排列
l r 4  //list results 4的缩写，选择模式4，即以质量分数格式输出
set axis 1 T 800 1800 10  //选择温度作为坐标轴作图，取值范围是[800,1800]，步长10
l ax  //list axis的缩写，列出设置的坐标轴
step  //沿着坐标轴计算平衡值
l line  //在step命令中列出存储的平衡值
l eq  //列出平衡值
plot //做相分数图
T  //水平坐标轴是温度
NP( *) // 垂直坐标轴是相的摩尔数
title  //在顶部给出图形的名称
step 1 fig 1 //图形名称
render // 最终作图
plot
position outside right //将线条的注释放在图的外面
title step 1 fig 2
render
plot
T
w(*,cr)  //作稳定相中Cr的含量
title step 1 fig 3
render
plot
T
H  //垂直坐标轴是焓
title step 1 fig 4
render
ent sym cp=hm.t;  //enter symbol的缩写，定义cp这个符号，它的表达式是hm.t，点号是导数
plot
T
cp //垂直坐标轴是热容
title step 1 fig 5
render //注意此时的热容是负数，原因是终点的平衡值没有设置正确，下面是修改y轴的范围
plot
T
cp
yr  //yrange的缩写
N
0
200
title step 1 fig 6
position off
render
```
# Fe-C相图
```cpp
r t steel1
fe c
set cond t=1000 p=1e5 n=1 x(c)=.2
c e
l r 1
set ax 1 x(c) 0 1 ,,,
set ax 2 t 500 2000 25
l ax
l sh
map  //开始相图计算
plot
x(*,c)
T
title map 3 fig 1
render
plot
x(*,c)
T
xr  //xrange的缩写，改变x轴的范围
n
0
.2
title map 3 fig 2
render
```
