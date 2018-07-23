---
title: 对ICME的理解 
tags: [ICME]
categories: computational material science 
date: 2017-11-29
---

# 什么是ICME
在TMS报告中，ICME的定义是“the integration of personnel (e.g., engineers, designers, etc.), computational models, experiments, design, and manufacturing processes across the product development cycle, for the purpose of accelerating and reducing the cost of development of a materials system or manufacturing process”，个人理解为将整个产品生产流程中的几何设计、材料选择、产品制造、性能测试、实际生产等环节全部连接在一起，缺一不可，并发挥计算机模拟在各个环节的作用，目的是为了节约时间和成本。
ICME与其他几个工业中的材料制造的概念有什么区别呢？
## ICME VS Digital Twin
Digital Twin是“数字化双胞胎”，意在将生产流程做一个数字化的“复制”。但我觉得相对于ICME，Digital Twin反而忽略了材料和测试这一块，而是更加重视设计和制造这两部分。即，它认为制造过程中所用的材料都是满足要求的。比如它用CAD软件画出了产品的几何设计，然后就将其放入服役条件中，分析其受力情形等，其中设置的材料属性也是宏观属性，没有考虑材料中的缺陷等，只是用到了结构力学的知识；另一种情形，使用CFD软件计算了材料制造过程中的流体性能，但是仍然没有下探到材料层面，只是用到了流体力学的知识，而忽略了流动过程中的气孔、共晶相、析出相等。而ICME强调材料这块同样重要。
## ICME VS Multiscale Modeling
Multiscale Modeling是多尺度模拟，即将宏观、介观、微观模拟全部考虑进去，比如从原子电子尺度的计算，经过晶粒尺度的计算，再到工件尺度的计算。ICME的模拟这块目的是为了实现多尺度模拟，但是限于目前的计算能力，ICME允许模拟“分而治之”，比如材料成分的选择就可以只用第一性原理计算，材料的性能模拟就可以只用有限元等，且为了保证计算效率，有的影响不明显的尺度可以不用模拟。不同尺度的模拟输入输出可以互相作为初始条件和边界条件，以及参数传递等。
# ICME的特点
ICME有以下几个特点：
## 各个环节同等重要，缺一不可
设计、材料、制造、测试等环节在ICME都同等重要，不能忽视其中某一个，认为其中某一个不重要。因此，ICME需要多个领域小组的协同配合，不能各自为战。
## 流程非线性化
ICME中的整个生产流程不是线性的，即一条龙流水线模式，而是不断反馈和迭代的过程。比如产品原型设计好后，通过Process Modeling和Material Modeling发现设计不合理，那就及时返工，在其他环节也是这样，只要某一环节出现问题，就打断整个流程，返回到前面进行修改。
## 计算机模拟与V&V验证
ICME强调在各个环节中都发挥计算机模拟的重要性，这样能够有效地减少实验成本和时间成本等。比如设计使用CAD软件和各种优化软件，材料使用ThermoCalc等数据库软件，制造使用CFD等软件，性能使用有限元软件等。但是也不盲从计算机模拟结果，而是及时使用Verification和Validation来验证模拟的正确性和准确性。
## 定量预测与经验推测
以前的产品生产流程也都包含以上环节，但很多时候都是停留在经验基础上，但这种模式在变换产品后可能就不适用了，只能重新摸索。但ICME能够给出定量的理论基础，方便问题的排查和经验的迁移。
# ICME目前存在的问题
ICME目前仅有几个成功的案例，还没有大规模的在工业中应用。比如缺乏软件支持等。
# 我的角色
开发基于材料微观组织的定量预测模型（目前的模拟都没有Microstructure-Based）、协助指定V&V标准、数据库的整合
