---
title: Pandas可视化数据分析工具D-Tale详解
tags: [dtale]
categories: data analysis
date: 2022-12-17
---

# 简介
最近在狂补数据分析方面的知识，有一个趁手的“兵器”非常重要（就像在图像处理领域，我一直使用[`ImagePy`](https://github.com/Image-Py/imagepy)，能够快速处理图像和测试算法）。经过搜索，在数据分析领域，发现了一个非常强大的工具——['D-Tale'](https://github.com/man-group/dtale)。
直接看一下官方简介（下面这段翻译来自最近大火的`ChatGPT`，对比了一下`DeepL`的翻译，发现`ChatGPT`效果更好，`DeepL`把`Pandas`直接翻译成了`潘达斯`）：
> D-Tale是一个Flask后端和React前端的结合，为您提供了一种简单的方法来查看和分析Pandas数据结构。它与ipython笔记本和python / ipython终端无缝集成。目前，该工具支持DataFrame、Series、MultiIndex、DatetimeIndex和RangeIndex等Pandas对象。

# 安装
安装方式非常简单：
```sh
pip install dtale
```
# 启动
`D-Tale`有很多种启动方法。总体上来说，可以按在图形界面或者后台启动来分为两类。
## 图形界面启动
`D-Tale`支持在多种人机交互的图形界面中启动。比如常用的`Jupyter Notebook`、`JupyterHub`、`Google Colab`、`Kaggle`。
以`Jupyter Notebook`为例（需要先有已经通过`pandas`所读取的`df`数据）：
```python
import dtale

dtale.show(df)
# 也可以直接在新的浏览器标签页中打开
# dtale.show(df, open_browser=True)
```

## 后台
`D-Tale`也支持在后台直接启动，此时又可以分为两种情形：
### 脚本运行
可以在`Python`脚本中直接启动`D-Tale`，即在`dtale.show`方法中加入`subprocess=False`参数：
```python
import dtale

dtale.show(subprocess=False)
```
### 命令行运行
也可以直接在命令行终端中运行：
```sh
dtale
```
可以看出，在后台中启动时是“无数据”方式启动的（当然也可以直接读取数据），此时会打开一个“加载数据”的界面供读取数据。
上面启动的方式都是可以的，取决于你的使用场景。

# 使用
启动后，复制所输出文字的最后一行到浏览器中，比如：
```sh
http://qi-air.local:40000/
```
如果是“无数据”直接启动，那么就是如下界面：
![no-data](https://user-images.githubusercontent.com/6218739/208232720-0f117142-11f8-462b-b207-c550f5da7f0d.png)
如果是直接读取了数据启动，那么直接就呈现了数据表格：
![](https://user-images.githubusercontent.com/6218739/208232835-75d6996e-a58a-42da-8975-2f2320d5c153.png)
下面将以[这里](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv)提供的`COVID-19`数据作为数据。
同时将界面调整一下，点击左上角的箭头标记，打开“主菜单”：
![](https://user-images.githubusercontent.com/6218739/208233861-7b49c089-6b57-4b48-9ab2-6f2caab491e8.png)
在该列的倒数第二项和第三项中，点击`Pin Menu`从而固定该菜单，以及将语言设置为中文。
于是，整个页面变成了：
![](https://user-images.githubusercontent.com/6218739/208233967-1daa211b-b73a-4313-b22c-545a8229413f.png)

## 编辑单元格
可以编辑网格中的任何单元格（除了行索引或标题，后者可以用`Rename`功能来编辑）。
为了编辑一个单元格，只需双击它。这将把它转换成一个文本输入域，此时可以看到一个闪烁的光标。除了将该单元格变成一个输入框外，它还会在屏幕顶部显示一个输入框，以便更好地查看长字符串。这里应该保证你输入的值与你编辑的列的数据类型相符。例如。
`integers` -> 应该是一个有效的正数或负数的整数
`float` -> 应该是一个有效的正数或负数的浮点数
`string` -> 任何有效的字符串都可以
`category` -> 要么是一个预先存在的类别，要么这将创建一个新的类别（所以要注意！）。
`date`, `timestamp`, `timedelta` -- 应该是相应的有效的字符串。
`boolean` -- 输入的任何字符串将被转换成小写字母，如果它等于 "true"，那么它将使单元格变成 "True"，否则就是 "False"。
也可以使用这两个预留值：
`"nan"` -> `numpy.nan` 
`"inf"` -> `numpy.inf` 
## 复制一定范围内的单元格到剪贴板
（1）按住`Shift`，点击某单元格；
（2）不要松开`Shift`，滑动鼠标，选择要复制的单元格，此时这些单元格变为灰色；
（3）点击另一个单元格，此时就会蹦出复制对话框。

## Describe描述
可以使用主菜单中的`Describe`来查看所有列及其数据类型，以及每列的统计细节。
也可以直接点击每一列，通过`Describe (Column Analysis)`来查看统计细节（建议采用这一种方式，上一种虽然能一下看全部列的，但有时会有bug）。
![](https://user-images.githubusercontent.com/6218739/208235366-0c70de2e-4af8-4c0e-b8af-668f0d5442fc.png)

## 离群值检测
在上面的`Describe`页面的下方，可以看到`Outliers`离群值的统计，这些值是通过如下代码统计获得：
```python
s = df[column]
q1 = s.quantile(0.25)
q3 = s.quantile(0.75)
iqr = q3 - q1
iqr_lower = q1 - 1.5 * iqr
iqr_upper = q3 + 1.5 * iqr
outliers = s[(s < iqr_lower) | (s > iqr_upper)]
```
## 自定义过滤
自定义过滤可以在主菜单中进行设置，也可以在每一列的列菜单的最下方进行。
![](https://user-images.githubusercontent.com/6218739/208236531-94563c20-c79a-43d1-81fb-39395c7e7db1.png)
## DataFrame函数
主菜单中的`DataFrame Functions`可以通过一系列的DataFrame函数来新增列或改变已有的列。
![](https://user-images.githubusercontent.com/6218739/208236688-d34b54b3-87cb-4e13-b62c-c85ed2d28820.png)
## 合并与堆叠
这个功能允许用户合并或堆叠（即垂直连接）已经加载到`D-Tale`中的dataframes，或者上传额外的数据。上面显示的演示涉及到以下操作。（该功能类似数据库中的`Join`功能，内有例子可供理解）
![](https://user-images.githubusercontent.com/6218739/208236890-5798f5a0-d8b8-428d-b755-a12ac81fde19.png)
![](https://user-images.githubusercontent.com/6218739/208236915-3b8d9e2e-e206-4cef-8a04-ecb842644fbd.png)

## 汇总数据
这是一个非常强大的功能，允许用户从当前加载的数据中创建一个新的数据。目前可用的操作有：
（1）聚合：通过在特定索引的列上运行不同的聚合来整合数据。
（2）透视`Pivot`：这是对`pandas.Dataframe.pivot`和`pandas.pivot_table`的简单封装。
（3）转置：在一个索引上转置你的数据（如果你的索引有很多唯一的值，请注意dataframe会变得非常宽）
![](https://user-images.githubusercontent.com/6218739/208237144-e29582e6-7049-4d8c-9f20-540fb4d0054a.png)
![](https://user-images.githubusercontent.com/6218739/208237176-26d99910-3a24-469b-92e5-e2e9e8fa1152.png)

## 重复项
从数据中删除重复的列/值，并将重复的数据提取到单独的实例中。
![](https://user-images.githubusercontent.com/6218739/208237326-595e2a18-8b2b-4c94-80f7-c0233c841acf.png)

## 缺失值分析
使用[`missingno`](https://github.com/ResidentMario/missingno)软件包，显示分析数据集中存在的缺失（`NaN`）数据的图表。也可以在一个标签中单独打开它们，或者使用右上角的链接将它们导出为静态PNG。
![](https://user-images.githubusercontent.com/6218739/208237505-d5a13ab1-e9e7-4c7f-926f-cfff954fb45f.png)
## 图表
基于数据建立自定义图表（由[`plotly/dash`](https://github.com/plotly/dash)提供）。
（1）图表将在一个新的选项卡中打开，因为功能太多了，可能希望能够在原始选项卡中引用主网格数据。
（2）要建立一个图表，必须为`X`和`Y`的输入选择一个值，这将有效地驱动X和Y轴上的数据。
如果正在处理一个三维图表（热图、三维散点图、表面图），还需要为`Z`轴输入一个值。
（3）一旦输入了所有需要的坐标轴，一个图表就会被建立。
（4）如果X轴（或3D图表中的X和Y的组合）上的数据有重复的，有三个选择：
（4.1）指定一个组，这将为每个组创建序列。
（4.2）指定一个聚合，可以从以下选项中选择一个：计数、首数、尾数、平均值、中位数、最小值、最大值、标准差、方差、平均绝对偏差、所有项目的乘积、总和、滚动。
（a）指定一个 "滚动 "聚合也需要一个窗口和一个计算（相关性、计数、协方差、峰度、最大值、平均值、中位数、最小值、偏度、标准差、总和或方差）。
（b）对于热图，也可以使用 "相关Correlation"聚合，因为在热图中查看相关矩阵是非常有用的。其他地方不支持这种聚合。
（4.3）同时指定一个组和一个聚合。
（5）可以在不同的图表类型之间进行切换：线形、条形、饼形、文字云、热图、3D散点和曲面。
（6）如果指定了一个组，那么可以在一个图表中显示所有序列，或者将每个序列分成自己的图表 "`Ghart per Group`"。
![](https://user-images.githubusercontent.com/6218739/208238269-bf845720-acbc-4bbd-aa18-ade0f9a7e650.png)
## 网络查看器
可以查看有向图。
![](https://user-images.githubusercontent.com/6218739/208245158-bfc3e474-ba4c-40ee-bd54-7a29237a45e8.png)
## 相关性
显示所有数字列与所有其他数字列的Pearson相关矩阵
（1）默认情况下，它将显示一个pearson相关的网格（可通过使用下拉菜单进行过滤）。
（2）如果有一个日期类型的列，可以点击一个单独的单元格，看到该列组合的pearson相关的时间序列。
目前，如果有多个日期类型的列，将有能力通过下拉的方式在它们之间进行切换。
（3）此外，可以点击时间序列中的单个点来查看进入该相关的点的散点图。
在散点图部分，也可以通过悬停在 "PPS "旁边的数字来查看图表中这些数据点的PPS的细节。
（4）当在D-Tale中查看的数据有日期或时间戳列，但每个日期/时间戳列只有一行数据时，相关性弹出窗口的行为有点不同：
用户得到的不是一个时间序列的相关图，而是一个滚动的相关图，可以改变窗口（默认：10）。
当用户点击滚动相关图中的一个点时，散点图将被创建。散点图中显示的数据将是该日期的滚动相关中涉及的日期范围。
![](https://user-images.githubusercontent.com/6218739/208245202-e72ed666-2c45-4124-b1d8-f50a8c6772bf.png)
## 预测能力得分
预测力得分（使用软件包[`ppscore`](https://github.com/8080labs/ppscore)）是一个不对称的、与数据类型有关的得分，可以检测两列之间的线性或非线性关系。该分数范围从0（无预测能力）到1（完全预测能力）。它可以作为相关关系（矩阵）的替代。警告：这可能需要一段时间来加载。
这个页面的工作原理与相关性页面类似，但使用PPS计算来填充网格，通过点击单元格，可以查看这两列问题的PPS的细节。
![](https://user-images.githubusercontent.com/6218739/208245265-256ffaa8-fa17-4900-8887-45cf9b722de0.png)
## 热力图
这将隐藏任何非浮点或非int列（右侧的索引除外），并对每个单元格的背景应用一种颜色。
每个浮点被重新规范化为0到1.0之间的值。
对于重正化，有两个选项
（1）按列：每个值都是根据其列的最小/最大值计算的。
（2）整体：每个值都是根据数据集中所有非隐藏的浮点数/int列的整体最小/最大值来计算的。
每个重新规范化的值都被传递到一个色标，即红色（0）-黄色（0.5）-绿色（1.0）。
![image](https://user-images.githubusercontent.com/6218739/208245349-57176009-cc59-443f-96cf-6addbcabbf62.png)
## 高亮显示Dtypes
这是一个快速检查的方法，看看数据是否被正确归类了。通过点击这个菜单选项，它将为特定数据类型的每一列分配一个特定的背景颜色。
|category|timedelta|float|int|date|string|bool|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|purple|orange|green|light blue|pink|white|yellow

![image](https://user-images.githubusercontent.com/6218739/208245622-8f2e5ec5-2b0d-4507-843a-3341c9fbf0e7.png)
## 高亮显示缺失
任何包含`nan`值的单元格将以黄色突出显示。
任何字符串列的单元格如果是空字符串或仅由空格组成的字符串，将以橙色突出显示。
❗将被添加到任何包含缺失值的列标题中。
![image](https://user-images.githubusercontent.com/6218739/208245719-e0e4f331-7915-4acf-ab4a-d2af39f3366e.png)
## 高亮显示离群值
突出显示超过自定义离群值计算的上界或下界的数字列的任何单元格。
下限离群值将以红色标示，其中较深的红色将接近该列的最大值。
上界离群值将以蓝色标示，深蓝色将接近该列的最小值。
⭐将被添加到任何包含离群值的列标题中。
![image](https://user-images.githubusercontent.com/6218739/208245793-a3c026f4-776e-4183-84cc-d4d4da4f703e.png)
## 高亮显示范围
根据三个不同的标准，突出显示任何数字单元格的范围：等于、大于、小于。
可以随意激活这些条件，它们将被视为一个 "或 "的表达式。例如，`(x == 0) or (x < -1) or (x > 1)`。

## 低方差标志
在这两个条件都是真的情况下，在列标题上显示标志。
（1）唯一值数量/唯一列数量<10%
（2）最常见值的计数/第二常见值的计数 > 20

## 代码导出
代码导出的是一些小的代码片段，代表了正在查看的网格的当前状态，包括以下内容。
（1）建立的列
（2）过滤
（3）排序

其他可导出的代码有：
（1）Desrcibe描述（`Column Analysis`）
（2）相关性（网格、时间序列图和散点图）
（3）使用图表生成器构建的图表

## 导出CSV
将当前数据导出为`CSV`或`TSV`。

## 加载数据
无论是在没有加载数据的情况下启动D-Tale，还是在已经加载一些数据之后，现在都可以直接从GUI中加载数据或选择一些样本数据集。

## Instances
这将给出关于其他D-Tale实例在当前Python进程下运行的信息。

## 刷新列宽
这个通常是在你的列不再对齐的情况下的一个故障保护。点击它应该能修复这个问题。

