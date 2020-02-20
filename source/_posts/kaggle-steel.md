---
title: Kaggle钢铁赛：基于PyTorch/Unet算法的钢材表面缺陷检测——（1）数据集分析和可视化
tags: [Machine Learning, PyTorch]
categories: programming
date: 2020-2-15
---

# 简介
Kaggle上有一个[钢材表面缺陷检测的竞赛](https://www.kaggle.com/c/severstal-steel-defect-detection/overview)，是一个很好的将深度学习应用于传统材料检测的例子。对该赛题和解法的剖析，可以辅助理解深度学习的流程，及其应用于具体问题的套路。

这次解析分为两部分：
（1）第一部分，即本文，是一个预备性工作，即对该竞赛的数据集的分析和可视化，参考的是这个notebook——[clear mask visualization and simple eda](https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda)。感谢GoldFish的分享。
（2）第二部分，参见另一篇文章，即算法分析，参考的是Rishabh Agrahari的[使用PyTorch框架及Unet算法的notebook](https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88)。

其他参考文献：
- [kaggle数据挖掘比赛基本流程](https://kakack.github.io/2017/09/%E8%BD%AC-Kaggle%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E6%AF%94%E8%B5%9B%E5%9F%BA%E6%9C%AC%E6%B5%81%E7%A8%8B/)
- [Jupyter Notebook 使用技巧彙整：與 Kaggle 資料集互動](https://medium.com/pyradise/jupyter-notebook-tricks-kaggle-public-api-a578b99341d0)
- [Setting Up Kaggle in Google Colab](https://towardsdatascience.com/setting-up-kaggle-in-google-colab-ebb281b61463)
 
# 将kaggle数据迁移到Google Colab
这一步其实不用做，可以将数据集直接下载下来，用自己的电脑训练。但此时中国境内新型非冠病毒肆虐，按要求在家隔离（希望这场疫情赶紧过去，中国加油！），手头只有一个工作用的笔记本，无法胜任该训练任务。所以考虑云端训练。
（备注：使用Google Colab需要自备梯子）
当然也可以使用Kaggle的notebook，但此时发现在Kaggle运行notebook非常慢，根本加载不出来。而Google Colab跟Kaggle是一家，Colab中GPU训练也很方便快捷，同时迁移数据速度也很快，所以做这一步在当下是一个很好的选择。

## 获取Kaggle账户的Token
```python
kaggle-> my account->Create New API Token
```
## 将该文件上传到Google Colab的root账户下
打开一个Colab notebook，然后：
```python
import json
!mkdir /root/.kaggle
token = {"username": "YOUR-USERNAME", "key": "YOUR-KEY"}
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)

!chmod 600 /root/.kaggle/kaggle.json
```

## 下载数据集
第一种方法：（Attention！该方法下载的数据不全）
到kaggle钢铁赛的Data一栏中找到下载数据集的kaggle API 命令，，即[这里](https://www.kaggle.com/c/severstal-steel-defect-detection/data)，然后在Colab中执行：
```python
!kaggle competitions download -c severstal-steel-defect-detection
```
为了保证是下载到/content目录下，最好在该命令后面加上-p /content选项。
虽然是官方页面上给出的API，但是下载后发现仅有几十张图片，明显不是完整的数据集。
第二种方法：（有效）
（1）首先搜索kaggle与Sevelstal有关的数据集：
```python
!kaggle datasets list -s severstal
```
此时会列出很多带有该关键字的数据集名称，通过与该竞赛Data页面上的数据集大小对比，发现lyubovrogovaya/severstal数据集大小是2G，所以猜测该数据集是正确的，但实际上下载下来看了一下（10秒下载完成），不是原始的数据集，又重新找了一下，发现duongnh1/severstal 这个数据集是正确的。
（2）下载该数据集：
```python
!kaggle datasets download -d duongnh1/severstal
```
也是10秒就下载完了。
（3）解压查看该数据集：
```python
!unzip severstal.zip
!ls severstal
```
可以发现该数据集包含了赛题中完整的数据集信息：
```python
sample_submission.csv test_images train.csv train_images
```
这四个文件和文件夹的意义分别是：
- train_images：该文件夹中存储训练图像
- test_images：该文件夹中存储测试图像
- train.csv：该文件中存储训练图像的缺陷标注，有4类缺陷，ClassId = [1, 2, 3, 4]
- sample_submission.csv：该文件是一个上传文件的样例，每个ImageID要有4排，每一排对应一类缺陷


（4）将数据集转存到Google Drive中
Google会重置在临时的这个空间中存储的数据，因此第二天一看原来下载的数据都没了。所以要把这个数据集转存到Google Drive中。
首先要先在左侧Mount Drive，这样就会出现drive这个文件夹，然后：
```python
!mv severstal drive/"My Drive"/
```

# 数据集分析
这一部分主要就是根据这个notebook[clear mask visualization and simple eda](https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda)来探究的。
## 加载必要的Python模块
```python
import numpy as np # linear algebra
import pandas as pd
pd.set_option("display.max_rows", 101)
import os
print(os.listdir("drive/My Drive/severstal"))
import cv2
import json
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["font.size"] = 15
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
```

## 读取和分析文本数据
### 读取数据
```python
train_df = pd.read_csv("drive/My Drive/severstal/train.csv")
sample_df = pd.read_csv("drive/My Drive/severstal/sample_submission.csv")
```
初步查看一下里面的数据：
```python
train_df.head()
```
结果为：
```python
     ImageId_ClassId        EncodedPixels
0    0002cc93b.jpg_1        29102 12 29346 24 29602 24 29858 24 30114 24 3...
1    0002cc93b.jpg_2        NaN
2    0002cc93b.jpg_3        NaN
3    0002cc93b.jpg_4        NaN
4    00031f466.jpg_1        NaN
```
以及sample的开头：
```python
ImageId_ClassId    EncodedPixels
0                  004f40c73.jpg_1   1 1
1                  004f40c73.jpg_2   1 1
2                  004f40c73.jpg_3   1 1
3                  004f40c73.jpg_4   1 1
4                  006f39c41.jpg_1   1 1
```

### 有无缺陷及每类缺陷的图像数量
```python
class_dict = defaultdict(int)
kind_class_dict = defaultdict(int)
no_defects_num = 0
defects_num = 0

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        no_defects_num += 1
    else:
        defects_num += 1

    kind_class_dict[sum(labels.isna().values == False)] += 1

    for idx, label in enumerate(labels.isna().values.tolist()):
        if label == False:
            class_dict[idx+1] += 1

print("the number of images with no defects: {}".format(no_defects_num))
print("the number of images with defects: {}".format(defects_num))
```
得到的输出为：
```python
the number of images with no defects: 5902
the number of images with defects: 6666
```
即无缺陷的图像有5902张，有缺陷的图像有6666张。
再对有缺陷的图像进行缺陷分类：
```python
fig, ax = plt.subplots()
sns.barplot(x=list(class_dict.keys()), y=list(class_dict.values()), ax=ax)
ax.set_title("the number of images for each class")
ax.set_xlabel("class")
class_dict
```
得到：
```python
defaultdict(int, {1: 897, 2: 247, 3: 5150, 4: 801})
```
以及可视化结果：
![defect_class](https://user-images.githubusercontent.com/6218739/74599423-90b3d600-50bc-11ea-84b7-a9955d762c00.png)

从这一步得出的结论有两个：
（1）有缺陷和无缺陷的图像数量大致相当；
（2）缺陷的类别是不平衡的。

### 一张图像中包含的缺陷数量
```python
fig, ax = plt.subplots()
sns.barplot(x=list(kind_class_dict.keys()), y=list(kind_class_dict.values()), ax=ax)
ax.set_title("Number of classes included in each image");
ax.set_xlabel("number of classes in the image")
kind_class_dict
```
得到：
```python
defaultdict(int, {0: 5902, 1: 6239, 2: 425, 3: 2})
```
以及可视化结果：
![classes_in_one_image](https://user-images.githubusercontent.com/6218739/74599478-76c6c300-50bd-11ea-870c-ab2032635f04.png)
这一步得到的结论是：
大多数图像没有缺陷或仅含一种缺陷。

## 读取和分析图像数据
### 读取数据
```python
train_size_dict = defaultdict(int)
train_path = Path("drive/My Drive/severstal/train_images/")
for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1
```
看一下训练集中图像的尺寸和数目：
```python
train_size_dict
```
得到：
```python
defaultdict(int, {(1600, 256): 12568})
```
即，训练集中图像大小为1600乘以256大小，一共有12568张。
再读取和查看一下测试集中的图像：
```python
test_size_dict = defaultdict(int)
test_path = Path("drive/My Drive/severstal/test_images/")

for img_name in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1

test_size_dict
```
得到：
```python
defaultdict(int, {(1600, 256): 1801})
```
测试集中的图像也是1600乘以256，共1801张。

## 可视化标注
### 为不同的缺陷类别设置颜色显示
```python
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
for i in range(4):
    ax[i].axis('off')
    ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * palet[i])
    ax[i].set_title("class color: {}".format(i+1))

fig.suptitle("each class colors")
plt.show()
```

不同的缺陷类别用如下颜色表示：
![class-color](https://user-images.githubusercontent.com/6218739/74599680-ca86db80-50c0-11ea-8872-27216caa4df1.png)

### 将不同的缺陷标识归类
```python
idx_no_defect = []
idx_class_1 = []
idx_class_2 = []
idx_class_3 = []
idx_class_4 = []
idx_class_multi = []
idx_class_triple = []

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
        
    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    elif (labels.isna() == [False, True, True, True]).all():
        idx_class_1.append(col)
    elif (labels.isna() == [True, False, True, True]).all():
        idx_class_2.append(col)
    elif (labels.isna() == [True, True, False, True]).all():
        idx_class_3.append(col)
    elif (labels.isna() == [True, True, True, False]).all():
        idx_class_4.append(col)
    elif labels.isna().sum() == 1:
        idx_class_triple.append(col)
    else:
        idx_class_multi.append(col)

```
即将有不同缺陷的图像进行归类，同时注意最后有两个类别id_class_triple和id_class_multi用于存储同时有三类缺陷和同时有两类缺陷的图像。

### 创建可视化标注的函数
```python
def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(1600*256, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos-1:pos+le-1] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names[0], mask

def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_path / name))
    fig, ax = plt.subplots(figsize=(15, 15))
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title(name)
    ax.imshow(img)
    plt.show()
```
第一个函数name_and_mask是得到图像的名称及其标注信息，第二个函数show_mask_image是使用opencv的findContours函数将标注画出来。

### 无缺陷的图像展示
首先看一下五张没有任何缺陷的图像：
```python
for idx in idx_no_defect[:5]:
    show_mask_image(idx)
```
如图：
![no_defects_imgs_1](https://user-images.githubusercontent.com/6218739/74599826-5f8ad400-50c3-11ea-909e-b0d12a6e4fdc.png)
![no_defects_imgs_2](https://user-images.githubusercontent.com/6218739/74599827-61549780-50c3-11ea-9506-f537d46ce518.png)
![no_defects_imgs_3](https://user-images.githubusercontent.com/6218739/74599828-61ed2e00-50c3-11ea-816b-a3f0e190764c.png)
![no_defects_imgs_4](https://user-images.githubusercontent.com/6218739/74599829-6285c480-50c3-11ea-8be5-c57a63e1db5c.png)
![no_defects_imgs_5](https://user-images.githubusercontent.com/6218739/74599830-631e5b00-50c3-11ea-9a5e-8f13228f00f1.png)

### 仅含第1类缺陷的图像展示
看一下五张仅含第1类缺陷的图像：
```python
for idx in idx_class_1[:5]:
    show_mask_image(idx)
```

如图：
![image](https://user-images.githubusercontent.com/6218739/74599879-0f604180-50c4-11ea-9116-84bf75fb2295.png)
![image](https://user-images.githubusercontent.com/6218739/74599880-171fe600-50c4-11ea-8e99-8795de3504c7.png)
![image](https://user-images.githubusercontent.com/6218739/74599882-1b4c0380-50c4-11ea-848b-f7d9b8376c5f.png)
![image](https://user-images.githubusercontent.com/6218739/74599885-1edf8a80-50c4-11ea-9026-b9033a2bcc31.png)
![image](https://user-images.githubusercontent.com/6218739/74599886-22731180-50c4-11ea-8c27-fa84fd41d847.png)

### 仅含第2类缺陷的图像展示
看一下五张仅含第2类缺陷的图像：
```python
for idx in idx_class_2[:5]:
    show_mask_image(idx)
```
如图：
![image](https://user-images.githubusercontent.com/6218739/74599907-87c70280-50c4-11ea-81ca-50662bc02c0a.png)
![image](https://user-images.githubusercontent.com/6218739/74599908-8b5a8980-50c4-11ea-9951-c268e8e0c55a.png)
![image](https://user-images.githubusercontent.com/6218739/74599910-91506a80-50c4-11ea-9d1a-7f8fb60cb9d8.png)
![image](https://user-images.githubusercontent.com/6218739/74599912-96adb500-50c4-11ea-8d3d-2ab3bb835ebe.png)
![image](https://user-images.githubusercontent.com/6218739/74599914-9ca39600-50c4-11ea-897e-59f238b2c918.png)

### 仅含第3类缺陷的图像展示
看一下五张仅含第3类缺陷的图像：
```python
for idx in idx_class_3[:5]:
    show_mask_image(idx)
```

如图：
![image](https://user-images.githubusercontent.com/6218739/74599934-e3918b80-50c4-11ea-91d5-f4ba53775314.png)
![image](https://user-images.githubusercontent.com/6218739/74599937-e68c7c00-50c4-11ea-988c-0138a6cb3155.png)
![image](https://user-images.githubusercontent.com/6218739/74599944-fa37e280-50c4-11ea-8219-d2df60d4fc5b.png)
![image](https://user-images.githubusercontent.com/6218739/74599939-eee4b700-50c4-11ea-9bb5-8d65494fd248.png)
![image](https://user-images.githubusercontent.com/6218739/74599941-f310d480-50c4-11ea-8444-cf8f85fe410b.png)
 

### 仅含第4类缺陷的图像展示
看一下五张仅含第4类缺陷的图像：
```python
for idx in idx_class_4[:5]:
    show_mask_image(idx)
```
如图：
![image](https://user-images.githubusercontent.com/6218739/74599955-2bb0ae00-50c5-11ea-9cc9-b7402a480ea5.png)
![image](https://user-images.githubusercontent.com/6218739/74599957-2eab9e80-50c5-11ea-8946-051e81198e72.png)
![image](https://user-images.githubusercontent.com/6218739/74599959-323f2580-50c5-11ea-8cd2-1cfcf6c23017.png)
![image](https://user-images.githubusercontent.com/6218739/74599960-366b4300-50c5-11ea-85f2-e3f1e1b53003.png)
![image](https://user-images.githubusercontent.com/6218739/74599964-39feca00-50c5-11ea-8f7f-a783740fdb1f.png)

### 同时含有两类缺陷的图像展示
看一下五张同时含有两类缺陷的图像：
```python
for idx in idx_class_multi[:5]:
    show_mask_image(idx)
```
如图：
![image](https://user-images.githubusercontent.com/6218739/74599988-864a0a00-50c5-11ea-8260-9507a1834174.png)
![image](https://user-images.githubusercontent.com/6218739/74599989-88ac6400-50c5-11ea-9874-d6e4447fd49d.png)
![image](https://user-images.githubusercontent.com/6218739/74599990-8d711800-50c5-11ea-968d-f0b9d5bccfd5.png)
![image](https://user-images.githubusercontent.com/6218739/74599993-94982600-50c5-11ea-9079-9dfb217e9d14.png)
![image](https://user-images.githubusercontent.com/6218739/74599995-98c44380-50c5-11ea-8370-52dda33ecca5.png)


### 同时含有三类缺陷的图像展示

同时含有三类缺陷的图像只有两张，所以都显示出来了：
```python
for idx in idx_class_triple:
    show_mask_image(idx)
```
如图：
![image](https://user-images.githubusercontent.com/6218739/74600020-fd7f9e00-50c5-11ea-8a30-fa06e6642f88.png)
![image](https://user-images.githubusercontent.com/6218739/74600021-007a8e80-50c6-11ea-805c-1061fdc62a2c.png)

### 是否有像素属于多个缺陷
这一步查看是否有某个像素属于多个缺陷：
```python
for col in tqdm(range(0, len(train_df), 4)):
    name, mask = name_and_mask(col)
    if (mask.sum(axis=2) >= 2).any():
        show_mask_image(col)
```
可以看出，所有的像素都是仅对应一个或0个缺陷。
