---
title: Kaggle细胞赛：基于PyTorch/UNet算法的细胞核识别
tags: [Machine Learning, PyTorch]
categories: programming
date: 2020-2-29
---

从2015年开始，Kaggle每年都举办一次Data Science Bowl，旨在召集众多力量开发算法，来解决当前某一特定领域的迫切问题。2018年的数据碗的任务是识别细胞的细胞核nuclei，从而使得更加方便地进行药物测试，使得新药的上市时间缩短。

Yun Chen分享了他的使用PyTorch/UNet算法的notebook，见[这里](https://www.kaggle.com/cloudfall/pytorch-tutorials-on-dsb2018)，本文是对该notebook代码的详细解析和再现，并适当做了一些修改。

再分享一篇挺好的背景文章：
[基于深度学习的图像语义分割算法综述](https://www.zybuluo.com/Team/note/1205894)


# 数据集分析
## 下载数据集
```python
!kaggle competitions download -c data-science-bowl-2018
```
## 解压并迁移数据
解压数据：
```python
!unzip stage1_sample_submission.csv.zip
!unzip stage1_solution.csv.zip
!unzip stage1_train_labels.csv.zip
!unzip stage2_sample_submission_final.csv.zip
 
!mkdir stage1_test
!unzip stage1_test.zip -d stage1_test

!mkdir stage1_train
!unzip stage1_train.zip -d stage1_train

!mkdir stage2_test_final
!unzip stage2_test_final.zip -d stage2_test_final
```

迁移数据到Google Drive：
```python
!mkdir nuclei
!mv stage1_test nuclei
!mv stage1_train nuclei
!mv stage2_test_final/ nuclei
!mv *.csv nuclei
!mv nuclei /content/drive/My\ Drive
```
这样就做到了持久化，防止notebook重启时数据丢失。

## 数据文件描述
（1）/stage1_train/*：该文件夹是训练集，包含训练用的图像及其掩膜图像
（2）/stage1_test/*：该文件夹是测试集，仅包含图像
（3）/stage2_test/*：这是第二阶段的测试集，仅包含图像
（4）stage1_sample_submission.csv：在第一阶段需要提交的文件格式
（5）stage2_sample_submission.csv：在第二阶段需要提交的文件格式
（6）stage1_train_labels.csv：该文件是训练集中的掩膜图像的游程编码RLE

# 加载必要的Python包
```python
import os
from pathlib import Path
from PIL import Image
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils import data
from torchvision import transforms
TRAIN_PATH = './train.pth'
TEST_PATH = './test.tph'
%matplotlib inline
```

# 数据集加载
数据集加载是至关重要的一步，也是非常繁琐的一步。因为这一步无法标准化，必须针对特定的数据集进行解析。
## 数据预处理
下面是对该竞赛的数据集的处理方法：
```python
def process(file_path, has_mask=True):
  file_path = Path(file_path)
  files = sorted(list(file_path.iterdir()))
  datas = []
  for file in tqdm(files):
    item = {}
    imgs = []
    for image in (file/'images').iterdir():
      img = io.imread(image)
      imgs.append(img)

    assert len(imgs) == 1

    if img.shape[2] > 3:
      assert (img[:, :, 3]!=255).sum() == 0
    img = img[:, :, :3]

    if has_mask:
      mask_files = list((file/'masks').iterdir())
      masks = None

      for i, mask in enumerate(mask_files):
        mask = io.imread(mask)
        assert (mask[(mask!=0)] == 255).all()

        if masks is None:
          H, W = mask.shape
          masks = np.zeros((len(mask_files), H, W))
        masks[i] = mask

      total_mask = masks.sum(0)
           assert (total_mask[(total_mask!=0)] == 255).all()

      item['mask'] = torch.from_numpy(total_mask)

    item['name'] = str(file).split('/')[-1]
    item['img'] = torch.from_numpy(img)
    datas.append(item)
    return datas
 
test = process("/content/drive/My Drive/nuclei/stage1_test", False)
torch.save(test, TEST_PATH)
train = process("/content/drive/My Drive/nuclei/stage1_train")
torch.save(train, TRAIN_PATH)
```

具体来看：
```python
  file_path = Path(file_path)
  files = sorted(list(file_path.iterdir()))
```
这里用了Python3的pathlib库来读入文件夹路径，之前大家常用的是os.path，但现在普遍推荐使用pathlib库来替代os.path，因为其采用面向对象的方式，且用法更简单，参加资料如下：
[pathlib路径库使用详解](https://xin053.github.io/2016/07/03/pathlib%E8%B7%AF%E5%BE%84%E5%BA%93%E4%BD%BF%E7%94%A8%E8%AF%A6%E8%A7%A3/)

然后再用list转换一下，是为了下面使用tqdm库来可视化进度条。
```python
    for image in (file/'images').iterdir():
      img = io.imread(image)
      imgs.append(img)

    assert len(imgs) == 1

    if img.shape[2] > 3:
      assert (img[:, :, 3]!=255).sum() == 0

    img = img[:, :, :3]
```
这一部分是读取具体的图像，但这里注意两点：
（1）首先对每一子文件夹下的图像数量进行判断，确保只有一张图像；
（2）这个数据集中的图像有个特点，它是4通道的，最后一个通道的值都是255，所以这里会有shape的判断，并且使用了assert来确保最后一个通道值都是255。
最后取原图像的前三个通道存入新图像中。

```python
    if has_mask:
      mask_files = list((file/'masks').iterdir())
      masks = None

      for i, mask in enumerate(mask_files):
        mask = io.imread(mask)
        assert (mask[(mask!=0)] == 255).all()

        if masks is None:
          H, W = mask.shape
          masks = np.zeros((len(mask_files), H, W))

        masks[i] = mask
```
这一步是逐个读取掩膜文件，其中的assert语句是保证mask确实是0和255二值的。
```python
      tmp_mask = masks.sum(0)
      assert (tmp_mask[(tmp_mask!=0)] == 255).all()
```
这一步是将masks中的同一位置上的元素进行加和，然后通过assert语句保证加和后不为0的元素都是255，这一步是保证每个像素上都最多只有一个掩膜值，即两个掩膜没有重叠。
```python
      total_mask = masks.sum(0)
```
因为masks变量实际有多个通道，即多个掩膜，这一步是将每个通道上的掩膜值根据序号重新赋值，然后组合在一起，使得所有掩膜都在一张图像上。(这一步与原notebook不同，原notebook是不同的掩膜有不一样的值)

比如这张细胞核图像：
![1f84ac0d-1df9-42c9-b4ee-ca08102cd715](https://user-images.githubusercontent.com/6218739/75554371-2b28f780-5a75-11ea-9357-928dde25f866.png)
它的掩膜就是：
![012a8162-4eaa-489c-8f35-e196651a8071](https://user-images.githubusercontent.com/6218739/75554405-3d0a9a80-5a75-11ea-9663-e85c11cf5df3.png)

```python
      item['mask'] = torch.from_numpy(total_mask)

    item['name'] = str(file).split('/')[-1]
    item['img'] = torch.from_numpy(img)
    datas.append(item)
```
然后将图像img、文件名name和掩膜mask（如果有的话）以字典的形式存入datas这个列表中。
```python
test = process("/content/drive/My Drive/nuclei/stage1_test", False)
torch.save(test, TEST_PATH)
train = process("/content/drive/My Drive/nuclei/stage1_train")
torch.save(train, TRAIN_PATH)
```
最后，将这个列表用PyTorch存储模型的方式持久化存储下来。
这一步需要的时间很长，所以最好是将这个存储数据的列表持久化后，将其挪动到Google Drive中，防止下次重启丢失。
但实际操作下来，发现过程不是那么美好，首先是Google Colab给分配的RAM有点小，而训练集中的数据特别多，尤其是mask分开存储的方式，使得数据量巨多，后期直接把内存撑爆了，而且执行速度非常慢，于是将代码稍微改了以下，每隔5步就torch save一下，然后将datas清零，这样就能保证及时释放内存：

```python
    if k % 5 == 0:
      torch.save(datas, name + np.str(k)+".pt")
      datas = []
```
然后再把所有save的数据load以后串接起来就行：
```python
def data_catenate(file_path):
    path = Path(file_path)
    data = []

    for i in tqdm(path.iterdir()):
        # print(i)
        temp = torch.load(i)
        data += temp
    return data
```

## 数据加载
首先定义数据集格式，自定义的数据集格式需要继承torch.utils.data.Dataset，然后重载以下两个方法：
（1）__len__：这样len(dataset)就可以返回整个数据集的大小，
（2）__getitem__：这样就可以使用dataset[i]来对数据进行索引。
针对这里的具体训练数据集：
```python
class TrainDataset(data.Dataset):
    def __init__(self, data, source_transform, target_transform):
        self.datas = data
        self.s_transform = source_transform
        self.t_transform = target_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:, :, None].byte().numpy()
        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask
    def __len__(self):
        return len(self.datas)
```
可以看出，对img和mask分别又做了一些变换。
对于这些变换操作来说，最佳实践是不要写函数，而是写可调用的类，这样参数就不必每次都要传递。因此，只需实现__call__方法和__init__方法（如有必要）。然后如下调用：
```python
tsfm = Transform(params)
transformed_sample = tsfm(sample)
```
这里因为所有的变换在torchvision的transforms中是自带的，所以不需要自定义变换，只调用即可，而且使用了Compose将这些变换组合起来：
```python
import PIL
s_trans = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((128, 128)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

t_trans = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((128, 128), interpolation=PIL.Image.NEAREST),
                              transforms.ToTensor()
])
```
然后将变换规则传入数据集中：
```python
train_dataset = TrainDataset(train, s_trans, t_trans)
```

具体使用该数据集时，可以使用for循环来逐个读取数据，但这样势必会丧失一些功能：
（1）批量处理数据；
（2）打乱数据顺序；
（3）并行加载数据。
因此，PyTorch提供了torch.utils.data.DataLoader类作为迭代器，提供上述功能。
将数据集放入DataLoader中，并指定参数：
```python
train_dataloader = data.DataLoader(train_dataset, num_workers=2, batch_size=4)
```

# UNet网络结构
崔家华的一篇博文对UNet的网络结构和代码实现讲得挺好，见[这里](https://cuijiahua.com/blog/2019/12/dl-15.html)。
本部分是对崔同学的博文的摘抄学习，代码部分中的参数是结合上面的notebook中的参数进行了修改。
## 网络结构原理
> UNet最早发表在2015的MICCAI会议上，5年多的时间，论文引用量已经达到了接近12000次。
> UNet成为了大多做医疗影像语义分割任务的baseline，同时也启发了大量研究者对于U型网络结构的研究，发表了一批基于UNet网络结构的改进方法的论文。
> UNet网络结构，最主要的两个特点是：U型网络结构和Skip Connection跳层连接。

![image](https://user-images.githubusercontent.com/6218739/75503501-bde37b00-5a10-11ea-99ac-9c9015aa1bb0.png)
> UNet是一个对称的网络结构，左侧为下采样，右侧为上采样。
> 按照功能可以将左侧的一系列下采样操作称为encoder，将右侧的一系列上采样操作称为decoder。
> Skip Connection中间四条灰色的平行线，Skip Connection就是在上采样的过程中，融合下采样过过程中的feature map。
> Skip Connection用到的融合的操作也很简单，就是将feature map的通道进行叠加，俗称Concat。


## 代码实现
将整个UNet结构拆分为多个模块进行分析。（后面的文字依然是摘抄自崔家华博客，不再加引用标识，见谅）
### DoubleConv模块
从UNet网络中可以看出，不管是下采样过程还是上采样过程，每一层都会连续进行两次卷积操作，这种操作在UNet网络中重复很多次，可以单独写一个DoubleConv模块。
```python
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```
上述的Pytorch代码：torch.nn.Sequential是一个时序容器，Modules 会以它们传入的顺序被添加到容器中。比如上述代码的操作顺序：卷积->BN->ReLU->卷积->BN->ReLU。
DoubleConv模块的in_channels和out_channels可以灵活设定，以便扩展使用。
输出矩阵的高度和宽度（即输出的特征图feature map）这两个维度的尺寸由输入矩阵、卷积核、扫描方式所共同决定，计算公式为：
![image](https://user-images.githubusercontent.com/6218739/75505152-71e70500-5a15-11ea-9784-2b27b6731e80.png)

### Down模块
UNet网络一共有4次下采样过程，模块化代码如下：
```python
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
```
这里的代码很简单，就是一个maxpool池化层，进行下采样，然后接一个DoubleConv模块。
其中，池化层选的是2乘以2的窗口大小，那么默认获得的也是这样大小的填充步长，池化以后的feature map的大小计算方式跟上面卷积的相同。

至此，UNet网络的左半部分的下采样过程的代码都写好了，接下来是右半部分的上采样过程。

### Up模块
上采样过程用到的最多的当然就是上采样了，除了常规的上采样操作，还有进行特征的融合。
```python
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```
代码复杂一些，我们可以分开来看。
首先是__init__初始化函数里定义的上采样方法以及卷积采用DoubleConv。
上采样，定义了两种方法：Upsample和ConvTranspose2d，也就是双线性插值和反卷积。
双线性插值很好理解，示意图：
![image](https://user-images.githubusercontent.com/6218739/75517269-49710200-5a39-11ea-8ae3-7316d43727cf.png)

简单地讲：已知Q11、Q12、Q21、Q22四个点坐标，通过Q11和Q21求R1，再通过Q12和Q22求R2，最后通过R1和R2求P，这个过程就是双线性插值。对于一个feature map而言，其实就是在像素点中间补点，补的点的值是多少，是由相邻像素点的值决定的。
反卷积，顾名思义，就是反着卷积。卷积是让featuer map越来越小，反卷积就是让feature map越来越大，示意图：

![image](https://user-images.githubusercontent.com/6218739/75517361-76bdb000-5a39-11ea-85a2-a869f7443195.png)

下面蓝色为原始图片，周围白色的虚线方块为padding结果，通常为0，上面绿色为卷积后的图片。 这个示意图，就是一个从2*2的feature map->4*4的feature map过程。
在forward前向传播函数中，x1接收的是上采样的数据，x2接收的是特征融合的数据。
如果两个feature map大小不同，那么特征融合方法可以有两种：
（1）将大的feature进行裁剪，再进行concat；
（2）将小的feature进行填充，再进行concat。

这里是使用的第二种，先对小的feature map进行padding，再进行concat。

### OutConv模块
用上述的DoubleConv模块、Down模块、Up模块就可以拼出UNet的主体网络结构了。UNet网络的输出需要根据分割数量，整合输出通道。具体操作就是channel的变换：
```python
class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.conv(x)
```
需要注意的是原notebook中没有加入Sigmoid层，但实测加入后精度提高很多。这样得到的就是一个0到1的概率分布。
### UNet网络
这一部分是将上面的模块组合起来形成整个UNet网络：
```python
""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

model = UNet(n_channels=1, n_classes=1).cuda()
```
这里还有另外一位网友写的UNet网络进行辅助理解，[点击这里](https://github.com/JavisPeng/u_net_liver/blob/master/unet.py)。

# 定义损失函数和优化器
这里使用Dice系数定义损失：
```python
def soft_dice_loss(inputs, targets):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score
```

具体的原理可以参见如下一篇博客：
[医学图像分割之 Dice Loss](https://www.aiuai.cn/aifarm1159.html)

优化器选择Adam：
```python
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
```

# 训练和保存模型
```python
for epoch in range(200):
    running_loss = 0.0

    for i, data in enumerate(train_dataloader, 0):
        x_train, y_train = data
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        optimizer.zero_grad()
        predict = model(x_train)
        loss = soft_dice_loss(predict, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 20 == 19:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print("Finish training")
```
这里在GPU上进行训练。
训练完后及时地将模型保存下来：
```python
MODEL_PATH = 'model.pth'
torch.save(model.state_dict(), MODEL_PATH)
```

#测试
## 创建测试数据集和加载器
```python
class TestDataset():

    def __init__(self, path, source_transform):
        self.datas = torch.load(path)
        self.s_transform = source_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        img = self.s_transform(img)
        return img

    def __len__(self):
        return len(self.datas)

test_dataset = TestDataset(TEST_PATH, s_trans)
test_dataloader = data.DataLoader(test_dataset,num_workers=2,batch_size=2)
```

## 查看一下测试集
```python
dataiter = iter(test_dataloader)
imgs = dataiter.next()
```
注意上面的imgs的size()是这样的：
```python
torch.Size([2, 1, 128, 128])
```
最前面的2是batch size，说明dataloader中一个元素包含两张图像，所以保存时需要这样：
```python
io.imsave("1.png", imgs[0][0].data.numpy())
```

## 加载模型（可选）
如有必要的话，先从模型文件中加载模型：
```python
model = UNet(1, 1).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
```

## 测试
```python
output = []
with torch.no_grad():
    for data in test_dataloader:
        data = data.cuda()
        # io.imsave("train.png", data[1][0].data.cpu().numpy())
        predict = model(data)
        # io.imsave("test.png", o[1][0].data.cpu().numpy())
        output.append(predict)

    result = torch.cat(output, dim=0)
```
然后得到的result可以通过设置阈值来二值化显示最终结果，比如第i张图像：
```python
pred = result[i][0].data.cpu().numpy()
pred[pred>0.7] = 255
pred[pred<=0.7] = 0
io.imsave("i.png", pred)
```
