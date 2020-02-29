---
title: Kaggle钢铁赛：基于PyTorch/UNet算法的钢材表面缺陷检测——（2）算法分析
tags: [Machine Learning, PyTorch]
categories: programming
date: 2020-2-16
---

# 简介

Kaggle上有一个[钢材表面缺陷检测的竞赛](https://www.kaggle.com/c/severstal-steel-defect-detection/overview)，是一个很好的将深度学习应用于传统材料检测的例子。对该赛题和解法的剖析，可以辅助理解深度学习的流程，及其应用于具体问题的套路。

这次解析分为两部分：

（1）第一部分，即[上一篇文章](https://qixinbo.info/2020/02/15/kaggle-steel/)，是一个预备性工作，即对该竞赛的数据集的分析和可视化，参考的是这个notebook[clear mask visualization and simple eda](https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda)。感谢GoldFish的分享。

（2）第二部分，即本文，参考的是Rishabh Agrahari的[使用PyTorch框架及UNet算法的notebook](https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88)，中间穿插了很多背景知识介绍。

再次声明一下，本次训练是在Google Colab上进行，所以有的命令行命令有些许不同，比如sh命令都加上了叹号，不过不影响理解。

一些关于卷积神经网络的预备知识：
-[CNN笔记：通俗理解卷积神经网络](https://blog.csdn.net/v_JULY_v/article/details/51812459)
-[CNN中卷积层的计算细节](https://zhuanlan.zhihu.com/p/29119239)
-[CNN中的参数解释及计算](https://flat2010.github.io/2018/06/15/%E6%89%8B%E7%AE%97CNN%E4%B8%AD%E7%9A%84%E5%8F%82%E6%95%B0/)

# 加载预训练模型
这一部分的参考文献有：
-[Pytorch深度学习实战教程（一）：语义分割基础与环境搭建](https://mp.weixin.qq.com/s/KI-9z7FBjfoWfZK3PEPXJA)
-[Pytorch深度学习实战教程（二）：UNet语义分割网络](https://mp.weixin.qq.com/s/6tZVUbyEjLVewM8vGK9Zhw)
-[UNet以ResNet34为backbone in keras](https://blog.csdn.net/m0_37477175/article/details/83861678)
-[一大波PyTorch图像分割模型来袭，俄罗斯程序员出品新model zoo](https://www.qbitai.com/2019/05/2157.html)

这个notebook没有用UNet传统的编码器和解码器，而是用的预训练的resnet18网络作为编码器，再在此基础上，构建相应的解码器。这个带预训练resnet18的UNet是借用了这个开源库[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)。该库目前提供了5种模型结构，每个架构有46种可用的编码器，且所有的编码器都具有预训练权重，因此广受好评。以上内容的具体原理可以参见上面的参考文献的描述。

在原notebook中，Rishabh Agrahari在Kaggle的kernel中没有正确通过pip安装这个库，所以他把整个库下载下来然后传到kaggle数据集中了，但实测在Colab中可以通过pip成功安装，如下：

```python
!pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
测试一下：
```python
import segmentation_models_pytorch as smp
model = smp.Unet()
```
会显示成功下载预训练模型resnet34，这是因为没有给它传参，默认下载并使用该模型。
另外，如果下一次再使用该notebook时，会发现之前安装的包都没了，此时需要永久安装这些python包，见下面的参考文献：
-[How do I install a library permanently in Colab?](https://stackoverflow.com/questions/55253498/how-do-i-install-a-library-permanently-in-colab)

# 导入必要的Python包
```python
import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```

# 掩膜的编码及解码

```python
#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks
```
这一步是定义了两个函数：
mask2rle函数是将mask使用RLE编码，RLE的全称是Run-length encoding，称为游程编码，是一种无损数据压缩技术。
make_mask是为了对csv文件中mask进行RLE解码。
比如'1 3'代表起点是像素1，然后长度为3个像素，即像素索引为(1, 2, 3)。然后不同游程之间也是用空格分隔，比如'1 3 10 5'代表的就是(1, 2, 3, 10, 11, 12, 13, 14)这8个像素。

参考资料：
-[游程编码](https://zh.wikipedia.org/wiki/%E6%B8%B8%E7%A8%8B%E7%BC%96%E7%A0%81)
-[对mask进行rle编码然后进行解码-详细注释](https://blog.csdn.net/appleyuchi/article/details/102938491)
-[RLE functions - Run Lenght Encode & Decode](https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode)

#数据加载
```python
class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)

def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader
```

这一步是依据PyTorch的规范建立自己的数据集、数据变换、数据加载器等。
具体可以参考官方的创建自定义数据集及加载器一节：
[Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

# 衡量指标定义
```python
def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)
        p = (probability > threshold).float()
        t = (truth > 0.5).float()
        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)
        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))
        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
#         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
#         dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)
    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue

        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)

    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou
```
这一部分是计算衡量指标Dice和IoU。
各个指标的意义可以参考如下资料：
[图像分割评价指标](https://www.zdaiot.com/MachineLearning/%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87/)

# 模型初始化
```python
import segmentation_models_pytorch as smp
model = smp.Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
```
对UNet网络设定具体的参数，如backbone选择resnet18，预训练权重为imagenet，四分类，无激活函数等。
可以具体看一下该模型：
```python
model
```
输出结果太长了，不再具体显示。

# 模型训练和验证
```
class Trainer(object):

    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)

        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()

                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()
```

定义了Trainer类，用来处理模型的训练和验证。
```python
sample_submission_path = '/content/drive/My Drive/severstal/sample_submission.csv'
train_df_path = '/content/drive/My Drive/severstal/train.csv'
data_folder = "/content/drive/My Drive/severstal"
test_data_folder = "/content/drive/My Drive/severstal/test_images"
```
设定一系列路径。

```python
model_trainer = Trainer(model)
model_trainer.start()
```

下面就进入漫长的训练和验证阶段，因为这里是分析源码功能，不对精确性做考虑，所以这里将默认的20个epoches改为了2个，输出结果为：

```python
Starting epoch: 0 | phase: train | ⏰: 03:39:08
Loss: 0.0386 | IoU: 0.1436 | dice: 0.3747 | dice_neg: 0.5761 | dice_pos: 0.1963
Starting epoch: 0 | phase: val | ⏰: 03:59:18
Loss: 0.0205 | IoU: 0.3210 | dice: 0.5299 | dice_neg: 0.6596 | dice_pos: 0.4149
******** New optimal found, saving state ********

Starting epoch: 1 | phase: train | ⏰: 04:03:09
Loss: 0.0196 | IoU: 0.2905 | dice: 0.5555 | dice_neg: 0.7579 | dice_pos: 0.3763
Starting epoch: 1 | phase: val | ⏰: 04:21:04
Loss: 0.0166 | IoU: 0.2829 | dice: 0.6151 | dice_neg: 0.9035 | dice_pos: 0.3596
******** New optimal found, saving state ********
```

可以看出，一个epoch大约需要25分钟时间。
查看一下此时Colab所使用的GPU：
```python
torch.cuda.get_device_name(0)
```

可以看出此时是Tesla T4，Google还挺给力。
如果是需要长时间的训练，当前Google Colab还有一些小tricks，比如最好不要关闭浏览器窗口，因为关闭后90mins后就会该实例就会被切断。但一直保持浏览器窗口也只能最多训练12 hours。见：
[Can i run a google colab (free edition) script and then shutdown my computer?](https://stackoverflow.com/questions/55050988/can-i-run-a-google-colab-free-edition-script-and-then-shutdown-my-computer)
也有网友分享了一个简单的JS函数来模拟点击，自动重连，见：
[How to prevent Google Colab from disconnecting?](https://medium.com/@shivamrawat_756/how-to-prevent-google-colab-from-disconnecting-717b88a128c0)
 
# 得分作图
```python
# PLOT TRAINING
losses = model_trainer.losses
dice_scores = model_trainer.dice_scores # overall dice
iou_scores = model_trainer.iou_scores

def plot(scores, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
    plt.legend(); 
    plt.show()

plot(losses, "BCE loss")
plot(dice_scores, "Dice score")
plot(iou_scores, "IoU score")
```

如图，因为仅训练了2个epoches，所以这里的作图只是展示一下。
![image](https://user-images.githubusercontent.com/6218739/74802202-19787f00-5314-11ea-8ec0-bbe861ccfd1e.png)
![image](https://user-images.githubusercontent.com/6218739/74802207-1ed5c980-5314-11ea-88f3-f04125adf692.png)
![image](https://user-images.githubusercontent.com/6218739/74802212-2301e700-5314-11ea-83b6-d1a22c538149.png)

# 推理和提交
因为原文中作者是在Kaggle GPU上进行训练，整个训练过程约400分钟，超过了Kaggle的60min的限制，所以作者没有在这个notebook中进行推理和提交。他又写了两个一个notebook来进行后面的推理和提交，见[这里](https://www.kaggle.com/rishabhiitbhu/unet-pytorch-inference-kernel)。

因为这里没有实际的训练，所以这里将这一个后续的notebook也直接附在这里。
## 定义测试数据集
```python
class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples

# initialize test dataloader
best_threshold = 0.5
num_workers = 2
batch_size = 4
print('best_threshold', best_threshold)
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
```
为测试集创建PyTorch规范的Dataset和DataLoader。

## 后处理
```python
def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)

    num = 0

    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num
```
这一步是对掩膜进行后处理。

## 加载模型
```python
# Initialize mode and load trained weights
ckpt_path = "model.pth"
device = torch.device("cuda")
model = smp.Unet("resnet18", encoder_weights=None, classes=4, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])
```
这一步是初始化模型及加载模型。这也是持久化的需求。
该模型文件pth是之前训练时自动存储的。

## 推理并生成提交文件
```python
# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()

    for fname, preds in zip(fnames, batch_preds):
        for cls, pred in enumerate(preds):
            pred, num = post_process(pred, best_threshold, min_size)
            rle = mask2rle(pred)
            name = fname + f"_{cls+1}"
            predictions.append([name, rle])

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("submission.csv", index=False)
```
可以发现，此时当前目录下回生成submission.csv文件，里面的掩膜信息也是用RLE游程编码的。
