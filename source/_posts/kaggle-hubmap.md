---
title: Kaggle肾小球图像分割比赛全解析
tags: [kaggle]
categories: programming
date: 2021-2-14
---

# 概览

## 赛事描述
Kaggle上正在进行一项名为“HuBMAP: Hacking the Kidney”的竞赛，链接见[这里](https://www.kaggle.com/c/hubmap-kidney-segmentation)，总奖金是6万美金，该竞赛的目的是为了检测人体组织中的functional tissue units (FTUs)。FTU的定义是：three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block，感觉类似于细胞的概念，正中是细胞核，周围是细胞质。
 

## 算法评估标准
该竞赛使用Dice系数来评估算法的优劣。关于Dice系数，可以见如下博客解析：
[医学图像分割之 Dice Loss](https://www.aiuai.cn/aifarm1159.html)

竞赛所提交的文件使用游程编码方式（RLE，run-length encoding）来减小文件体积。
关于掩膜mask与rle编码与解码的代码，可以参见网友Paulo Pinto的notebook：
[RLE functions - Run Lenght Encode & Decode](https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode)

# 数据分析

## 数据集概览
该赛事中的数据一共有20张肾的图像，每一张都对其中的肾小球FTU进行了标注，有8张用于训练集，5张用于公榜测试集，剩下7张用于私榜测试集。
每一张图像都是非常大的TIFF格式，500MB-5GB大小。

训练集中的标注有两种形式：游程编码和未编码的JSON格式。
可以使用外部数据和/或预训练的机器学习模型，不过这些数据和模型必须在CC BY 4.0下授权。
下载数据集（这是在google colab上运行，colab上的机器性能较好；如果直接使用kaggle上的notebook，则数据集直接内置）
```python
!kaggle competitions download -c hubmap-kidney-segmentation
```
然后进行数据的探索性分析，该过程参考了以下notebook：
[HuBMAP - Exploratory Data Analysis](https://www.kaggle.com/ihelon/hubmap-exploratory-data-analysis)

## 导入必要的包
```python
import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import albumentations as A
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T
```

## 配置路径及超参数
```python
BASE_PATH = "../input/hubmap-kidney-segmentation/"
TRAIN_PATH = os.path.join(BASE_PATH, "train/")
TEST_PATH = os.path.join(BASE_PATH, "test/")

EPOCHES = 5
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 文件分析
（1）训练集文件分析：
```python
df_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
df_train
```

得到：
```python
           id                        encoding
0        2f6ecfcdf        296084587 4 296115835 6 296115859 14 296147109...
1        aaa6a05cc        30989109 59 31007591 64 31026074 68 31044556 7...
2        cb2d976f4        78144363 5 78179297 15 78214231 25 78249165 35...
3        0486052bb        101676003 6 101701785 8 101727568 9 101753351 ...
4        e79de561c        7464094 14 7480273 41 7496453 67 7512632 82 75...
5        095bf7a1f        113430380 22 113468538 67 113506697 111 113544...
6        54f2eec69        124601765 36 124632133 109 124662536 147 12469...
7        1e2425f28        49453112 7 49479881 22 49506657 31 49533433 40...
```
train.csv文件中包含了图像的id及其游程编码。可以看出图像的名称就是id名。
（2）提交文件分析：
```python
df_sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
df_sub
```
得到：
```python
   id                          predicted
0        b9a3865fc        NaN
1        b2dc8411c        NaN
2        26dc41664        NaN
3        c68fe75ea        NaN
4        afa5e8098        NaN
```
提交文件就是对公共测试集上的图像的预测，可以看出id就是公共测试集中的图像名称，predicted一栏需要后面填入。
（3）数据集大小分析：
```
print("number of train images: ", df_train.shape[0])
print("number of test images: ", df_sub.shape[0])
```
分别是8和5。
（4）元数据分析：
```python
df_meta = pd.read_csv(os.path.join(BASE_PATH, "HuBMAP-20-dataset_information.csv"))
df_meta.sample(3)
```
该文件中包含了数据集中的每一张图像额外的信息，比如它的主人的身体信息、性别、种族等。
同时指明训练集中除了肾小球的标注文件，比如1e2425f28.json，还有其他解剖组织的标注文件，比如1e2425f28-anatomical-structure.json。
该文件是为了辅助理解该赛题背后的医学知识，有可能对特征功能有用，但目前看没法直接使用。

## 工具函数

以下是关于游程编码和解码、读取图像、可视化的工具代码。
```python
# 图像分块
# min_overlap这个参数指的是有可能出现的最小的overlap，而不是保证这个overlap一定会出现
# 即自适应产生的overlap肯定会大于该min_overlap
def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]   
    return slices.reshape(nx*ny,4)

# 固定随机数，以保证可复现性
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 游程编码转为图像掩膜
def rle2mask(mask_rle, shape):
    # shape的形状是(width, height), width是通常理解的图像宽度
    # 原始的rle编码是str类型，里面的元素成对出现，即start起始像素及length长度
    s = mask_rle.split()

    # 将s中的start和length分别提取出来，因为它们在原始列表中是成对出现，所以这里对这两种数据都是每隔两个元素提取一次
    # 然后再通过python的列表生成式语法另存成numpy数组
    # https://www.liaoxuefeng.com/wiki/1016959663602400/1017317609699776
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # 原始csv文件中的游程编码是从绝对位置开始，因此转化为numpy数组时需要减1
    starts -= 1
    ends = starts + lengths

    # 根据原始图像大小建立一个空白图像
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    # 根据起始像素和结束像素，在该空白图像上创建掩膜
    # 1为mask，0为背景
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1

    # 将img按输入shape变换形状，这里就体现了shape中元素顺序的关键
    # 因为RLE编码是先从上到下，然后再从左到右进行的，所以分割时是先满足高度要求，即先按一列一列地来分组
    # 因为输入的shape是宽度在前，高度在后，正好reshape就按这个shape来变换形状
    # 比如原图如果宽为5，高为2，那么就reshape((5, 2))，即分成5组，每组2个元素
    # 然后因为图像存成numpy数组时是行数乘以列数，即转置一下即可
    return img.reshape(shape).T

# 图像掩膜转为游程编码
# https://blog.csdn.net/qq_35985044/article/details/104332577
def mask2rle(img):
    # 将掩膜按从上到下、从左到右的顺序压平
    pixels = img.T.flatten()

    # 前后各加一个0作为缓冲区
    pixels = np.concatenate([[0], pixels, [0]])

    # 以下记录的是掩膜值开始发生变化的位置，使用的方法是将数组错移一位，并与原数组比较
    # 这样每一段重复的序列在前后位置都有一个变化的位置记录，前后位置是成对出现的
    # +1是为了做位置调整
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 这一步比较抽象，首先记住runs记录了像素值发生变化的位置
    # runs[1::2]是从第二个位置开始，每隔两个元素提取一次，即该表达式提取的是“成对出现的前后位置”中的“后”这一位置
    # runs[::2]则是从第一个位置开始，每隔两个元素提取一次，即该表达式提取的是“成对出现的前后位置”中的“前”这一位置
    # 然后两者相减，并在原来“后”这一位置存储差值，即每个重复序列的长度
    runs[1::2] -= runs[::2]

    # 将上述元素逐个取出，并且使用空格符连接成字符串
    return ' '.join(str(x) for x in runs)

# 使用numba加速
# 可以参考如下nb里的讨论部分
# https://www.kaggle.com/leighplt/pytorch-fcn-resnet50/comments
@numba.njit()
def mask2rle_numba_1d(pixels):
    size = len(pixels)
    points = []
    if pixels[0] == 1: points.append(1)
    for i in range(1, size):
        if pixels[i] != pixels[i-1]:
            if len(points) % 2 == 0:
                points.append(i+1)
            else:
                points.append(i+1 - points[-1])

    if pixels[-1] == 1: points.append(size-points[-1]+1)   
    return points

# 该函数必须得与上面的分开，因为最后的join函数不支持numba
def mask2rle_numba(image):
    pixels = image.T.flatten()
    points = mask2rle_numba_1d(pixels)
    return ' '.join(str(x) for x in points)

# 读取训练集图像
# 对于大型tif图像，推荐使用rasterio来读取，读取速度会提升很多倍
def read_image(image_id, scale=None, verbose=1):
    # python3.6引入的f-string，用于格式化字符串
    # https://blog.csdn.net/sunxb10/article/details/81036693
    image = tifffile.imread(os.path.join(TRAIN_PATH, f"{image_id}.tiff"))

    # 数据集中有几张图像是(1, 1, 3, X, Y)这样的格式，需要将其转为正确的(X, Y, 3)格式
    if len(image.shape) == 5:
        image = image.sequeeze().transpose(1, 2, 0)

    # 这个地方用到了pandas对象的布尔数组索引
    # https://www.pypandas.cn/docs/user_guide/indexing.html#%E7%B4%A2%E5%BC%95%E7%9A%84%E4%B8%8D%E5%90%8C%E9%80%89%E6%8B%A9
    # 特别需要注意的是第二个参数的顺序，这里是图像的宽度在前，高度在后
    mask = rle2mask(df_train[df_train["id"] == image_id]['encoding'].values[0], (image.shape[1], image.shape[0]))

    # 开启详细显示信息模式
    if verbose:
        print(f"[{image_id}] Image shape: {image.shape}")
        print(f"[{image_id}] Mask shape: {mask.shape}")

    # 缩放
    if scale:
        # 注意opencv的resize函数要求的参数是先宽后高，注意顺序
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        mask = cv2.resize(mask, new_size)

        if verbose:
            print(f"[{image_id}] Image shape: {image.shape}")
            print(f"[{image_id}] Mask shape: {mask.shape}")

    return image, mask

# 读取测试集图像
def read_test_image(image_id, scale=None, verbose=1):
    image = tifffile.imread(os.path.join(TEST_PATH, f"{image_id}.tiff"))

    if len(image.shape) == 5:
        image = image.sequeeze().tranpose(1, 2, 0)

    if verbose:
        print(f"[{image_id}] Image shape: {image.shape}")

    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)

        if verbose:
            print(f"[{image_id}] Image shape: {image.shape}")

    return image

# 可视化图像及其掩膜
# 主要是应用了matplotlib库，其用法可参考如下链接
# https://lijin-thu.github.io/06.%20matplotlib/06.01%20pyplot%20tutorial.html
def plot_image_and_mask(image, mask, image_id):
    # 产生一幅图，指定其大小
    plt.figure(figsize=(16, 10))

    # 创建一行三列的子图
    # 这里是第一个子图
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image {image_id}", fontsize=18)

    # 第二个子图
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(mask, cmap="hot", alpha=0.5)
    plt.title(f"Image {image_id} + mask", fontsize=18)

    # 第三个子图
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="hot")
    plt.title(f"Mask", fontsize=18)

    plt.show()

# 可视化图像及其掩膜的一部分
def plot_slice_image_and_mask(image, mask, start_h, end_h, start_w, end_w):
    plt.figure(figsize=(16, 5))
    sub_image = image[start_h : end_h, start_w : end_w, :]
    sub_mask = mask[start_h : end_h, start_w : end_w]
    plt.subplot(1, 3, 1)
    plt.imshow(sub_image)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sub_image)
    plt.imshow(sub_mask, cmap="hot", alpha=0.5)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(sub_mask, cmap="hot")
    plt.axis("off")
    plt.show()
```

依据这些工具函数，可以很方便地进行数据集的读取和调用。
以训练集中的某张图像为例，观察其部分原图及其标注：
```python
image_id = "0486052bb"
image, mask = read_image(image_id, 2)
plot_image_and_mask(image, mask, image_id)

plot_slice_image_and_mask(image, mask, 5000, 7500, 2500, 5000)
plot_slice_image_and_mask(image, mask, 5250, 5720, 3500, 4000)
```
结果如下图：
![vis](https://user-images.githubusercontent.com/6218739/100980887-e7719a00-3580-11eb-9dcd-2921740efa80.png)

## 无网络连接安装依赖
因为这个比赛的notebook不允许使用Internet，因此，如果调用了kaggle默认环境所没有的模块和包时，需要事先自己在线下准备好。
具体做法可以参考[该教程](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/113195)：
（1）首先打开notebook的网络，然后使用pip下载所依赖的包的whl文件：
```python
!pip download [package_name]
```
记住此时download的顺序以及版本号（这点一定要注意），后面要按该顺序的逆序进行安装。
（2）将这些whl文件上传到kaggle的dataset中；
这一步又涉及怎样将这些文件先下载下来，此时可以参考[该教程](https://www.kaggle.com/getting-started/168312)：
一种是直接在右侧的output侧栏中点击下载；
一种是通过命令下载：
```python
%cd /kaggle/working
from IPython.display import FileLink -> FileLink(r'*name of file*')
```
注意，在kaggle上上传非whl的安装包时，比如上传的时.tar.gz格式的源码包，kaggle会自动将其解压成文件夹。
因此，需要将此种文件后缀名需要更改为.xyz，然后上传。
上传时如果发现其他dataset仓库已经有相同的软件包，可以直接用那里的，也可以选择include duplicates上传自己的。
（3）在该notebook中引用上面的dataset，然后按逆序安装这些whl。
对于非whl的文件，注意将其copy到本地路径后，再修改为.tar.gz后缀名，然后正常pip install即可。

# 正确提交一次和跑一遍模型
## 正确提交一次
这里首先通过正确提交一次，保证提交文件是正确的，否则可能花了很多时间搭建算法和训练，最后却卡在提交上，没法及时提交。
因为Kaggle的submission目前看像是一个玄学，大家都在尝试怎样提交成功，比如下面的讨论：
[Scoring error on submission - even with sample_submission.csv](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/116409)
[Scoring error again](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/123466)
原因就是如第一个帖子中Julia所说，kaggle官方为了确保测试集不会被大家hack或产生leak，将测试集藏得很深。
想要提交成功，要假设测试集就在那里，且文件的ID千万不能硬编码，要做到“自适应”，同时里面的内容一开始可以全部设为0，但也注意有些竞赛对于数值也有要求，比如下面的帖子中，因为最后的score要计算Spearman系数，所以每一个array中至少要有两个不同的值：
[Unable to fix submission scoring error](https://www.kaggle.com/c/google-quest-challenge/discussion/126777)
针对于此例，一个很小的提交如下：
```python
DATA_PATH = '../input/hubmap-kidney-segmentation'
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

p = pathlib.Path(DATA_PATH)
subm = {}

for i, filename in enumerate(p.glob('test/*.tiff')):
    dataset = rasterio.open(filename.as_posix(), transform = identity)
    preds = np.zeros(dataset.shape, dtype=np.uint8)
    subm[i] = {'id':filename.stem, 'predicted': mask2rle_numba(preds)}
    del preds
    gc.collect();

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('submission.csv', index=False)
```

## 跑一遍模型
这里实际是用随机权重的模型在测试集上跑一遍，保证整个流程是通路的，然后再进行模型的训练，即“以终为始”。

```python
def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(False)

    # 将原来的用于多类分割的模型最后一层改为目前的两类分割
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

# 这一步也非常重要，因为不能有网络连接，如果不事先下载好这些权重文件，notebook运行过程中会联网下载，导致提交不成功
!mkdir -p /root/.cache/torch/hub/checkpoints/
!cp ../input/pytorch-pretrained-models/resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/
!cp ../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth /root/.cache/torch/hub/checkpoints/

model = get_model()
model.to(DEVICE);

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

# 这个变换一定要与训练集的变换相同
trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(NEW_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])
 
p = pathlib.Path(DATA_PATH)

subm = {}

## 真正使用过程中，在训练模型后，在这里要加载训练好的模型
# model.load_state_dict(torch.load("../input/models/HuBMAP_model_best.pth"))
model.eval()

# 遍历测试集中的文件
for i, filename in enumerate(p.glob('test/*.tiff')):
    # 使用rasterio来打开tiff文件，实测速度会很快
    # transform参数是用来进行仿射变换，这里不涉及坐标系变换，可以不用
    # 常用用法可见：
    # https://theonegis.github.io/geos/%E4%BD%BF%E7%94%A8Rasterio%E8%AF%BB%E5%8F%96%E6%A0%85%E6%A0%BC%E6%95%B0%E6%8D%AE/index.html
    dataset = rasterio.open(filename.as_posix(), transform = identity)

    # 因为原始图像很大，为防止爆内存，将其分块处理
    slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
    # 创建一个存储预测结果的缓存
    preds = np.zeros(dataset.shape, dtype=np.uint8)
    # 对分块图像进行遍历
    for (x1,x2,y1,y2) in slices:
        # read方法将rasterio的数据集格式转为numpy.ndarray
        image = dataset.read([1,2,3], window=Window.from_slices((x1,x2),(y1,y2)))
        # 改变一下维度次序
        image = np.moveaxis(image, 0, -1)
        # 与训练集采用同样的tranform变换
        image = trfm(image)
        with torch.no_grad():
            # 将image放入DEVICE中，cpu或gpu
            image = image.to(DEVICE)[None]
            # 模型对图像进行预测
            # 这个地方需要注意的是model的返回值
            # torchvision模型库中的classification和segmentation、detection的模型返回值不同
            # 比如，用于classification的模型，比如AlexNet，其model(X)返回的就是预测值y的tensor
            # 而segmentation的模型，比如fcn_resnet50，它的返回值是一个有序字典，其中的key有out和aux
            # 所以下面的代码需要使用out这个key来获得预测值tensor
            # 可以参考：
            # https://pytorch.org/docs/stable/torchvision/models.html
            # https://github.com/pytorch/vision/blob/d0063f3d83beac01e85f3027c4de6499a8985469/torchvision/models/segmentation/fcn.py#L9
            # https://colab.research.google.com/github/spmallick/learnopencv/blob/master/PyTorch-Segmentation-torchvision/intro-seg.ipynb#scrollTo=ZsIngeXleQ1H
            score = model(image)['out'][0][0]

            # 这个得分有可能是负的，为了一致性，将其使用sigmoid激活，转为0到1
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = cv2.resize(score_sigmoid, (WINDOW, WINDOW))

            # 以0.5作为阈值，输出mask
            preds[x1:x2,y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)
   
    # 将预测值转为RLE编码
    subm[i] = {'id':filename.stem, 'predicted': mask2rle_numba(preds)}
    # 删除临时变量
    del preds
    # 回收内存，防止内部爆掉，这一步非常重要
    gc.collect();

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('submission.csv', index=False)
```

# 制作数据集
因为原始图像太大了，单张甚至达到5G大小，极易爆内存，因此需要将其切分成小图像，才能进行可行的训练。
## 在线制作数据集
这一节之所以称为“在线制作”，是因为数据集的加载和切分都是在程序运行时才开始进行，与之对应的，下面一节采用的是事先将原来的数据集切分好，等到用的时候直接读入即可。
这一节的在线制作是直接制作了适用于PyTorch的数据集格式，如下：
```python
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

# 自定义数据集的制作可以参考如下链接：
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class HubDataset(D.Dataset):
    def __init__(self, root_dir, transform,
                 window=256, overlap=32, threshold = 100):
        # 加载路径使用pathlib，后续推荐使用它来替代os.path
        # 具体使用方法可以参考：
        # https://docs.python.org/zh-cn/3/library/pathlib.html
        self.path = pathlib.Path(root_dir)
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'train.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold
        self.x, self.y = [], []

        # 运行下面的分块函数
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            # ToTensor函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255，
            # 从而将数据范围变换到[0, 1]之间
            # https://www.cnblogs.com/ocean1100/p/9494640.html
            T.ToTensor(),

            # 标准化的操作是减去均值并除以标准差，即将数据变换为均值为0、标准差为1的标准正态分布
            # 之所以标准化，常用的解释是：（1）突出特征；（2）方便反向传播的计算，具体讨论可以参考：
            # http://www.soolco.com/post/62169_1_1.html
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    # 分块
    def build_slices(self):
        self.masks = []
        self.files = []
        self.slices = []

        for i, filename in enumerate(self.csv.index.values):
            filepath = (self.path /'train'/(filename+'.tiff')).as_posix()
            self.files.append(filepath)

            print('Transform', filename)

            with rasterio.open(filepath, transform = identity) as dataset:
                self.masks.append(rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape))
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)

                # 排除掉一些没有目标对象的分块
                for slc in tqdm(slices):
                    x1,x2,y1,y2 = slc
                    if self.masks[-1][x1:x2,y1:y2].sum() > self.threshold or np.random.randint(100) > 120:
                        self.slices.append([i,x1,x2,y1,y2])

                        image = dataset.read([1,2,3],
                            window=Window.from_slices((x1,x2),(y1,y2)))

                         # if image.std().mean() < 10:
                         #     continue

                         # print(image.std().mean(), self.masks[-1][x1:x2,y1:y2].sum())

                        image = np.moveaxis(image, 0, -1)
                        self.x.append(image)
                        self.y.append(self.masks[-1][x1:x2,y1:y2])

    # get data operation
    def __getitem__(self, index):
        image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

WINDOW=1024
MIN_OVERLAP=32
NEW_SIZE=256

# albumentations是一个基于OpenCV的数据增强库，拥有非常简单且强大的可以用于多种任务（分割、检测）的接口，易于定制且添加其他框架非常方便。
# 它可以对数据集进行逐像素的转换，如模糊、下采样、高斯噪声、高斯模糊、动态模糊、RGB转换、随机雾化等；
# 也可以进行空间转换（同时也会对目标标签进行转换），如裁剪、翻转、随机裁剪等
# 数据增强的方式是这样的：
# 比如在一个epoch之内，我是把所有的图片都过一遍，对于每张图片我都是进行一个transform的操作，比如transform内部有0.5的概率进行左右flip，
# 那么这张图片左右flip的概率就是0.5，可能这一个epoch不flip，下一个epoch就会flip.
# 换句话说，现有的数据增强是带有随机性的，比如是否随机镜像翻转，随机crop的时候选择哪块区域，加多强的噪声等，每次增强的结果可能都不一样，
# 这样模型相当于看到了很多份不同的图像。如果没有概率性的操作，即p=1， 做一次增强之后便不再变化，则和不做增强是等价的，即模型在整个训练过程中只能看到一份相同的不断重复的数据。
# 可以详见
# https://discuss.gluon.ai/t/topic/1666
trfm = A.Compose([
    A.Resize(NEW_SIZE,NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                   saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),

    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.0),
    A.ShiftScaleRotate(),
])
 

# 使用模型库里的模型时，有的模型所需要的输入图像的尺寸是固定的，比如必须是64*64等，所以在图像增强后要保证图像的尺寸满足该要求。
# 可以采取以下方法来满足任意图像尺寸的输入：
#（1）传统办法：预处理，也就是边缘补0，或者切割，或者重采样、resize来得到相同大小的输入图片作为input。
#（2）模型方法：使用Kaiming He大佬的SPP-Net可以输入任意大小的图片，原文arxiv地址：https://arxiv.org/abs/1406.4729
#（3）优雅方法：加一层torch.nn.AdaptiveMaxPool2d。
# 具体讨论可以见
# https://www.zhihu.com/question/45873400

ds = HubDataset(DATA_PATH, window=WINDOW, overlap=MIN_OVERLAP, transform=trfm)

# 划分验证集和训练集
valid_idx, train_idx = [], []
for i in range(len(ds)):
    # 挑出第8张图片的所有切片作为验证集
    if ds.slices[i][0] == 7:
        valid_idx.append(i)
    else:
        train_idx.append(i)

train_ds = D.Subset(ds, train_idx)
valid_ds = D.Subset(ds, valid_idx)

# define training and validation data loaders
loader = D.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

vloader = D.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
```

## 离线制作数据集
如上所述，上述制作数据集的方式是离线进行的，该数据集准备过程参考了如下notebook：
[256x256 images](https://www.kaggle.com/iafoss/256x256-images)
具体切分方式与上面在线制作时不同，但表达的意思相同。

```python
# 对迭代器进行tqdm封装，传入total参数来指明预计迭代次数
# https://ptorch.com/news/170.html
# 如果想选择dataframe某一行作为测试，可以参考dataframe各种索引方法
# https://www.jianshu.com/p/32bfb327bf07

for index, rles in tqdm(df_mask.iterrows(), total=len(df_mask)):
    print(index)
    img, mask = read_image(index)
    shape = img.shape
    tile_before_compress = compress * tile_size

    # 计算填充量，使得可以整除
    pad0 = tile_before_compress - shape[0] % tile_before_compress
    pad1 = tile_before_compress - shape[1] % tile_before_compress

    # 用0填充
    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    # 对于img，x和y两个维度都填充，第三维度则不填充，所以第三维度设为(0, 0)
    img = np.pad(img, ((pad0 // 2, pad0 - pad0 // 2), (pad1 // 2, pad1 - pad1 // 2), (0, 0)), constant_values=0)
    mask = np.pad(mask, ((pad0 // 2, pad0 - pad0 // 2), (pad1 // 2, pad1 - pad1 // 2)), constant_values=0)

    # 压缩图像
    img = cv2.resize(img, (img.shape[1]//compress, img.shape[0]//compress), interpolation = cv2.INTER_AREA)

    # 将图像分块成tile大小
    img = img.reshape(img.shape[0] // tile_size, tile_size, img.shape[1] // tile_size, tile_size, 3)

    # 将图像的通道调整顺序，横纵个数放在前面，然后让两者相乘，得到分块总个数
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    mask = cv2.resize(mask, (mask.shape[1]//compress, mask.shape[0]//compress), interpolation = cv2.INTER_AREA)
    mask = mask.reshape(mask.shape[0] // tile_size, tile_size, mask.shape[1] // tile_size, tile_size)
    mask = mask.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)

    # 使用zip函数将img和mask中的元素打包在一块，统一调用
    # https://www.runoob.com/python3/python3-func-zip.html
    for i, (im, m) in enumerate(zip(img, mask)):
        x_tot.append((im / 255.0).reshape(-1, 3).mean(0))
        x2_tot.append((im / 255.0).reshape(-1, 3).mean(0))
        cv2.imwrite(OUT_IMG + f'{index}_{i}.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.imwrite(OUT_MASK + f'{index}_{i}.png', m)

img_avr = np.array(x_tot).mean(0)
img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print("mean: ", img_avr, ", std:", img_std)

```

# 训练模型
首先通过上面的get_model()加载模型，然后再定义一系列的必要步骤，包括优化器、损失评价等，如下：
```python
# 自定义Soft Dice Loss
# 网友总结了用于图像分割的常用的损失函数的PyTorch实现，见：
# https://github.com/JunMa11/SegLoss
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

# 损失函数1
bce_fn = nn.BCEWithLogitsLoss()

# 损失函数2
dice_fn = SoftDiceLoss()

# 最终损失是两种损失函数的加权平均
def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8*bce+ 0.2*dice

# 在验证集上运行模型
@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()

    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()

### 为了让中间输出结果好看，专门定制了一个显示效果
header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'

best_loss = 10
EPOCHES = 10

# 开始迭代训练
for epoch in range(1, EPOCHES+1):
    losses = []
    # 计时开始
    start_time = time.time()

    # 这里显式地设置model是train模式，以使dropout和batchnorm等网络层中的参数不固定
    # 它仅仅是一个flag，与之类似的：model.eval()是设定eval模式，即这些网络层中的参数固定，以保证测试结果的可重复性
    # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    model.train()

    # 开始在训练集数据中迭代
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)['out']
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # 在验证集上运行模型
    vloss = validation(model, vloader, loss_fn)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time()-start_time)/60**1))
    losses = []
    # 及时地将最佳模型存储下来
    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), 'model_best.pth')

# 训练结束后删除这些存储数据地变量，并回收内存
del loader, vloader, train_ds, valid_ds, ds
gc.collect();
```

训练完后的模型就开始接下来在上面的测试集上进行推导，并提交结果文件（注意在kaggle上提交时，可以把前面的训练过程去掉，直接提交训练好的模型）。

