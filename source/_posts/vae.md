---
title: 变分自编码器的原理和程序解析
tags: [Machine Learning]
categories: programming
date: 2018-7-24
---

# 参考文献
变分自编码器（Variational Auto-Encoder）可以说是深度学习领域的一股清流，没有采用“盲目堆砌各种神经层而乱碰瞎试”的套路，而是将神经网络与贝叶斯概率图结合，是理论指导模型结构设计的范例。深入了解它的原理，可以有助于建立良好的算法设计思想。
本文是对“科学空间”博主苏剑林的三篇博客的摘抄总结（话说苏博主真是沉得下心来研究算法啊。。科普得还那么好。。）：
- [变分自编码器（一）：原来是这么一回事](https://kexue.fm/archives/5253)
- [变分自编码器（二）：从贝叶斯观点出发](https://kexue.fm/archives/5343)
- [变分自编码器（三）：这样做为什么能成？](https://kexue.fm/archives/5383)

以及结合变分自编码VAE的PyTorch实现[VAE in PyTorch](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py),来对VAE进行理解。

# VAE的原理
## 问题描述
我们有一批数据样本${X\_1, X\_2, ..., X\_n}$，其整体用$X$来描述。我们想通过这些数据样本来得到$X$的分布$p(X)$，这样就可以得到所有可能的$X$。
但直接通过这些样本点来得到分布是不现实的，因为我们也不知道它符合什么样的分布。机器学习中的“生成模型”，比如VAE和GAN，对于这个问题的解决方式是引入一个中间隐变量$Z$，然后构建一个从隐变量$Z$生成数据$\hat{X}$的模型。那么，生成数据的边缘分布就可以如下计算：
$$
p(\hat{X})=\int\_Z p(\hat{X}|Z)p(Z)dz
$$
理想情况下，该生成分布就是原始样本点的概率分布，即$p(\hat{X})=p(X)$。如果这个目标达到了，这样就既得到了$p(X)$这个概率分布，又得到了生成模型$p(X|Z)$，一举两得。
具体计算时，条件概率$p(X|Z)$和先验概率$p(Z)$都可以事先假定，但此时生成的数据是采样结果，并不知道它们的分布，所以生成模型的难题就是判断生成分布与真实分布的相似度。
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fu4dv5emcdj30or0c7myy.jpg)

## 损失函数
**注意，原论文中的优化目标是最大化原始样本的对数似然函数，即：**
$$
\ln p\_\theta (X\_1, X\_2,..., X\_n) = \sum\_{i=1}^N \ln p\_\theta (X\_i)
$$
**而经过推导以后，得到：该似然函数是编码模型$q(Z|X)$与后验概率$p(Z|X)$的KL散度与另一个loss的和，而直接优化这个似然函数不可行，且这个KL散度最小为0，因此转而优化这个loss，即似然函数的下界。而下方的推导，一开始的优化目标就是与原论文不同，下方的优化目标是真实联合分布和生成模型的联合分布的KL散度最小。实际上，两方一对比可以发现，原论文要优化的下界loss就是下方推导的联合分布的KL散度。
所以，下方的loss只是原论文的一个中间步骤，VAE的理想目标还是对原始数据的极大似然估计，但发现该目标实现不了，因此VAE只能说是一个近似模型。**

假设$p(X)$就是要求的真实分布，因为：
$$
p(X)=\int\_Z p(X|Z)p(Z)dz=\int\_Z p(X,Z)dZ
$$
所以从联合分布的角度来看，假设有一个任意的联合概率分布$q(X,Z)$，用KL散度来度量两个联合分布之间的距离：
$$
KL(p(x,z)||q(x,z))= \iint p(X,Z) \ln\frac{p(X,Z)}{q(X,Z)}dZdX
$$
我们希望这个KL散度越小越好，因此，损失函数就是以这个KL散度为基本。由于我们手头上只有$X$的样本，因此利用$p(X,Z)=\tilde{p}(X)p(Z|X)$对上式进行改写：
\begin{aligned}
KL(p(X,Z)\parallel q(X,Z)) =& \int \tilde{p}(X)\Big[ \int p(Z|X) \ln \frac{\tilde{p}(X)p(Z|X)}{q(X,Z)}dZ \Big]dX \\\\
=& \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\frac{\tilde{p}(X)p(Z|X)}{q(X,Z)}dZ \Big] \\\\
=& \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \big[ \ln\tilde{p}(X)+\ln\frac{p(Z|X)}{q(X,Z)} \big]dZ \Big] \\\\
=& \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\tilde{p}(X)dZ \Big] + \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\frac{p(Z|X)}{q(X,Z)} dZ \Big] \\\\
=& \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \ln\tilde{p}(X)\int p(Z|X) dZ \Big] + \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\frac{p(Z|X)}{q(X,Z)} dZ \Big] \\\\
=& \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \ln\tilde{p}(X)\Big] + \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\frac{p(Z|X)}{q(X,Z)} dZ \Big] \\\\
\end{aligned}
注意，这里的$\tilde{p}(X)$是根据样本$X\_1,X\_2,...,X\_n$确定的关于$X$的先验分布，尽管我们不一定能准确写出它的形式，但它是确定的、存在的，因此第一项只是一个常数，所以，损失函数中可以去掉这一项，只包含KL散度的第二部分：
\begin{aligned}
L&= KL(p(X|Z)||q(X,Z))-CONSTANT \\\\
&= \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\frac{p(Z|X)}{q(X,Z)} dZ \Big] \\\\
\end{aligned}
因为KL散度最小为0，所以这里的损失函数的下界就是$-\mathbb{E}\_{\tidle{p}(X)} \Big[ \ln \tilde{p}(X) \Big]$，注意到$\tilde{p}(X)$不一定是概率，在连续情形时它是概率密度，所以它可以大于1也可以小于1，所以这个下界有可能为负，即loss可能为负。实际比较损失与下界的接近程度就可以比较生成器的相对质量。
进一步地，把$q(X,Z)$利用联合概率$q(X,Z)=q(X|Z)q(Z)$进行变换：
\begin{aligned}
L &= \mathbb{E}\_{X\sim\tilde{p}(X)}\Big[ \int p(Z|X) \ln\frac{p(Z|X)}{q(X|Z)q(Z)} dZ \Big] \\\\
&= \mathbb{E}\_{X\sim\tilde{p}(X)} 
\Big[ -\int p(Z|X)\ln q(X|Z)dZ + \int p(Z|X) \ln\frac{p(Z|X)}{q(Z)} dZ \Big] \\\\
&= \mathbb{E}\_{X\sim\tilde{p}(X)} 
\Big[ \mathbb{E}\_{Z\sim p(Z|X)}\big[ -\ln q(X|Z) \big] + \mathbb{E}\_{z\sim p(Z|X)} \big[ \ln\frac{p(Z|X)}{q(Z)} \big] \Big] \\\\
&= \mathbb{E}\_{X\sim\tilde{p}(X)} 
\Big[ \mathbb{E}\_{Z\sim p(Z|X)}\big[ -\ln q(X|Z) \big] + KL \big( p(Z|X) || q(Z) \big) \Big] \\\\
\end{aligned}
注意，第一项中$Z$是根据$p(Z|X)$来采样的，而计算的却是$-\ln q(X|Z)$的期望。

## 实验
现在$q(X|Z), q(Z|X), q(Z)$全都是未知的，连形式都没有确定，而为了实际算法，就得明确地把它们写出来。
首先，为了便于采样，假设$Z\sim N(0,1)$，即隐参量符合标准正态分布，这就解决了$q(Z)$。
然后，对于$q(Z|X)$，也是假设它是（各分量独立的）正态分布，其均值和方差由$X$决定，这个决定由神经网络算出。具体来说就是设计两个神经网络，它们接收$X$，分别输出均值$\mu(X)$和方差的对数$\ln \sigma^2$。这两个参数训练出来以后，就可以得到$p(Z|X)$的形式（注意是多元正态分布的概率密度）：
$$
p(Z|X)=\frac{1}{\prod\limits_{k=1}^d \sqrt{2\pi \sigma_{(k)}^2(X)}}\exp\left(-\frac{1}{2}\left\Vert\frac{Z-\mu(X)}{\sigma(X)}\right\Vert^2\right)
$$
其中，$d$是隐参量$Z$的维度。以$Z$的维度为2为例，上式的具体写法就是：
$$
p(Z|X)=\frac{1}{2\pi \sigma_{(1)}(X)\sigma_{(2)}(X)}\exp\left(-\frac{1}{2}\left((\frac{Z-\mu\_1(X)}{\sigma\_1(X)})^2+(\frac{Z-\mu\_2(X)}{\sigma\_2(X)})^2\right)\right)
$$
这部分实际就是Encoder的作用。
既然假定了$q(Z)$和$q(Z|X)$都是正态分布，那么它们的KL散度也就可以计算出来（由于考虑的是各分量独立的多元正态分布，因此只需要推导一元正态分布的情形即可）：
\begin{aligned}
&KL\Big(N(\mu,\sigma^2) || N(0,1)\Big) \\\\
&= \int \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2} \left(\ln \frac{e^{-(x-\mu)^2/2\sigma^2}/\sqrt{2\pi\sigma^2}}{e^{-x^2/2}/\sqrt{2\pi}}\right)dx \\\\
&= \int \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2} \ln \Big[ \frac{1}{\sqrt{\sigma^2}}\exp ( \frac{1}{2}\big[x^2-(x-\mu)^2/\sigma^2\big] ) \Big] dx \\\\ 
&=\frac{1}{2}\int \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2} \Big[-\ln \sigma^2+x^2-(x-\mu)^2/\sigma^2 \Big] dx \\\\
&=\frac{1}{2}\Big( \mu^2+\sigma^2-\ln \sigma^2-1) \\\\
\end{aligned}

现在只剩下$q(X|Z)$。原始论文中给出了两种分布供选择，分别对应于不同的情形。
### 伯努利分布
伯努利分布，即二元分布，只适用于$X$是一个多元的二值向量（只能是0或1）的清醒，比如$X$是二值图像，如MNIST数据集。这个分布的唯一参数就是$X=1$时的概率$\rho$，此时还是构建神经网络来训练这个参数。而后验分布$q(X|Z)$的表达式是：
$$
q(X|Z)=\prod_{k=1}^D \Big(\rho_{(k)}(Z)\Big)^{X_{(k)}} \Big(1 - \rho_{(k)}(Z)\Big)^{1 - X_{(k)}}
$$
这里，$D$是$X$的维度。
这时候就可以算出损失函数的第一项中的要求期望的那部分：
$$
-\ln q(X|Z) = \sum_{k=1}^D \Big[- X_{(k)} \ln \rho_{(k)}(Z) - (1-X_{(k)}) \ln \Big(1 -\rho_{(k)}(Z)\Big)\Big]
$$
可以看出，这一项正好是交叉熵的形式！所以，这部分可以直接调用软件中的交叉熵的计算函数。另外，要注意的是，$\rho(Z)$正好是用作Decoder，它要压缩到$0\sim 1$之间，这可以用sigmoid函数来整流。
### 正态分布
如果假设$q(X|Z)$服从正态分布，那么它的形式就跟之前的$P(Z|X)$是一样的，只不过是$X$和$Z$交换位置（同时别忘了将维度也更改了）：
$$
p(X|Z)=\frac{1}{\prod\limits_{k=1}^D \sqrt{2\pi \sigma_{(k)}^2(Z)}}\exp\left(-\frac{1}{2}\left\Vert\frac{X-\mu(Z)}{\sigma(Z)}\right\Vert^2\right)
$$
这里的均值和方差的对数也是通过构建神经网络来训练得到（从下面可以看出，一般固定方差，所以就不用训练方差这个网络了），这里也是Decoder的作用。
那么，此时损失函数的第一项中的要求期望的那部分：
$$
-\ln q(X|Z) = \frac{1}{2}\left\Vert\frac{X-\mu(Z)}{\sigma(Z)}\right\Vert^2 + \frac{D}{2}\ln 2\pi + \frac{1}{2}\sum_{k=1}^D \ln \sigma_{(k)}^2(Z)
$$
如果我们将方差固定为一个常数$\sigma^2$，此时：
$$
-\ln q(X|Z) \sim \frac{1}{2\sigma^2}\left\Vert X-\mu(Z)\right\Vert^2
$$
上式就是将$\mu(Z)$作为Decoder时的MSE损失函数。
因此，对于二值数据，我们选择$\rho(Z)$作为Decoder（注意用sigmoid函数整流），然后用交叉熵作为损失函数，这对应于$q(X|Z)$是伯努利分布的情形；而对于一般数据，我们选择训练出来的均值$\mu(Z)$作为decoder，然后用MSE作为损失函数，这对应于$q(X|Z)$是固定方差的正态分布时的情形。具体这个方差取多少，是有一定人为性的，这个方差也决定了重构误差和KL散度的比例是多少。

### 采样
对于损失函数：
$$
L = \mathbb{E}\_{X\sim\tilde{p}(X)} 
\Big[ \mathbb{E}\_{Z\sim p(Z|X)}\big[ -\ln q(X|Z) \big] + KL \big( p(Z|X) || q(Z) \big) \Big]
$$
已经知道了$-\ln q(X|Z)$ 和$KL \big( p(Z|X) || q(Z) \big)$，还涉及到$\mathbb{E}\_{Z\sim p(Z|X)}\big[ -\ln q(X|Z) \big]$怎么采样计算的问题。VAE直接给出了“只采一个样本就可以”的结论，因此损失函数变为：
$$
L = \mathbb{E}\_{X\sim\tilde{p}(X)} \Big[ -\ln q(X|Z) + KL \big( p(Z|X) || q(Z) \big) \Big]
$$
事实上我们会运行多个epoch，每次的隐参量都是随机生成的，因此当epoch数足够多时，事实上是可以保证采样的充分性的。

# VAE的形象化理解
## VAE的错误理解
关于VAE的一个常见的错误理解如下图所示：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fu4eolz910j30qt0dyq5e.jpg)
即，先根据原始的样本数据算出$Z$所符合的标准正态分布的均值和方差，然后再从该分布中采样一个$Z$，根据$Z$算出一个$X$。但问题是：究竟经过重新采样出来的$Z\_k$及其算出来的$X\_k$，是不是还对应着原来的$X\_k$，所以此时如果直接最小化$D(\hat{X}\_k, X\_k)$是很不科学的（$D$代表某种距离函数），事实上代码也不是这么写的。

## VAE的正确理解
为了保证经过重新采样出来的$Z\_k$及其算出来的$X\_k$，还对应着原来的$X\_k$，我们可以假定这个采样点$Z\_k$所遵循的分布$p(Z)$是专属于$X\_k$的，即$p(Z|X\_k)$，而且该分布还假定是独立的、多元的正态分布。这样，每一个$X\_k$都配备了一个专属的正态分布，这样才方便后面的生成器做还原（即每个样本$X\_k$都有自己的均值和方差）。但这样一来，多少个$X$就有多少个正态分布。我们知道正态分布有两组参数：均值$\mu$和方差$\sigma^2$（多元的话，它们都是向量），那怎样确定专属于$X\_k$的正态分布的均值和方差呢？方法就是用神经网络来拟合。这就是神经网络时代的哲学：难算的都用神经网络来拟合。注意，这里是构建两个分别含有两个全连接层的神经网络$\mu\_k=f\_1(X\_k)$和$log\sigma^2=f\_2(X\_k)$来计算均值和方差。注意，$f\_2$是训练的方差的对数。
示意图如下：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fu4jl5x0o9j30rf0imtcm.jpg)

## VAE的生成能力体现在哪
在最小化$D(\hat{X}\_k,X\_k)$时，结果会受到噪声的影响，这是因为$Z\_k$是通过重新采样得到的（即通过一个随机数生成）。显然噪声会增加训练的难度，不过好在这个噪声强度（也就是方差）是通过一个神经网络算出来的，所以最终模型为了降低训练难度（只拟合均值，肯定比拟合多个来得容易），会想尽办法让方差为0。而方差为0的话，也就丧失了随机性，所以不管如何采样，其实都只是取得了均值。即，模型会慢慢退化成普通的AutoEncoder，没有了生成能力。
而VAE之所以是生成模型，是因为它还让所有的$p(Z|X)$都向标准正态分布看齐。因为此时：
$$
p(Z)=\sum\_X p(Z|X)p(X)=\sum\_X N(0,1)p(X)=N(0,1)\sum\_X p(X)=N(0,1)
$$
这样就能推出这样的先验假设，即$p(Z)$是标准正态分布，这就意味着，隐参量符合了正态分布，这样就可以放心地从这个分布中随机采样，保证了生成能力。
示意图如下：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fu4ku70trbj30rd0nmwiv.jpg)
那么，问题就变成了怎样让$p(Z|X)$向标准正态分布看齐。最直接的一个方法就是在损失函数中加入额外的loss：
$$
L\_\mu = ||f\_1(X\_k)||^2 \quad and \quad L\_{\sigma^2}=||f\_2(X\_K)||^2
$$
因为$f\_1$和$f\_2$分别代表了均值和方差的对数，达到标准正态分布意味着二者都为零。但是这种方法面临着这两个损失怎样选取的问题。所以，原文中直接算了一般的正态分布和标准正态分布的KL散度作为额外的损失，即
$$
L\_{\mu,\sigma^2}=\frac{1}{2}\sum\_{i=1}^d(\mu\_{(i)}^2+\sigma\_{(i)}^2-log\sigma\_{(i)}^2-1)
$$
这里$d$是隐变量$Z$的维度，而$\mu\_{(i)}$和$\sigma\_{(i)}^2$分别表示一般正态分布的均值向量和方差向量的第i个分量。

## VAE的本质
VAE本质上就是在我们常规的自编码器的基础上，对encoder的结果（在VAE中对应着计算均值的网络）加上了“高斯噪声”，使得结果decoder能够对噪声有鲁棒性；而那个额外的KL loss（目的是让均值为0，方差为1），事实上就是相当于对encoder的一个正则项，希望encoder出来的东西均有零均值。
那另外一个encoder（对应着计算方差的网络）的作用呢？它是用来动态调节噪声的强度的。直觉上来想，当decoder还没有训练好时（重构误差远大于KL loss），就会适当降低噪声（KL loss增加，注意KL loss等于0表示分布就是标准正态分布），使得拟合起来容易一些（重构误差开始下降）；反之，如果decoder训练得还不错时（重构误差小于KL loss），这时候噪声就会增加（KL loss减少），使得拟合更加困难了（重构误差又开始增加），这时候decoder就要想办法提高它的生成能力了。
说白了，重构的过程是希望没噪声的，而KL loss则希望有高斯噪声的，两者是对立的。所以，VAE跟GAN一样，内部其实是包含了一个对抗的过程，只不过它们两者是混合起来，共同进化的。

# 条件VAE
因为目前的VAE是无监督训练的，因此很自然想到：如果有标签数据，那么能不能把标签信息加进去辅助生成样本呢？这个问题的意图，往往是希望能够实现控制某个变量来实现生成某一类图像。当然，这是肯定可以的，我们把这种情况叫做Conditional VAE，或者叫CVAE。（相应地，在GAN中我们也有个CGAN。）

# 程序实现
## 导入必要的包
```cpp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
```
导入PyTorch和TorchVision，并根据实际情形看使用GPU还是CPU。建立samples文件夹放置图片。

## 设置超参数和导入数据
```cpp
image_size = 784 # 这个是输入图片的长宽像素之积，MNIST图片是28*28的，所以是784
h_dim = 400 # 计算均值方差的神经网络是一个含有两个全连接层和一个ReLU层的网络。这里是第一个全连接层的输出神经元个数 
z_dim = 20 # 这里是第二个全连接层的输出神经元个数
num_epochs = 15 # 迭代次数
batch_size = 128 # 批处理样本时每批的个数
learning_rate = 1e-3 # 学习速率

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)
```

## 模型构建
```cpp
# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    def encode(self, x): # 编码器
        h = F.relu(self.fc1(x)) # 计算第一个全连接层，然后用ReLU整流
        return self.fc2(h), self.fc3(h) # 对均值和方差(实际是$log\sigma^2$)都计算第二个全连接层
    
    def reparameterize(self, mu, log_var): # 重参数化
        std = torch.exp(log_var/2)  # 这一步是将方差的对数转换为标准差。
        eps = torch.randn_like(std) # 生成一个跟std同样大小的取自标准正态分布的随机数。
        return mu + eps * std  # 根据标准正态分布的采样得到之前正态分布中的采样。

    def decode(self, z): # 解码器，也是两个全连接层，中间有个ReLU整流，最后有个Sigmoid层进行激活。
        h = F.relu(self.fc4(z)) 
        return F.sigmoid(self.fc5(h))
    
    def forward(self, x): # 前向计算
        mu, log_var = self.encode(x) # 对输入变量进行编码，得到隐变量所满足的正态分布的均值和方差
        z = self.reparameterize(mu, log_var) # 重参数化
        x_reconst = self.decode(z) # 解码器
        return x_reconst, mu, log_var
```
有几个注意点：
(1) 对每个输入的x都生成一个正态分布，然后再从该分布中采样得到z，然后再根据生成器得到新x。所以，输入x、隐参量z和输出x是一一对应的。
(2) 计算方差的神经网络实际计算的不是方差，而是它的对数，即$log(\sigma^2)$。
(3) 重参数reparameterize的作用：得到隐参量z的正态分布后，还要对其采样得到离散值。但这个“采样”操作是不可导的，因此没法应用于梯度下降的训练过程。而根据公式：
$$
\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(z-\mu)^2}{2\sigma^2})dz = 
\frac{1}{\sqrt{2\pi}}exp[-\frac{1}{2}(\frac{z-\mu}{\sigma})^2]d(\frac{z-\mu}{\sigma})
$$
可以看出变量$\epsilon=\frac{z-\mu}{\sigma}$服从标准正态分布。因此，从原正态分布中采样一个z，就等同于从标准正态分布中采样一个$\epsilon$，然后让$z=\mu+\epsilon\sigma$。这样，就可以不用把采样这个操作加入到反向传播中，而只需要将结果放进来即可。

## 训练和测试模型
```cpp
model = VAE().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):
        # Forward pass
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)  # 前向计算，得到新的变量x，以及正态分布的均值和方差
        
        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False) # 计算输入变量和新变量之间的二进制交叉熵
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # 计算KL散度，公式见下方。
        
        # Backprop and optimize
        loss = reconst_loss + kl_div # 总损失等于交叉熵和KL散度之和
        optimizer.zero_grad() # 梯度置零
        loss.backward() # 反向传播
        optimizer.step() # 步进
        
        if (i+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
    
    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
```
注意：
总损失函数等于交叉熵和KL散度。
KL散度是为了度量两个概率分布之间的差异。如果两个分布相等，那么KL散度为0。KL散度的一个主要性质是非负性，因此最小化KL散度的结果就是使得两个分布尽可能相等，这一点的严格证明要用到变分法，这里正是VAE中的V的来源。
KL散度的计算公式是：
$$
D\_{KL}(p(x)||q(x))=\int p(x)\ln\frac{p(x)}{q(x)}dx
$$
上面的x是连续随机变量。如果是离散的随机变量，可以有两种方式计算：
一是使用数值计算，即
$$
D\_{KL}(p(x)||q(x))=\sum{p(x_i) \ln \frac{p(x_i)}{q(x_i)}}\Delta x
$$
二是使用采样计算，即
$$
D\_{KL}(p(x)||q(x))=E\_{x \in p(x)} [\ln \frac{p(x_i)}{q(x_i)}]
$$
其中，E是期望，有：
$$
E\_{x~p(x)}[f(x)]=\int f(x)p(x)dx \approx \frac{1}{n}\sum\_{i=1}^n f(x\_i), \qquad x\_i \in p(x)
$$
注意，上式中的$x\_i$是从概率分布$p\_i$中采样，所以采样结果已经包含在了$p\_i$中，故形式上与数值计算不同。
