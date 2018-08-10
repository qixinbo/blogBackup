---
title: 变分自编码器的原理和程序解析
tags: [Machine Learning]
categories: programming
date: 2018-7-24
---

# 参考文献
本文是结合博客[变分自编码器（一）：原来是这么一回事](https://kexue.fm/archives/5253)和变分自编码VAE的PyTorch实现[VAE in PyTorch](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py),对VAE进行理解。
“科学空间”博客上除了上面那一篇blog，还有后续两篇，也是精辟。见：
- [变分自编码器（二）：从贝叶斯观点出发](https://kexue.fm/archives/5343)
- [变分自编码器（三）：这样做为什么能成？](https://kexue.fm/archives/5383)

# VAE的原理
## 问题描述
我们有一批数据样本${X\_1, X\_2, ..., X\_n}$，其整体用$X$来描述。我们想通过这些数据样本来得到$X$的分布$p(X)$，这样就可以得到所有可能的$X$。
直接通过这些样本点来得到分布是不现实的，因为我们也不知道它符合什么样的分布。机器学习中的“生成模型”是一个解法。生成模型，比如VAE和GAN，方式是引入一个中间隐变量$Z$，然后构建一个从隐变量$Z$生成目标数据$X$的模型。具体来说，先事先假设$Z$服从某个常见的分布（比如正态分布或均匀分布）（实际上VAE并没有直接使用这个先验分布，而是使用了$p(Z|X)$这个后验分布是正态分布），然后希望训练出一个模型或函数$X=g(Z)$，这样就可以将$Z$的分布映射到一个生成分布，理想上该生成分布就是训练集的概率分布。用数学语言来说就是：
$$
p(X)=\sum\_Z p(X|Z)p(Z)
$$

那么，生成模型的难题就是判断生成分布与真实分布的相似度，因为我们只知道两者的采样结果，不知道它们的分布表达式。
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fu4dv5emcdj30or0c7myy.jpg)
## VAE的错误理解
关于VAE的一个常见的错误理解如下图所示：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fu4eolz910j30qt0dyq5e.jpg)
即，先根据原始的样本数据算出$Z$所符合的标准正态分布的均值和方差，然后再从该分布中采样一个$Z$，根据$Z$算出一个$X$。但问题是：究竟经过重新采样出来的$Z\_k$及其算出来的$X\_k$，是不是还对应着原来的$X\_k$，所以此时如果直接最小化$D(\hat{X}\_k, X\_k)$是很不科学的（$D$代表某种距离函数），事实上代码也不是这么写的。

## VAE的正确理解
为了保证经过重新采样出来的$Z\_k$及其算出来的$X\_k$，还对应着原来的$X\_k$，我们可以假定这个采样点$Z\_k$所遵循的分布$p(Z)$是专属于$X\_k$的，即$p(Z|X\_k)$，而且该分布还假定是独立的、多元的正态分布。因此，VAE中所用到的分布是该后验分布$p(Z|X\_k)$是正态分布，而不是先验分布$p(Z)$是正态分布。

这样，每一个$X\_k$都配备了一个专属的正态分布，这样才方便后面的生成器做还原。但这样一来，多少个$X$就有多少个正态分布。我们知道正态分布有两组参数：均值$\mu$和方差$\sigma^2$（多元的话，它们都是向量），那怎样确定专属于$X\_k$的正态分布的均值和方差呢？方法就是用神经网络来拟合。这就是神经网络时代的哲学：难算的都用神经网络来拟合。注意，这里是构建两个分别含有两个全连接层的神经网络$\mu\_k=f\_1(X\_k)$和$log\sigma^2=f\_2(X\_k)$来计算均值和方差。注意，$f\_2$是训练的方差的对数。
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
(1) 实际用到的分布是$p(z|x)$这个后验分布是正态分布，而$p(z)$这个先验分布是正态分布是自然推导得到的。
(2) 对每个输入的x都生成一个正态分布，然后再从该分布中采样得到z，然后再根据生成器得到新x。所以，输入x、隐参量z和输出x是一一对应的。
(3) 计算方差的神经网络实际计算的不是方差，而是它的对数，即$log(\sigma^2)$。
(4) 重参数reparameterize的作用：得到隐参量z的正态分布后，还要对其采样得到离散值。但这个“采样”操作是不可导的，因此没法应用于梯度下降的训练过程。而根据公式：
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
D\_{KL}(p(x)||q(x))=\int p(x)ln\frac{p(x)}{q(x)}dx
$$
上面的x是连续随机变量。如果是离散的随机变量，可以有两种方式计算：
一是使用数值计算，即
$$
D\_{KL}(p(x)||q(x))=\sum{p(x_i) ln \frac{p(x_i)}{q(x_i)}}\Delta x
$$
二是使用采样计算，即
$$
D\_{KL}(p(x)||q(x))=E\_{x \in p(x)} [ln \frac{p(x_i)}{q(x_i)}]
$$
其中，E是期望，有：
$$
E\_{x~p(x)}[f(x)]=\int f(x)p(x)dx \approx \frac{1}{n}\sum\_{i=1}^n f(x\_i), \qquad x\_i \in p(x)
$$
注意，上式中的$x\_i$是从概率分布$p\_i$中采样，所以采样结果已经包含在了$p\_i$中，故形式上与数值计算不同。
