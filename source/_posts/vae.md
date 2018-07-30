---
title: 变分自编码器的程序实现与对应原理
tags: [Machine Learning]
categories: programming
date: 2018-7-24
---

# 概述
本文是结合博客[变分自编码器（一）：原来是这么一回事](https://kexue.fm/archives/5253)和变分自编码VAE的PyTorch实现[VAE in PyTorch](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py),对VAE进行理解。
“科学空间”博客上除了上面那一篇blog，还有后续两篇，也是精辟。见：
- [变分自编码器（二）：从贝叶斯观点出发](https://kexue.fm/archives/5343)
- [变分自编码器（三）：这样做为什么能成？](https://kexue.fm/archives/5383)

# 导入必要的包
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

# 设置超参数和导入数据
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

# 模型构建
VAE的模型大致是这样的：
![变分自编码器模型图示](http://7xrm8i.com1.z0.glb.clouddn.com/vae.png)
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

# 训练和测试模型
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
这里的KL散度是计算了正态分布和标准正态分布之间的关系，推导结果为：
$$
KL(N(\mu,\sigma^2)||N(0,1))=\frac{1}{2}(-log\sigma^2+\mu^2+\sigma^2-1)
$$
