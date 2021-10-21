---
title: GAN系列算法原理及极简代码解析
tags: [GAN]
categories: machine learning 
date: 2021-10-21
---

# 基本思想
[生成对抗网络——原理解释和数学推导](https://alberthg.github.io/2018/05/05/introduction-gan/)
首先有一个“生成器(Generator)”：其实就是一个神经网络，或者是更简单的理解，他就是一个函数(Function)。输入一组向量，经由生成器，产生一组目标矩阵（如果你要生成图片，那么矩阵就是图片的像素集合，具体的输出视你的任务而定）。它的目的就是使得自己造样本的能力尽可能强，强到什么程度呢，强到你判别网络没法判断我是真样本还是假样本。
同时还有一个“判别器(Discriminator)”：判别器的目的就是能判别出来一张图它是来自真实样本集还是假样本集。假如输入的是真样本，网络输出就接近 1，输入的是假样本，网络输出接近 0，那么很完美，达到了很好判别的目的。
那为什么需要这两个组件呢？GAN在结构上受博弈论中的二人零和博弈 （即二人的利益之和为零，一方的所得正是另一方的所失）的启发，系统由一个生成模型（G）和一个判别模型（D）构成。G 捕捉真实数据样本的潜在分布，并生成新的数据样本；D 是一个二分类器，判别输入是真实数据还是生成的样本。生成器和判别器均可以采用深度神经网络。GAN的优化过程是一个极小极大博弈(Minimax game)问题，优化目标是达到纳什均衡。

# 原始GAN
这里用的网络非常简单，仅有二层，且还不是卷积神经网络，而是全连接层。后面的GAN变种会使用更加强大的深度网络。
首先先看一下判别器和生成器的分别的损失函数：
![loss-d](https://user-images.githubusercontent.com/6218739/136900996-05cfa37c-96f4-48cf-9f5b-7f7aa6a074b8.png)
![loss-g](https://user-images.githubusercontent.com/6218739/136901466-4f4a85d8-92db-4eaf-9e05-6b8b206df7d7.png)

最终实现的效果就是：生成器能够凭空（也不是完全凭空，它的输入是一个具有隐参量维度的噪声图像）生成一张与训练图片极为类似的虚假图片。

代码在：
[Machine-Learning-Collection/ML/Pytorch/GANs](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 非常小的网络
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# 生成器
class Generator(nn.Module):
    # z是隐空间参量
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            # 输入会标准化为[-1, 1]，所以这里的输出也要标准化到[-1, 1]
            nn.Tanh(), 
        )

    def forward(self, x):
        return self.gen(x)


# 超参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # 学习率
z_dim = 64 # 隐参量的维度
image_dim = 28 * 28 * 1  # 784，MNIST
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
# 这里加入噪声是为了看出在迭代过程中的变化
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    # 按道理应该采用与MNist相同的均值和标准差(0.1307, 0.3081)
    # 但上面的超参数的设置是作者用(0.5, 0.5)时调出来的，所以这里如果改了就会发散
    # 这也说明GAN对参数非常敏感，非常难以训练
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 判别器的优化算法
opt_disc = optim.Adam(disc.parameters(), lr=lr)
# 生成器的优化算法
opt_gen = optim.Adam(gen.parameters(), lr=lr)
# 损失函数设为Binary Cross Entropy
# 公式为-[y*logx + (1-y)log(1-x)]
# 注意公式前面的负号，后面计算损失时该负号将最大化改为了最小化
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# 迭代训练
for epoch in range(num_epochs):
    # 从加载器里取出的是real图像
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ########## 训练判别器：最大化log(D(x)) + log(1 - D(G(z))) ###############
        ## 即：log(D(real)) + log(1 - D(G(latent_noise)))

        # 事先准备隐空间的噪声数据
        noise = torch.randn(batch_size, z_dim).to(device)
        # 将噪声数据传给生成器，生成假的图像
        fake = gen(noise)

        #### 计算log(D(x))损失 ####
        # 将真实图像传给判别器，即计算D(x)
        disc_real = disc(real).view(-1)
        # 将D(x)与1分别作为预测值和目标值放到BCE中进行计算
        # 根据BCE的公式-[y*logx + (1-y)log(1-x)]，这里y为1，因此此处计算的就是-log(D(x))
        # 也可以这样理解，此处的损失就是看看判别器对于真实图像的预测是不是接近1，即判别器对于真实图像的性能怎么样
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        #### 计算log(1-D(G(z)))损失
        # 将生成器生成的虚假图像传给判别器，即计算D(G(z))
        disc_fake = disc(fake).view(-1)
        # 将D(G(z))与0分别作为预测值和目标值放到BCE中进行计算
        # 根据BCE的公式-[y*logx + (1-y)log(1-x)]，这里y为0，因此此处计算的就是-log(1-D(G(z)))
        # 也可以这样理解，此处的损失就是看看判别器对于虚假图像的预测是不是接近0，即判别器对于虚假图像的性能怎么样
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # 总损失
        lossD = (lossD_real + lossD_fake) / 2
        # 判别器的反向传播
        disc.zero_grad()
        # 注意，这里将retain_graph设为True，是为了保留该过程中计算的梯度，后续生成器网络更新时使用
        # 否则这里判别器网络构建了正向计算图后，反向传播结束后就将其销毁
        lossD.backward(retain_graph=True)
        opt_disc.step()


        ########## 训练生成器：最小化log(1 - D(G(z)))，等价于最大化log(D(G(z)) ###############
        ## 第二种损失不会遇到梯度饱和的问题

        # 将生成器生成的虚假图像传给判别器，即计算D(G(z))
        # 这里的disc是经过了升级后的判别器，所以与第99行的D(G(z))计算不同
        # 但fake这个量还是上面的fake = gen(noise)
        output = disc(fake).view(-1)
        # 将D(G(z))与1分别作为预测值和目标值放到BCE中进行计算
        # 根据BCE的公式-[y*logx + (1-y)log(1-x)]，这里y为1，因此此处计算的就是-log(D(G(z)))
        # 也可以这样理解，此处的损失就是看看判别器对于生成器生成的虚假图像的预测是不是接近1，即生成器有没有骗过判别器
        # 这里log(D(G(z))的计算与上面的log(D(G(z))的计算不重复，是因为生成器和判别器是分开训练的，两者都要有各自的损失函数
        lossG = criterion(output, torch.ones_like(output))
        # 生成器的反向传播
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # 下面就是用于tenshorboard的可视化
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
```

# DCGAN
DCGAN，深度卷积生成对抗网络，全名“Deep Convolutional Generative Adversarial Networks”。
DCGAN的生成器和判别器的网络架构如下：
![dcgan](https://user-images.githubusercontent.com/6218739/136920307-cfbf4981-6576-4bad-9a31-866b8465ece4.png)
与上面的原生的GAN类似，DCGAN是将网络架构换成了深度卷积网络，其参数也要小心调节。

# WGAN
[令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)
（该文章下面的评论也很有见解）
自从2014年Ian Goodfellow提出以来，GAN就存在着训练困难、生成器和判别器的loss无法指示训练进程、生成样本缺乏多样性等问题。从那时起，很多论文都在尝试解决，但是效果不尽人意，比如最有名的一个改进DCGAN依靠的是对判别器和生成器的架构进行实验枚举，最终找到一组比较好的网络架构设置，但是实际上是治标不治本，没有彻底解决问题。而Wasserstein GAN（下面简称WGAN）成功地做到了以下爆炸性的几点：
（1）彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
（2）基本解决了collapse mode的问题，确保了生成样本的多样性（collapse mode就是模式倒塌。比如我们知道人民币有好几个面额的纸币。假钞制造团伙发现如果他们将全部精力都放在制造一种面值的货币时最容易获得成功。而这时候，模式倒塌也就发生了。虽然这个假钞制造团伙能够制造出十分真实的货币，但却只有一种，而这有时并不是我们希望的。我们希望假钞制造团伙能生成所有的币值人民币。）
（3）训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高
（4）以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到。
那以上好处来自哪里？这就是令人拍案叫绝的部分了——实际上作者整整花了两篇论文，在第一篇《Towards Principled Methods for Training Generative Adversarial Networks》里面推了一堆公式定理，从理论上分析了原始GAN的问题所在，从而针对性地给出了改进要点；在这第二篇《Wasserstein GAN》里面，又再从这个改进点出发推了一堆公式定理，最终给出了改进的算法实现流程，而改进后相比原始GAN的算法实现流程却只改了四点：
（1）判别器最后一层去掉sigmoid
（2）生成器和判别器的loss不取log
（3）每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
（4）不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行。

上述是工程上的改进，但为什么这样改进，需要非常深厚的数学知识推导。从宏观上理解就是如下图所示：
![distance](https://user-images.githubusercontent.com/6218739/137093439-169e8c77-debc-46cb-982f-d91502574eb4.png)
GAN的目的是使得生成图像的样本分布与真实图像的样本分布尽可能相近，即数学上怎样表达这两种分布的距离。原始GAN使用了JS散度来衡量该距离（更新loss后的GAN使用的是KL散度），WGAN则使用的是Wasserstein距离。
因为对于JS散度，无论真实样本分布跟生成样本分布是远在天边，还是近在眼前，只要它们俩没有一点重叠或者重叠部分可忽略，JS散度就固定是常数$log 2$，而这对于梯度下降方法意味着——梯度为0！此时对于最优判别器来说，生成器肯定是得不到一丁点梯度信息的；即使对于接近最优的判别器来说，生成器也有很大机会面临梯度消失的问题。 原始GAN不稳定的原因是：判别器训练得太好，生成器梯度消失，生成器loss降不下去；判别器训练得不好，生成器梯度不准，四处乱跑。只有判别器训练得不好不坏才行，但是这个火候又很难把握，甚至在同一轮训练的前后不同阶段这个火候都可能不一样，所以GAN才那么难训练。
Ian Goodfellow提出的原始GAN两种形式有各自的问题，第一种形式等价在最优判别器下等价于最小化生成分布与真实分布之间的JS散度，由于随机生成分布很难与真实分布有不可忽略的重叠以及JS散度的突变特性，使得生成器面临梯度消失的问题；第二种形式在最优判别器下等价于既要最小化生成分布与真实分布直接的KL散度，又要最大化其JS散度，相互矛盾，导致梯度不稳定，而且KL散度的不对称性使得生成器宁可丧失多样性也不愿丧失准确性，导致collapse mode现象。
Wasserstein距离相比KL散度、JS散度的优越性在于，即便两个分布没有重叠，Wasserstein距离仍然能够反映它们的远近。
其具体定义如下：
![em](https://user-images.githubusercontent.com/6218739/137243663-49a15547-9972-4daf-95ea-5a78751e10af.png)
（注意，Wasserstein距离本身是一个距离的概念，上式中的下界意思是从这两个分布中采样时取得的最小距离，就是Wasserstein距离）
在实际使用Wasserstein距离时，无法直接应用，作者将其通过某一定理改变成了如下条件：
![wgan-1](https://user-images.githubusercontent.com/6218739/137256384-4c8ed599-a4e0-46b3-9b1f-d8546bfc53fb.png)
![wgan-2](https://user-images.githubusercontent.com/6218739/137256914-76caca0f-c705-4973-aa5e-b492ba1910f6.png)
即这里不知道函数f的具体形式，用一组参数w来定义这些f，这里就是深度学习中的常用套路，“什么东西如果不知道，就用神经网络来学到”，因此f就是参数为w的一套网络，这里原作者命名为Critic。（至于上式中K施加的限制，是通过对权重参数w的裁剪来实现的。）
注意上式中是尽可能最大化才能获得Wasserstein距离，因此网络f的作用是一个测距网络，它的最大化max的过程就是为了找到Wasserstein距离，即该网络逐步优化成为一个准确的Wasserstein距离测量尺。
因此，最好不把它称为判别网络，而是称为测距网络。它的训练过程就是最小化下面的损失（因为它是为了近似拟合Wasserstein距离，属于回归任务，所以要把最后一层的sigmoid拿掉）：
![wgan-loss-1](https://user-images.githubusercontent.com/6218739/137257689-06b20ed0-fd71-411e-b3c2-17798a574b3d.png)
那么接下来，因为测距网络已经将Wasserstein距离计算了出来，下一步就是将该距离尽可能地减小，从而使得生成图像的分布尽可能地与真实图像的分布类似，这就是生成器网络要干的事情。
回到公式14中Wasserstein距离的定义，该距离由两部分构成，第一项是真实图像分布的贡献，第二项是生成图像分布的贡献，而第一项与生成器网络无关，因此要使得Wasserstein距离变得小一些（注意不要受式中的max影响，max是为了计算Wasserstein距离），就要使第二项（注意带前面的负号）变小，即：
![wgan-loss-2](https://user-images.githubusercontent.com/6218739/137263288-aa414595-7a8d-4b78-97ed-83c134835975.png)
至于代码实现，大部分都可以复用之前的，只是针对上面说的四点改动一下即可。
```python
# 测距网络，看着跟原始GAN的判别网络类似，但核心意义不同，这里仍然沿用那个网络架构，但改个名字以示区别
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        ### 训练测距网络，最大化E[critic(real)] - E[critic(fake)]
        ### 即最小化-(E[critic(real)] - E[critic(fake)])
        # Critic训练得越好，对Generator的提升更有利，因此多训练几轮Critic。
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            # 测距网络的损失函数
            # 两个期望值相减
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # 裁剪网络权重
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ########## 训练生成器网络 ################
        ## 最大化E[critic(gen_fake)]
        ## 即最小化-E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
```
## WGAN-GP
[WGAN的来龙去脉](https://zhuanlan.zhihu.com/p/58260684)
作者们发现WGAN有时候也会伴随样本质量低、难以收敛等问题。WGAN为了保证Lipschitz限制，采用了weight clipping的方法，然而这样的方式可能过于简单粗暴了，因此他们认为这是上述问题的罪魁祸首。
具体而言，他们通过简单的实验，发现weight clipping会导致两大问题：模型建模能力弱化，以及梯度爆炸或消失。
他们提出的替代方案是给Critic loss加入gradient penalty (GP)，这样，新的网络模型就叫WGAN-GP。
新的Loss为：
![wgan-gp-loss](https://user-images.githubusercontent.com/6218739/137277767-786bce8c-7f5c-4c2a-aca2-41cb80fcde85.png)
另一个值得注意的地方是，用于计算GP的样本是生成样本和真实样本的线性插值。
GP部分的代码为：
```python
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    # 线性插值
    interpolated_images = real * alpha + fake * (1 - alpha)

    # 计算判别网络（测距网络）得分
    mixed_scores = critic(interpolated_images)

    # 计算梯度
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    # 2范数
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
```

# CGAN
[李宏毅GAN2018笔记 Conditional GAN](https://zhuanlan.zhihu.com/p/61464846)
Conditional，意思是条件，所以 Conditional GAN 的意思就是有条件的GAN。Conditional GAN 可以让 GAN 产生的结果符合一定的条件，即可以通过人为改变输入的向量（记不记得我们让生成器生成结果需要输入一个低维向量），控制最终输出的结果。
这种网络与普通 GAN 的区别在于输入加入了一个额外的 condition（比如在 text-to-image 任务中的描述文本），并且在训练的时候使得输出的结果拟合这个 condition。
所以现在判别器不仅要对生成结果的质量打分，还要对结果与输入 condition 的符合程度打分。
Conditional GAN 的判别器有两种常见架构，前者更为常用，但李宏毅老师认为后者更加合理，它用两个神经网络分别对输出结果的质量以及条件符合程度独立进行判别。
![cgan](https://user-images.githubusercontent.com/6218739/137661146-99fb337c-9d64-4740-ab9a-5b232a1aafdb.png)

# Pix2Pix
[pix2pix算法笔记](https://blog.csdn.net/u014380165/article/details/98453672)
[Pix2Pix图图转换网络原理分析与pytorch实现](https://zhuanlan.zhihu.com/p/90300175)
自动图图转换任务被定义为：在给定充足的数据下，从一种场景转换到另一种场景。从功能实现上来看，网络需要学会“根据像素预测像素”（predict pixels from pixels）。
CNNs的研究已经给图图转换问题提供了一种简便的思路，比如设计一个编码解码网络AE，AE的输入 是白天的图像，AE的期望输出是黑夜的图像。那么可以使用MSE损失，来最小化网络输出的黑夜图像和真实黑夜图像之间的差异，实现白天到黑夜的图图转换。
然而，CNN需要我们去设计特定的损失函数，比如使用欧氏距离会导致预测的图像出现模糊。所以，需要去设计一种网络，这种网络不需要精心选择损失函数。
更确切地说，是用一种通用的损失函数形式来自动学习特定任务的损失函数，即GAN架构里判别器和生成器的损失函数是通用形式，它可以用来作为所有图图转换任务的统一损失，而具体任务的损失则是在训练过程中自动学习到的，这样就不用手动准确设定损失函数。

论文Image-to-Image Translation with Conditional Adversarial Networks发表在CVPR2017，简称pix2pix，是将GAN应用于有监督的“图像到图像”翻译的经典论文，有监督表示训练数据是成对的。图像到图像翻译（image-to-image translation）是GAN很重要的一个应用方向，什么叫图像到图像翻译呢？其实就是基于一张输入图像得到想要的输出图像的过程，可以看做是图像和图像之间的一种映射（mapping），我们常见的图像修复、超分辨率其实都是图像到图像翻译的例子。（节选为CSDN博主「AI之路」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明）
pix2pix基于GAN实现图像翻译，更准确地讲是基于cGAN（conditional GAN，也叫条件GAN），因为cGAN可以通过添加条件信息来指导图像生成，因此在图像翻译中就可以将输入图像作为条件，学习从输入图像到输出图像之间的映射，从而得到指定的输出图像。而其他基于GAN来做图像翻译的，因为GAN算法的生成器是基于一个随机噪声生成图像，难以控制输出，因此基本上都是通过其他约束条件来指导图像生成，而不是利用cGAN，这是pix2pix和其他基于GAN做图像翻译的差异。

生成器采用U-Net，这是在图像分割领域应用非常广泛的网络结构，能够充分融合特征；而原本GAN中常用的生成器结构是encoder-decoder类型。
判别器采用PatchGAN。通常判断都是对生成样本整体进行判断，比如对一张图片来说，就是直接看整张照片是否真实。而且Image-to-Image Translation中很多评价是像素对像素的，所以在这里提出了分块判断的算法，在图像的每个块上去判断是否为真，最终平均给出结果。PatchGAN的差别主要是在于Discriminator上，一般的GAN是只需要输出一个true or fasle 的矢量，这是代表对整张图像的评价；但是PatchGAN输出的是一个N x N的矩阵，这个N x N的矩阵的每一个元素，比如a(i,j) 只有True or False 这两个选择（label 是 N x N的矩阵，每一个元素是True 或者 False），这样的结果往往是通过卷积层来达到的，因为逐次叠加的卷积层最终输出的这个N x N 的矩阵，其中的每一个元素，实际上代表着原图中的一个比较大的感受野，也就是说对应着原图中的一个Patch，因此具有这样结构以及这样输出的GAN被称之为Patch GAN。这么设计的原因是依靠L1项来保证低频的准确性。为了对高频信息建模（即细节），关注对局部图像块（patches）就已经足够了。

损失函数沿用了最原始的GAN的损失，即有log作用：
![pix2pix-loss-1](https://user-images.githubusercontent.com/6218739/137694646-69faa7ab-913e-4cdc-9f6f-6e97faf30da7.png)
同时加入了一个L1损失，使生成图像不仅要像真实图片，也要更接近于输入的条件图片：
![pix2pix-loss-2](https://user-images.githubusercontent.com/6218739/137695208-19578014-4dcf-448d-8781-35a33a519441.png)
将对抗损失和L1损失相加，就得到了最终的整体损失函数：
![pix2pix-loss](https://user-images.githubusercontent.com/6218739/137695374-7ee625bb-6593-4391-8d0b-10a407d1642d.png)

pix2pix的代码实现与之前的GAN大同小异，不同的地方就是上面的模型架构和损失函数，不再赘述。

# CycleGAN
[异父异母的三胞胎：CycleGAN, DiscoGAN, DualGAN](https://zhuanlan.zhihu.com/p/26332365)
[CycleGAN](https://zhuanlan.zhihu.com/p/26995910)

pix2pix的模型是在成对的数据上训练的，也就是说，对于线条到猫的应用，我们训练的时候就需要提供一对一对的数据：一个线条画，和对应的真实的猫图片。
然而在很多情况下，我们并没有这样完美的成对的训练数据。比如说如果你想把马变成斑马，并没有这样对应的一个马对应一个斑马。然而，马的图片和斑马的图片却很多。所以这篇论文就是希望，能够通过不成对的训练数据，来学到变换。
一个普通的GAN只有一个生成器和一个判别器。而在CycleGAN里，分别有两个生成器和判别器。如下图所示：
![cyclegan](https://user-images.githubusercontent.com/6218739/138195026-c3033357-122f-4448-90b8-bd4814cf2e9c.png)
一个生成器将X域的图片转换成Y域的图片（用G表示），而另一个生成器做相反的事情，用F表示。而两个判别器$D_x$和$D_y$试图分辨两个域中真假图片。（这里假图片指的是从真照片transform来的）
看上图，X通过G生成Y，Y再通过F生成X，构成了一个循环，所以叫CycleGAN。整个cycle可以看成是一个autoencoder，两个generator看成是encoder和decoder。而两个discriminator则是准则。
损失函数分为两部分：
（1）对抗损失Adversarial Loss：
从X到Y的对抗损失为：
![cyclegan-loss-1](https://user-images.githubusercontent.com/6218739/138197675-43f6916f-3730-4836-b962-dce030489cfe.png)
从Y到X的对抗损失反之亦然。
（2）Cycle Consistency 损失
Cycle consistency是为了使得transform能成功。讲道理，如果你能从X转换到Y，然后再从Y转换到X，最后的结果应该和输入相似。这里他们用最后输出和输入的L1距离来作为另外的惩罚项。
这个惩罚项防止了mode collapse的问题。如果没有这个cycle consistency项，网络会输出更真实的图片，但是无论什么输入，都会是一样的输出。而如果加了cycle consistency，一样的输出会导致cycle consistency的直接失败。所以这规定了在经过了变换之后的图片不仅需要真实，且包含原本图片的信息。
![cyclegan-loss-2](https://user-images.githubusercontent.com/6218739/138223001-01e048e8-1f0d-4ae0-aa39-5cb518809809.png)

制作数据集时，比如想把马和斑马进行转换，那么就准备马的数据集X，斑马的数据集Y，两者不需要数量相等，也不需要一一对应。训练时，上面的损失会保证马转换成相同体型和姿态的斑马。

# ProGAN
[ProGAN：Step by step, better than better](https://zhuanlan.zhihu.com/p/93748098)
ProGAN 中的 Pro 并非 Professional，而是 Progressive，即逐渐的意思，这篇 paper 主要想解决的问题是高清图像难以生成的问题，图像生成主要的技术路线有：（1）Autoregressive Model: PixelRNN，（2）VAEs，（3）GANs。
GAN最大的好处在于生成的图像十分Sharp，而弱点则在于训练麻烦，容易崩，而且生成的数据分布只是训练集数据分布的一个子集，即多样性不足。ProGAN 最大的贡献在于提出了一种新的训练方式，即，我们不要一上来就学那么难的高清图像生成，这样会让 Generator 直接崩掉，而是从低清开始学起，学好了再提升分辨率学更高分辨率下的图片生成。从4x4到8x8一直提升到1024x1024，循序渐进，即能有效且稳定地训练出一个高质量的高分辨率生成器模型。
这样做的好处主要有二：
（1）毫无疑问，比直接学生成 1024x1024 的图像稳定多了。
（2）另外，节省时间，训练低分辨率阶段下的生成器快得不知道哪里去了，大大节省整体训练时间。

# SRGAN
[SRGAN 详解](https://perper.site/2019/03/01/SRGAN-%E8%AF%A6%E8%A7%A3/)
SRGAN目标从一个低分辨率的图片中生成它的高分辨率版本。
传统CNN方法：基于深度学习的高分辨率图像重建已经取得了很好的效果，其方法是通过一系列低分辨率图像和与之对应的高分辨率图像作为训练数据，学习一个从低分辨率图像到高分辨率图像的映射函数。但是当图像的放大倍数在4以上时，很容易使得到的结果显得过于平滑，而缺少一些细节上的真实感。这是因为传统的方法使用的代价函数一般是最小均方差（MSE），使得生成的图像有较高的信噪比，但是缺少了高频信息，出现过度平滑的纹理。作者还做了实验，证明并不是信噪比越高超分辨率效果越好。
本文的做法：应当使重建的高分辨率图像与真实的高分辨率图像无论是低层次的像素值上，还是高层次的抽象特征上，和整体概念和风格上，都应当接近。因此在loss部分，SRGAN加上了feature map部分的MSE loss。
网络结构有：
生成网络部分：SRResnet，输入是低分辨率图像（注意与原始GAN输入是噪声进行对比），由残差结构，BN，PReLU组成，用于实现高分辨率的生成。
判别器部分：由大量卷积层，Leaky ReLU,BN等结构组成，用于判别图像的真实性。
损失函数由两部分组成：（1）content loss：传统算法使用的是还原图像与GT图像之间的MSE损失，作者为了避免放大后特征过于平滑，认为高层次（features map）也应当相似。因此定义了VGG feature map loss。（2）adversarial loss：对抗网络部分的loss为判别器判别loss，即当生成器生成的图片，判别器认为为真实的图片时，该loss取得最小。
因此，SRGAN是一个监督式算法，它需要Ground Truth的输入。

## ESRGAN
[ESRGAN超分辨网络](https://zhuanlan.zhihu.com/p/338646051)
ESRGAN就是Enhanced Super-Resolution Generative Adversarial Networks，作者主要从三个方面对SRGAN进行改进：网络结构、对抗损失、感知损失。
（1）网络结构：引入了 Residual-in-Residual Dense Block (RRDB)来代替SRGAN中的resblock；移除了网络单元的BN层；增加了residual scaling，来消除部分因移除BN层对深度网络训练稳定性的影响。
（2）对抗损失：SRGAN的对抗损失的目的是为了让真实图像的判决概率更接近1，让生成图像的判决概率更接近0。而改进的ESRGAN的目标是，让生成图像和真实图像之间的距离保持尽可能大，这是引入了真实图像和生成图像间的相对距离（Relativistic average GAN简称RaGAN），而不是SRGAN中的衡量和0或1间的绝对距离。（具体说来，ESRGAN目的是：让真实图像的判决分布减去生成图像的平均分布，再对上述结果做sigmoid处理，使得结果更接近于1；让生成图像的判决分布减去真实图像的平均分布，再对上述结果做sigmoid处理，使得结果更接近于0。）
（3）感知损失：（基于特征空间的计算，而非像素空间）使用VGG网络激活层前的特征图，而不像SRGAN中使用激活层后的特征图。因为激活层后的特征图有更稀疏的特征，而激活前的特征图有更详细的细节，因此可以带来更强的监督。并且，通过使用激活后的特征图作为感知损失的计算，可以带来更加锐化的边缘和更好视觉体验。
