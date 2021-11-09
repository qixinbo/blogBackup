---
title: NLP霸主Transformer及CV新秀Vision Transformer解析
tags: [Transformer]
categories: machine learning 
date: 2021-11-9
---

# 参考文献
[保姆级教程：图解Transformer](https://cuijiahua.com/blog/2021/01/dl-basics-3.html)
[Transformer模型详解](https://terrifyzhao.github.io/2019/01/11/Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3.html)
[Transformer 详解](https://wmathor.com/index.php/archives/1438/)
[盘点 | 2021年paper大热的Transformer (ViT)](https://picture.iczhiku.com/weixin/message1610942723056.html)
["未来"的经典之作ViT：transformer is all you need!](https://zhuanlan.zhihu.com/p/356155277)
[ViT( Vision Transformer)](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/classification/ViT.html)


# 简介
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)是一篇Google于2017年提出的将Attention思想发挥到极致的论文。这篇论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN ，目前大热的Bert、GPT和DALL-E就是基于Transformer构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。
![1](https://user-images.githubusercontent.com/6218739/140851930-34149770-8d8b-4fed-bdf2-43cce255f0b1.png)

相比于NLP领域，在CV领域中，卷积神经网络CNN却是绝对的霸主。对于图像问题，CNN具有天然的先天优势（inductive bias）：平移不变性（translation equivariance）和局部性（locality）。而transformer虽然不并具备这些优势，但是transformer的核心self-attention的优势不像卷积那样有固定且有限的感受野，self-attention操作可以获得long-range信息（相比之下CNN要通过不断堆积Conv layers来获取更大的感受野），但训练的难度就比CNN要稍大一些。
仍然是Google，[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)这篇2020年的论文将Transformer引入了CV中，形成了Vision Transformer，简称为ViT。
本文尝试理解一下原始Transformer及其衍生品ViT。


# Transformer架构
Transformer 的内部，在本质上是一个 Encoder-Decoder 的结构，即 编码器-解码器。
![2](https://user-images.githubusercontent.com/6218739/140852001-0fbc70ae-83f7-40c7-b784-4ed205ac30a9.png)
Transformer 中抛弃了传统的 CNN 和 RNN，整个网络结构完全由 Attention 机制组成，并且采用了 6 层 Encoder-Decoder 结构。
![3](https://user-images.githubusercontent.com/6218739/140852095-d48c5aaa-d167-4169-8b56-5451a8db4416.png)
显然，Transformer 主要分为两大部分，分别是编码器和解码器。
整个 Transformer 是由 6 个这样的结构组成，为了方便理解，我们只看其中一个Encoder-Decoder 结构。
以一个简单的例子进行说明：
![4](https://user-images.githubusercontent.com/6218739/140852188-a8f1dee3-5b59-489b-ba4e-df8d6664f6da.png)

Why do we work?，我们为什么工作？
左侧红框是编码器，右侧红框是解码器，
编码器负责把自然语言序列映射成为隐藏层（上图第2步），即含有自然语言序列的数学表达。
解码器把隐藏层再映射为自然语言序列，从而使我们可以解决各种问题，如情感分析、机器翻译、摘要生成、语义关系抽取等。
简单说下，上图每一步都做了什么：
（1）输入自然语言序列到编码器: Why do we work?；
（2）编码器输出的隐藏层，再输入到解码器；
（3）输入 $<𝑠𝑡𝑎𝑟𝑡>$符号到解码器；
（4）解码器得到第一个字"为"；
（5）将得到的第一个字"为"落下来再输入到解码器；
（6）解码器得到第二个字"什"；
（7）将得到的第二字再落下来，直到解码器输出 $<𝑒𝑛𝑑>$，即序列生成完成。


## 编码器
编码器即是把自然语言序列映射为隐藏层的数学表达的过程。
为了方便学习，编码器可以分为 4 个部分：
![5](https://user-images.githubusercontent.com/6218739/140853558-6b30feb1-7ca3-495b-8c6f-a1456d67be8d.png)

### 位置嵌入（𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛𝑎𝑙 𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔）
我们输入数据 X 维度为[batch size, sequence length]的数据，比如我们为什么工作。
batch size 就是 batch 的大小，这里只有一句话，所以 batch size 为 1，sequence length 是句子的长度，一共 7 个字，所以输入的数据维度是 [1, 7]。
我们不能直接将这句话输入到编码器中，因为 Tranformer 不认识，我们需要先进行字嵌入，即得到图中的$X_{embedding}$。

简单点说，就是文字到字向量的转换，这种转换是将文字转换为计算机认识的数学表示，用到的方法就是Word2Vec，Word2Vec的具体细节，对于初学者暂且不用了解，这个是可以直接使用的。

得到的$X{embedding}$的维度是[batch size, sequence length, embedding dimension]，embedding dimension 的大小由 Word2Vec 算法决定，Tranformer 采用 512 长度的字向量。所以$X_{embedding}$的维度是[1, 7, 512]。

至此，输入的我们为什么工作，可以用一个矩阵来简化表示。
![6](https://user-images.githubusercontent.com/6218739/140853602-a6f68d15-8308-4b0d-b067-abb5d9a7e098.png)
我们知道，文字的先后顺序，很重要。
比如吃饭没、没吃饭、没饭吃、饭吃没、饭没吃，同样三个字，顺序颠倒，所表达的含义就不同了。
文字的位置信息很重要，Tranformer 没有类似 RNN 的循环结构，没有捕捉顺序序列的能力。
为了保留这种位置信息交给 Tranformer 学习，我们需要用到位置嵌入。
加入位置信息的方式非常多，最简单的可以是直接将绝对坐标 0,1,2 编码。
Tranformer 采用的是 sin-cos 规则，使用了 sin 和 cos 函数的线性变换来提供给模型位置信息：
$$
\begin{aligned} P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\ P E_{(\text {pos }, 2 i+1)} &=\cos \left(\text { pos } / 10000^{2 i / d_{\text {model }}}\right) \end{aligned}
$$
上式中 pos 指的是句中字的位置，取值范围是 [0, 𝑚𝑎𝑥 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑙𝑒𝑛𝑔𝑡ℎ)，i 指的是字嵌入的维度, 取值范围是 [0, 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 𝑑𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛)。
上面有 sin 和 cos 一组公式，也就是对应着 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 𝑑𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛 维度的一组奇数和偶数的序号的维度，从而产生不同的周期性变化。
可以用代码，简单看下效果。
```python
# 导入依赖库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    # 归一化, 用位置嵌入的每一行除以它的模长
    # denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    # position_enc = position_enc / (denominator + 1e-8)
    return positional_encoding
    
positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=16)
plt.figure(figsize=(10,10))
sns.heatmap(positional_encoding)
plt.title("Sinusoidal Function")
plt.xlabel("hidden dimension")
plt.ylabel("sequence length")
```
可以看到，位置嵌入在 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 𝑑𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛 （也是hidden dimension ）维度上随着维度序号增大，周期变化会越来越慢，而产生一种包含位置信息的纹理。
![embed](https://user-images.githubusercontent.com/6218739/140854019-3e363d94-cefe-4772-a7f2-fa87de1eb41b.png)
就这样，产生独一的纹理位置信息，模型从而学到位置之间的依赖关系和自然语言的时序特性。
最后，将$X_{\text {embedding }}$和 位置嵌入 相加（维度相同，可以直接相加），得到该字真正的向量表示，然后送给下一层。

### 自注意力层（𝑠𝑒𝑙𝑓 𝑎𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 𝑚𝑒𝑐ℎ𝑎𝑛𝑖𝑠𝑚）
这部分介绍来自于[这篇博客](https://terrifyzhao.github.io/2019/01/11/Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3.html)
self-attention，其思想和attention类似，但是self-attention是Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路，我们看个例子： The animal didn't cross the street because it was too tired 这里的it到底代表的是animal还是street呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把it和animal联系起来，接下来我们看下详细的处理过程。
（1）首先，self-attention会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512），注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是64低于embedding维度的。
![qkv](https://user-images.githubusercontent.com/6218739/140857139-71ca395e-ec3a-40c4-8c0d-4c815ceb9e09.png)
那么Query、Key、Value这三个向量又是什么呢？这三个向量对于attention来说很重要，当你理解了下文后，你将会明白这三个向量扮演者什么的角色。
（2）计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点乘，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即$q1·k1$，然后是针对于第二个词即$q1·k2$。
![score](https://user-images.githubusercontent.com/6218739/140857357-91caf198-e459-471b-a800-8ae705ef1434.png)
（3）接下来，把点乘的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会很大。
![score2](https://user-images.githubusercontent.com/6218739/140857512-c22f2546-25e5-449d-8632-6cdffd1dc4fe.png)
（4）下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attention在当前节点的值。
![score3](https://user-images.githubusercontent.com/6218739/140857653-468ca02a-3dfb-4d33-a8b9-36f8596f9f15.png)
在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵Q与K相乘，乘以一个常数，做softmax操作，最后乘上V矩阵：
![attention](https://user-images.githubusercontent.com/6218739/140857883-d06b029d-99ec-4a28-8711-35b0d4425f53.png)
![attention2](https://user-images.githubusercontent.com/6218739/140857941-c6f16054-2104-4205-850b-d15bc96a6659.png)
这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为scaled dot-product attention。

#### Multi-Headed Attention
这篇论文更牛的地方是给self-attention加入了另外一个机制，被称为“multi-headed” attention，该机制理解起来很简单，就是说不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，tranformer是使用了8组，所以最后得到的结果是8个矩阵。
![multihead1](https://user-images.githubusercontent.com/6218739/140868505-278beff7-e9dc-440d-87ad-b2a021310a59.png)
![multihead2](https://user-images.githubusercontent.com/6218739/140868544-311c039e-252c-425f-b957-3d1f2eece542.png)
这给我们留下了一个小的挑战，前馈神经网络没法输入8个矩阵呀，这该怎么办呢？所以我们需要一种方式，把8个矩阵降为1个，首先，我们把8个矩阵连在一起，这样会得到一个大的矩阵，再随机初始化一个矩阵和这个组合好的矩阵相乘，最后得到一个最终的矩阵。
![multihead3](https://user-images.githubusercontent.com/6218739/140868642-638e7a9f-6543-4068-b569-12984bc7b5be.png)
这就是multi-headed attention的全部流程了，这里其实已经有很多矩阵了，我们把所有的矩阵放到一张图内看一下总体的流程。
![multihead4](https://user-images.githubusercontent.com/6218739/140868817-3f900670-e211-4395-bb5a-2e9625ed3644.png)

### 残差链接和层归一化
加入了残差设计和层归一化操作，目的是为了防止梯度消失，加快收敛。
#### 残差设计
我们在上一步得到了经过注意力矩阵加权之后的 $𝑉$， 也就是$𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑄, 𝐾, 𝑉)$，我们对它进行一下转置，使其和$X_{\text {embedding }}$ 的维度一致, 也就是 [𝑏𝑎𝑡𝑐ℎ 𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑙𝑒𝑛𝑔𝑡ℎ, 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 𝑑𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛]，然后把他们加起来做残差连接，直接进行元素相加，因为他们的维度一致:
$$
X_{embedding} + Attention(Q, \ K, \ V)
$$
在之后的运算里，每经过一个模块的运算，都要把运算之前的值和运算之后的值相加，从而得到残差连接，训练的时候可以使梯度直接走捷径反传到最初始层：
$$
X + SubLayer(X)
$$
#### 层归一化
Normalization有很多种，但是它们都有一个共同的目的，那就是把输入转化成均值为0方差为1的数据。我们在把数据送入激活函数之前进行normalization（归一化），因为我们不希望输入数据落在激活函数的饱和区。
说到 normalization，那就肯定得提到 Batch Normalization。BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。
BN的具体做法就是对每一小批数据，在批这个方向上做归一化。如下图所示：
![BN](https://user-images.githubusercontent.com/6218739/140869917-059093b6-d8c7-462e-9926-ecdb6101f898.png)
可以看到，右半边求均值是沿着数据 batch_size的方向进行的，其计算公式如下：
$$
BN(x_i)=\alpha × \frac{x_i-\mu_b}{\sqrt{\sigma^2_B+\epsilon}}+\beta
$$
那么什么是 Layer normalization 呢？它也是归一化数据的一种方式，不过 LN 是在每一个样本上计算均值和方差，而不是BN那种在批方向计算均值和方差！
![LN](https://user-images.githubusercontent.com/6218739/140870020-f7fb5f4e-d6de-4838-9dbb-e809e0f2fdec.png)
LN的公式为：
$$
LN(x_i)=\alpha × \frac{x_i-\mu_L}{\sqrt{\sigma^2_L+\epsilon}}+\beta
$$

### 前馈网络
前馈网络FeedForward，其实就是两层线性映射并用激活函数激活。
然后经过这个网络激活后，再经过一个残差连接和层归一化，即可输出。
直接看代码可能更直观：
```python
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 两个fc层，对最后的512维度进行变换
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
```

### 编码器总结
经过上面 3 个步骤，我们已经基本了解了 Encoder 的主要构成部分。
用一个更直观的图表示如下：
![encoder](https://user-images.githubusercontent.com/6218739/140872426-955ab9c0-842e-444c-947b-b6d9292a4fe1.png)
文字描述为：
输入$x_1,x_2$经 self-attention 层之后变成$z_1,z_2$ ，然后和输入$x_1,x_2$进行残差连接，经过 LayerNorm 后输出给全连接层。全连接层也有一个残差连接和一个 LayerNorm，最后再输出给下一个 Encoder（每个 Encoder Block 中的 FeedForward 层权重都是共享的）
公式描述为：
（1）字向量与位置编码
$$
X = \text{Embedding-Lookup}(X) + \text{Positional-Encoding}
$$
（2）自注意力机制
$$
\begin{align}
Q &= \text{Linear}_q(X) = XW_{Q}\\
K &= \text{Linear}_k(X) = XW_{K}\\
V &= \text{Linear}_v(X) = XW_{V}\\
X_{attention} &= \text{Self-Attention}(Q,K,V)
\end{align}
$$
（3）self-attention 残差连接与 Layer Normalization
$$
\begin{align}
X_{attention} &= X + X_{attention}\\
X_{attention} &= \text{LayerNorm}(X_{attention})
\end{align}
$$
（4）前馈网络FeedForward
$$
X_{hidden} = \text{Linear}(\text{ReLU}(\text{Linear}(X_{attention})))
$$
（5）FeedForward 残差连接与 Layer Normalization
$$
\begin{align}
X_{hidden} &= X_{attention} + X_{hidden}\\
X_{hidden} &= \text{LayerNorm}(X_{hidden})
\end{align}
$$
其中：
$$
X_{hidden} \in \mathbb{R}^{batch\_size  \ * \  seq\_len. \  * \  embed\_dim}
$$

## 解码器
见[原文](https://wmathor.com/index.php/archives/1438/)。
Decoder架构如下：
![decoder](https://user-images.githubusercontent.com/6218739/140873135-92be1693-5efe-4998-9c6b-9071a3c96fae.png)
我们先从 HighLevel 的角度观察一下 Decoder 结构，从下到上依次是：
（1）Masked Multi-Head Self-Attention
（2）Multi-Head Encoder-Decoder Attention
（3）FeedForward Network
和 Encoder 一样，上面三个部分的每一个部分，都有一个残差连接，后接一个 Layer Normalization。Decoder 的中间部件并不复杂，大部分在前面 Encoder 里我们已经介绍过了，但是 Decoder 由于其特殊的功能，因此在训练时会涉及到一些细节。

### Masked Self-Attention
具体来说，传统 Seq2Seq 中 Decoder 使用的是 RNN 模型，因此在训练过程中输入$t$时刻的词，模型无论如何也看不到未来时刻的词，因为循环神经网络是时间驱动的，只有当$t$时刻运算结束了，才能看到$t+1$时刻的词。而 Transformer Decoder 抛弃了 RNN，改为 Self-Attention，由此就产生了一个问题，在训练过程中，整个 ground truth 都暴露在 Decoder 中，这显然是不对的，我们需要对 Decoder 的输入进行一些处理，该处理被称为 Mask。
举个例子，Decoder 的 ground truth 为 "start起始符号 I am fine"，我们将这个句子输入到 Decoder 中，经过 WordEmbedding 和 Positional Encoding 之后，将得到的矩阵做三次线性变换$W_Q,W_K,W_V$。然后进行 self-attention 操作，首先通过$\frac {Q\times K^T}{\sqrt {d_k}}$得到 Scaled Scores，接下来非常关键，我们要对 Scaled Scores 进行 Mask，举个例子，当我们输入 "I" 时，模型目前仅知道包括 "I" 在内之前所有字的信息，即 "start起始符号" 和 "I" 的信息，不应该让其知道 "I" 之后词的信息。道理很简单，我们做预测的时候是按照顺序一个字一个字的预测，怎么能这个字都没预测完，就已经知道后面字的信息了呢？Mask 非常简单，首先生成一个下三角全 0，上三角全为负无穷的矩阵，然后将其与 Scaled Scores 相加即可：
![mask](https://user-images.githubusercontent.com/6218739/140873707-8ab175f3-df87-40bc-901f-62305d01b2cd.png)
之后再做 softmax，就能将 - inf 变为 0，得到的这个矩阵即为每个字之间的权重：
![unmask](https://user-images.githubusercontent.com/6218739/140873785-a14ea99e-09c5-4610-a97f-39039bd99ad6.png)
Multi-Head Self-Attention 无非就是并行的对上述步骤多做几次，前面 Encoder 也介绍了，这里就不多赘述了。

### Masked Encoder-Decoder Attention
其实这一部分的计算流程和前面 Masked Self-Attention 很相似，结构也一摸一样，唯一不同的是这里的$K,V$为 Encoder 的输出（不然Encoder辛辛苦苦做的输出就没用了），$Q$为 Decoder 中 Masked Self-Attention 的输出。
![e-d](https://user-images.githubusercontent.com/6218739/140874351-f075901e-7869-4f2d-8c03-c94b12116938.png)


# Vision Transformer
使用Transformer进行视觉任务的研究已经成了一个新的热点，大家为了更低的模型复杂度以及训练的效率，都在研究如何将这一技术应用在视觉任务上。
通常来说，在所有的关于Transformer的论文以及工作中，有两个比较大的架构，其中一个就是传统的CNNs加Transformer组合而成的结构，另一种是纯粹的Transformers。
ViT使用的就是纯粹的Transformer去完成视觉任务，也就是说，它没有使用任何的CNNs。这也是它的价值所在，谷歌大脑团队在几乎没有修改任何基于NLP的Transformer的结构基础之上，只是将输入进行了一个适配，将图片切分成许多的小格，然后将这些作为序列输入到模型，最终完成了分类任务，并且效果可以直追基于CNNs的SOTA。
ViT的思路很简单：直接把图像分成固定大小的patchs，然后通过线性变换得到patch embedding，这就类比NLP的words和word embedding，由于transformer的输入就是a sequence of token embeddings，所以将图像的patch embeddings送入transformer后就能够进行特征提取从而分类了。ViT模型原理如下图所示，其实ViT模型只是用了transformer的Encoder来提取特征（原始的transformer还有decoder部分，用于实现sequence to sequence，比如机器翻译）。
![ViT](https://user-images.githubusercontent.com/6218739/140877989-64b07f86-fe4f-45df-8f4c-a30a11391cb0.png)

ViT架构相对于原始Transformer，有一些特殊处理：
## 图像分块嵌入
考虑到在Transformer结构中，输入是一个二维的矩阵，矩阵的形状可以表示为$(N,D)$，其中$N$是sequence的长度，而$D$是sequence中每个向量的维度。因此，在ViT算法中，首先需要设法将$H \times W \times C$的三维图像转化为$(N,D)$的二维输入。
ViT中的具体实现方式为：将$H \times W \times C$的图像，变为一个$N \times (P^2 \times C)$的序列。这个序列可以看作是一系列展平的图像块，也就是将图像切分成小块后，再将其展平。该序列中一共包含了$N=HW/P^2$个图像块，每个图像块的维度则是$(P^2\times C)$。其中$P$是图像块的大小，$C$是通道数量。经过如上变换，就可以将$N$视为sequence的长度了。但是，此时每个图像块的维度是$(P^2\times C)$，而我们实际需要的向量维度是$D$，因此我们还需要对图像块进行 Embedding。这里 Embedding 的方式非常简单，只需要对每个$(P^2 \times C)$的图像块做一个线性变换，将维度压缩为$D$即可。

## Class Token
ViT借鉴BERT增加了一个特殊的class token。transformer的encoder输入是a sequence patch embeddings，输出也是同样长度的a sequence patch features，但图像分类最后需要获取image feature，简单的策略是采用pooling，比如求patch features的平均来获取image feature，但是ViT并没有采用类似的pooling策略，而是直接增加一个特殊的class token，其最后输出的特征加一个linear classifier就可以实现对图像的分类（ViT的pre-training时是接一个MLP head），所以输入ViT的sequence长度是$N+1$。
class token对应的embedding在训练时随机初始化，然后通过训练得到。

## Positional Encoding
按照 Transformer 结构中的位置编码习惯，这个工作也使用了位置编码。不同的是，ViT 中的位置编码没有采用原版 Transformer 中的 sin-cos编码，而是直接设置为可学习的 Positional Encoding。对训练好的 Positional Encoding 进行可视化，可以看到，位置越接近，往往具有更相似的位置编码。此外，出现了行列结构，同一行/列中的 patch 具有相似的位置编码。
![vit-pos](https://user-images.githubusercontent.com/6218739/140884681-87fb368e-87a6-4e6a-a9df-3ea19a31012b.png)