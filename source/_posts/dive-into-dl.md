---
title: 《动手学深度学习》PyTorch版学习笔记
tags: [Machine Learning, PyTorch]
categories: programming
date: 2020-3-8
---

Attention预警！：
时刻铭记“Garbage in, garbage out!”，因此，涉及到data时，一定注意实际查看，确保计算时Input和Output的一致性和准确性。

原书MXNet版在[这里](https://github.com/d2l-ai/d2l-zh)。
PyTorch版在[这里](https://github.com/ShusenTang/Dive-into-DL-PyTorch)。

# 深度学习简介
目前的机器学习和深度学习应用共同的核心思想：用数据编程。
通俗来说，机器学习是一门讨论各式各样的适用于不同问题的函数形式，以及如何使用数据来有效地获取函数参数具体值的学科。
深度学习是具有多级表示的表征学习方法（表征学习关注如何自动找出表示数据的合适方式，以便更好地将输入变换为正确的输出）。在每一级（从原始数据开始），深度学习通过简单的函数将该级的表示变换为更高级的表示。因此，深度学习模型也可以看作是由许多简单函数复合而成的函数。当这些复合的函数足够多时，深度学习模型就可以表达非常复杂的变换。值得一提的是，作为表征学习的一种，深度学习将自动找出每一级表示数据的合适方式。
（1）深度学习的一个外在特点是端到端的训练。也就是说，并不是将单独调试的部分拼凑起来组成一个系统，而是将整个系统组建好之后一起训练。比如说，计算机视觉科学家之前曾一度将特征抽取与机器学习模型的构建分开处理，像是Canny边缘探测和SIFT特征提取曾占据统治性地位达10年以上，但这也就是人类能找到的最好方法了。当深度学习进入这个领域后，这些特征提取方法就被性能更强的自动优化的逐级过滤器替代了。
（2）除端到端的训练以外，我们也正在经历从含参数统计模型转向完全无参数的模型。当数据非常稀缺时，我们需要通过简化对现实的假设来得到实用的模型。当数据充足时，我们就可以用能更好地拟合现实的无参数模型来替代这些含参数模型。这也使我们可以得到更精确的模型，尽管需要牺牲一些可解释性。

# 预备知识
## 数据操作
在PyTorch中，torch.Tensor是存储和变换数据的主要工具。Tensor和NumPy的多维数组非常类似。然而，Tensor提供GPU计算和自动求梯度等更多功能，这些使Tensor更加适合深度学习。
### 创建
```python
x = torch.empty(5, 3) # 创建一个5x3的未初始化的Tensor
x = torch.rand(5, 3) # 创建一个5x3的随机初始化的Tensor
x = torch.zeros(5, 3, dtype=torch.long) # 创建一个5x3的long型全0的Tensor
x = torch.tensor([5.5, 3]) # 接根据数据创建
x = x.new_ones(5, 3, dtype=torch.float64) # 可以通过现有的Tensor来创建，此方法会默认重用输入Tensor的一些属性，例如torch.dtype和torch.device
x = torch.randn_like(x, dtype=torch.float) # 除非自定义数据类型
print(x.shape) # 可以通过shape或者size()来获取Tensor的形状
print(x.size()) # 注意：返回的torch.Size其实就是一个tuple, 支持所有tuple的操作。
```
还有很多函数可以创建Tensor，如eye、arange、linspace，这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu)。

### 操作
（1）算术操作，以加法为例：
```python
y = torch.rand(5, 3)
print(x + y) # 加法形式一
print(torch.add(x, y)) # 加法形式二
result = torch.empty(5, 3)
torch.add(x, y, out=result) # 可以指定输出
print(result)
y.add_(x) # 加法形式三 inplace，PyTorch操作inplace版本都有后缀_
print(y)
```

（2）索引
可以使用类似NumPy的索引操作来访问Tensor的一部分，需要注意的是：索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
```python
y = x[0, :]
y += 1
print(y)
print(x[0, :])
```
PyTorch还提供了一些高级的选择函数，如index_select、masked_select等。
（3）改变形状
用view()来改变Tensor的形状：
```python
y = x.view(15)
z = x.view(-1, 5) # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
```
注意view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
另外注意：虽然view返回的Tensor与源Tensor是共享data的，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者id（内存地址）并不一致。
所以如果我们想返回一个真正新的副本（即不共享data内存）该怎么办呢？Pytorch还提供了一个reshape()可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。推荐先用clone创造一个副本然后再使用view。
```python
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)
```
使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。
另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number：
```python
x = torch.randn(1)
print(x)
print(x.item())
```
PyTorch中的Tensor支持超过一百种操作，包括转置、索引、切片、数学运算、线性代数、随机数等等，具体可参考官方API。

### 广播机制
当对两个形状不同的Tensor按元素运算时，可能会触发广播（broadcasting）机制：先适当复制元素使这两个Tensor形状相同后再按元素运算。
```python
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)
```
这部分的广播机制可以参照numpy的广播来理解。

### 运算的内存开销
前面说了，索引操作是不会开辟新内存的，而像y = x + y这样的运算是会新开内存的，然后将y指向新内存。可以使用Python自带的id函数来验证：如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同。
```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = x + y
print(id(y) == id_before)
```
如果想指定结果到原来的y的内存，可以使用前面介绍的索引来进行替换操作，如下：
```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = x + y
print(id(y) == id_before)
```
还可以使用运算符全名函数中的out参数或者自加运算符+=(也即add_())达到上述效果：
```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
# torch.add(x, y, out = y) 
# y += x
y.add_(x)
print(id(y) == id_before)
```

### Tensor和Numpy相互转换
我们很容易用numpy()和from_numpy()将Tensor和NumPy中的数组相互转换。但是需要注意的一点是： 这两个函数所产生的的Tensor和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！
还有一个常用的将NumPy中的array转换成Tensor的方法就是torch.tensor(), 需要注意的是，此方法总是会进行数据拷贝（就会消耗更多的时间和空间），所以返回的Tensor和原来的数据不再共享内存。
（1）Tensor转Numpy
```python
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)
```

（2）Numpy转Tensor
```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)
```
所有在CPU上的Tensor（除了CharTensor）都支持与NumPy数组相互转换。

### Tensor on GPU
用方法to()可以将Tensor在CPU和GPU（需要硬件支持）之间相互移动。
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device) # 直接创建一个在GPU上的Tensor
    x = x.to(device)  # 等价于.to('cuda')
    z = x + y
    print(z)
    print(z.to('cpu'), torch.double)
```

## 自动求梯度
PyTorch提供的autograd包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。
### 概念
Tensor是autograd包的核心类，如果将其属性.requires_grad设置为True，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用.backward()来完成所有梯度计算。此Tensor的梯度将累积到.grad属性中。
注意在y.backward()时，如果y是标量，则不需要为backward()传入任何参数；否则，需要传入一个与y同形的Tensor。
如果不想要被继续追踪，可以调用.detach()将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用with torch.no_grad()将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（requires_grad=True）的梯度。

Function是另外一个很重要的类。Tensor和Function互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的，若是，则grad_fn返回一个与这些运算相关的对象，否则是None。

### Tensor
```python
x = torch.ones(2, 2, requires_grad=True) # 创建一个Tensor并设置requires_grad=True
print(x)
print(x.grad_fn) # None, x是直接创建的，所以它没有grad_fn
y = x + 2 
print(y)
print(y.grad_fn) # y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn
print(x.is_leaf, y.is_leaf) # 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None
z = y * y * 3
out = z.mean()
print(z, out) # z的grad_fn是MulBackward，out的grad_fn是MeanBackward
a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True) # 通过.requires_grad_()来用in-place的方式改变requires_grad属性
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn) # b的grad_fn是SumBackward
```

### 梯度
```python
out.backward() # 因为out是一个标量，所以调用backward()时不需要指定求导变量
print(x.grad)

out2 = x.sum()
out2.backward() # grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度
print(x.grad)

out3 = x.sum()
x.grad.data.zero_() # 所以一般在反向传播之前需把梯度清零
out3.backward()
print(x.grad)
```
首先，有一个向量x，它经过某个运算得到了向量y，那么y对x的导数就是一个Jacobian矩阵，记为J。
再者，始终记着，且抛开诸如神经网络等所有的知识点，从计算上来说，torch.autograd就是一个纯粹的计算引擎，它是计算向量与Jacobian矩阵的乘积的一个运算库（Jacobian矩阵是在底层计算的），即vector-Jabobian product计算引擎。这个vector可以是任意向量，记为v，注意是任意的。
最后，向量y经过运算又得到了标量l。此时，如果上面的向量v恰好是l对y的导数，那么根据链式法则，autograd所计算的vector-Jacobian product恰好就是标量l对向量x的导数。所以，autograd的效果就是只要知道y=f(x)和l=g(y)或者l对y的导数v，就可以得到l对x的导数。

那么，具体来说，autograd在反向传播计算梯度时，out.backward()其实是需要一个grad_tensors，就是上面那个向量v，则有两种情形：
（1）如果out是标量，该参数就可以为空，此时out就是l。因为通常是最后的损失函数调用backward()，而损失函数又通常是标量，所以此时backward()不需要参数。
（2）如果out不是标量，比如上面的y，y.backward(v)，那么就需要指定这个v，而这个v是与y同型的，这是为了让y乘以v是一个标量，这样就保证了始终是标量对张量求导。再说一遍，这个v可以是任意的，但如果恰巧是未知的标量l对y的导数，这样往前计算的梯度就有了意义，比如x中的梯度就是遥远的l对x的梯度。
那么，总的计算梯度的方式就是这样：
```python
x -> y -> z -> m -> n -> l
设l就是一个标量，那么：
l.backward()
就会先计算dl/dn, 然后：
n.backward(dl/dn)
这样，仍然n*dl/dn是一个标量，就是l，那么继续往前传，先得到dl/dm，再传入：
m.backward(dl/dm)
所以，每一次backward()时总是一个标量，而且是l，所以，传到最后，就可以得到：
dl/dx
```

如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我们可以对tensor.data进行操作。
```python
x = torch.ones(1, requires_grad=True)
print(x.data) # 还是一个Tensor
print(x.data.requires_grad) # False，此时x已经独立于计算图以外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图中，所以不会影响梯度传播
y.backward()
print(x) # x的值会发生变化
print(x.grad) # 但梯度不变
```

# 深度学习基础
## 线性回归
线性回归输出是一个连续值，因此适用于回归问题。
与回归问题不同，分类问题中模型的最终输出是一个离散值。softmax回归则适用于分类问题。
## 线性回归的从零开始实现
尽管强大的深度学习框架可以减少大量重复性工作，但若过于依赖它提供的便利，会导致我们很难深入理解深度学习是如何工作的。因此，本节将介绍如何只利用Tensor和autograd来实现一个线性回归的训练。
首先，导入本节中实验所需的包或模块，其中的matplotlib包可用于作图，且设置成嵌入显示。
```python
%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython import display
```

### 生成数据集
```python
# 构造一个简单的人工训练数据集，它可以使我们能够直观比较学到的参数和真实的模型参数的区别
num_inputs = 2 # 输入特征数为2
num_examples = 100 # 训练集样本数为100
true_w = [2, -3.4] # 真实的权重
true_b = 4.2 # 真实的偏差

features = torch.randn(num_examples, num_inputs, dtype=torch.float32) # 根据训练集个数生成随机的批量样本
labels = true_w[0]*features[:, 0] + true_w[1]*features[:, 1] + true_b # 真实的标签
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32) # 在真实标签上加上服从正态分布的噪声项

print(features[0], labels[0]) # 打印第一个样本的输入特征和标签

plt.scatter(features[:, 1].numpy(), labels.numpy(), 1) # 将第二个特征与标签进行作图，看一下它们的线性关系
```

### 读取数据
在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回batch_size（批量大小）个随机样本的特征和标签。
```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # 生成读取索引
    random.shuffle(indices) # 打乱索引顺序
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)]) # 将索引包装成一个Tensor，同时注意最后一次可能不足一个batch，要取实际的大小
        yield features.index_select(0, j), labels.index_select(0, j) # yield关键字将data_iter()函数做成了一个迭代器

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break # 读取第一个小批量数据样本并打印。每个批量的特征形状为(10, 2)，分别对应批量大小和输入个数；标签形状为批量大小。
```
对这里面用的yield用法，可以参看这篇博文辅助理解：[python中yield的用法详解——最简单，最清晰的解释](https://blog.csdn.net/mieleizhi0522/article/details/82142856)

### 初始化模型参数
```python
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32) # 将权重初始化为均值为0、标准差为0.01的正态随机数
b = torch.zeros(1, dtype=torch.float32) # 将偏差初始化为0
w.requires_grad_(requires_grad=True) # 模型训练过程中需要对这些参数求梯度来迭代取值，所以需要将requires_grad置为True
b.requires_grad_(requires_grad=True)
```

### 定义模型
```python
def linreg(X, w, b):
    return torch.mm(X, w) + b # 线性回归的矢量计算表达式
```
### 定义损失函数
```python
def squared_loss(y_hat, y):
    # 注意这里返回的是向量，把y变成了y_hat的样子，注意这个地方与PyTorch中MSELoss不同
       # MSELoss返回的是标量，即将这里的向量再做一个操作，默认是取平均值，也可以取和
    return(y_hat - y.view(y_hat.size())) ** 2 / 2
```
### 定义优化算法
以下的sgd函数实现了小批量随机梯度下降算法，它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和（这是因为上面的损失函数是带小批量的，后面的loss.sum()会将这一批样本上的损失都加和，如果是像PyTorch的MSELoss取平均值的话，这里就不需要除以批量大小），将它除以批量大小来得到平均值。
```python
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param用的是param.data
```
下图的计算公式一目了然：
![image](https://user-images.githubusercontent.com/6218739/75660181-4d558c00-5ca6-11ea-95a8-a343ea73ad35.png)

### 训练模型
在训练中，我们将多次迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征X和标签y），通过调用反向函数backward计算小批量随机梯度，并调用优化算法sgd迭代模型参数。由于我们之前设批量大小batch_size为10，每个小批量的损失l的形状为(10, 1)。由于变量l并不是一个标量，所以我们可以调用.sum()将其求和得到一个标量，再运行l.backward()得到该变量有关模型参数的梯度。注意在每次更新完参数后不要忘了将参数的梯度清零。
在一个迭代周期（epoch）中，我们将完整遍历一遍data_iter函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。
```python
lr = 0.03 # 学习率
num_epochs = 30 # 迭代次数
net = linreg # 线性回归模型
loss = squared_loss # 平方损失

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum() # l是有关小批量X和y的损失
        l.backward() # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels) # 每次迭代后都将模型测试一下
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
```
打印一下最终的参数，它们应该很接近于真实的参数：
```python
print(true_w, '\n', w)
print(true_b, '\n', b)
```
## 线性回归的简洁实现
在本节中，将介绍如何使用PyTorch更方便地实现线性回归的训练。
### 生成数据集
这一步跟上面的相同，不再赘述。
### 读取数据
PyTorch提供了data包来读取数据。由于data常用作变量名，将导入的data模块用Data代替。
```python
import torch.utils.data as Data
batch_size = 10 # 一个小批量设为10个
dataset = Data.TensorDataset(features, labels) # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```
这里data_iter的使用跟上一节中的一样。让我们读取并打印第一个小批量数据样本。
```python
for X, y in data_iter:
    print(X, y)
    break
```
### 定义模型
首先，导入torch.nn模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。之前我们已经用过了autograd，而nn就是利用autograd来定义模型。nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。下面先来看看如何用nn.Module实现一个线性回归模型。
```python
import torch.nn as nn
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x): # 定义前向传播
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)
```
事实上我们还可以用nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中：

```python
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 还可以再添加层
)
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
                                 ('linear', nn.Linear(num_inputs, 1))
                                 # 还可以再添加层
]))
print(net) # 注意这个地方的输出与前面自定义net的不同
print(net[0]) # 这里的net是Sequential实例，所以需要加索引
```
可以通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个生成器。
```python
for param in net.parameters():
    print(param) # 这里面的参数的个数和尺寸是根据网络层的结构及输入输出自动确定的。
```
注意：torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用input.unsqueeze(0)来添加一维。

### 初始化模型参数
在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。PyTorch在init模块中提供了多种参数初始化方法。这里的init是initializer的缩写形式。我们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。
```python
from torch.nn import init
# net[0]这样根据下标访问子模块的写法只有当net是个ModuleList或者Sequential实例时才可以
# 如果net是像第一种那样自定义的，需要将索引去掉

init.normal_(net[0].weight, mean=0, std=0.1)
init.constant_(net[0].bias, val=0)
```
### 定义损失函数
PyTorch在nn模块中提供了各种损失函数，这些损失函数可看作是一种特殊的层，PyTorch也将这些损失函数实现为nn.Module的子类。
```python
loss = nn.MSELoss() # 使用均方误差损失作为损失函数
```

### 定义优化算法
同样，我们也无须自己实现小批量随机梯度下降算法。torch.optim模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。
```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03) # 设置学习率为0.03
print(optimizer)
```
还可以为不同子网络设置不同的学习率，这在finetune时经常用到。例：
```python
# 这里不能直接运行，因为subnet1都是假的
optimizer = optim.SGD([
                       {'params': net.subnet1.parameters()}, # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                       {'params': net.subnet2.parameters(), 'lr': 0.01}
], lr=0.03)

```
有时候我们不想让学习率固定成一个常数，那如何调整学习率呢？主要有两种做法。一种是修改optimizer.param_groups中对应的学习率，另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。
```python
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率调整为之前的0.1倍
```
### 训练模型
通过调用optim实例的step函数来迭代模型参数：
```python
num_epochs = 30

for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1)) # output的size是[10, 1]，而y的size是[10]，所以y要改变一下形状
        # 同时，这里的l是算的这一批次上的样本的损失的平均值
        optimizer.zero_grad() # 梯度清零， 等价于net.zero_grad()
        l.backward()
        optimizer.step()

    print('epoch %d, loss %f' % (epoch, l.item()))
```
将学习到的权重和偏差与真实值进行一下对比：
```python
print(true_w, net[0].weight)
print(true_b, net[0].bias)
```
## softmax回归
前几节介绍的线性回归模型适用于输出为连续值的情景。在另一类情景中，模型输出可以是一个像图像类别这样的离散值。对于这样的离散值预测问题，我们可以使用诸如softmax回归在内的分类模型。和线性回归不同，softmax回归的输出单元从一个变成了多个，且引入了softmax运算使输出更适合离散值的预测和训练。

虽然我们仍然可以使用回归模型来进行建模，并将预测值就近定点化到1、2和3这样的离散值之一，但这种连续值到离散值的转化通常会影响到分类质量。因此我们一般使用更加适合离散值输出的模型来解决分类问题。
（1）softmax运算的必要性
既然分类问题需要得到离散的预测输出，一个简单的办法是将输出值当作预测类别是i的置信度，并将值最大的输出所对应的类作为预测输出。然而，直接使用输出层的输出有两个问题。一方面，由于输出层的输出值的范围不确定，我们难以直观上判断这些值的意义。另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。
softmax运算符（softmax operator）解决了以上两个问题。它通过softmax操作将输出值变换成值为正且和为1的概率分布，这样某一概率大了，也等价于它的输出是大的，这样就不改变预测类别输出。
（2）交叉熵损失函数
softmax运算将输出变换成一个合法的类别预测分布。另一方面，真实标签也可以用类别分布表达，比如构造一个向量，使其第i个元素为1，其余为0，就代表样本i类别的离散数值。这样，训练目标就可以设为预测概率分布尽可能接近于真实的标签概率分布。
因此，就是寻找适合衡量两个概率分布差异的测量函数，其中，交叉熵（cross entropy）是一个常用的衡量方法，它只关心对正确类别的预测概率，只要其值足够大，就可以确保分类结果正确，因为如果不是正确类别，这个交叉熵就是0。
（3）模型预测及评价
在训练好softmax回归模型后，给定任一样本特征，就可以预测每个输出类别的概率。通常把预测概率最大的类别作为输出类别。如果它与真实类别（标签）一致，说明这次预测是正确的。下面将使用准确率（accuracy）来评价模型的表现。它等于正确预测数量与总预测数量之比。

## 图像分类数据集Fashion-MNIST
本节将使用torchvision包来加载Fashion-MNIST数据集，
它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。torchvision主要由以下几部分构成：
torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。

### 获取数据集
```python
# 通过train参数指定获取训练集或测试集，download指定是否下载
# transforms.ToTensor()使所有数据转换为Tensor，如果不进行转换则返回的是PIL图片
mnist_train = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transforms.ToTensor())
```
transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor。
这个地方有个坑：如果输入的数组是0到255之间，但数据类型不是np.uint8，那么ToTensor()只会更改通道顺序，而不会除以255变换到0到1之间，所以，如果用像素值(0-255整数)表示图片数据，那么一律将其类型设置成uint8，避免不必要的bug。一定要确保input和output都是心里有数的。

见如下几篇相关帖子：
[Bugs with torchvision.transforms.ToPILImage()?](https://discuss.pytorch.org/t/bugs-with-torchvision-transforms-topilimage/26109)
[2.2.4 图像数据的一个坑](https://tangshusen.me/2018/12/05/kaggle-doodle-reco/)
 
查看数据集中的数据：
```python
from skimage import io
from skimage import img_as_ubyte
# mnist_train和mnist_test都是torch.utils.data.Dataset的子类
print(len(mnist_train))
feature, label = mnist_train[0]
print(feature.shape, label) # shape是通道*高*宽
io.imsave("1.png", img_as_ubyte(feature.view((28, 28)).numpy()))
```
feature中的图像已经是[0.0, 1.0]范围，所以需要转化成[0, 255]范围才能正常显示。
参考见[这篇文章](https://www.cnblogs.com/denny402/p/5122328.html)。

将数值标签与文本标签对应起来：
```python
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```
### 读取小批量
在实践中，数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时。PyTorch的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取。
```python
batch_size = 256
num_workers = 4
# mnist_train是torch.utils.data.Dataset的子类，所以我们可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例
# 通过参数num_workers来设置4个进程读取数据
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle=False, num_workers=num_workers)
```
查看一下，读取一遍训练数据需要的时间：
```python
import time
start = time.time()
for X, y in train_iter:
    continue

print('%.2f sec' % (time.time()-start))
```
## softmax回归的从零开始实现
### 获取和读取数据
这一步就是按照上一节中下载和小批量读取Fashion-MNIST数据集的方式。
### 初始化模型参数
跟线性回归中的例子一样，我们将使用向量表示每个样本。
```python
import numpy as np
num_inputs = 784 # 这里用向量来表示图像，28*28=784，相当于784个输入特征
num_outputs = 10 # 输出为10个类别
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float) # 权重为784*10，初始为正态分布
b = torch.tensor(np.zeros(num_outputs), dtype=torch.float) # 偏差为10， 初始为0
W.requires_grad_(requires_grad=True) # 设置需要计算梯度
b.requires_grad_(requires_grad=True)
```

### 实现softmax运算
```python
# 先演示一下怎样对多维Tensor按维度进行操作
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim = 0, keepdim=True)) # tensor([[5, 7, 9]])，对同一列的元素求和，并保持原来的维度
print(X.sum(dim = 1, keepdim=True)) # tensor([[6], [15]])，对同一行的元素求和，并保持原来的维度
# 定义softmax运算
def softmax(X): # 矩阵X的行数是样本数，列数是输出个数
    X_exp = X.exp() # 先对每个元素做指数运算
    partition = X_exp.sum(dim = 1, keepdim = True) # 然后对exp矩阵同一行的元素求和
    return X_exp / partition # 最后令exp矩阵的每行各元素与该行元素之和相除，最终使得每行元素的和为1且非负，所以每行都是合法的概率分布
    # 这个除法也应用了广播机制
```

### 定义模型
```python
def net(X):
    # 注意，这里需要使用view函数将图像转换为向量
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
```
### 定义损失函数
```python
# 先演示一下gather函数的用法
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 这是虚构的两个样本在3个类别下的预测概率
y = torch.LongTensor([0, 2]) # 这是虚构的两个样本的标签类别
y_hat.gather(1, y.view(-1, 1)) # tensor([[0.1000], [0.5000]]) 这样就得到了两个样本的标签的预测概率
```
关于gather函数，其实就是索引，具体解析可以参看[这篇博文](https://www.cnblogs.com/HongjianChen/p/9451526.html)。
那么交叉熵损失函数就是：
```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
```

### 计算分类准确率
给定一个类别的预测概率分布y_hat，我们把预测概率最大的类别作为输出类别。如果它与真实类别y一致，说明这次预测是正确的。分类准确率即正确预测数量与总预测数量之比。
```python
def accuracy(y_hat, y):
    # argmax返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同
    # 相等条件判断式(y_hat.argmax(dim=1) == y)是一个类型为ByteTensor的Tensor
    # 用float()将其转换为值为0（相等为假）或1（相等为真）的浮点型Tensor
    return (y_hat.argmax(dim=1) == y).float().mean().item()

print(accuracy(y_hat, y))
# 继续使用在演示gather函数时定义的变量y_hat和y，并将它们分别作为预测概率分布和标签
# 可以看到，第一个样本预测类别为2（该行最大元素0.6在本行的索引为2），与真实标签0不一致
# 第二个样本预测类别为2（该行最大元素0.5在本行的索引为2），与真实标签2一致
# 因此，这两个样本上的分类准确率为0.5。
```

评价模型net在数据集data_iter上的准确率：
```python
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item() # 注意这个地方先求和
        n +=  y.shape[0]

    return acc_sum /n # 再求平均
```

### 训练模型
```python
num_epochs, lr = 5, 0.1
def sgd(params, lr, batch_size): # 定义优化算法
    for param in params:
        param.data -= lr * param.grad / batch_size 

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum() # 这个地方是求和，所以后面的SGD算法中除以batch size
            if optimizer is not None: # 这里看是否传入了优化器，默认是不传入
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None: # 如果不传入优化器，就显式地对参数梯度置为0
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' 
              % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
```

### 预测
```python
X, y = iter(test_iter).next()
print(net(X).argmax(dim=1)[:10], y[:10])
```

## softmax回归的简洁实现
### 获取和读取数据
与上一节相同
### 定义和初始化模型
```python
import torch
import torch.nn as nn

num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x): # x的形状是(batch, 1, 28, 28)，所以要view一下
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)
```
还可以将形状转换这一块单独提出来：
```python
class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

from collections import OrderedDict
net = nn.Sequential(
    OrderedDict([
                 ('flatten', FlattenLayer()),
                 ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
```
然后，我们使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。
```python
from torch.nn import init
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
```

### softmax和交叉熵损失函数
在上一节的练习中，分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。因此，PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳定性更好。
```python
loss = nn.CrossEntropyLoss()
```
### 定义优化算法
```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```
### 训练模型
使用上一节定义的训练函数，只是传入不同的参数：
```python
num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

## 多层感知机
多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。
由于输入层不涉及计算，因此，多层感知机的层数等于隐藏层的层数加上输出层。
全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。
（1）激活函数
对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function）。
常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。
（2）多层感知机
多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。
多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。

## 权重衰减
过拟合现象，即模型的训练误差远小于它在测试集上的误差。虽然增大训练数据集可能会减轻过拟合，但是获取额外的训练数据往往代价高昂。
应对过拟合问题的常用方法：权重衰减（weight decay）和丢弃法（Dropout，下一节介绍）。
权重衰减等价于L2范数正则化（regularization）。正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。
L2范数正则化在模型原损失函数基础上添加L2范数惩罚项，从而得到训练所需要最小化的函数。L2范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积。
L2范数正则化令权重先自乘小于1的数，再减去不含惩罚项的梯度。因此，L2范数正则化又叫权重衰减。
权重衰减通过惩罚绝对值较大的模型参数为需要学习的模型增加了限制，这可能对过拟合有效。实际场景中，我们有时也在惩罚项中添加偏差元素的平方和。
可以直接在构造优化器实例时通过weight_decay参数来指定权重衰减超参数。默认下，PyTorch会对权重和偏差同时衰减。我们可以分别对权重和偏差构造优化器实例，从而只对权重衰减。

## 丢弃法
当对某隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。
丢弃概率是丢弃法的超参数。
丢弃法不改变其输入的期望值。
在测试模型时，我们为了拿到更加确定性的结果，一般不使用丢弃法。
在PyTorch中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时（即model.eval()后），Dropout层并不发挥作用。

## 正向传播、反向传播和计算图
正向传播是指对神经网络沿着从输入层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）。
我们通常绘制计算图来可视化运算符和变量在计算中的依赖关系，其中方框代表变量，圆圈代表运算符，箭头表示从输入到输出之间的依赖关系。
反向传播指的是计算神经网络参数梯度的方法。总的来说，反向传播依据微积分中的链式法则，沿着从输出层到输入层的顺序，依次计算并存储目标函数有关神经网络各层的中间变量以及参数的梯度。

在训练深度学习模型时，正向传播和反向传播之间相互依赖：
一方面，正向传播的计算可能依赖于模型参数的当前值，而这些模型参数是在反向传播的梯度计算后通过优化算法迭代的；
另一方面，反向传播的梯度计算可能依赖于各变量的当前值，而这些变量的当前值是通过正向传播计算得到的。

因此，在模型参数初始化完成后，我们交替地进行正向传播和反向传播，并根据反向传播计算的梯度迭代模型参数。既然我们在反向传播中使用了正向传播中计算得到的中间变量来避免重复计算，那么这个复用也导致正向传播结束后不能立即释放中间变量内存。这也是训练要比预测占用更多内存的一个重要原因。另外需要指出的是，这些中间变量的个数大体上与网络层数线性相关，每个变量的大小跟批量大小和输入个数也是线性相关的，它们是导致较深的神经网络使用较大批量训练时更容易超内存的主要原因。

## 数值稳定性和模型初始化
深度模型有关数值稳定性的典型问题是衰减（vanishing）和爆炸（explosion）。
在神经网络中，通常需要随机初始化模型参数。
随机初始化模型参数的方法有很多。之前的实现中，我们使用torch.nn.init.normal_()使模型net的权重参数采用正态分布的随机初始化方式。不过，PyTorch中nn.Module的模块参数都采取了较为合理的初始化策略（不同类型的layer具体采样的哪一种初始化方法的可参考源代码），因此一般不用我们考虑。

还有一种比较常用的随机初始化方法叫作Xavier随机初始化，它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

# 深度学习计算
## 模型构造
有多种模型构造的方法：
### 继承Module类来构造模型
Module类是nn模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。
### Module的子类
Module类是一个通用的部件。事实上，PyTorch还实现了继承自Module的可以方便构建模型的类: 如Sequential、ModuleList和ModuleDict等等。不过虽然Sequential等类可以使模型构造更加简单，但直接继承Module类可以极大地拓展模型构造的灵活性。
（1）Sequential类：当模型的前向计算为简单串联各个层的计算时，Sequential类可以通过更加简单的方式定义模型。这正是Sequential类的目的：它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加Module的实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。
（2）ModuleList类：ModuleList接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作。
既然Sequential和ModuleList都可以进行列表化构造网络，那二者区别是什么呢。ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现；而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。
ModuleList的出现只是让网络定义前向传播时更加灵活；另外，ModuleList不同于一般的Python的list，加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中。
（3）ModuleDict类：ModuleDict接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作；和ModuleList一样，ModuleDict实例仅仅是存放了一些模块的字典，并没有定义forward函数需要自己定义。同样，ModuleDict也与Python的Dict有所不同，ModuleDict里的所有模块的参数会被自动添加到整个网络中。

## 模型参数的访问、初始化和共享
### 访问模型参数
```python
import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
) # Pytorch 已进行默认初始化

print(net)

X = torch.rand(2, 4) # 构造数据集的输入
Y = net(X).sum() # 根据默认初始化的参数计算输出

# 可以通过Module类的parameters()或者named_parameters方法来访问所有参数（以迭代器的形式返回）
# 后者除了返回参数Tensor外还会返回其名字
print(type(net.named_parameters()))

for name, param in net.named_parameters():
    print(name, param.size()) # 返回的名字自动加上了层数的索引作为前缀

# 对于使用Sequential类构造的神经网络，可以通过方括号[]来访问网络的任一层。
# 索引0表示隐藏层为Sequential实例最先添加的层。
# 返回的param的类型为torch.nn.parameter.Parameter，其实这是Tensor的子类
# 和Tensor不同的是如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param)) # 这里是单层的参数，所以名字中没有层数索引的前缀

weight_0 = list(net[0].parameters())[0] # 要使用list将这个迭代器转化一下
# 因为Parameter是Tensor，即Tensor拥有的属性它都有，比如可以根据data来访问参数数值，用grad来访问参数梯度。
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)
```

### 初始化模型参数
PyTorch中nn.Module的模块参数采取了默认的较为合理的初始化策略（不同类型的layer具体采样的哪一种初始化方法的可参考源代码）。但我们经常需要使用其他方法来初始化权重。PyTorch的init模块里提供了多种预设的初始化方法。
```python
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01) # 用正态分布初始化权重
        print(name, param.data)

    if 'bias' in name:
        init.constant_(param, val=0) # 将偏差置0
        print(name, param.data)
```

### 自定义初始化方法
有时候我们需要的初始化方法并没有在init模块中提供。这时，可以实现一个初始化方法，从而能够像使用其他初始化方法那样使用它。在这之前我们先来看看PyTorch是怎么实现这些初始化方法的，例如torch.nn.init.normal_：
```python
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)
```
可以看到这就是一个inplace改变Tensor值的函数，而且这个过程是不记录梯度的。 类似的我们来实现一个自定义的初始化方法。
```python
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)
```
还可以通过改变这些参数的data来改写模型参数值同时不会影响梯度：
```python
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
```

### 共享模型参数
在有些情况下，我们希望在多个层之间共享模型参数。共享模型参数的方式有: Module类的forward函数里多次调用同一个层。此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的。
不过注意，因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的。

## 自定义层
虽然PyTorch提供了大量常用的层，但有时候我们依然希望自定义层。本节将介绍如何使用Module来自定义层，从而可以被重复调用。

### 不含模型参数的自定义层
```python
import torch
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    # 将计算放在了forward函数里，所以，该自定义层没有模型参数
    def forward(self, x):
        return x - x.mean()

# 可以实例化这个层，然后做前向计算
layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))

# 也可以用它来构造更复杂的模型
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
y.mean().item()
```

### 含模型参数的自定义层
我们还可以自定义含模型参数的自定义层。其中的模型参数可以通过训练学出。
之前介绍了Parameter类其实是Tensor的子类，如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里。所以在自定义含模型参数的层时，我们应该将参数定义成Parameter，除了像之前那样直接定义成Parameter类外，还可以使用ParameterList和ParameterDict分别定义参数的列表和字典。
```python
# ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数
# 另外也可以使用append和extend在列表后面新增参数。
class MyDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyDense()
print(net)
# 而ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用了。
# 例如使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等等
class MyDictDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })

        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

# 这样就可以根据传入的键值来进行不同的前向传播
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))
# 可以使用自定义层构造模型。它和PyTorch的其他层在使用上很类似
net = nn.Sequential(
    MyDictDense(),
    MyDense()
)
print(net)
print(net(x))
```

## 读取和存储

在实际中，我们有时需要把训练好的模型部署到很多不同的设备。在这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用。
### 读写Tensor
我们可以直接使用save函数和load函数分别存储和读取Tensor。save使用Python的pickle实用程序将对象进行序列化，然后将序列化的对象保存到disk，使用save可以保存各种对象,包括模型、张量和字典等。而load使用pickle unpickle工具将pickle的对象文件反序列化为内存。
```python
import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt') # 保存到文件
x2 = torch.load('x.pt') # 读取文件数据到内存
y = torch.zeros(4)
torch.save([x, y], 'xy.pt') # 存储一个Tensor列表

xy_list = torch.load('xy.pt') # 读取该Tensor列表
torch.save({'x':x, 'y':y}, 'xy_dict.pt') # 存储一个从字符串映射到Tensor的字典

xy_dict = torch.load('xy_dict.pt')
xy_dict
```

### 读写模型
在PyTorch中，Module的可学习参数，即权重和偏差，模块模型包含在参数中(通过model.parameters()访问)。state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.relu(self.hidden(X))
        return self.output(a)

net = MLP()
net.state_dict() # relu层没有可学习的参数
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer.state_dict()
```

PyTorch中保存和加载训练模型有两种常见的方法:
（1） 仅保存和加载模型参数(state_dict)，这是推荐方式，形式为：
```python
torch.save(model.state_dict(), PATH) # 保存，推荐的文件后缀名是pt或pth
model = TheModelClass(*args, **kwargs) # 加载
model.load_state_dict(torch.load(PATH))
```

例子：
```python
X = torch.randn(2, 3)
Y = net(X)
PATH = './net.pt'
torch.save(net.state_dict(), PATH) # 存储模型参数

net2 = MLP() # net2和net一样都是MLP()类，所以模型参数相同
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
Y2 == Y # 两者计算结果相同
```
（2）保存和加载整个模型，形式为：
```python
torch.save(model, PATH) # 保存
model = torch.load(PATH) # 加载
```

## GPU计算
对复杂的神经网络和大规模的数据来说，使用CPU来计算可能不够高效。在本节中，将介绍如何使用单块NVIDIA GPU来计算。需要确保已经安装好了PyTorch GPU版本。
```python
!nvidia-smi # 查看显卡信息
```

### 计算设备
PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。
```python
import torch
torch.cuda.is_available() # 查看GPU是否可用
torch.cuda.device_count() # 查看GPU数目
torch.cuda.current_device() # 查看当前GPU索引号，从0开始
torch.cuda.get_device_name(0) # 根据索引号查看GPU名称
```

### Tensor的GPU计算
默认情况下，Tensor会被存在内存上。因此，之前我们每次打印Tensor的时候看不到GPU相关标识。
使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。如果有多块GPU，我们用.cuda(i)来表示第 iii 块GPU及相应的显存（iii从0开始）且cuda(0)和cuda()等价。
```python
x = torch.tensor([1, 2, 3])
x = x.cuda(0)
print(x)
print(x.device) # 可以通过Tensor的device属性来查看该Tensor所在的设备

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device) # 可以在创建时就指定设备
x = torch.tensor([1, 2, 3]).to(device) # 第二种方法
y = x**2 # 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上
print(y)
```
需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。

### 模型的GPU计算
同Tensor类似，PyTorch模型也可以通过.cuda转换到GPU上。我们可以通过检查模型的参数的device属性来查看存放模型的设备。
同时，需要保证模型输入的Tensor和模型都在同一设备上，否则会报错。
```python
from torch import nn

net = nn.Linear(3, 1)
list(net.parameters())[0].device
net.cuda() # 将模型转换到CPU上
list(net.parameters())[0].device
x = torch.randn(2, 3).cuda()
net(x)
```

# 卷积神经网络
卷积神经网络中涉及到输入和输出图像的形状的转换，这里将后面模型构造时用到的一段通用测试代码摘出来，供以后参考：
```python
net = vgg(conv_arch, fc_features, fc_hidden_units)
# 构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
X = torch.rand(1, 1, 224, 224)
# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
for name, blk in net.named_children(): 
    X = blk(X)
    print(name, 'output shape: ', X.shape)
```

## 二维卷积层
卷积神经网络（convolutional neural network）是含有卷积层（convolutional layer）的神经网络。
虽然卷积层得名于卷积（convolution）运算，但我们通常在卷积层中使用更加直观的互相关（cross-correlation）运算。
二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。即，可以通过数据来学习卷积核。
实际上，卷积运算与互相关运算类似。为了得到卷积运算的输出，我们只需将核数组左右翻转并上下翻转，再与输入数组做互相关运算。可见，卷积运算和互相关运算虽然类似，但如果它们使用相同的核数组，对于同一个输入，输出往往并不相同。
那么，你也许会好奇卷积层为何能使用互相关运算替代卷积运算。其实，在深度学习中核数组都是学出来的：卷积层无论使用互相关运算或卷积运算都不影响模型预测时的输出。
为了与大多数深度学习文献一致，如无特别说明，本书中提到的卷积运算均指互相关运算。
二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征，也叫特征图（feature map）。影响元素x的前向计算的所有可能输入区域（可能大于输入的实际尺寸）叫做X的感受野（receptive field）。
我们可以通过更深的卷积神经网络使特征图中单个元素的感受野变得更加广阔，从而捕捉输入上更大尺寸的特征。
比如，输入层是一个3乘3的图像，记为X，经过一个2乘2的卷积核，得到的输出层是一个2乘2的图像，记为Y。那么Y中每个元素的感受野是X中的2乘2的范围大小，即这个元素仅与X中的这四个元素相关。此时考虑一个更深的卷积网络：将Y与另一个形状为2乘2的卷积核做互相关运算，输出单个元素z，那么，z在Y上的感受野包括Y的全部四个元素，则在输入X上的感受野包括其中全部的9个元素（X的这9个元素是由Y的四个元素所感受的）

## 填充和步幅
卷积层的输出形状由输入形状和卷积核窗口形状决定。卷积层的两个超参数，即填充和步幅，它们可以对给定形状的输入和卷积核改变输出形状。
填充（padding）是指在输入高和宽的两侧填充元素（通常是0元素）。
卷积窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动。我们将每次滑动的行数和列数称为步幅（stride）。

## 多输入通道和多输出通道
（1）多输入通道
当输入数据含多个通道时，我们需要构造一个输入通道数与输入数据的通道数相同的卷积核，从而能够与含多通道的输入数据做互相关运算。
即，卷积核的通道数由输入数据的通道数所决定。
（2）多输出通道
当输入通道有多个时，因为我们对各个通道的结果做了累加，所以不论输入通道数是多少，输出通道数总是为1。
如果希望得到含多个通道的输出，我们可以为每个输出通道分别创建形状为c乘以h乘以w的卷积核，将它们在输出通道上进行连结。
即，输出数据的通道数由卷积核的个数所决定。
（3）1乘1卷积层
1乘1卷积层通常用来调整网络层之间的通道数（可以类比于全连接层的隐藏神经元个数），并控制模型复杂度。
因为使用了最小窗口，1乘1卷积失去了卷积层可以识别高和宽维度上相邻元素构成的模式的功能。实际上，1乘1卷积的主要计算发生在通道维上。值得注意的是，输入和输出具有相同的高和宽。输出中的每个元素来自输入中在高和宽上相同位置的元素在不同通道之间的按权重累加。
假设将通道维当作特征维，将高和宽维度上的元素当成数据样本，那么1乘1卷积层的作用与全连接层等价。但它又相比于全连接层有一个优点：它仍然保留了输入图像的空间信息，即不是一个很长的向量，从而使空间信息能够自然传递到后面的层中去。

## 池化层
池化（pooling）层的提出是为了缓解卷积层对位置的过度敏感性。
不同于卷积层里计算输入和核的互相关性，池化层直接计算池化窗口内元素的最大值或者平均值。该运算也分别叫做最大池化或平均池化。
同卷积层一样，池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状。池化层填充和步幅与卷积层填充和步幅的工作机制一样。
默认情况下，PyTorch里的池化层的步幅和池化窗口形状相同。
当然，我们也可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅。
在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。

## 卷积神经网络（LeNet）
在之前对Fashion-MNIST数据集分类时，使用的方法是对图像中的像素全部展开得到一个很长的向量，然后输入进全连接层中。
然而，这种分类方法有一定的局限性。
（1）图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别。
（2）对于大尺寸的输入图像，使用全连接层容易造成模型过大。这带来过复杂的模型和过高的存储开销。
卷积层尝试解决这两个问题。一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；另一方面，卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

### 定义模型
LeNet交替使用卷积层和最大池化层后接全连接层来进行图像分类。
```python
import torch
from torch import nn, optim
# 卷积操作的维度变化公式为:
# Height_out = (Height_in - Height_kernal + 2*padding) / stride +1 
# LeNet当时的输入图片是单通道的32*32像素的灰度图
# 但现在的Fashion-MNIST是28*28

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # (输入通道，输出通道，卷积核尺寸)，所以输出尺寸为(28-5)+1=24，通道为6
            nn.Sigmoid(), # Sigmoid激活
            nn.MaxPool2d(2, 2), # (卷积核尺寸，步幅)，所以输出尺寸为(24-2)/2+1=12，通道仍然为6
            nn.Conv2d(6, 16, 5), # 所以输出尺寸为(12-5)+1=8，通道为16
            nn.Sigmoid(), # Sigmoid激活
            nn.MaxPool2d(2, 2) # 所以输出尺寸为(8-2)/2+1=4，通道仍然为16
        )

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), # 输入为16*4*4，输出为120个神经元
            nn.Sigmoid(), # Sigmoid激活
            nn.Linear(120, 84), # 又一个全连接层
            nn.Sigmoid(), # Sigmoid激活
            nn.Linear(84, 10) # 输出层，类别为10
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1)) # view一下形状，第一维为batch_size，剩下的就是图像转换成的向量
        return output

net = LeNet()
print(net)
```

### 获取数据和训练模型
（1）还是使用之前的数据下载和加载方式：
```python
import torchvision
import torchvision.transforms as transforms

mnist_train = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transforms.ToTensor())

batch_size = 256
num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle=False, num_workers=num_workers)
```
（2）修改分类准确度计算代码，使其支持GPU计算：
```python
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device，则使用net的device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式，这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n
```
（3）修改训练过程，使其支持GPU计算：
```python
import time

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on', device)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' 
              % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc, time.time()-start))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```

## 深度卷积神经网络（AlexNet）
我们在上一节看到，神经网络可以直接基于图像的原始像素进行分类。这种称为端到端（end-to-end）的方法节省了很多中间步骤。然而，在很长一段时间里更流行的是研究者通过勤劳与智慧所设计并生成的手工特征。这类图像分类研究的主要流程是：
获取图像数据集；
使用已有的特征提取函数生成图像的特征；
使用机器学习模型对图像的特征分类。
当时认为的机器学习部分仅限最后这一步。如果那时候跟机器学习研究者交谈，他们会认为机器学习既重要又优美。优雅的定理证明了许多分类器的性质。机器学习领域生机勃勃、严谨而且极其有用。然而，如果跟计算机视觉研究者交谈，则是另外一幅景象。他们会告诉你图像识别里“不可告人”的现实是：计算机视觉流程中真正重要的是数据和特征。也就是说，使用较干净的数据集和较有效的特征甚至比机器学习模型的选择对图像分类结果的影响更大。

 
### 学习特征表示
既然特征如此重要，它该如何表示呢？
我们已经提到，在相当长的时间里，特征都是基于各式各样手工设计的函数从数据中提取的。事实上，不少研究者通过提出新的特征提取函数不断改进图像分类结果。这一度为计算机视觉的发展做出了重要贡献。

然而，另一些研究者则持异议。他们认为特征本身也应该由学习得来。他们还相信，为了表征足够复杂的输入，特征本身应该分级表示。持这一想法的研究者相信，多层神经网络可能可以学得数据的多级表征，并逐级表示越来越抽象的概念或模式。以图像分类为例，以物体边缘检测为例。在多层神经网络中，图像的第一级的表示可以是在特定的位置和⻆度是否出现边缘；而第二级的表示说不定能够将这些边缘组合出有趣的模式，如花纹；在第三级的表示中，也许上一级的花纹能进一步汇合成对应物体特定部位的模式。这样逐级表示下去，最终，模型能够较容易根据最后一级的表示完成分类任务。需要强调的是，输入的逐级表示由多层模型中的参数决定，而这些参数都是学出来的。

### AlexNet
2012年，AlexNet横空出世。AlexNet使用了8层卷积神经网络，并以很大的优势赢得了ImageNet 2012图像识别挑战赛。它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。
AlexNet跟LeNet结构类似，但使用了更多的卷积层和更大的参数空间来拟合大规模数据集ImageNet。它是浅层神经网络和深度神经网络的分界线。
两者具体对比如下：
第一，与相对较小的LeNet相比，AlexNet包含8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
AlexNet第一层中的卷积窗口形状是11×11。因为ImageNet中绝大多数图像的高和宽均比MNIST图像的高和宽大10倍以上，ImageNet图像的物体占用更多的像素，所以需要更大的卷积窗口来捕获物体。第二层中的卷积窗口形状减小到5×5，之后全采用3×3。此外，第一、第二和第五个卷积层之后都使用了窗口形状为3×3、步幅为2的最大池化层。而且，AlexNet使用的卷积通道数也大于LeNet中的卷积通道数数十倍。
紧接着最后一个卷积层的是两个输出个数为4096的全连接层。这两个巨大的全连接层带来将近1 GB的模型参数。由于早期显存的限制，最早的AlexNet使用双数据流的设计使一个GPU只需要处理一半模型。幸运的是，显存在过去几年得到了长足的发展，因此通常我们不再需要这样的特别设计了。
第二，AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数。一方面，ReLU激活函数的计算更简单，例如它并没有sigmoid激活函数中的求幂运算。另一方面，ReLU激活函数在不同的参数初始化方法下使模型更容易训练。这是由于当sigmoid激活函数输出极接近0或1时，这些区域的梯度几乎为0，从而造成反向传播无法继续更新部分模型参数；而ReLU激活函数在正区间的梯度恒为1。因此，若模型参数初始化不当，sigmoid函数可能在正区间得到几乎为0的梯度，从而令模型无法得到有效训练。
第三，AlexNet通过丢弃法来控制全连接层的模型复杂度。而LeNet并没有使用丢弃法。
第四，AlexNet引入了大量的图像增广，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。

虽然看上去AlexNet的实现比LeNet的实现也就多了几行代码而已，但这个观念上的转变和真正优秀实验结果的产生令学术界付出了很多年。
（1）定义模型：
```python
import torch
from torch import nn, optim

# AlexNet所使用的数据集是ImageNet
# 这里使用的输入图像是单通道的尺寸为224*224
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 使用较大的11 x 11窗口来捕获物体。同时使用步幅4来较大幅度减小输出高和宽
            # 这里使用的输出通道数比LeNet中的也要大很多
            nn.Conv2d(1, 96, 11, 4), # (输入通道，输出通道，卷积核尺寸，步幅，填充)，所以输出尺寸为(224-11)/4+1=54 这里不能整除，所以向下取整
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # 所以输出尺寸为(54-3)/2+1=26，通道数仍为96，池化层是默认向下取整，可以改变ceil_mode参数来改成向上取整
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，这是因为(x-5+2*2)/1+1=x
            # 且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2), # 输出为(26-5+2*2)/1+1=26
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # (26-3)/2+1 = 12
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1), # 输出为(12-3+2*1)/1+1=12
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1), # 12
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), # 12
            nn.ReLU(),
            nn.MaxPool2d(3, 2) # (12-3)/2+1=5
        )

        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = AlexNet()
print(net)
```

（2）读取数据
虽然论文中AlexNet使用ImageNet数据集，但因为ImageNet数据集训练时间较长，我们仍用前面的Fashion-MNIST数据集来演示AlexNet。读取数据的时候我们额外做了一步将图像高和宽扩大到AlexNet使用的图像高和宽224。这个可以通过torchvision.transforms.Resize实例来实现。也就是说，我们在ToTensor实例前使用Resize实例，然后使用Compose实例来将这两个变换串联以方便调用。
```python
# 将读取数据的步骤封装成一个函数，方便调用
def load_data_fashion_mnist(batch_size, resize=None, root='.'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))

    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
```

（3）训练
```python
# 分类准确度测量函数
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device，则使用net的device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式，这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n

# 将训练过程封装起来，方便调用
import time

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' 
              % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc, time.time()-start))

# 开始训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
# 相对于LeNet，由于图片尺寸变大了而且模型变大了，所以需要更大的显存，也需要更长的训练时间了。
```

## 使用重复元素的网络（VGG）

AlexNet在LeNet的基础上增加了3个卷积层。但AlexNet作者对它们的卷积窗口、输出通道数和构造顺序均做了大量的调整。虽然AlexNet指明了深度卷积神经网络可以取得出色的结果，但并没有提供简单的规则以指导后来的研究者如何设计新的网络。

接下来会介绍几种不同的深度网络设计思路。

本节介绍VGG，它的名字来源于论文作者所在的实验室Visual Geometry Group。VGG提出了可以通过重复使用简单的基础块来构建深度模型的思路。

### VGG块
VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为3乘3的卷积层后接上一个步幅为2、窗口形状为2乘2的最大池化层。卷积层保持输入的高和宽不变（因为(h-3+2x1）+1=h），而池化层则对其减半（因为(h-2)/2+1=h/2）。
对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核优于采用大的卷积核，因为可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。例如，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。
```python
import time
import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)
```

### VGG网络
```python
# 这个自定义层是将(n, c, h, w)拉伸成(n, c*h*w)，即将图像转成向量
class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

# 整个vgg网络前面是若干个卷积块，后面是3个全连接层
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 整个网络的卷积层部分
    # conv_arch参数包含了上面的vgg块的三个参数：块的数目、输入通道、输出通道

    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i+1), vgg_block(num_convs, in_channels, out_channels))

    # 整个网络的全连接层部分
    net.add_module("fc", nn.Sequential(
        FlattenLayer(),
        # fc_features就是前面经过卷积操作后图像的尺寸c*h*w, fc_hidden_units是隐藏层的神经元个数
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net
# 有5个卷积块，前2块是单卷积层，后3块是双卷积层，即连续做两次卷积
# 经过5个vgg_block，宽和高会减半5次，变成224/(2^5)=224/32=7
# 同时，通道数也在翻倍，起始是1个通道，之后不断翻倍，直到512个通道
# 因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11。 
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))

# 经过上面的卷积操作后，输出通道为512，图像尺寸为7，所以输入到下面的全连接层的向量就是512*7*7
fc_features = 512*7*7
# 全连接层的隐藏层的神经元个数，这个可以任意设定
fc_hidden_units = 4096
# 构造网络
net = vgg(conv_arch, fc_features, fc_hidden_units)
```
模型加载和训练过程与上一节的AlexNet相同。

## 网络中的网络（NiN）
前几节介绍的LeNet、AlexNet和VGG在设计上的共同之处是：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。其中，AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽（增加通道数）和加深。
网络中的网络（NiN）则提出了另外一个思路，即重复使用由卷积层和代替全连接层的1乘1卷积层构成的NiN块来构建深层网络。
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。
NiN的以上设计思想影响了后面一系列卷积神经网络的设计。

加粗！！：NiN因为使用了全局平均池化层，从而使得每个通道的宽和高都为1，这样就使得输出与输入图片的尺寸无关，极大地提高了模型的灵活性，而之前的LeNet、AlexNet和VGG必须要给定特定尺寸的图片才能运行，否则会与全连接层的输入不匹配。

### NiN块
卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维。而1乘1卷积层可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征。因此，NiN使用1乘1卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。
```python
# NiN块是NiN中的基础块。
# 它由一个卷积层加两个充当全连接层的1×11×1卷积层串联而成。
# 其中第一个卷积层的超参数可以自行设置，而第二和第三个卷积层的超参数一般是固定的。
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    return blk
```

### NiN网络
```python
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    # 这样就可以将输入的图像平均池化成一个1*1大小的元素
    # 再配合上下面的FlattenLayer，就起到了最后全连接输出的效果
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size = x.size()[2:])

# 这个自定义层是将(n, c, h, w)拉伸成(n, c*h*w)，即将图像转成向量
class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    # NiN是在AlexNet问世不久后提出的。它们的卷积层设定有类似之处。
    # NiN使用卷积窗口形状分别为11×1111×11、5×55×5和3×33×3的卷积层，相应的输出通道数也与AlexNet中的一致。
    nin_block(1, 96, kernel_size=11, stride=4, padding=0), # 输出为(224-11)/4+1=54
    # 每个NiN块后接一个步幅为2、窗口形状为3×33×3的最大池化层
    nn.MaxPool2d(kernel_size=3, stride=2), # 输出为(54-3)/2+1=26 
    nin_block(96, 256, kernel_size=5, stride=1, padding=2), # 输出为(26-5+2*2)/1+1=26
    nn.MaxPool2d(kernel_size=3, stride=2), # 输出为(26-3)/2+1=12
    nin_block(256, 384, kernel_size=3, stride=1, padding=1), # (12-3+2*1)/1+1 = 12
    nn.MaxPool2d(kernel_size=3, stride=2), # (12-3)/2+1=5
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1), # (5-3+2*1)/1+1=5
    GlobalAvgPool2d(), # (5-5)/1+1=1
    FlattenLayer()
)
print(net)
```

## 含并行连结的网络（GoogLeNet）
在2014年的ImageNet图像识别挑战赛中，一个名叫GoogLeNet的网络结构大放异彩。它虽然在名字上向LeNet致敬，但在网络结构上已经很难看到LeNet的影子。GoogLeNet吸收了NiN中网络串联网络的思想，并在此基础上做了很大改进。
### Inception块
GoogLeNet中的基础卷积块叫作Inception块，得名于同名电影《盗梦空间》（Inception）。其结构如图所示：
![image](https://user-images.githubusercontent.com/6218739/75971838-4e3c2700-5f0d-11ea-8d34-cbe8e187f07b.png)

可以看出，Inception块里有4条并行的线路。前3条线路使用窗口大小分别是1乘1、3乘3和5乘5的卷积层来抽取不同空间尺寸下的信息，其中中间2个线路会对输入先做1乘1卷积来减少输入通道数，以降低模型复杂度。第四条线路则使用3乘3最大池化层，后接1乘1卷积层来改变通道数。4条线路都使用了合适的填充来使输入与输出的高和宽一致。最后我们将每条线路的输出在通道维上连结，并输入接下来的层中去。
其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。
```python
# Inception块中可以自定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度。
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    # c2和c3的内部因为都有两个层，所以输出通道也要有两个，来分别设定
    # c4的最大池化层不需要通道设定，所以c4的通道也只有一个即可
    # 假设输入图像的尺寸为h，经过下面的计算可得，输出图像的尺寸仍为h
    def __init__(self, in_c, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1) # (h-1)+1=h
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1) # (h-1)+1=h
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) # (h-3+2*1)+1=h
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1) # (h-1)+1=h
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2) # (h-5+2*2)+1=h
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # (h-3+2*1)/1+1=h
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1) # (h-1)+1=h

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x)))) # 注意这里输入的也是x
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x)))) # 注意这里输入的也是x
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1) # 在通道维上进行连结
```

### GoogLeNet
```python
# GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），每个模块之间使用步幅为2的3×3最大池化层来减小输出高宽。
# 图片尺寸就以文中给出的96为例
# 第一模块使用一个64通道的7×77×7卷积层，输入通道为1，输出通道为64
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), # (96-7+2*3)/2+1=48
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (48-3+2*1)/2+1=24
)
# 第二模块使用2个卷积层和1个池化层，输入通道为64，输出通道为192
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1), # (24-1)+1=24
    nn.Conv2d(64, 192, kernel_size=3, padding=1), #(24-3+2*1)+1=24
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (24-3+2*1)/2+1=12
)
# 第三模块串联2个完整的Inception块
b3 = nn.Sequential(
    # 第1个Inception块的输入通道为192，输出通道为64+128+32+32=256
    Inception(192, 64, (96, 128), (16, 32), 32), # 12
    # 第1个Inception块的输入通道为256，输出通道为128+192+96+64=480
    Inception(256, 128, (128, 192), (32, 96), 64), # 12
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (12-3+2*1)/2+1=6
)
# 第四模块串联5个Inception块
b4 = nn.Sequential(
    # 第1个Inception块的输入通道为480， 输出通道为192+208+48+64=512
    Inception(480, 192, (96, 208), (16, 48), 64), # 6
    # 输入512， 输出160+224+64+64=512
    Inception(512, 160, (112, 224), (24, 64), 64), # 6
    # 输入512， 输出128+256+64+64=512
    Inception(512, 128, (128, 256), (24, 64), 64), # 6
    # 输入512， 输出112+288+64+64=528
    Inception(512, 112, (144, 288), (32, 64), 64), # 6
    # 输入528， 输出256+320+128+128=832
    Inception(528, 256, (160, 320), (32, 128), 128), # 6
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (6-3+2*1)/2+1=3 
)
# 第五模块串联2个Inception块
b5 = nn.Sequential(
    # 输入832， 输出256+320+128+128=832
    Inception(832, 256, (160, 320), (32, 128), 128), # 3
    # 输入832， 输出384+384+128+128=1024
    Inception(832, 384, (192, 384), (48, 128), 128), # 3
    # 这一步非常重要，使用全局平均池化层来将每个通道的高和宽变成1，这样就与输入图像的尺寸无关
    GlobalAvgPool2d() # 1
)
net = nn.Sequential(
    b1, b2, b3, b4, b5,
    FlattenLayer(), # (N, 1024, 1, 1)转化成向量(N, 1024)
    nn.Linear(1024, 10) # 全连接层输出类别，这里的1024是之前的通道数，与图像尺寸无关
)
```
加载数据时，这里的尺寸改成了96，也可以继续用之前的224，GoogLeNet因为使用了全局平均池化，所以对输入图片尺寸不敏感：
```python
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
```
## 批量归一化
在之前的例子里，我们对输入数据做了标准化处理：处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。标准化处理输入数据使各个特征的分布相近：这往往更容易训练出有效的模型。
通常来说，数据标准化预处理对于浅层模型就足够有效了。随着模型训练的进行，当每层中参数更新时，靠近输出层的输出较难出现剧烈变化。但对深层神经网络来说，即使输入数据已做标准化，训练中模型参数的更新依然很容易造成靠近输出层输出的剧烈变化。这种计算数值的不稳定性通常令我们难以训练出有效的深度模型。
批量归一化的提出正是为了应对深度模型训练的挑战。在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。
批量归一化和下一节将要介绍的残差网络为训练和设计深度模型提供了两类重要思路。
对全连接层和卷积层做批量归一化的方法稍有不同。
（1）对全连接层做批量归一化：将批量归一化层置于全连接层中的仿射变换和激活函数之间，批量归一化层引入了两个可以学习的模型参数，拉伸（scale）参数$\gamma$和偏移（shift）参数$\beta$。
（2）对卷积层做批量归一化：对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且每个通道都拥有独立的拉伸和偏移参数，并均为标量。
（3）预测时的批量归一化：使用批量归一化训练时，我们可以将批量大小设得大一点，从而使批量内样本的均值和方差的计算都较为准确。将训练好的模型用于预测时，我们希望模型对于任意输入都有确定的输出。因此，单个样本的输出不应取决于批量归一化所需要的随机小批量中的均值和方差。一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。可见，和丢弃层一样，批量归一化层在训练模式和预测模式下的计算结果也是不一样的。

Pytorch中nn模块定义的BatchNorm1d和BatchNorm2d类使用起来非常简单，二者分别用于全连接层和卷积层，都需要指定输入的num_features参数值，对于全连接层来说该值应为输出个数，对于卷积层来说则为输出通道数。

## 残差网络
让我们先思考一个问题：对神经网络模型添加新的层，充分训练后的模型是否只可能更有效地降低训练误差？理论上，原模型解的空间只是新模型解的空间的子空间。也就是说，如果我们能将新添加的层训练成恒等映射f(x)=x，新模型和原模型将同样有效。由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。然而在实践中，添加过多的层后训练误差往往不降反升。即使利用批量归一化带来的数值稳定性使训练深层模型更加容易，该问题仍然存在。针对这一问题，何恺明等人提出了残差网络（ResNet）。它在2015年的ImageNet图像识别挑战赛夺魁，并深刻影响了后来的深度神经网络的设计。
残差块通过跨层的数据通道从而能够训练出有效的深度神经网络。

### 残差块
在残差块中，输入可通过跨层的数据线路更快地向前传播。
```python
# 残差块可以设定输出通道数、是否使用额外的1×11×1卷积层来修改通道数以及卷积层的步幅，即可以改变输出大小。
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        # ResNet沿用了VGG全3×33×3卷积层的设计。
        # 残差块里首先有2个有相同输出通道数的3×33×3卷积层。
        # 每个卷积层后接一个批量归一化层和ReLU激活函数。
        # 然后将输入跳过这两个卷积运算后直接加在最后的ReLU激活函数前。
        # 这样的设计要求两个卷积层的输出与输入形状一样，从而可以相加。如果想改变通道数，就需要引入一个额外的1×1卷积层来将输入变换成需要的形状后再做相加运算。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
```
### ResNet模型
```python
# ResNet的前两层跟GoogLeNet中的一样：在输出通道数为64、步幅为2的7×7卷积层后接步幅为2的3×3的最大池化层。
# 不同之处在于ResNet每个卷积层后增加的批量归一化层。
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# GoogLeNet在后面接了4个由Inception块组成的模块。ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        # 第一个模块的通道数同输入通道数一致
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 后面的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            # 在第一个模块中，由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
# 为ResNet加入所有残差块。这里每个模块使用两个残差块。
net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
net.add_module('resnet_block2', resnet_block(64, 128, 2))
net.add_module('resnet_block3', resnet_block(128, 256, 2))
net.add_module('resnet_block4', resnet_block(256, 512, 2))
# 最后，与GoogLeNet一样，加入全局平均池化层后接上全连接层输出。
net.add_module('global_avg_pool', GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module('fc', nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))
# 这里每个模块里有4个卷积层（不计算1×11×1卷积层），加上最开始的卷积层和最后的全连接层，共计18层。这个模型通常也被称为ResNet-18。
# 通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。
# 虽然ResNet的主体架构跟GoogLeNet的类似，但ResNet结构更简单，修改也更方便。这些因素都导致了ResNet迅速被广泛使用。
```

拿个测试数据跑一下：
```python
X = torch.rand(1, 1, 224, 224)
for name, layer in net.named_children():
    X = layer(X)
    print(name, 'output shape = ', X.shape)
```

最后，加载数据和训练模型都跟之前的一样。
 
## 稠密连接网络（DenseNet）
ResNet中的跨层连接设计引申出了数个后续工作，比如这里的稠密连接网络（DenseNet）。
假设将部分前后相邻的运算抽象为模块A和模块B。与ResNet的主要区别在于，DenseNet里模块B的输出不是像ResNet那样和模块A的输出相加，而是在通道维上连结。这样模块A的输出可以直接传入模块B后面的层。在这个设计里，模块A直接跟模块B后面的所有层连接在了一起。这也是它被称为“稠密连接”的原因。
DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。
### 稠密块
```python
# DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构，首先在conv_block函数里实现这个结构。
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk
# 稠密块由多个conv_block组成，每块使用相同的输出通道数。
# 卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）
# 比如，定义一个有2个输出通道数为10的卷积块。使用通道数为3的输入时，我们会得到通道数为3+2×10=23的输出，增长率就是10
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i* out_channels
            net.append(conv_block(in_c, out_channels))

        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 在通道维上将输入和输出连结
        return X
```

### 过渡层
```python
# 由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。
# 过渡层用来控制模型复杂度。它通过1×11×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk
```

### DenseNet模型
```python
# DenseNet首先使用同ResNet一样的单卷积层和最大池化层。
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块。
# 同ResNet一样，可以设置每个稠密块使用多少个卷积层。这里我们设成4，从而与上一节的ResNet-18保持一致。
# 稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。
num_channels, growth_rate = 64, 32 # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4] # length为4，表明有4个稠密块，每个元素都是4，表明每个稠密块有4个卷积层

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module('DenseBlock_%d' % i, DB)
    # 上一个稠密块的输出通道，这里用的是类.属性的用法
    num_channels = DB.out_channels
    # 在稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module('transition_block_%d' % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net.add_module('BN', nn.BatchNorm2d(num_channels))
net.add_module('relu', nn.ReLU())
# 与ResNet一样，最后加入全局平均池化层后接上全连接层输出。
net.add_module('global_avg_pool', GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
net.add_module('fc', nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))
```
最后，加载数据和训练模型都跟之前的一样。

# 循环神经网络
本章先略过

# 优化算法
## 优化与深度学习
虽然优化为深度学习提供了最小化损失函数的方法，但本质上，优化与深度学习的目标是有区别的。
由于优化算法的目标函数通常是一个基于训练数据集的损失函数，优化的目标在于降低训练误差。 而深度学习的目标在于降低泛化误差。为了降低泛化误差，除了使用优化算法降低训练误差以外，还需要注意应对过拟合。
在一个深度学习问题中，我们通常会预先定义一个损失函数。有了损失函数以后，我们就可以使用优化算法试图将其最小化。在优化中，这样的损失函数通常被称作优化问题的目标函数（objective function）。依据惯例，优化算法通常只考虑最小化目标函数。其实，任何最大化问题都可以很容易地转化为最小化问题，只需令目标函数的相反数为新的目标函数即可。
优化在深度学习中有很多挑战，比如局部最小值和鞍点。
（1）深度学习模型的目标函数可能有若干局部最优值。当一个优化问题的数值解在局部最优解附近时，由于目标函数有关解的梯度接近或变成零，最终迭代求得的数值解可能只令目标函数局部最小化而非全局最小化。
（2）梯度接近或变成零可能是由于当前解在局部最优解附近造成的。事实上，另一种可能性是当前解在鞍点（saddle point）附近。比如在鞍点位置，目标函数在x轴方向上是局部最小值，但在y轴方向上是局部最大值。
假设一个函数的输入为k维向量，输出为标量，那么它的海森矩阵（Hessian matrix）有k个特征值。该函数在梯度为0的位置上可能是局部最小值、局部最大值或者鞍点。
当函数的海森矩阵在梯度为零的位置上的特征值全为正时，该函数得到局部最小值。
当函数的海森矩阵在梯度为零的位置上的特征值全为负时，该函数得到局部最大值。
当函数的海森矩阵在梯度为零的位置上的特征值有正有负时，该函数得到鞍点。
随机矩阵理论告诉我们，对于一个大的高斯随机矩阵来说，任一特征值是正或者是负的概率都是0.5。那么，以上第一种情况的概率为 0.5的k次方。由于深度学习模型参数通常都是高维的（k很大），目标函数的鞍点通常比局部最小值更常见。

## 梯度下降和随机梯度下降
下图中的公式清晰地解释了为什么梯度下降能降低目标函数的数值：
![image](https://user-images.githubusercontent.com/6218739/76134337-5bb1f800-6058-11ea-908d-a03ebaa1a44f.png)
在深度学习里，目标函数通常是训练数据集中有关各个样本的损失函数的平均。当训练数据样本数很大时，梯度下降每次迭代的计算开销很高。
随机梯度下降（stochastic gradient descent，SGD）减少了每次迭代的计算开销。

## 小批量随机梯度下降
在每一次迭代中，梯度下降使用整个训练数据集来计算梯度，因此它有时也被称为批量梯度下降（batch gradient descent）。而随机梯度下降在每次迭代中只随机采样一个样本来计算梯度。我们还可以在每轮迭代中随机均匀采样多个样本来组成一个小批量，然后使用这个小批量来计算梯度。
在实际中，（小批量）随机梯度下降的学习率可以在迭代过程中自我衰减。
当批量大小为1时，小批量随机梯度下降算法即为随机梯度下降；当批量大小等于训练数据样本数时，该算法即为梯度下降。当批量较小时，每次迭代中使用的样本少，这会导致并行处理和内存使用效率变低。这使得在计算同样数目样本的情况下比使用更大批量时所花时间更多。当批量较大时，每个小批量梯度里可能含有更多的冗余信息。为了得到较好的解，批量较大时比批量较小时需要计算的样本数目可能更多，例如增大迭代周期数。

## 动量法
目标函数有关自变量的梯度代表了目标函数在自变量当前位置下降最快的方向。因此，梯度下降也叫作最陡下降（steepest descent）。在每次迭代中，梯度下降根据自变量当前位置，沿着当前位置的梯度更新自变量。然而，如果自变量的迭代方向仅仅取决于自变量当前位置，这可能会带来一些问题。举个例子：同一位置上，假设目标函数在竖直方向比在水平方向的斜率的绝对值更大。因此，给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大。那么，我们需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解。然而，这会造成自变量在水平方向上朝最优解移动变慢。
动量法的提出是为了解决梯度下降的上述问题。
（1）动量法使用了指数加权移动平均的思想。它将过去时间步的梯度做了加权平均，且权重按时间步指数衰减。
（2）动量法使得相邻时间步的自变量更新在方向上更加一致。
在PyTorch中，只需要通过参数momentum来指定动量超参数即可使用动量法。

## AdaGrad算法
在之前介绍过的优化算法中，目标函数自变量的每一个元素在相同时间步都使用同一个学习率来自我迭代。
动量法依赖指数加权移动平均使得自变量的更新方向更加一致，从而降低发散的可能。而AdaGrad算法根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。
AdaGrad算法在迭代过程中不断调整学习率，并让目标函数自变量中每个元素都分别拥有自己的学习率。
使用AdaGrad算法时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。
通过名称为Adagrad的优化器方法，我们便可使用PyTorch提供的AdaGrad算法来训练模型。

## RMSProp算法
RMSProp算法和AdaGrad算法的不同在于，RMSProp算法使用了小批量随机梯度按元素平方的指数加权移动平均来调整学习率。
通过名称为RMSprop的优化器方法，我们便可使用PyTorch提供的RMSProp算法来训练模型。注意，超参数$\gamma$通过alpha指定。

## AdaDelta算法
除了RMSProp算法以外，另一个常用优化算法AdaDelta算法也针对AdaGrad算法在迭代后期可能较难找到有用解的问题做了改进。有意思的是，AdaDelta算法没有学习率这一超参数。AdaDelta算法没有学习率超参数，它通过使用有关自变量更新量平方的指数加权移动平均的项来替代RMSProp算法中的学习率。
通过名称为Adadelta的优化器方法，我们便可使用PyTorch提供的AdaDelta算法。它的超参数可以通过rho来指定。

## Adam算法
Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。所以Adam算法可以看做是RMSProp算法与动量法的结合。
Adam算法使用了偏差修正。
通过名称为“Adam”的优化器实例，我们便可使用PyTorch提供的Adam算法。

# 计算性能
## 命令式编程和符号式编程
（1）命令式编程更方便。当我们在Python里使用命令式编程时，大部分代码编写起来都很直观。同时，命令式编程更容易调试。这是因为我们可以很方便地获取并打印所有的中间变量值，或者使用Python的调试工具。
（2）符号式编程更高效并更容易移植。一方面，在编译的时候系统容易做更多优化（因为在编译时系统能够完整地获取整个程序）；另一方面，符号式编程可以将程序变成一个与Python无关的格式，从而可以使程序在非Python环境下运行，以避开Python解释器的性能问题。
截止目前（2020年3月），PyTorch仅采用了命令式编程方式。
## 异步计算
以下一段是唐树森同学对PyTorch官网上的翻译。
默认情况下，PyTorch中的 GPU 操作是异步的。当调用一个使用 GPU 的函数时，这些操作会在特定的设备上排队但不一定会在稍后立即执行。这就使我们可以并行更多的计算，包括 CPU 或其他 GPU 上的操作。 一般情况下，异步计算的效果对调用者是不可见的，因为（1）每个设备按照它们排队的顺序执行操作，（2）在 CPU 和 GPU 之间或两个 GPU 之间复制数据时，PyTorch会自动执行必要的同步操作。因此，计算将按每个操作同步执行的方式进行。 可以通过设置环境变量CUDA_LAUNCH_BLOCKING = 1来强制进行同步计算。当 GPU 产生error时，这可能非常有用。（异步执行时，只有在实际执行操作之后才会报告此类错误，因此堆栈跟踪不会显示请求的位置。）

## 自动并行计算
PyTorch能有效地实现在不同设备上（比如两块GPU）自动并行计算。

## 多GPU计算
因为目前手头只有一块GPU，所以本节没法实战，故略过。
需要注意单主机多GPU计算与分布式计算的区别。
 
# 计算机视觉
## 图像增广
这个地方有几点提前注意：
（1）下面介绍的都是torchvision自带的函数，关于图像增广可以借助更专业的第三方库（比如可以将标注一块增广），比如：
[Image augmentation for machine learning experiments](https://github.com/aleju/imgaug)
（2）为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广。此外，在实际PyTorch应用时，注意使用ToTensor将小批量图像转成PyTorch需要的格式，即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数。

图像增广（image augmentation）技术通过对训练图像做一系列随机改变，来产生相似但又不同的训练样本，从而扩大训练数据集的规模。图像增广的另一种解释是，随机改变训练样本可以降低模型对某些属性的依赖，从而提高模型的泛化能力。例如，我们可以对图像进行不同方式的裁剪，使感兴趣的物体出现在不同位置，从而减轻模型对物体出现位置的依赖性。我们也可以调整亮度、色彩等因素来降低模型对色彩的敏感度。可以说，在当年AlexNet的成功中，图像增广技术功不可没。

这部分涉及PIL、skimage、OpenCV、PyTorch Tensor，这四个有类似的地方，也有很多小区别，比如：
（1）PIL、skimage、OpenCV的图像通道都是h乘w乘c，即高乘宽乘通道，而PyTorch的ToTensor自己会转化为c乘h乘w，同时转为float后除以255（这里一定注意，如果ToTensor接收的是numpy的array，一定保证它是uint8格式，否则可能仅是通道顺序变化，而数值没有除以255）
（2）PIL的数据类型是Image对象，skimage和OpenCV都是numpy。

torchvision.transforms模块有大量现成的转换方法，不过需要注意的是有的方法输入的是PIL图像，如Resize；有的方法输入的是tensor，如Normalize；而还有的是用于二者转换，如ToTensor将PIL图像转换成tensor。一定要注意这点，使用时看清文档。

具体可以参考如下文章：
[OpenCV，PIL，Skimage你pick谁](https://zhuanlan.zhihu.com/p/52344534)
[opencv-PIL-matplotlib-Skimage-Pytorch图片读取区别与联系](https://www.jianshu.com/p/dd08418c306f)
[pytorch图像基本操作](https://zhuanlan.zhihu.com/p/27382990)

### 常用的图像增广方法
```python
%matplotlib inline
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from skimage import io
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 这里使用PIL读取图像
img = Image.open('drive/My Drive/cat1.jpg')
# 翻转
# 左右翻转图像通常不改变物体的类别。它是最早也是最广泛使用的一种图像增广方法。
horizon_flip = transforms.RandomHorizontalFlip()
# 上下翻转不如左右翻转通用。但是至少对于样例图像，上下翻转不会造成识别障碍。
vertical_flip = transforms.RandomVerticalFlip()
# 裁剪
# 可以通过对图像随机裁剪来让物体以不同的比例出现在图像的不同位置
# 这同样能够降低模型对目标位置的敏感性。
# 每次随机裁剪出一块面积为原面积10%∼100%的区域，且该区域的宽和高之比随机取自0.5∼2，
# 然后再将该区域的宽和高分别缩放到200像素。
shape_aug = transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# 变化颜色
# 可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
# 将图像的亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）
brightness_change = transforms.ColorJitter(brightness=0.5)
# 随机变化图像的色调
hue_change = transforms.ColorJitter(hue=0.5)
# 随机变化图像的对比度
contrast_change = transforms.ColorJitter(contrast=0.5)
# 综合设置颜色变化
color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# 叠加多个图像增广方法
# 实际应用中我们会将多个图像增广方法叠加使用。
# 我们可以通过Compose实例将上面定义的多个图像增广方法叠加起来，再应用到每张图像之上。
compose = transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              color_aug,
                              shape_aug
])
# 大部分图像增广方法都有一定的随机性。
# 为了方便观察图像增广的效果，需要多次运行转换函数
for i in range(10):
    img_trans = horizon_flip(img)
    img_trans = vertical_flip(img)
    img_trans = shape_aug(img)
    img_trans = brightness_change(img)
    img_trans = hue_change(img)
    img_trans = contrast_change(img)
    img_trans = color_aug(img)
    img_trans = compose(img)
    # 存储时使用skimage，借此看看skimage与PIL的转换
    io.imsave(np.str(i)+'.jpg', np.array(img_trans))
```

## 微调
迁移学习是解决小数据集的一个有效方法。
本节介绍迁移学习中的一种常用技术：微调（fine tuning）。微调由以下4步构成。
在源数据集（如ImageNet数据集）上预训练一个神经网络模型，即源模型。
创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。
当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力。

注: 在使用预训练模型时，一定要和预训练时作同样的预处理。 如果你使用的是torchvision的models，那就要求: All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 如果你使用的是pretrained-models.pytorch仓库，请务必阅读其README，其中说明了如何预处理。

接下来我们来实践一个具体的例子：热狗识别。我们将基于一个小数据集对在ImageNet数据集上训练好的ResNet模型进行微调。该小数据集含有数千张包含热狗和不包含热狗的图像。我们将使用微调得到的模型来识别一张图像中是否包含热狗。
导入必要的包：
```python
%matplotlib inline
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 获取数据集
```python
# 下载数据集
!wget https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip
# 解压，得到两个文件夹hotdog/train和hotdog/test。
# 这两个文件夹下面均有hotdog和not-hotdog两个类别文件夹，每个类别文件夹里面是图像文件。
!unzip hotdog.zip
# 查看一下数据
# 关于ImageFolder的用法可以参考下面的链接：
# https://discuss.pytorch.org/t/questions-about-imagefolder/774/3
# https://blog.csdn.net/TH_NUM/article/details/80877435
print(train_imgs.classes) # ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字。
print(train_imgs.class_to_idx) # 字符串类别所对应的数值类别
train_imgs[0][0] # 前面的是正类图像，即热狗
train_imgs[-1][0] # 后面的是负类图像，即非热狗

# 指定RGB三个通道的均值和方差来将图像通道归一化 (每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出)
# 这个地方一定与预训练模型所做的处理保持一致！！
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 训练集所做的预处理
train_augs = transforms.Compose([
                                 # 先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入
                                 transforms.RandomResizedCrop(size=224),
                                 # 左右翻转
                                 transforms.RandomHorizontalFlip(),
                                 # 转为PyTorch所需的形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数的数据
                                 transforms.ToTensor(),
                                 # 归一化
                                 normalize
])
# 测试集所做的预处理
test_augs = transforms.Compose([
                                # 将图像的高和宽均缩放为256像素
                                transforms.Resize(size=256),
                                # 然后从中裁剪出高和宽均为224像素的中心区域作为输入
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                normalize
])
```

### 定义和初始化模型
```python
# 使用在ImageNet数据集上预训练的ResNet-18作为源模型。
# 这里指定pretrained=True来自动下载并加载预训练的模型参数。在第一次使用时需要联网下载模型参数。
# 不管你是使用的torchvision的models还是pretrained-models.pytorch仓库，默认都会将预训练好的模型参数下载到你的home目录下.torch文件夹。
# 你可以通过修改环境变量$TORCH_MODEL_ZOO来更改下载目录。
# 另一个比较常用的方法是，在其源码中找到下载地址直接浏览器输入地址下载，下载好后将其放到环境变量$TORCH_MODEL_ZOO所指文件夹即可，这样比较快。
pretrained_net = models.resnet18(pretrained=True)

# 下面打印源模型的成员变量fc。
# 作为一个全连接层，它将ResNet最终的全局平均池化层输出变换成ImageNet数据集上1000类的输出。
# 如果你使用的是其他模型，那可能没有成员变量fc（比如models中的VGG预训练模型），
# 所以正确做法是查看对应模型源码中其定义部分，这样既不会出错也能加深我们对模型的理解。
print(pretrained_net.fc)

# 这里应该将最后的fc成修改我们需要的输出类别数:
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
# 此时fc层中的参数已经被初始化了，但是其他层依然保存着预训练得到的参数。
print(list(pretrained_net.fc.parameters()))

# 由于是在很大的ImageNet数据集上预训练的，所以非fc层的参数已经足够好，因此一般只需使用较小的学习率来微调这些参数
# 而fc中的随机初始化参数一般需要更大的学习率从头训练。
# PyTorch可以方便的对模型的不同部分设置不同的学习参数
# 将fc层的参数的id取出
output_params = list(map(id, pretrained_net.fc.parameters()))
# 从整个net的所有参数中剔除fc的参数，保留非fc中的参数放入feature_params中
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01 # 默认的学习率设为0.01
optimizer = optim.SGD([
                       {'params': feature_params}, # 非fc层的参数使用默认的学习率，即外层的学习率
                       {'params': pretrained_net.fc.parameters(), 'lr': lr*10} # fc层参数的学习率设为已训练过的部分的10倍
                       ],
                      lr=lr,
                      weight_decay=0.001)
```

### 微调模型
```python
# 先定义一个统一的训练过程函数
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device，则使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式，这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式

            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' 
              % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc, time.time()-start))
# 再定义一个使用微调的训练函数train_fine_tuning以便多次调用
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder('hotdog/train', transform=train_augs), batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder('hotdog/test', transform=test_augs), batch_size)
    loss = torch.nn.CrossEntropyLoss()
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# 根据前面的设置，我们将以10倍的学习率从头训练目标模型的输出层参数
train_fine_tuning(pretrained_net, optimizer)
```

## 目标检测
很多时候图像里有多个我们感兴趣的目标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。在计算机视觉里，我们将这类任务称为目标检测（object detection）或物体检测。
在目标检测里，我们通常使用边界框（bounding box）来描述目标位置。边界框是一个矩形框，可以由矩形左上角的x和y轴坐标与右下角的x和y轴坐标确定。
目标检测相关知识暂时略过，包括下面的锚框、目标检测数据集、SSD、区域卷积神经网络R-CNN系列（R-CNN、Fast R-CNN、Faster R-CNN、Mask R-CNN）。
值得一提的是，Mask R-CNN在Faster R-CNN的基础上做了修改。Mask R-CNN将兴趣区域池化层替换成了兴趣区域对齐层，即通过双线性插值（bilinear interpolation）来保留特征图上的空间信息，从而更适于像素级预测。兴趣区域对齐层的输出包含了所有兴趣区域的形状相同的特征图。它们既用来预测兴趣区域的类别和边界框，又通过额外的全卷积网络预测目标的像素级位置。

## 语义分割和数据集
在目标检测问题中，我们一直使用方形边界框来标注和预测图像中的目标。而语义分割（semantic segmentation）问题，它关注如何将图像分割成属于不同语义类别的区域。值得一提的是，这些语义区域的标注和预测都是像素级的。与目标检测相比，语义分割标注的像素级的边框显然更加精细。
### 图像分割和实例分割
计算机视觉领域还有2个与语义分割相似的重要问题，即图像分割（image segmentation）和实例分割（instance segmentation）。在这里将它们与语义分割简单区分一下。
（1）图像分割将图像分割成若干组成区域。这类问题的方法通常利用图像中像素之间的相关性。它在训练时不需要有关图像像素的标签信息，在预测时也无法保证分割出的区域具有我们希望得到的语义。比如，图像分割后不知道分割出来的东西是什么，并不知道哪个是狗，哪个是猫。
（2）实例分割又叫同时检测并分割（simultaneous detection and segmentation）。它研究如何识别图像中各个目标实例的像素级区域。与语义分割有所不同，实例分割不仅需要区分语义，还要区分不同的目标实例。如果图像中有两只狗，实例分割需要区分像素属于这两只狗中的哪一只。

### Pascal VOC2012语义分割数据集
语义分割的一个重要数据集叫作Pascal VOC2012。
（1）下载并读取数据集
```python
# 下载数据集
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image

def read_voc_images(root="drive/My Drive/VOCdevkit/VOC2012", is_train=True, max_num=None):
    # ImageSets/Segmentation路径包含了指定训练和测试样本的文本文件
    txt_fname = '%s/ImageSets/Segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()

    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None]*len(images), [None]*len(images)
    for i, fname in enumerate(images):
        # JPEGImages和SegmentationClass路径下分别包含了样本的输入图像和标签
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert('RGB')
        # 这里的标签也是图像格式，其尺寸和它所标注的输入图像的尺寸相同。标签中颜色相同的像素属于同一个语义类别。
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert('RGB')

    return features, labels
voc_dir = 'drive/My Drive/VOCdevkit/VOC2012'
train_features, train_labels = read_voc_images(voc_dir, max_num=100)

# 在标签图像中，白色和黑色分别代表边框和背景，而其他不同的颜色则对应不同的类别。
# 接下来，我们列出标签中每个RGB颜色的值及其标注的类别。
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 创建一个256*256*256长度的tensor
colormap2label = torch.zeros(256**3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    # 将颜色索引与类别索引一一对应起来
    # 注意colormap2label是一个一维向量，所以不同通道的颜色值传入后要乘以256
    # 具体的数值大小无意义，只要是不同类别能分辨开即可，同时与下面的
    # (colormap[:, :, 0]*256+colormap[:, :, 1])*256+colormap[:,:,2]要对应起来
    colormap2label[(colormap[0]*256+colormap[1])*256+colormap[2]] = i

def voc_label_indices(colormap, colormap2label):
    # 将PIL Image转成numpy，然后数据类型改为int32位，colormap仍然是h*w*c的样式
    colormap = np.array(colormap.convert('RGB')).astype('int32')
    # 将不同的通道乘以256转化成索引值
    # 这样，idx的shape就是h*w
    idx = ((colormap[:, :, 0]*256+colormap[:, :, 1])*256+colormap[:,:,2])
    # 找到该索引矩阵所对应的标签类别
    # 因为idx是索引矩阵，其大小与图像大小相同，那么返回的也正是每个像素所对应的类别，即与图像同样大小的类别矩阵
    return colormap2label[idx]
```
（2）预处理数据
在之前的章节中，我们通过缩放图像使其符合模型的输入形状。然而在语义分割里，这样做需要将预测的像素类别重新映射回原始尺寸的输入图像。这样的映射难以做到精确，尤其在不同语义的分割区域。为了避免这个问题，我们将图像裁剪成固定尺寸而不是缩放。具体来说，我们使用图像增广里的随机裁剪，并对输入图像和标签裁剪相同区域。
```python
def voc_rand_crop(feature, label, height, width):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label

# 比如随机裁剪200*300大小的区域
img = voc_rand_crop(train_features[0], train_labels[0], 200, 300)
```
（3）自定义语义分割数据集
```python
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       mean=self.rgb_mean,
                                                       std=self.rgb_std)
        ])
        self.crop_size = crop_size
        features, labels = read_voc_images(
            root = voc_dir,
            is_train = is_train,
            max_num = max_num
        )
        # 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的filter函数所移除。
        self.features = self.filter(features)
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid samples')

    # 这个地方需要特别注意，PIL.size返回的是(width, height)，即宽在前，高在后，而我们输入的参数是高在前
    # https://liam.page/2015/04/22/pil-tutorial-basic-usage/
    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and 
            img.size[0] >= self.crop_size[1]
        )]

    # 通过实现__getitem__函数，我们可以任意访问数据集中索引为idx的输入图像及其每个像素的类别索引
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return (self.tsf(feature), # feature是float32的tensor
                voc_label_indices(label, self.colormap2label) # label是uint8的tensor
                )
    def __len__(self):
        return len(self.features)

crop_size = (320, 480)
max_num = 100
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)
# 设批量大小为64，分别定义训练集和测试集的迭代器。
batch_size = 64
num_workers = 4
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)

test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                             num_workers=num_workers)
```

## 全卷积网络
全卷积网络（fully convolutional network，FCN）采用卷积神经网络实现了从图像像素到像素类别的变换。与之前介绍的卷积神经网络有所不同，全卷积网络通过转置卷积（transposed convolution）层将中间层特征图的高和宽变换回输入图像的尺寸，从而令预测结果与输入图像在空间维（高和宽）上一一对应：给定空间维上的位置，通道维的输出即该位置对应像素的类别预测。
全卷积网络先使用卷积神经网络抽取图像特征，然后通过 1乘1 卷积层将通道数变换为类别个数，最后通过转置卷积层将特征图的高和宽变换为输入图像的尺寸，从而输出每个像素的类别。
在全卷积网络中，可以将转置卷积层初始化为双线性插值的上采样。

# 自然语言处理
暂时略过。
