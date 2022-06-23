---
title: 轻量级PyTorch通用训练模板pytorch-accelerated解析：0 -- 介绍及入门
tags: [PyTorch]
categories: machine learning 
date: 2022-5-28
---
# 介绍
[`pytorch-accelerated`](https://github.com/Chris-hughes10/pytorch-accelerated)是一个轻量级的库，旨在通过提供一个最小但可扩展的训练循环（封装在一个`Trainer`对象中）来加速`PyTorch`模型的训练过程；它足够灵活，可以处理大多数用例，并且能够利用不同的硬件选项而不需要修改代码。更多介绍见原作者的[博文](https://towardsdatascience.com/introducing-pytorch-accelerated-6ba99530608c)。
`pytorch-accelerated`最大的两个特点就是：简单`simplicity`和透明`transparency`。怎么理解这两个词呢（个人理解）：
（1）简单体现在它是一套可复用的`PyTorch`的训练代码，每次有新模型和新场景需要训练时不必将原来的代码拷过来拷过去，`pytorch-accelerated`提供了一套通用但不失灵活性的代码模板；
（2）透明体现在它基于纯正的`PyTorch`语法和`API`，而不是对`PyTorch`进行过度封装（此处`PyTorch Lightning`中枪），否则用户会感觉新学习了一套框架和语法。
这里可以再扩展说一下，作者的[介绍博文](https://towardsdatascience.com/introducing-pytorch-accelerated-6ba99530608c)后有一条评论：“这个库与`PyTorch Lighting`对比怎么样，我已经用了`Lightning`，为啥要用这个库呢？”作者给出了一个长长的回答，解释得挺清晰的：
> 我使用`Lightning`很长时间了，但越来越觉得我花在理解和调试`Lightning`上的时间比花在任务上的时间多。这在一开始可能很简单，但由于选项和指标太多，我发现自己花了很多时间阅读文档，并发现源代码对我来说并不容易一目了然。
> 另外，从一开始我就发现，`Lightning`的设计，即模型和训练循环绑在一起，并不适合我想为同一个任务轻松切换不同的模型和优化器的场景；我个人更喜欢训练循环与模型分开。这意味着模型仍然是纯正的`PyTorch`；我可以插入其他库的模型，如`timm`，而不需要做任何修改，也不需要在部署时有任何额外的依赖。
> 从本质上讲，我想要一个简单易懂、易于调试的库，同时保持足够的灵活性以满足我的需要；学习曲线足够浅，我可以把它介绍给其他人，他们也可以很快上手。根据我自己的要求，我创建了PyTorch-accelerated。它在简单的情况下开箱即用，而且很容易定制行为的任何部分。

再再扩展说一下，对于`PyTorch Lightning`，知乎上有很多真香帖（比如[1](https://zhuanlan.zhihu.com/p/353985363)、[2](https://zhuanlan.zhihu.com/p/157742258)），也有很多劝退帖（比如[1](https://zhuanlan.zhihu.com/p/492703063)、[2](https://zhuanlan.zhihu.com/p/363045412)（这篇虽然从标题和全文上看是“真香”，实则在文章最后和评论中作者表示弃坑了，采用了原生的`PyTorch`））。个人感觉采用`PyTorch Lightning`有两个很大的弊端（严格来说其实是一个）：
（1）与原生`PyTorch`相比，`PyTorch Lightning`进行了过度封装，感觉像是在学习另一个框架；
（2）假设使用`PyTorch Lightning`编写代码，如果是个人使用还好，但如果是一个团队共同维护代码，很难说服别人也采用该写法。
于是，本文的主角`pytorch-accelerated`就有了用武之地。

## 目标用户
什么类型的用户/开发者可以尝试使用`pytorch-accelerated`呢：
（1）熟悉PyTorch的用户，但希望避免编写常见的训练循环模板，以专注于其他更重要的部分。
（2）喜欢并乐于选择和创建自己的模型、损失函数、优化器和数据集的用户。
（3）喜欢简单但高效的功能的用户，这些功能的行为需要很容易调试、理解和推理（`PyTorch Lightning`：你报我身份证得了）。

另一方面，`pytorch-accelerated`不适合什么类型的用户/开发者呢？
（1）如果你正在寻找一个端到端的解决方案，包括从加载数据到推理的所有内容，该方案帮助你选择模型、优化器或损失函数，此时可能更适合使用[fastai](https://github.com/fastai/fastai)。 `pytorch-accelerated`只关注训练过程，其他所有问题都由用户负责。
（2）如果你想自己编写整个训练循环，但是不想涉及恼人的设备管理问题（比如多机多卡并行），你可能最适合使用`Hugging Face`的[`accelerate`](https://github.com/huggingface/accelerate)！ `accelerate`专注于将`PyTorch`的分布式训练和混合精度训练变得简单，不过整个训练循环得自己写。
（3）如果你正在研究一个定制的、高度复杂的、不符合通常训练循环模式的用例，并且想在你选择的硬件上挤出每一点性能，你可能最好坚持使用原生`PyTorch`；在高度专业化的情况下，任何高级`API`都会成为一种开销。
# 安装
（注意提前安装好`pytorch`）
使用`pip`安装：
```sh
pip install pytorch-accelerated
```
其实看`pytorch-accelerated`的`requirements.txt`，它只依赖于`Hugging Face`的`accelerate`库及`tqdm`（在终端下显示进度条），因此可以说是它的`API`就是`PyTorch`的原生`API`加上`accelerate`的分布式训练的`API`。

## accelerate库

`Hugging Face`的`🤗accelerate`库（[地址](https://github.com/huggingface/accelerate)），可以无痛地对`Pytorch`进行多`GPU`、`TPU`、[混合精度训练](https://zhuanlan.zhihu.com/p/103685761)。见[机器之心的报道](https://mp.weixin.qq.com/s/ST0mWd4E7ZxMMl04_yTKzA)。
> 多数 PyTorch 高级库都支持分布式训练和混合精度训练，但是它们引入的抽象化往往需要用户学习新的 API 来定制训练循环。许多 PyTorch 用户希望完全控制自己的训练循环，但不想编写和维护训练所需的复杂的样板代码。Hugging Face 最近发布的新库 Accelerate 解决了这个问题。
> `🤗accelerate`提供了一个简单的`API`，将与多`GPU`、`TPU`、`fp16`相关的样板代码抽离了出来，保持其余代码不变。`PyTorch`用户无须使用不便控制和调整的抽象类或编写、维护样板代码，就可以直接上手多`GPU`或`TPU`。

`🤗accelerate`库有两大特点：
（1）提供了一套简单的`API`来处理分布式训练和混合精度训练，无需在不同情形下对代码进行大的改动；
（2）提供了一个命令行接口工具来快速配置并并行脚本。

## 可选
如果是想为了直接运行`pytorch-accelerated`提供的`examples`，则可以这样安装[其他所依赖的包](https://stackoverflow.com/questions/46775346/what-do-square-brackets-mean-in-pip-install)：
```sh
pip install pytorch-accelerated[examples]
```

# 配置和运行
可以使用`🤗accelerate`的`CLI`工具来生成配置文件：
```sh
accelerate config --config_file accelerate_config.yaml
```
然后运行：
```sh
accelerate launch --config_file accelerate_config.yaml train.py [--training-args]
```
值得注意的是，也不是必须要用`CLI`工具。如果想更精细地控制启动命令和参数，仍然可以通过通常的方式来运行脚本：
```sh
python train.py / python -m torch.distributed ...
```

# MNIST例子
`MNIST`手写字符分类可以说是深度学习领域的`Hello World`。
这一节将以`MNIST`来看看`pytorch-accelerated`是怎么使用的。

## 数据和模型准备
因为`pytorch-accelerated`专注于训练模型部分，因此数据的加载和模型的构建、配置都是使用的原生的`PyTorch`的代码：
```python
# examples/train_mnist.py
import os

from torch import nn, optim
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# 加载数据集
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# 拆分数据集
train_dataset, validation_dataset, test_dataset = random_split(dataset, [50000, 5000, 5000])

# 定义神经网络模型
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, input):
        return self.main(input.view(input.shape[0], -1))

# 实例化模型
model = MNISTModel()
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 定义损失函数
loss_func = nn.CrossEntropyLoss()
	
```

## 训练循环
加载了模型和数据后，下一步就是编写训练循环。
这里就是`pytorch-accelerated`的用武之地，唯一要做的就是导入`Trainer`。
```python
from pytorch_accelerated import Trainer

# Trainer被设计成用来封装一个完整的训练循环
# 将模型、损失函数和优化器都传入Trainer中
trainer = Trainer(
		model,
		loss_func=loss_func,
		optimizer=optimizer,
)

# 入口点是train方法，定义了该怎样训练
# 可以设置训练集、验证集（注意是验证集，不是测试集）、迭代次数、批处理大小
# 还可以设置学习率策略、梯度累加等，这里没有用到
trainer.train(
	train_dataset=train_dataset,
	eval_dataset=validation_dataset,
	num_epochs=8,
	per_device_batch_size=32,
)

# 评估模型
trainer.evaluate(
	dataset=test_dataset,
	per_device_batch_size=64,
)
```

## 训练
### 生成配置文件
```sh
accelerate config --config_file train_mnist.yaml
```

### 开始训练
```sh
accelerate launch --config_file train_mnist.yaml train_mnist.py
```

## 增加指标
上述例子中追踪的指标仅仅是每次迭代的损失`loss`，为了对训练结果有更深入的洞察，可以增加更多的指标。
对于指标的计算，这里引入了[`torchmetrics`](https://github.com/Lightning-AI/metrics)（发现这个库还是`PyTorch-Lightning`社区开发的，这波`PyTorch-Lightning`被人摘桃了，血亏🤗），该库兼容分布式训练，因此就不需要手动从不同进程中聚合计算结果。
计算指标有两种方式：
（1）定义一个继承自`Trainer`的子类，
（2）使用回调`callback`。
具体使用哪种方式极大地取决于用户的喜好。
不过作者有如下建议：因为计算指标实际上是不能影响训练代码的，因此使用`callback`可能是一个好的方式，因为使用`Trainer`子类的话，它也会间接地参与训练过程。不过还是具体情况具体分析。注意，因为`callbacks`都是顺序执行的，必须保证在打印指标之前就调用这些回调。
下面是对比这两种实现方式。

### 使用Trainer子类
`Trainer`有很多方法可以被重载，具体的文档在[这里](https://pytorch-accelerated.readthedocs.io/en/latest/trainer.html#customizing-trainer-behaviour)。主要的一个特点是有动词前缀（比如`create`和`calculate`）的方法都是期望能返回一个数值，其他的方法（比如`optimizer.step()`）则是用来设置内部状态。
示例代码为：
```python
# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a model on the MNIST Dataset

# This example demonstrates how the default trainer class can be overridden
# so that we can record classification metrics
#
# Note, this example requires installing the torchmetrics package
########################################################################

import os

from torch import nn, optim
from torch.utils.data import random_split
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_accelerated import Trainer


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        return self.main(x.view(x.shape[0], -1))


class TrainerWithMetrics(Trainer):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # this will be moved to the correct device automatically by the
        # MoveModulesToDeviceCallback callback, which is used by default
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(num_classes=num_classes),
                "precision": Precision(num_classes=num_classes),
                "recall": Recall(num_classes=num_classes),
            }
        )

    def calculate_eval_batch_loss(self, batch):
        batch_output = super().calculate_eval_batch_loss(batch)
        preds = batch_output["model_outputs"].argmax(dim=-1)

        self.metrics.update(preds, batch[1])

        return batch_output

    def eval_epoch_end(self):
        metrics = self.metrics.compute()
        self.run_history.update_metric("accuracy", metrics["accuracy"].cpu())
        self.run_history.update_metric("precision", metrics["precision"].cpu())
        self.run_history.update_metric("recall", metrics["recall"].cpu())

        self.metrics.reset()


def main():
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    num_classes = len(dataset.class_to_idx)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [50000, 5000, 5000]
    )
    model = MNISTModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    trainer = TrainerWithMetrics(
        model=model, loss_func=loss_func, optimizer=optimizer, num_classes=num_classes
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        num_epochs=2,
        per_device_batch_size=32,
    )

    trainer.evaluate(
        dataset=test_dataset,
        per_device_batch_size=64,
    )


if __name__ == "__main__":
    main()
```

### 使用回调
上面的代码可能已经有所显示，对于“增加指标”这种小的微调，如果使用`Trainer`子类的话会显得有些用力过猛。
此例中，可以保持默认的`trainer`不变，而使用回调来扩展功能。
为了创建一个新的回调，需要创建`TrainerCallback`的子类，然后重载相关方法，文档见[这里](https://pytorch-accelerated.readthedocs.io/en/latest/callbacks.html#creating-new-callbacks)。为了避免与`Trainer`的方法混淆，所有的回调方法都有`on_`前缀。
创建新的回调：
```python
from torchmetrics import MetricCollection, Accuracy, Precision, Recall

from pytorch_accelerated.callbacks import TrainerCallback


class ClassificationMetricsCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(num_classes=num_classes),
                "precision": Precision(num_classes=num_classes),
                "recall": Recall(num_classes=num_classes),
            }
        )

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.metrics.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        trainer.run_history.update_metric("accuracy", metrics["accuracy"].cpu())
        trainer.run_history.update_metric("precision", metrics["precision"].cpu())
        trainer.run_history.update_metric("recall", metrics["recall"].cpu())

        self.metrics.reset()
```
需要注意的一点是，在训练或验证之前，需要手动地将指标移动到正确地设备上。不过`Trainer`已经将这一步做了非常简单的处理，可以依据上下文返回正确的设备，即：
```python
from pytorch_accelerated.trainer import DEFAULT_CALLBACKS

# 将自定义的回调传入Trainer
# 因为想保持默认行为，所以将自定义的回调放在所有的默认回调之前。
trainer = Trainer(
	model,
	loss_func=loss_func,
	optimizer=optimizer,
	callbacks=(
		ClassificationMetricsCallback(
			num_classes=num_classes,
		),
		*DEFAULT_CALLBACKS,
	),
)
```
然后将上面代码复制进入最开始的代码中即可，无需变动其他地方的代码（只是需要在创建数据集后，计算一下其分类数目以传入回调`num_classes = len(dataset.class_to_idx)`）。

# 后记
实际只从这个`MNIST`例子还不能看出`pytorch-accelerated`能在多大程度上提高效率，后面需要多研究一下它的例子，毕竟`通用模板`的意义在于能适用于多种情形。留坑待填。