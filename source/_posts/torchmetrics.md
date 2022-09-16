---
title: PyTorch指标计算库TorchMetrics详解
tags: [PyTorch]
categories: machine learning 
date: 2022-9-17
---

参考资料：
[TorchMetrics Docs](https://torchmetrics.readthedocs.io/en/latest/index.html)
[TorchMetrics — PyTorch Metrics Built to Scale](https://devblog.pytorchlightning.ai/torchmetrics-pytorch-metrics-built-to-scale-7091b1bec919)
[Improve Your Model Validation With TorchMetrics](https://pub.towardsai.net/improve-your-model-validation-with-torchmetrics-b457d3954dcd)

# 什么是指标
一般来说，指标（`metrics`）的目的是监控和量化训练过程。如果你使用了一些技术，如学习率调度`learning-rate scheduling`或提前停止`early stopping`，你可能正在使用“指标”。虽然也可以为此使用损失`loss`，但指标是首选，因为它们能更好地代表训练目标。这意味着，当损失（如交叉熵）将网络的参数沿着正确的方向更新时，指标也可以指明一些关于网络的行为的额外见解。
与损失相反，指标不需要是可微的（事实上很多都不是），但其中一些是可微的。如果指标本身是可微的，并且它是基于纯`PyTorch`实现，那么它也跟损失一样可以用来进行反向传播。

# 简介
[`TorchMetrics`](https://github.com/Lightning-AI/metrics)对`80`多个`PyTorch`指标进行了代码实现，且其提供了一个易于使用的`API`来创建自定义指标。主要特点有：
- 一个标准化的接口，以提高可重复性
- 兼容分布式训练
- 经过了严格的测试
- 在批次`batch`之间自动累积
- 在多个设备之间自动同步

# 安装
使用`pip`：
```sh
pip install torchmetrics
```
或使用`conda`：
```sh
conda install -c conda-forge torchmetrics
```
# 使用
与`torch.nn`类似，大多数指标都有一个基于类的版本和一个基于函数的版本。
## 函数版本
函数版本的指标实现了计算每个度量所需的基本操作。它们是简单的`python`函数，接收`torch.tensors`作为输入，然后返回`torch.tensor`类型的相对应的指标。
一个简单的示例如下：
```python
import torch
# import our library
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target)
```
## 模块版本
几乎所有的函数版本的指标都有一个相应的基于类的版本，该版本在实际代码中调用对应的函数版本。基于类的指标的特点是具有一个或多个内部状态（类似于`PyTorch`模块的参数），使其能够提供额外的功能：
- 对多个批次的数据进行累积
- 多个设备之间的自动同步
- 指标运算

一个示例如下：
```python
import torch
# import our library
import torchmetrics

# initialize metric
metric = torchmetrics.Accuracy()

n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    # metric on current batch
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")

# metric on all batches using custom accumulation
acc = metric.compute()
print(f"Accuracy on all data: {acc}")

# Reseting internal state such that metric ready for new data
metric.reset()
```

# 自定义指标
自定义指标就像子类化一个`torch.nn.Module`一样简单。简单地说，子类化`Metric`并做以下工作。
- 实现`__init__`方法，在这里为每一个指标计算所需的内部状态调用`self.add_state`；
- 实现`update`方法，在这里进行更新指标状态所需的逻辑；
- 实现`compute`方法，在这里进行最终的指标计算。

关于实现自定义指标的实际例子和更多信息，看[这个页面](https://torchmetrics.readthedocs.io/en/latest/pages/implement.html#implement)。