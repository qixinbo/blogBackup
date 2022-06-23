---
title: 轻量级PyTorch通用训练模板pytorch-accelerated解析：3 -- API之Schedulers
tags: [PyTorch]
categories: machine learning 
date: 2022-6-5
---

# 简介
这一章将研究`pytorch-accelerated`的调度器`API`。
`PyTorch-accelerated`提供了一些调度器的实现，可以在任何`PyTorch`训练循环中使用。然而，与`PyTorch`的原生调度器不同——这些原生调度器可以在训练循环的不同点上被调用——所有`Pytorch-accelerated`调度器都期望在每次优化器更新后被调用。

# 内置调度器
`PyTorch-accelerated`内置了一个有状态的余弦退火学习率调度器，基于[这篇论文](https://arxiv.org/abs/1608.03983)，但没有论文中的`restart`。
这个调度器与`PyTorch`的`CosineAnnealingLR`不同，它提供了增加`warmup`和`cooldown`的`epoch`的选项。此外，可以通过调整`k-decay`参数来修改退火率，可详见[这篇论文](https://arxiv.org/abs/2004.05909)。
关于该调度器的具体细节，暂不深究，具体可以查看[这个文档](https://pytorch-accelerated.readthedocs.io/en/latest/schedulers.html#pytorch_accelerated.schedulers.cosine_scheduler.CosineLrScheduler)。

# 调度器基类
`PyTorch-accelerated`提供了两种类型的调度器的基类。
## 有状态的调度器
有状态的调度器维护一个内部计数，其对应于调度器的`step()`方法被调用了多少次。由于这些调度器与原生的`PyTorch`调度器具有相同的接口，因此`Trainer`默认支持这些调度器。
基类为：
```python
class pytorch_accelerated.schedulers.scheduler_base.StatefulSchedulerBase(optimizer, param_group_field: str = 'lr')
```
一个有状态的参数调度器基类，可用于更新优化器的参数组中的任意字段。这方面最常见的用例是学习率调度。
与`PyTorch`的调度器（它可以在训练循环的不同点被调用）不同的是，该类的目的是在每次优化器更新结束时被一致调用。
这个类负责维护更新的数量，在每次计算调度器的步骤时增加一个内部计数。
该类的一个用法如下：
```python
for current_epoch, epoch in enumerate(num_epochs):
    for batch in train_dataloader:
        xb, yb = batch
        predictions = model(xb)
        loss = loss_func(predictions, yb)

        loss.backward()
        optimizer.step()

        scheduler.step()
```

## 无状态的调度器
无状态的调度器没有维护关于当前训练运行的内部状态，因此需要在调用时明确提供当前的更新数量。如果要在训练器中使用无状态调度器，这就需要对`Trainer`进行子类化，并重写`scheduler_step()`方法。
基类是：
```python
class pytorch_accelerated.schedulers.scheduler_base.SchedulerBase(optimizer: Optimizer, param_group_field: str = 'lr')
```
该类也可用于更新优化器参数组中的任何字段。这方面最常见的用例是学习率调度。
与`PyTorch   的调度器（它可以根据实现方式在训练循环的不同点被调用）不同的是，这个类的目的是在每次优化器更新结束时被一致调用。
由于这个类在默认情况下是无状态的，它希望更新的数量是明确提供的，如下所示：
```python
for current_epoch, epoch in enumerate(num_epochs):
    num_updates = current_epoch * num_update_steps_per_epoch
    for batch in train_dataloader:
        xb, yb = batch
        predictions = model(xb)
        loss = loss_func(predictions, yb)

        loss.backward()
        optimizer.step()

        num_updates +=1
        scheduler.step_update(num_updates)
```

# 创建新的调度器
虽然调度器通常用于调度学习率，但`PyTorch-accelerated`中的调度器基类可用于调度优化器参数组中的任意参数。
要创建一个新的调度器，在大多数情况下，只需要对其中一个基类进行子类化，并重写`get_updated_values()`方法。
## 案例1：创建一个简单的目标学习率调度器
下面是一个例子，说明如何实现一个调度器，在每次达到一个`milestone`目标时，以一个系数`gamma`调整每个参数组的学习率：
```python
from pytorch_accelerated.schedulers import StatefulSchedulerBase

class MilestoneLrScheduler(StatefulSchedulerBase):
    def __init__(
        self, optimizer, gamma=0.5, epoch_milestones=(2, 4, 5), num_steps_per_epoch=100
    ):
        super().__init__(optimizer, param_group_field="lr")
        self.milestones = set(
            (num_steps_per_epoch * milestone for milestone in epoch_milestones)
        )
        self.gamma = gamma

    def get_updated_values(self, num_updates: int):
        if num_updates in self.milestones:
            lr_values = [
                group[self.param_group_field] for group in self.optimizer.param_groups
            ]
            updated_lrs = [lr * self.gamma for lr in lr_values]
            return updated_lrs
```

## 案例2：对权重衰减进行调度
下面是一个例子，说明可以定义一个调度器，每隔`n`步就递增一个系数`gamma`来增加权重衰减量：
```python
from pytorch_accelerated.schedulers import StatefulSchedulerBase

class StepWdScheduler(StatefulSchedulerBase):
    def __init__(self, optimizer, n=1000, gamma=1.1):
        super().__init__(optimizer, param_group_field="weight_decay")
        self.n = n
        self.gamma = gamma

    def get_updated_values(self, num_updates: int):
        if num_updates % self.n == 0 and num_updates > 0:
            wd_values = [
                group[self.param_group_field] for group in self.optimizer.param_groups
            ]
            updated_wd_values = [wd * self.gamma for wd in wd_values]
            return updated_wd_values
```