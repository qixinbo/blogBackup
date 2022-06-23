---
title: 轻量级PyTorch通用训练模板pytorch-accelerated解析：4 -- 其他API
tags: [PyTorch]
categories: machine learning 
date: 2022-6-12
---

# 简介
这一章将研究`pytorch-accelerated`的其他`API`，包括追踪`Tracking`、运行配置`Run Config`、微调` Fine tuning`。

# Tracking
## RunHistory抽象基类
`RunHistory`抽象基类定义了`Trainer`运行历史的`API`：
```python
class pytorch_accelerated.tracking.RunHistory
```
（1）获得当前`epoch`的数值：
```python
def current_epoch(self) -> int
```
（2）获得指定指标最近的记录值：
```python
def get_latest_metric(self, metric_name)
```
（3）获得在追踪的所有指标的名字：
```python
def get_metric_names(self) -> Iterable
```
返回的是一个集合，所以名称都是独一无二的。
（4）获得指定指标的所有值：
```python
def get_metric_values(self, metric_name) -> Iterable
```
（5）重置：
```python
def reset(self)
```
重置`RunHistory`的状态。
（6）记录指定指标的值：
```python
def update_metric(self, metric_name, metric_value)
```

## 具体实现
上面的`RunHistory`是抽象基类，其定义的方法都没有具体的实现。
`InMemoryRunHistory`是对`RunHistory`的一个具体实现。
```python
class InMemoryRunHistory(RunHistory):
    """
    An implementation of :class:`RunHistory` which stores all recorded values in memory.
    """

    def __init__(self):
        self._current_epoch = 1
        self._metrics = defaultdict(list)

    def get_metric_names(self):
        return set(self._metrics.keys())

    def get_metric_values(self, metric_name):
        return self._metrics[metric_name]

    def get_latest_metric(self, metric_name):
        if len(self._metrics[metric_name]) > 0:
            return self._metrics[metric_name][-1]
        else:
            raise ValueError(
                f"No values have been recorded for the metric {metric_name}"
            )

    def update_metric(self, metric_name, metric_value):
        self._metrics[metric_name].append(metric_value)

    @property
    def current_epoch(self):
        return self._current_epoch

    def _increment_epoch(self):
        self._current_epoch += 1

    def reset(self):
        self._current_epoch = 1
        self._metrics = defaultdict(list)
```

# Run Config
`TrainerRunConfig`是一个不可变的数据类，包含训练器`Trainer`当前状态的数值。
```python
@dataclass(frozen=True)
class TrainerRunConfig:
```
其属性有：
- `num_epochs`：当前训练的迭代次数
- `train_per_device_batch_size`：训练时每个设备上的批大小
- `train_dl_kwargs`：创建训练集数据加载器的所需参数
- `eval_per_device_batch_size`：评估时每个设备上的批大小
- `eval_dl_kwargs`：创建验证集数据加载器的所需参数
- `gradient_accumulation_steps`：训练时梯度累加的步数
- `gradient_clip_value`：模型参数的梯度修剪的阈值
- `train_total_batch_size`：训练时总批大小
- `eval_total_batch_size`：评估时总批大小
- `num_update_steps_per_epoch`：训练时当模型参数更新时的步数
- `max_num_train_steps`：训练的总步数，如果指定的话，会覆盖掉`num_epochs`参数
- `is_local_process_zero`：如果当前进程是当前节点上的主进程，则为`True`；否则为`False`
- `is_world_process_zero`：如果当前进程是横跨所有节点的主进程，则为`True`，否则为`False`
- `is_distributed`：如果`trainer`是分布式训练，则为`True`，否则为`False`
- `mixed_precision`：包含所用的混合精度类型的字符串，否则为`no`

# Fine tuning
`ModelFreezer`是一个用来冻结和解冻一个模型的不同部分的类，其用来简化迁移学习中微调的操作。
```python
class pytorch_accelerated.finetuning.ModelFreezer(model, freeze_batch_norms=False)
```
该类使用以下的抽象定义：
- `Layer`：是一个深度为1的`torch.nn.Module`的子类，即这个特定的`module`不是嵌套的
- `LayerGroup`：是模型类的属性，可以是网络层`layers`或嵌套的`modules`

举个例子，如下的模型：
```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(100, 100)
        self.block_1 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.Sequential(
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
            ),
        )
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = self.input(x)
        x = self.block_1(x)
        out = self.output(x)
        return out
```
该模型的`LayerGroup`就是`[input, block_1, output]`，而`Layers`则是一个有序的、压平的`Linear`、`BatchNorm`、`ReLU`模块列表。

主要方法有：
（1）冻结指定索引的`LayerGroup`：
```python
freeze(from_index=0, to_index=-2, set_modules_as_eval=False)
```
默认情况下，这将冻结所有层组，除了最后一个层组。
参数有：
- `from_index`：第一个被冻结的`LayerGroup`的索引
- `to_index`：最后一个被冻结的`LayerGroup`的索引
- `set_modules_as_eval`：若为`True`，这些冻结的模块也会被置为`eval`模式。默认是`False`

（2）返回模型的所有`LayerGroups`：
```python
get_layer_groups() → List[LayerGroup]
```
（3）返回模型的所有`Layer`：
```python
get_layers() → List[Layer]
```
（4）返回所有未被冻结的模型参数：
```python
get_trainable_parameters()
```
这些参数将在训练中被更新。
（5）解冻指定索引的`LayerGroup`：
```python
unfreeze(from_index=-1, to_index=0, set_modules_as_training=True)
```
默认情况下，这将解冻所有`LayerGroups`。对于一个`LayerGroup`，任何已经解冻的参数都会被返回，这样如果需要的话，它们可以被添加到一个优化器中。
参数有：
- `from_index`：第一个被解冻的`LayerGroup`的索引
- `to_index`：最后一个被解冻的`LayerGroup`的索引
- `set_modules_as_training`：若为`True`，这些解冻的模块也会被置为`train`模式。默认是`True`。

返回值是：包含每一个解冻的`layer Group`的参数的字典。
