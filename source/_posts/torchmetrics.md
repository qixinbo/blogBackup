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
弄清楚需要评估哪些指标（`metrics`）是深度学习的关键。有各种指标，我们就可以评估ML算法的性能。
一般来说，指标（`metrics`）的目的是监控和量化训练过程。在一些技术中，如学习率调度`learning-rate scheduling`或提前停止`early stopping`，指标是用来调度和控制的关键。虽然也可以在这里使用损失`loss`，但指标是首选，因为它们能更好地代表训练目标。
与损失相反，指标不需要是可微的（事实上很多都不是），但其中一些是可微的。如果指标本身是可微的，并且它是基于纯`PyTorch`实现，那么它也跟损失一样可以用来进行反向传播。

# 简介
[`TorchMetrics`](https://github.com/Lightning-AI/metrics)对`80`多个`PyTorch`指标进行了代码实现，且其提供了一个易于使用的`API`来创建自定义指标。对于这些已实现的指标，如准确率`Accuracy`、召回率`Recall`、精确度`Precision`、`AUROC`、`RMSE`、`R²`等，可以开箱即用；对于尚未实现的指标，也可以轻松创建自定义指标。主要特点有：
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
每次调用指标的前向计算时，一方面对当前看到的一个批次的数据进行指标计算，另一方面更新内部指标状态，该状态记录了当前看到的所有数据。内部状态需要在`epoch`之间被重置，并且不应该在训练、验证和测试之间混淆。因此，强烈建议按不同的模式重新初始化指标，如下例所示：
```python
from torchmetrics.classification import Accuracy

train_accuracy = Accuracy()
valid_accuracy = Accuracy()

for epoch in range(epochs):
    for x, y in train_data:
        y_hat = model(x)

        # training step accuracy
        batch_acc = train_accuracy(y_hat, y)
        print(f"Accuracy of batch{i} is {batch_acc}")

    for x, y in valid_data:
        y_hat = model(x)
        valid_accuracy.update(y_hat, y)

    # total accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()

    # total accuracy over all validation batches
    total_valid_accuracy = valid_accuracy.compute()

    print(f"Training acc for epoch {epoch}: {total_train_accuracy}")
    print(f"Validation acc for epoch {epoch}: {total_valid_accuracy}")

    # Reset metric states after each epoch
    train_accuracy.reset()
    valid_accuracy.reset()
```


# 自定义指标
如果想使用一个尚不支持的指标，可以使用`TorchMetrics`的`API`来实现自定义指标，只需将`torchmetrics.Metric`子类化并实现以下方法：
- 实现`__init__`方法，在这里为每一个指标计算所需的内部状态调用`self.add_state`；
- 实现`update`方法，在这里进行更新指标状态所需的逻辑；
- 实现`compute`方法，在这里进行最终的指标计算。

## RMSE例子
以[均方根误差(RMSE, Root mean squared error)](https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE)为例，来看怎样自定义指标。
均方根误差的计算公式为：
$$
RMSE=\sqrt{\frac{1}{N}\sum_{n=1}^{N}\left ( \widehat{y}_i-y_i \right )^2}
$$
为了正确计算`RMSE`，我们需要两个指标状态：`sum_squared_error`来跟踪目标$\widehat{y}$和预测$y$之间的平方误差；`n_observations`来统计我们进行了多少次观测。
```python
from torchmetrics.metric import Metric

class MeanSquaredError(Metric):

    def __init__(self):
        super().__init__()
		# 添加状态，dist_reduce_fx指定了用来在多进程之间聚合状态所用的函数
        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # 更新状态
        self.sum_squared_error += torch.sum((preds-target)**2)
        self.n_observations += preds.numel()

    def compute(self):
        """Computes mean squared error over state."""
        return torch.sqrt(self.sum_squared_error/self.n_observations)
```

关于实现自定义指标的实际例子和更多信息，看[这个页面](https://torchmetrics.readthedocs.io/en/latest/pages/implement.html)。

# 指标运算
`TorchMetrics`支持大多数`Python`内置的算术、逻辑和位操作的运算符。
比如：
```python
first_metric = MyFirstMetric()
second_metric = MySecondMetric()

new_metric = first_metric + second_metric
```
这种运算模式可以适用于以下运算符（`a`是指标，`b`可以是指标、张量、整数或浮点数）：
- 加法（`a` `+` `b`)
- 按位与(`a` `&` `b`)
- 等价(`a` `==` `b`)
- 向下取整除`floor division` (`a` `//` `b`)
- 大于等于 (`a` `>=` `b`)
- 大于 (`a` `>` `b`)
- 小于等于 (`a` `<=` `b`)
- 小于 (`a` `<` `b`)
- 矩阵乘法 (`a` `@` `b`)
- 取模（`Modulo`，即取余）(`a` `%` `b`)
- 乘法 (`a` `*` `b`)
- 不等于 (`a` `!=` `b`)
- 按位或 (`a` `|` `b`)
- 乘方 (`a` `**` `b`)
- 减法 (`a` `-` `b`)
- 除法 (`a` `/` `b`)
- 按位异或 (`a` `^` `b`)
- 绝对值 (`abs(a)`)
- 取反 (`~a`)
- 负值 (`neg(a)`)
- 正值 (`pos(a)`)
- 索引 (`a[0]`)

# 指标集合
在很多情况下，用多个指标来评估模型的输出是很有好处的。在这种情况下，`MetricCollection`类可能会派上用场。它接受一连串的指标，并将这些指标包装成一个可调用的指标类，其接口与任一单一指标相同。
比如：
```python
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
metric_collection = MetricCollection([
    Accuracy(),
    Precision(num_classes=3, average='macro'),
    Recall(num_classes=3, average='macro')
])
print(metric_collection(preds, target))
```
输出为：
```python
{'Accuracy': tensor(0.1250),
 'Precision': tensor(0.0667),
 'Recall': tensor(0.1111)}
```
使用`MetricCollection`对象的另一个好处是，它将自动尝试通过寻找共享相同基础指标状态的指标组来减少所需的计算。如果找到了这样的指标组，实际上只有其中一个指标被更新，而更新的状态将被广播给组内的其他指标。在上面的例子中，与禁用该功能相比，这将导致计算成本降低2-3倍。然而，这种速度的提高伴随着前期的固定成本，即在第一次更新后必须确定状态组。这个开销可能会大大高于在很低的步数（大约100步）下获得的速度提升，但仍然会导致超过这个步数的整体速度提升。如果事先知道分组，也可以手动设置，以避免动态搜索的这种额外成本。关于这个主题的更多信息，请看该类文档中的`compute_groups`参数。
# 指标可微性
如果在指标计算中涉及的所有计算都是可微的，那么该指标就支持反向传播。所有的类形式的指标都有一个属性`is_differentiable`，它指明该指标是否是可微的。
然而，请注意，一旦缓存的状态从计算图中分离出来，它就不能被反向传播。如果不分离的话就意味着每次更新调用都要存储计算图，这可能会导致内存不足的错误。具体到实际操作时，意味着：
```python
MyMetric.is_differentiable  # returns True if metric is differentiable
metric = MyMetric()
val = metric(pred, target)  # this value can be back-propagated
val = metric.compute()  # this value cannot be back-propagated
```
# 超参数优化
如果想直接优化一个指标，它需要支持反向传播（见上节）。然而，如果只是想对使用的指标进行超参数调整，此时如果不确定该指标应该被最大化还是最小化，那么可以参考指标类的`higher_is_better`属性：
```python
# returns True because accuracy is optimal when it is maximized
torchmetrics.Accuracy.higher_is_better

# returns False because the mean squared error is optimal when it is minimized
torchmetrics.MeanSquaredError.higher_is_better
```

# 常用指标
## 回归问题
### MSE
均方误差`MSE`，即`mean squared error`，计算公式为：
$$
\text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2
$$
其中，$y$是目标值的张量，而$\hat{y}$是预测值的张量。
示例代码：
```python
import torch
from torchmetrics import MeanSquaredError

target = torch.tensor([0., 1, 2, 3])
preds = torch.tensor([0., 1, 2, 2])

mean_squared_error = MeanSquaredError()
mean_squared_error(preds, target)
```
输出为：
```python
tensor(0.2500)
```
### MSLE
均方对数误差`MSLE`，即`mean squared logarithmic error`，计算公式为：
$$
\text{MSLE} = \frac{1}{N}\sum_i^N (\log_e(1 + y_i) - \log_e(1 + \hat{y_i}))^2
$$
示例代码：
```python
from torchmetrics import MeanSquaredLogError
target = torch.tensor([2.5, 5, 4, 8])
preds = torch.tensor([3, 5, 2.5, 7])
mean_squared_log_error = MeanSquaredLogError()
mean_squared_log_error(preds, target)
```
输出为：
```python
tensor(0.0397)
```
### MAE
平均绝对误差`MAE`，即`Mean Absolute Error`，计算公式为：
$$
\text{MAE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} |
$$
示例代码：
```python
import torch
from torchmetrics import MeanAbsoluteError

target = torch.tensor([3.0, -0.5, 2.0, 7.0])
preds = torch.tensor([2.5, 0.0, 2.0, 8.0])

mean_absolute_error = MeanAbsoluteError()
mean_absolute_error(preds, target)
```
输出为：
```python
tensor(0.5000)
```