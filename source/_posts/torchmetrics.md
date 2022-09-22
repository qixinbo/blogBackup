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

### MAPE
平均绝对百分比误差`MAPE`，即`Mean Absolute Percentage Error`，计算公式为：
$$
\text{MAPE} = \frac{1}{n}\sum_{i=1}^n\frac{|   y_i - \hat{y_i} |}{\max(\epsilon, | y_i |)}
$$
示例代码：
```python
from torchmetrics import MeanAbsolutePercentageError
target = torch.tensor([1, 10, 1e6])
preds = torch.tensor([0.9, 15, 1.2e6])
mean_abs_percentage_error = MeanAbsolutePercentageError()
mean_abs_percentage_error(preds, target)
>>> tensor(0.2667)
```
### WMAPE
加权平均绝对百分比误差`WMAPE`，即`Weighted Mean Absolute Percentage Error`，计算公式为：
$$
\text{WMAPE} = \frac{\sum_{t=1}^n | y_t - \hat{y}_t | }{\sum_{t=1}^n |y_t| }
$$
其与`MAPE`的区别可以参考[这篇文章](https://blog.csdn.net/zpf336/article/details/104374570)。
示例代码：
```python
from torchmetrics import WeightedMeanAbsolutePercentageError
target = torch.tensor([1, 10, 1e6])
preds = torch.tensor([0.9, 15, 1.2e6])
mean_abs_percentage_error = WeightedMeanAbsolutePercentageError()
mean_abs_percentage_error(preds, target)
>>> tensor(0.2000)
```
### SMAPE
对称平均绝对百分比误差`SMAPE`，即`symmetric mean absolute percentage error`，计算公式为：
$$
\text{SMAPE} = \frac{2}{n}\sum_1^n max(\frac{|   y_i - \hat{y_i} |}{| y_i | + | \hat{y_i} |, \epsilon})
$$
示例代码：
```python
from torchmetrics import SymmetricMeanAbsolutePercentageError
target = tensor([1, 10, 1e6])
preds = tensor([0.9, 15, 1.2e6])
smape = SymmetricMeanAbsolutePercentageError()
smape(preds, target)
>>> tensor(0.2290)
```
### 余弦相似度
余弦相似度，即`Cosine Similarity`，其含义可以参考其[维基百科](https://zh.wikipedia.org/zh-cn/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7)：
> 余弦相似性通过测量两个向量的夹角的余弦值来度量它们之间的相似性。0度角的余弦值是1，而其他任何角度的余弦值都不大于1；并且其最小值是-1。从而两个向量之间的角度的余弦值确定两个向量是否大致指向相同的方向。两个向量有相同的指向时，余弦相似度的值为1；两个向量夹角为90°时，余弦相似度的值为0；两个向量指向完全相反的方向时，余弦相似度的值为-1。这结果是与向量的长度无关的，仅仅与向量的指向方向相关。余弦相似度通常用于正空间，因此给出的值为0到1之间。
> 注意这上下界对任何维度的向量空间中都适用，而且余弦相似性最常用于高维正空间。例如在信息检索中，每个词项被赋予不同的维度，而一个文档由一个向量表示，其各个维度上的值对应于该词项在文档中出现的频率。余弦相似度因此可以给出两篇文档在其主题方面的相似度。
> 另外，它通常用于文本挖掘中的文件比较。此外，在数据挖掘领域中，会用到它来度量集群内部的凝聚力。

计算公式为：
$$
cos_{sim}(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||} = \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2}\sqrt{\sum_{i=1}^n y_i^2}}
$$
具体计算过程可以参考[该文章](https://clay-atlas.com/blog/2020/03/26/cosine-similarity-text-count/)。

示例代码：
```python
from torchmetrics import CosineSimilarity
target = torch.tensor([[0, 1], [1, 1]])
preds = torch.tensor([[0, 1], [0, 1]])
# reduction: how to reduce over the batch dimension using 'sum', 'mean' or 'none' (taking the individual scores)
cosine_similarity = CosineSimilarity(reduction = 'mean')
cosine_similarity(preds, target)
>>> tensor(0.8536)
```
### 可解释方差
可解释方差，即`explained variance`，解释可参考[维基百科](https://zh.wikipedia.org/zh-cn/%E5%8F%AF%E8%A7%A3%E9%87%8A%E5%8F%98%E5%BC%82)，计算公式为：
$$
\text{ExplainedVariance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
$$
示例代码为：
```python
from torchmetrics import ExplainedVariance
target = torch.tensor([3, -0.5, 2, 7])
preds = torch.tensor([2.5, 0.0, 2, 8])
# multioutput defines aggregation in the case of multiple output scores. 
explained_variance = ExplainedVariance(multioutput='uniform_average')
explained_variance(preds, target)
>>> tensor(0.9572)
```
### KL散度
KL散度，即`KL divergence`，解释可见[这里](https://zhuanlan.zhihu.com/p/39682125)，计算公式为：
$$
D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x) \log\frac{P(x)}{Q(x)}
$$
示例代码为：
```python
from torchmetrics import KLDivergence
p = torch.tensor([[0.36, 0.48, 0.16]])
q = torch.tensor([[1/3, 1/3, 1/3]])
kl_divergence = KLDivergence()
kl_divergence(p, q)
>>> tensor(0.0853)
```
### Tweedie偏差分数
Tweedie偏差分数，即`Tweedie Deviance Score`，可参考[这里的解释](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html)，计算公式为：
$$
        deviance\_score(\hat{y},y) =
        \begin{cases}
        (\hat{y} - y)^2, & \text{for }p=0\\
        2 * (y * log(\frac{y}{\hat{y}}) + \hat{y} - y),  & \text{for }p=1\\
        2 * (log(\frac{\hat{y}}{y}) + \frac{y}{\hat{y}} - 1),  & \text{for }p=2\\
        2 * (\frac{(max(y,0))^{2 - p}}{(1 - p)(2 - p)} - \frac{y(\hat{y})^{1 - p}}{1 - p} + \frac{(
            \hat{y})^{2 - p}}{2 - p}), & \text{otherwise}
        \end{cases}
$$
示例代码为：
```python
from torchmetrics import TweedieDevianceScore
targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
preds = torch.tensor([4.0, 3.0, 2.0, 1.0])
deviance_score = TweedieDevianceScore(power=2)
deviance_score(preds, targets)
>>> tensor(1.2083)
```
### Pearson相关性系数
Pearson相关性系数，即`Pearson Correlation Coefficient`，用于度量两组数据的变量X和Y之间的线性相关的程度，具体解释见[这里](https://zh.wikipedia.org/zh-cn/%E7%9A%AE%E5%B0%94%E9%80%8A%E7%A7%AF%E7%9F%A9%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0)，计算公式为：
$$
P_{corr}(x,y) = \frac{cov(x,y)}{\sigma_x \sigma_y}
$$
示例代码为：
```python
from torchmetrics import PearsonCorrCoef
target = torch.tensor([3, -0.5, 2, 7])
preds = torch.tensor([2.5, 0.0, 2, 8])
pearson = PearsonCorrCoef()
pearson(preds, target)
>>> tensor(0.9849)
```
### Spearman相关性系数
Spearman相关性系数，即`Spearman's rank correlation coefficient`，斯皮尔曼相关系数被定义成等级变量之间的皮尔逊相关系数，具体解释见[这里](https://zh.wikipedia.org/zh-cn/%E6%96%AF%E7%9A%AE%E5%B0%94%E6%9B%BC%E7%AD%89%E7%BA%A7%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0)，计算公式为：
$$
r_s = = \frac{cov(rg_x, rg_y)}{\sigma_{rg_x} * \sigma_{rg_y}}
$$
示例代码为：
```python
from torchmetrics import SpearmanCorrCoef
target = torch.tensor([3, -0.5, 2, 7])
preds = torch.tensor([2.5, 0.0, 2, 8])
spearman = SpearmanCorrCoef()
spearman(preds, target)
>>> tensor(1.0000)
```
### 决定系数
决定系数，即$R^2$、`Coefficient of determination`，在统计学中用于度量因变量的变异中可由自变量解释部分所占的比例，以此来判断回归模型的解释力。具体解释见[这里](https://zh.wikipedia.org/zh-cn/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0)。计算公式为：
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$
假设一数据集包括$y_1,...,y_n$共$n$个观察值，相对应的模型预测值分别为$f_1,...,f_n$。定义残差$e_i = y_i − f_i$，平均观察值为：
$$
\overline{y}=\frac{1}{n}\sum_{i=1}^n y_i
$$
于是得到总平方和为：
$$
SS_{tot}=\sum_i (y_i - \bar{y})^2
$$
残差平方和为：
$$
SS_{res}=\sum_i (y_i - \hat{y}_i)^2
$$
示例代码为：
```python
from torchmetrics import R2Score
target = torch.tensor([3, -0.5, 2, 7])
preds = torch.tensor([2.5, 0.0, 2, 8])
r2score = R2Score()
r2score(preds, target)
>>> tensor(0.9486)
```

## 分类问题
查看用于分类问题的各种指标之前，先看一下分类问题中指标计算时所需要的输入（包括预测值`predictions`和目标值`targets`）的形状和数据类型，其中`N`是批处理大小，`C`是类别数目。
一些背景资料：
[Logit](https://en.wikipedia.org/wiki/Logit)
[What does the logit value actually mean?](https://stats.stackexchange.com/questions/52825/what-does-the-logit-value-actually-mean)
[[原创] 用人话解释机器学习中的Logistic Regression（逻辑回归）](https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84logistic-regression%EF%BC%88%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%EF%BC%89/)
[【机器学习】逻辑回归（非常详细）](https://zhuanlan.zhihu.com/p/74874291)
[二分类、多分类、多标签分类的基础、原理、算法和工具](https://zhuanlan.zhihu.com/p/270458779)
[多分类模型Accuracy, Precision, Recall和F1-score的超级无敌深入探讨](https://zhuanlan.zhihu.com/p/147663370)

```python
| Type                                 | preds shape | preds dtype | target shape | target dtype |
| -----------                          | ----------- | ----------- | ----------- |----------- |
| 二分类                               | (N,)       |  float | (N,) | 二值，即0或1 |
| 多分类                               | (N,)        | int | (N,) | int |
| 带概率`p`或对数几率`logit`（$\text{logit}=ln\frac{p}{1-p}$）的多分类 | (N,C)        | float | (N,) | int |
| 多标签                               | (N,...)    | float | (N,...) | 二值 |
| 多维多分类                           | (N,...)   | int | (N,...) | int |
| 带概率`p`或对数几率`logit`的多维多分类 | (N,C,...)  | float | (N,...) | int |
```
以下是一些例子：
```python
# Binary inputs
binary_preds  = torch.tensor([0.6, 0.1, 0.9])
binary_target = torch.tensor([1, 0, 2])

# Multi-class inputs
mc_preds  = torch.tensor([0, 2, 1])
mc_target = torch.tensor([0, 1, 2])

# Multi-class inputs with probabilities
mc_preds_probs  = torch.tensor([[0.8, 0.2, 0], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
mc_target_probs = torch.tensor([0, 1, 2])

# Multi-label inputs
ml_preds  = torch.tensor([[0.2, 0.8, 0.9], [0.5, 0.6, 0.1], [0.3, 0.1, 0.1]])
ml_target = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 0]])
```
在某些情况下，可能有看起来是（多维）多类的输入，但实际上是二分类/多标签的输入——例如，如果预测和目标都是整型张量。或者相反的情形，想把二分类/多标签输入当作目前只显示二类的（多维）多类输入。
对于这些情况，在设定指标时，需要使用`multiclass`参数。
以`StatScores`指标为例看一下怎样使用这个参数。
首先，考虑有2个类别的标签预测的情况：
```python
from torchmetrics.functional import stat_scores

# These inputs are supposed to be binary, but appear as multi-class
preds  = torch.tensor([0, 1, 0])
target = torch.tensor([1, 1, 0])
```
由下面可以看出，默认是处理成“多分类”问题：
```python
stat_scores(preds, target, reduce='macro', num_classes=2)
>>> tensor([[1, 1, 1, 0, 1],
        [1, 0, 1, 1, 2]])
```
此时需要设定`multiclass=False`来使其处理成二分类问题。
```python
stat_scores(preds, target, reduce='macro', num_classes=1, multiclass=False)
>>> tensor([[1, 0, 1, 1, 2]])
```
上述处理方式跟事先将预测值的类型转为`float`的效果是相同的：
```python
stat_scores(preds.float(), target, reduce='macro', num_classes=1)
>>> tensor([[1, 0, 1, 1, 2]])
```
接下来考虑相反的情形：看起来像是二分类（因为预测值是`float`），但实际想处理成当前只有2类的多分类问题：
```python
preds  = torch.tensor([0.2, 0.7, 0.3])
target = torch.tensor([1, 1, 0])
```
通过设置`multiclass=True`来实现正确的效果：
```python
stat_scores(preds, target, reduce='macro', num_classes=1)
>>> tensor([[1, 0, 1, 1, 2]])
stat_scores(preds, target, reduce='macro', num_classes=2, multiclass=True)
>>> tensor([[1, 1, 1, 0, 1],
        [1, 0, 1, 1, 2]])
```
### 混淆矩阵
混淆矩阵，即`Confusion Matrix`，矩阵的每一列代表一个类的实例预测，而每一行表示一个实际的类的实例。之所以如此命名，是因为通过这个矩阵可以方便地看出机器是否将两个不同的类混淆了（比如说把一个类错当成了另一个）。 具体解释见[维基百科](https://en.wikipedia.org/wiki/Confusion_matrix)（中文版示例有些小错误）。
示例代码有：
（1）二分类：
```python
from torchmetrics import ConfusionMatrix
target = torch.tensor([1, 1, 0, 0])
preds = torch.tensor([0, 1, 0, 0])
confmat = ConfusionMatrix(num_classes=2)
confmat(preds, target)
>>> tensor([[2, 0],
>>>         [1, 1]])
```
（2）多分类：
```python
target = torch.tensor([2, 1, 0, 0])
preds = torch.tensor([2, 1, 0, 1])
confmat = ConfusionMatrix(num_classes=3)
confmat(preds, target)
>>> tensor([[1, 1, 0],
>>>         [0, 1, 0],
>>>         [0, 0, 1]])
```
（3）多标签：
```python
target = torch.tensor([[0, 1, 0], [1, 0, 1]])
preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
confmat = ConfusionMatrix(num_classes=3, multilabel=True)
confmat(preds, target)
>>> tensor([[[1, 0],
>>>          [0, 1]],
>>> 
>>>         [[1, 0],
>>>          [1, 0]],
>>> 
>>>         [[0, 1],
>>>          [0, 1]]])
```
### 准确率
准确率，即`Accuracy`，表明了分类正确的概率，计算公式为：
$$
\text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)
$$
如果用混淆矩阵中的数据来表达就是：
$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
$$
对于带概率或对数几率的多分类和多维多分类数据，参数`top_k`可以将该指标泛化为为`Top-K`准确度指标：对于每个样本，考虑前K个概率或对数几率最高的类别来判断是否找到了正确的标签。
对于多标签和多维多分类输入，该指标默认计算 "全局 "准确度，即单独计算所有标签或子样本。这可以通过设置`subset_accuracy=True`来改变为子集准确性（这需要样本中的所有标签或子样本都被正确预测）。
示例代码有：
（1）二分类：
```python
import torch
from torchmetrics import Accuracy
target = torch.tensor([0, 1, 2, 3])
preds = torch.tensor([0, 2, 1, 3])
accuracy = Accuracy()
accuracy(preds, target)
>>> tensor(0.5000)
```
（2）带概率的Top-K多分类：
```python
target = torch.tensor([0, 1, 2])
preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
accuracy = Accuracy(top_k=2)
accuracy(preds, target)
>>> tensor(0.6667)
```

### 精度
精度，即`Precision`，计算公式为：
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$
示例代码为：
```python
from torchmetrics import Precision
preds  = torch.tensor([2, 0, 2, 1])
target = torch.tensor([1, 1, 2, 0])
precision = Precision(average='macro', num_classes=3)
precision(preds, target)
>>> tensor(0.1667)
precision = Precision(average='micro')
precision(preds, target)
>>> tensor(0.2500)
```


### AUC
`AUC`，即`Area Under the Curve (AUC)`，torchmetrics提供了使用梯形公式`trapezoidal rule`计算某条曲线下的面积的方法，计算公式为：
$$
\int_{a}^{b}f(x)dx \approx (b-a)\frac{f(a)+f(b)}{2}
$$
注意，在离散的点形成的向量上进行梯形公式计算，其实际是每两点之间就计算一次，详见[`torch.trapezoid`函数](https://pytorch.org/docs/stable/generated/torch.trapezoid.html#torch.trapezoid)。
示例代码为：
```python
from torchmetrics.functional import auc
x = torch.tensor([0, 1, 2, 3])
y = torch.tensor([0, 1, 2, 2])
auc(x, y)
>>> tensor(4.)
```
### ROC
`ROC`，即`Receiver Operating Characteristic`，接收者操作特征曲线，是一种坐标图式的分析工具，具体解释可参见[维基百科](https://zh.wikipedia.org/zh-cn/ROC%E6%9B%B2%E7%BA%BF)。
`ROC`空间将伪阳性率（`FPR`、在所有实际为阴性的样本中，被错误地判断为阳性之比率）定义为 `X` 轴，真阳性率（`TPR`、在所有实际为阳性的样本中，被正确地判断为阳性之比率）定义为 `Y` 轴。
$$
TPR = \frac{TP}{TP+FN} \\
FPR = \frac{FP}{TN+FP}
$$
将同一模型每个阈值 的 (FPR, TPR) 坐标都画在ROC空间里，就成为特定模型的ROC曲线。
示例代码有：
（1）二分类：
```python
from torchmetrics import ROC
pred = torch.tensor([0, 1, 2, 3])
target = torch.tensor([0, 1, 1, 1])
roc = ROC(pos_label=1)
fpr, tpr, thresholds = roc(pred, target)
fpr
>>> tensor([0., 0., 0., 0., 1.])
tpr
>>> tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
thresholds
>>> tensor([4, 3, 2, 1, 0])
```
（2）多分类：
```python
pred = torch.tensor([[0.75, 0.05, 0.05, 0.05],
                     [0.05, 0.75, 0.05, 0.05],
                     [0.05, 0.05, 0.75, 0.05],
                     [0.05, 0.05, 0.05, 0.75]])
target = torch.tensor([0, 1, 3, 2])
roc = ROC(num_classes=4)
fpr, tpr, thresholds = roc(pred, target)
fpr
>>> [tensor([0., 0., 1.]),
>>>  tensor([0., 0., 1.]),
>>>  tensor([0.0000, 0.3333, 1.0000]),
>>>  tensor([0.0000, 0.3333, 1.0000])]
tpr
>>> [tensor([0., 1., 1.]),
>>>  tensor([0., 1., 1.]),
>>>  tensor([0., 0., 1.]),
>>>  tensor([0., 0., 1.])]
thresholds
>>> [tensor([1.7500, 0.7500, 0.0500]),
>>>  tensor([1.7500, 0.7500, 0.0500]),
>>>  tensor([1.7500, 0.7500, 0.0500]),
>>>  tensor([1.7500, 0.7500, 0.0500])]
```
（3）多标签：
```python
pred = torch.tensor([[0.8191, 0.3680, 0.1138],
                     [0.3584, 0.7576, 0.1183],
                     [0.2286, 0.3468, 0.1338],
                     [0.8603, 0.0745, 0.1837]])
target = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
roc = ROC(num_classes=3, pos_label=1)
fpr, tpr, thresholds = roc(pred, target)
fpr
>>> [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
>>>  tensor([0., 0., 0., 1., 1.]),
>>>  tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
tpr
>>> [tensor([0., 0., 1., 1., 1.]),
>>>  tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]),
>>>  tensor([0., 1., 1., 1., 1.])]
thresholds
>>> [tensor([1.8603, 0.8603, 0.8191, 0.3584, 0.2286]),
>>>  tensor([1.7576, 0.7576, 0.3680, 0.3468, 0.0745]),
>>>  tensor([1.1837, 0.1837, 0.1338, 0.1183, 0.1138])]
```
### AUC ROC
`AUC ROC`，即`Area under the Curve of ROC`，即`ROC`曲线下方的面积，具体解释可参见[维基百科](https://zh.wikipedia.org/zh-cn/ROC%E6%9B%B2%E7%BA%BF)，简单说：`AUC`值越大的分类器，正确率越高。
示例代码有：
（1）二分类：
```python
from torchmetrics import AUROC
preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
target = torch.tensor([0, 0, 1, 1, 1])
auroc = AUROC(pos_label=1)
auroc(preds, target)
>>> tensor(0.5000)
```
（2）多分类：
```python
preds = torch.tensor([[0.90, 0.05, 0.05],
                      [0.05, 0.90, 0.05],
                      [0.05, 0.05, 0.90],
                      [0.85, 0.05, 0.10],
                      [0.10, 0.10, 0.80]])
target = torch.tensor([0, 1, 1, 2, 2])
auroc = AUROC(num_classes=3)
auroc(preds, target)
>>> tensor(0.7778)
```