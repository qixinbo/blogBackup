---
title: 轻量级PyTorch通用训练模板pytorch-accelerated解析：1 -- API之Trainer
tags: [PyTorch]
categories: machine learning 
date: 2022-5-29
---

# 简介
从这一节开始，详细看一下`pytorch-accelerated`的`API`。
本文是对`Trainer`的`API`的解析。
# Trainer概览
训练器`Trainer`用来封装特定任务的整个训练循环，将模型、损失函数和优化器结合在一起，并为训练过程的每一步提供执行行为规范。
`Trainer`的实现是这样的：它提供了训练部分的（可重复的）实现，这些部分在被定义后很少发生变化——比如创建数据加载器，或如何将一批数据送入模型——同时与可能发生变化的部分保持解耦，比如模型、数据集、损失函数和优化器。
```python
class Trainer:
    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        callbacks=DEFAULT_CALLBACKS,
        run_history=None,
    ):
```
它的初始化函数中输入的参数有：
- `model`：用来训练的神经网络模型，是`nn.Module`的子类
- `loss_func`：用来训练模型的损失函数
- `optimizer`：用来更新模型参数的优化器
- `callbacks`：当训练进行时调用的回调函数列表`callbacks`，如果没有提供该列表，则使用默认的回调函数，包括`MoveModulesToDeviceCallback`、`TerminateOnNaNCallback`、`PrintProgressCallback`、`ProgressBarCallback`、`LogMetricsCallback`。
- `run_history`：`RunHistory`的子类的一个实例，如果不提供的话（即`None`），则会新建一个`InMemoryRunHistory`这个子类的实例。

# 训练模型
`Trainer`的主要入口函数是`train()`方法，其用来启动模型训练（如果提供了验证集，那么会同时包括训练和验证评估）。
```python
def train(
    self,
    train_dataset,
    num_epochs,
    eval_dataset=None,
    per_device_batch_size=8,
    max_num_train_steps=None,
    gradient_accumulation_steps=1,
    gradient_clip_value=None,
    create_scheduler_fn: Callable = None,
    train_dataloader_kwargs: dict = None,
    eval_dataloader_kwargs: dict = None,
    reset_run_history=True,
    collate_fn=None,
):
```
输入参数包括：
- `train_dataset`：训练集
- `num_epochs`：训练的迭代次数
- `eval_dataset`：验证集，如果不提供的话，则跳过模型的验证评估环节
- `per_device_batch_size`：每个设备上的批处理大小
- `max_num_train_steps`：最大迭代步数，如果提供该参数的话，会覆盖掉`num_epochs`
- `gradient_accumulation_steps`：对特定步数进行梯度累加来模拟一个更大的批处理大小，默认该参数为`1`
- `gradient_clip_value`：如果指定的话，模型参数的梯度将被修剪到`[-gradient_clip_value, gradient_clip_value]`之间。
- `create_scheduler_fn`：由于优化器需要在训练前就准备好，为了能够在优化器中使用学习率调度器，必须向`create_scheduler_fn`参数提供一个工厂函数。该工厂函数必须是一个接受优化器`optimizer`作为唯一参数的函数，并返回一个学习率调度器的实例。注意，这里不是传递一个学习率调度器的实例，而是传递一个工厂函数能返回这样的实例。
- `train_dataloader_kwargs`：用来传递给训练集数据加载器的构造函数的关键字参数字典，详情参见[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。
- `eval_dataloader_kwargs`：用来传递给验证集数据加载器的构造函数的关键字参数字典，详情参见[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。
- `reset_run_history`：重置`Trainer`保留的之前的运行历史
- `collate_fn`：训练集数据加载器和验证集数据加载器所使用的`collate_fn`函数

## 使用学习率调度器
由于`Pytorch`学习率调度器的调用方式不尽相同，为了实现最大的灵活性，`PyTorch-accelerated`的`Trainer`期望在每次优化器更新后都默认调用一个给定的调度器。
请注意，由于优化器和数据加载器需要在训练前进行准备，为了使用学习率调度器，必须向`train()`提供一个工厂函数作为`create_scheduler_fn`参数。这必须是一个接受优化器作为单一参数的函数，并返回一个学习率调度器的实例。再次注意，这里不是传递一个学习率调度器的实例，而是传递一个工厂函数能返回这样的实例。
创建调度器工厂函数的一个简单方法是使用`functools.partial()`，比如：
```python
from functools import Partial
from torch.optim import lr_scheduler

create_scheduler_fn = partial(lr_scheduler.StepLR, step_size=7, gamma=0.1)
```
特别注意：`Trainer`在每个批次之后都会调用一次所提供的调度器。这可能会导致意想不到的结果，因为一些`PyTorch`调度器预期是只在每一个`epoch`后进行调用。
例如，在上面的例子中，我们想要学习率在每个批次都会被乘以`0.1`。但是由于这个特殊的调度器被设计为每一个`epoch`调用一次，因此这不是我们想要的行为，此时可以通过下面的方法来解决这个问题：
```python
from functools import Partial
from torch.optim import lr_scheduler
from pytorch_accelerated import TrainerPlaceholderValues

epochs_step_size = 7
create_scheduler_fn = partial(
    lr_scheduler.StepLR,
    step_size=TrainerPlaceHolderValues.NUM_UPDATE_STEPS_PER_EPOCH * epochs_step_size
)
```
这里，为了确定每个`epoch`的`steps`的数值，使用了`TrainerPlaceholderValues`占位符，下面将介绍。

### 使用TrainerPlaceHolderValues
一些学习率调度器需要一些信息，比如在训练运行期间的总步数`steps`。由于在创建训练数据加载器之前无法获得这些信息——这将作为`train()`方法的一部分来完成——在这种情况下可以使用一个占位符值，如下所示：
```pytho
from functools import Partial
from pytorch_accelerated import TrainerPlaceholderValues
from torch.optim.lr_scheduler import OneCycleLR

create_scheduler_fn = partial(
            OneCycleLR,
            max_lr=config.lr,
            epochs=TrainerPlaceholderValues.NUM_EPOCHS,
            steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        )
```
这些占位符将由`trainer`在训练期间用正确的数值替换。
可用的占位符包括：
- `NUM_EPOCHS`
- `NUM_UPDATE_STEPS_PER_EPOCH`
- `TRAIN_DATALOADER_LEN`
- `EVAL_DATALOADER_LEN`

或者，可以通过重载`Trainer`的`create_scheduler()`方法来获得同样的结果。

### 使用PyTorch-accelerated内置的调度器
`PyTorch-accelerated`包括一些调度器的实现，这些调度器具有与`PyTorch`调度器相同的接口，还有一些基类可以轻松创建自定义调度器；这些将在后面的`Schedulers`一节中详细讨论。
这些调度器的实现有一个替代的构造函数，可以直接作为`create_scheduler_fn`参数传递给`train()``，如下所示：
```python
from pytorch_accelerated.schedulers import CosineLrScheduler
trainer.train(
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(num_warmup_epochs=5,
                                                                  warmup_starting_lr=1e-6,
                                                                  num_cooldown_epochs=5),
        )
```
### 使用timm调度器
`timm`中包含的调度器与原生的`PyTorch`调度器有不同的接口，所以默认情况下不能与`Trainer`一起使用。
`PyTorch-accelerated`包含了一个替代的`Trainer`，即`TrainerWithTimmScheduler`，它与`timm`的调度器兼容；`timm`调度器应作为工厂函数传递给这个训练器，方法与上述相同。


# 评估模型
一旦模型被训练好，或者从检查点`checkpoint`加载，训练器`Trainer`也可以用于评估，这包括使用`Trainer`的评估循环的逻辑，在给定的数据集上运行一个`epoch`。
```python
def evaluate(
    self,
    dataset=None,
    per_device_batch_size=8,
    dataloader_kwargs: dict = None,
    collate_fn=None,
):
```
它的参数有：
- `dataset`：要评估的数据集
- `per_device_batch_size`：每个设备上的批处理大小
- `dataloader_kwargs`：用来传递给数据加载器的构造函数的关键字参数字典，详情参见[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。
- `collate_fn`：数据加载器所使用的`collate_fn`函数。

注意：启动一个评估后，会重置`Trainer`的运行历史。
另外，在分布式评估过程中，如果`per_device_batch_size * 使用的进程数`不能精确划分数据集，并且`drop_last=False`没有传给`dataloader_kwargs`，那么`dataloader`将在耗尽批次的进程中从头重复。在计算指标时应考虑到这一点。

# 效用函数
`Trainer`提供了很多效用函数供使用：
## 保存检查点
```python
def save_checkpoint(
    self, save_path, checkpoint_kwargs=None, save_optimizer=True, save_per_node=True
):
```
在一个检查点`checkpoint`文件中保存模型、优化器及其他指定的对象。
输入参数有：
- `save_path`：存储检查点的路径，应该以`.pt`结尾
- `checkpoint_kwargs`：在检查点中应该包含的其他对象
- `save_optimizer`：指定是否保存优化器
- `save_per_node`：指定是否每个机器保存检查点，如果`False`，则仅在`0`号进程中保存。默认是`True`。

## 加载检查点
```python
def load_checkpoint(self, checkpoint_path, load_optimizer=True):
```
从检查点文件中加载模型和优化器。
参数有：
- `checkpoint_path`：要加载的检查点文件的路径
- `load_optimizer`：如果检查点文件中包含了优化器，指定是否加载它

## 打印
```python
def print(self, *args, **kwargs):
```
用来替代原生的`print()`，以每个节点只打印一次。

## 聚合
```python
def gather(self, tensor):
```
收集所有进程的张量值，并在第一个维度上将其连接起来。在进行评估时，这对重新组合所有进程的预测值很有用。
注意：该聚合操作将会在所有进程中进行。

# 自定义Trainer行为
虽然`Trainer`在简单的用例中能开箱即用，同时也鼓励对`Trainer`进行子类化并重载其方法。
以动词为前缀的方法，如`create`或`calculate`，期望返回一个值，所有其他方法都用于设置内部状态（如`optimizer.step()`）。

## 与构建相关的方法
（1）构建训练集数据加载器
```python
Trainer.create_train_dataloader(batch_size: int, train_dl_kwargs: Optional[dict] = None) → Iterable
```
创建一个在训练中使用的数据加载器，该数据加载器接收通过`Trainer`传入的`train_dataset`和`collate`函数。
参数有：
- `batch_size`：在每个设备上的批处理大小
- `train_dl_kwargs`：用来传递给数据加载器的构造函数的关键字参数字典，详情参见[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

注意：如果没有传入`train_dl_kwargs`参数，则使用`Trainer.get_default_train_dl_kwargs()`返回的参数。如果在`train_dl_kwargs`中同样加入了`batch_size`这一属性，则这里的`batch_size`会覆盖掉前面的`batch_size`。

（2）获得默认训练集数据加载器参数
```python
Trainer.get_default_train_dl_kwargs(batch_size) → dict
```
参数为`batch_size`，返回值为训练集数据加载器的默认参数字典。
（3）构建验证集数据加载器
```python
Trainer.create_eval_dataloader(batch_size: int, eval_dl_kwargs: Optional[dict] = None) → Iterable
```
创建一个在评估验证中使用的数据加载器，该数据加载器接收通过`Trainer`传入的`eval_dataset`和`collate`函数。
参数有：
- `batch_size`：在每个设备上的批处理大小
- `eval_dl_kwargs`：用来传递给数据加载器的构造函数的关键字参数字典，详情参见[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。

注意：如果没有传入`eval_dl_kwargs`参数，则使用`Trainer.get_default_eval_dl_kwargs()`返回的参数。如果在`eval_dl_kwargs`中同样加入了`batch_size`这一属性，则这里的`batch_size`会覆盖掉前面的`batch_size`。

（4）获得默认验证集数据加载器参数
```python
Trainer.get_default_eval_dl_kwargs(batch_size) → dict
```
参数为`batch_size`，返回值为验证集数据加载器的默认参数字典。
（5）创建调度器
```python
Trainer.create_scheduler()
```
基于传给`Trainer`的`create_scheduler_fn`函数创建一个学习率调度器，返回值是一个学习率调度器的实例。

## 与训练相关的方法
（1）训练开始时会调用如下方法：
```python
Trainer.training_run_start()
```
（2）每一个`epoch`训练和评估完成后会调用如下方法：
```python
Trainer.training_run_epoch_end()
```
（3）训练结束后会调用如下方法：
```python
Trainer.training_run_end()
```
### 训练步相关方法
（1）每一训练`epoch`开始时调用如下方法：
```python
Trainer.train_epoch_start()
```
该方法的默认行为是调用`self.model.train()`。
（2）在每一训练`epoch`后计算训练损失：
```python
Trainer.calculate_train_batch_loss(batch) → dict
```
参数为
- `batch`：训练集数据加载器的输出

返回值为一个包含训练损失、模型输出和批处理大小的字典，必须包含这三个`keys`，也能包含其他的`keys`，这些额外的返回值可以通过`~callbacks.TrainerCallback.on_train_step_end`这个回调获得。
（3）反向传播步
```python
Trainer.backward_step(loss)
```
使用加速器对`calculate_train_batch_loss()`返回的损失值进行反向传播。如果启用了梯度累积，该损失会被`1/累积步数`所缩放。
（4）优化步
```python
Trainer.optimizer_step()
```
执行一个单一的优化步骤，并更新之前传递给`self.optimizer`的参数。
（5）学习率调度步
```python
Trainer.scheduler_step()
```
如果指定了学习率调度器`self.scheduler`，则执行一次调度步。
（6）梯度置0步
```python
Trainer.optimizer_zero_grad()
```
将所有优化后的`torch.Tensor`的梯度置为0。
（7）每一训练`epoch`结束时调用如下方法：
```python
Trainer.train_epoch_end()
```

### 验证步相关方法
（1）每一验证`epoch`开始时调用如下方法：
```python
Trainer.eval_epoch_start()
```
该方法的默认行为是调用`self.model.eval()`。
（2）在每一验证`epoch`后计算验证损失：
```python
Trainer.calculate_eval_batch_loss(batch) → dict
```
参数为
- `batch`：验证集数据加载器的输出

返回值为一个包含验证损失、模型输出和批处理大小的字典，必须包含这三个`keys`，也能包含其他的`keys`，这些额外的返回值可以通过`~callbacks.TrainerCallback.on_eval_step_end`这个回调获得。
（3）每一验证`epoch`结束时调用如下方法：
```python
Trainer.eval_epoch_end()
```

## 与评估相关的方法
（1）模型评估开始时会调用如下方法：
```python
Trainer.evaluation_run_start()
```
（2）评估结束后会调用如下方法：
```python
Trainer.evaluation_run_end()
```

## 内部方法
内部方法都是带着下划线前缀。
本着`Python`的精神，在训练器`Trainer`中没有什么是真正隐藏的。然而，必须小心，因为通过覆盖这些内部方法，会从根本上改变了`Trainer`的内部工作方式，这可能会产生意想不到的后果。当修改一个或多个内部方法时，用户有责任确保训练器继续按预期的方式工作。
### 内部构建
（1）创建[`accelerate.Accelerator`](https://huggingface.co/docs/accelerate/main/en/accelerator#accelerate.Accelerator)的实例：
```python
Trainer._create_accelerator()
```
该实例将用来管理训练过程，后面简称为加速器实例。
（2）封装模型、优化器和数据加载器
```python
Trainer._prepare_model_optimizer_and_dataloaders()
```
使用加速器实例将模型、优化器和数据加载器封装在训练所需的任意封装器中(例如`Torch.nn.parallel.DistributedDataParallel`），并确保参数被放在适当的设备上。
默认情况下，这将把每个数据加载器转换为[`accelerate.data_loader.DataLoaderShard`](https://huggingface.co/docs/accelerate/main/en/internal#accelerate.data_loader.DataLoaderShard)的一个实例。根据数据加载器的`drop_last`属性的值，一种情况是迭代将停止在第一个太小/不能存在于所有进程的批次上，另一种情况是在数据耗尽的进程上从头开始循环批次，这样所有批次的大小都是一样的。注意：这可能会改变数据加载器的长度，所以应该在计算每个周期的更新步数（即初始化一个学习率调度器）之前调用。（这一段得再细细品）
（3）创建表示`trainer`当前状态的`TrainerRunConfig`的实例
```python
Trainer._create_run_config(per_device_batch_size, num_epochs, gradient_accumulation_steps, max_num_train_steps, gradient_clip_value) → TrainerRunConfig
```
参数有：
- `per_device_batch_size`：每个设备上的批大小
- `num_epochs`：在当前训练的`epoch`数目
- `gradient_accumulation_steps`：训练过程中使用的梯度累加的步数
- `max_num_train_steps`：指定训练的最大步数，如果指定该参数的话，会覆盖`num_epochs`参数
- `gradient_clip_value`：指定修剪模型参数梯度的阈值

### 与训练相关的内部方法
在训练开始时，会使用如下内部方法：
```python
Trainer._run_training()
```
### 与训练步相关的内部方法
（1）在每个训练步中的行为，使用如下内部方法：
```python
Trainer._run_train_epoch(train_dl)
```
参数是`train_dl`，即训练集的数据加载器。
（2）修剪模型参数的梯度：
```python
Trainer._clip_gradients()
```
该方法会根据之前传入`train()`方法的阈值来修剪模型参数的梯度。
默认情况下，使用[`accelerate.Accelerator.clip_grad_value_()`](https://huggingface.co/docs/accelerate/main/en/accelerator#accelerate.Accelerator.clip_grad_value_)来修剪梯度。

### 与验证/评估相关的内部方法
在每个验证步中的行为，使用如下内部方法：
```python
Trainer._run_eval_epoch(valid_dl, is_training: bool = True)
```
参数是
- `valid_dl`，即验证集或测试集的数据加载器。
- `is_training`：指定该评估是否是训练过程的一部分，即可以是训练过程中对验证集的评估，也可以是训练结束后在测试集上的评估。

# 记录指标
`Trainer`包含一个`RunHistory`的实例，它可以用来存储和获得训练期间要跟踪的任何指标的值。默认情况下，`Trainer`记录的唯一指标是训练和验证期间观察到的损失。
注意：如果使用了`PrintMetricsCallback`回调，那么运行历史中记录的所有指标将被自动打印到控制台。
`RunHistory`的`API`稍后会详细分析。
下面是一个例子，说明如何对`Trainer`进行子类化，并使用`RunHistory`来跟踪用`TorchMetrics`计算的指标：
```python
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from pytorch_accelerated import Trainer

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
```

# Trainer运行过程
`Trainer`内部到底干了啥？用伪代码的话，是这样表达：
```python
train_dl = create_train_dataloader()
eval_dl = create_eval_dataloader()
scheduler = create_scheduler()

training_run_start()
on_training_run_start()

for epoch in num_epochs:
    train_epoch_start()
    on_train_epoch_start()
    for batch in train_dl:
        on_train_step_start()
        batch_output = calculate_train_batch_loss(batch)
        on_train_step_end(batch, batch_output)
        backward_step(batch_output["loss"])
        optimizer_step()
        scheduler_step()
        optimizer_zero_grad()
    train_epoch_end()
    on_train_epoch_end()

    eval_epoch_start()
    on_eval_epoch_start()
    for batch in eval_dl:
        on_eval_step_start()
        batch_output = calculate_eval_batch_loss(batch)
        on_eval_step_end(batch, batch_output)
    eval_epoch_end()
    on_eval_epoch_end()

    training_run_epoch_end()
    on_training_run_epoch_end()

training_run_end()
on_training_run_end()
```

同样地，一个评估过程的表达可以这样表示：
```python
eval_dl = create_eval_dataloader()

evaluation_run_start()
on_evaluation_run_start()

eval_epoch_start()
on_eval_epoch_start()
for batch in eval_dl:
    on_eval_step_start()
    batch_output = calculate_eval_batch_loss(batch)
    on_eval_step_end(batch, batch_output)
eval_epoch_end()
on_eval_epoch_end()

evaluation_run_end()
on_evaluation_run_end()
```
了解`Trainer`内部如何工作的最好方法是查阅`train()`方法的源代码；为了使内部方法尽可能的干净和清晰，作者们做了大量的工作。