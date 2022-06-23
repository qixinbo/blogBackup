---
title: 轻量级PyTorch通用训练模板pytorch-accelerated解析：2 -- API之Callbacks
tags: [PyTorch]
categories: machine learning 
date: 2022-6-4
---

# 简介
这一章将研究`pytorch-accelerated`的回调机制。

# Callback概览
在`Trainer`中除了可重写的钩子（即自定义训练器的行为）之外，`Trainer`还包括一个回调系统。
建议将回调`Callbacks`用于对训练循环的操作不是很重要的代码中，比如日志，但这个决定由用户根据具体的使用情况进行判断。
注意：回调是按顺序执行的，所以如果一个回调被用来修改状态，比如更新一个指标，用户有责任确保这个回调被放在任何将读取这个状态的回调之前（即为了记录的目的）。
回调是在其相应的钩子之后被调用，例如，`on_train_epoch_end`回调方法在`pytorch_accelerated.trainer.train_epoch_end()`方法之后被调用。这样做是为了支持在一个方法中更新训练器的状态，然后再在回调中读取这个状态。

# 内置回调
（1）该回调用于在训练或验证过程中，如果检测到损失为`NaN`，则中止训练。
```python
class pytorch_accelerated.callbacks.TerminateOnNaNCallback
```
（2）该回调记录`Trainer`运行历史中更新的任何指标的最新值。
```python
class pytorch_accelerated.callbacks.LogMetricsCallback
```
默认情况下，它在每个机器上向命令行打印一次。
以`train`为前缀的指标在一个训练`epoch`结束时被记录，所有其他指标在验证评估后被记录。
通过重载`log_metrics()`方法，可以对其进行子类化以创建不同平台的记录器。
（3）该回调在训练的开始和结束时，以及在每个`epoch`开始时打印一个信息。
```python
class pytorch_accelerated.callbacks.PrintProgressCallback
```
（4）该回调使用一个进度条来显示每个训练和验证`epoch`的状态。
```python
class pytorch_accelerated.callbacks.ProgressBarCallback
```
（5）该回调根据一个给定的指标，在训练期间保存最佳模型。最佳模型的权重在训练结束时被加载。
```python
class pytorch_accelerated.callbacks.SaveBestModelCallback(save_path='best_model.pt', watch_metric='eval_loss_epoch', greater_is_better: bool = False, reset_on_train: bool = True)
```
参数有：
- `save_path`：保存检查点的路径，应该以`.pt`结尾
- `watch_metric`：该指标用来对比模型性能，它可从`trainer`的运行历史中获得
- `greater_is_better`：指定`watch_metric`怎样解释，是否是越大越好，默认是`False`
- `reset_on_train`：指定是否在后续训练中重置最佳指标。如果为`True`，将只比较当前训练运行期间观察到的指标。

（6）该回调用于提前终止。
```python
class pytorch_accelerated.callbacks.EarlyStoppingCallback(early_stopping_patience: int = 1, early_stopping_threshold: float = 0.01, watch_metric='eval_loss_epoch', greater_is_better: bool = False, reset_on_train: bool = True)
```
参数有：
- `early_stopping_patience`：设置指标没有改善的`epochs`数目，之后将停止训练
- `early_stopping_threshold`：指定在`watch_metric`上的最小变化，将其定义为`指标改善`，也就是说，绝对变化小于这个阈值，将被视为没有改善。
- `watch_metric`：用来评价模型性能的指标，可从`trainer`的运行历史中获得
- `great_is_better`：指定`watch_metric`是否是越大越好
- `reset_on_train`：指定是否在后续训练中重置最佳指标。如果为`True`，将只比较当前训练运行期间观察到的指标。

（7）该回调在训练或评估开始时将任意`Trainer`属性转移到适当的设备上。
```python
class pytorch_accelerated.callbacks.MoveModulesToDeviceCallback
```
这里的属性是`torch.nn.Module`的实例。
注意，这里不包括模型，因为它将由`Trainer`内部的`accelerate.Accelerator`实例单独准备。

# 创建新的回调
要创建一个包含自定义行为的新的回调，例如，将日志转移到一个外部平台，可以通过子类化`TrainerCallback`实现。为了避免与`Trainer`的方法相混淆，所有回调方法都以`on_`为前缀。
注意：为了获得最大的灵活性，`Trainer`的当前实例在每个回调方法中都是可用的。然而，在回调中改变`Trainer`的状态可能会产生意想不到的后果，因为这可能会影响训练运行的其他部分。如果使用回调来修改训练器的状态，用户有责任确保一切继续按计划进行。

## 回调基类
当创建新的回调时，需要使用如下的抽象基类。
```python
class pytorch_accelerated.callbacks.TrainerCallback
```
它的方法（以下称为事件`event`，原因是这些方法被后面的回调句柄的`call_event`方法调用，称为事件`event`也更容易被用户理解和联想）包括：
（1）`trainer`初始化结束后触发的事件：
```python
on_init_end(trainer, **kwargs)
```
（2）在训练开始时触发的事件：
```python
on_training_run_start(trainer, **kwargs)
```
（3）在每一个训练`epoch`开始时触发的事件（即对每一个`epoch`进行循环）：
```python
on_train_epoch_start(trainer, **kwargs)
```
（4）在每一个训练步`step`开始时触发的事件（即对每一个`batch`进行循环）：
```python
on_train_step_start(trainer, **kwargs)
```
（5）在每一个训练步`step`结束后触发的事件：
```python
on_train_step_end(trainer, batch, batch_output, **kwargs)
```
参数有：
- `batch`：训练集的当前`batch`
- `batch_output`：由`pytorch_accelerated.trainer.Trainer.calculate_train_batch_loss()`所返回的输出

（6）在每一个训练`epoch`结束后触发的事件：
```python
on_train_epoch_end(trainer, **kwargs)
```
（7）在每一个验证`epoch`开始时触发的事件（即对每一个`epoch`进行循环）：
```python
on_eval_epoch_start(trainer, **kwargs)
```
（8）在每一个验证步`step`开始时触发的事件（即对每一个`batch`进行循环）：
```python
on_eval_step_start(trainer, **kwargs)
```
（9）在每一个验证步`step`结束后触发的事件：
```python
on_eval_step_end(trainer, batch, batch_output, **kwargs)
```
参数有：
- `batch`：验证集的当前`batch`
- `batch_output`：由`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss()`所返回的输出

（10）在每一个验证`epoch`结束后触发的事件：
```python
on_eval_epoch_end(trainer, **kwargs)
```
（11）在训练结束后触发的事件：
```python
on_training_run_end(trainer, **kwargs)
```
（12）当出现训练错误后触发的事件：
```python
on_stop_training_error(trainer, **kwargs)
```
一个训练可能通过发出`StopTrainingError`异常来被提前停止。

## 案例1：使用回调来追踪指标
默认情况下，`pytorch_accelerated.trainer.Trainer`记录的唯一指标是训练和评估期间观察到的损失。为了跟踪其他指标，我们可以使用回调来扩展这一行为。
下面是一个例子，说明如何定义一个回调并使用`RunHistory`来跟踪用`TorchMetrics`计算的指标。
```python
from torchmetrics import MetricCollection, Accuracy, Precision, Recall

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
## 案例2：创建自定义日志的回调
建议使用回调来处理日志，以使训练循环集中在机器学习相关的代码上。通过对`LogMetricsCallback`回调的子类化，很容易为其他平台创建日志记录器。
例如，可以为`AzureML`（使用`MLFlow API`）创建一个记录器，如下所示：
```python
import mlflow

class AzureMLLoggerCallback(LogMetricsCallback):
    def __init__(self):
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    def on_training_run_start(self, trainer, **kwargs):
        mlflow.set_tags(trainer.run_config.to_dict())

    def log_metrics(self, trainer, metrics):
        if trainer.run_config.is_world_process_zero:
            mlflow.log_metrics(metrics)
```
## 案例3：自定义回调以在评估后保存结果
下面是一个自定义回调的例子，在评估期间记录预测结果，然后在评估周期结束时将其保存为`csv`。
```python
from collections import defaultdict
import pandas as pd

class SavePredictionsCallback(TrainerCallback):

    def __init__(self, out_filename='./outputs/valid_predictions.csv') -> None:
        super().__init__()
        self.predictions = defaultdict(list)
        self.out_filename = out_filename

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        input_features, targets = batch
        class_preds = trainer.gather(batch_output['model_outputs']).argmax(dim=-1)
        self.predictions['prediction'].extend(class_preds.cpu().tolist())
        self.predictions['targets'].extend(targets.cpu().tolist())

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer._accelerator.wait_for_everyone()
        if trainer.run_config.is_local_process_zero:
            df = pd.DataFrame.from_dict(self.predictions)
            df.to_csv(f'{self.out_filename}', index=False)
```

# 回调句柄
传递给`Trainer`的任何回调函数都是由一个内部回调句柄类`CallbackHandler`的实例来处理的。
```python
class pytorch_accelerated.callbacks.CallbackHandler(callbacks)
```
可以看出，回调句柄`CallbackHandler`的输入参数就是一系列的回调函数，然后该句柄顺序执行它们（执行顺序是按它们传入该句柄的顺序）。
主要方法有：
（1）添加单个回调函数：
```python
add_callback(callback)
```
参数为`callback`，类型是`TrainerCallback`的子类的实例。
（2）添加多个回调函数：
```python
add_callbacks(callbacks)
```
参数为`callbacks`，是一个回调函数列表。
（3）对于已添加注册的所有回调，根据特定事件`event`来顺序调用：
```python
call_event(event, *args, **kwargs)
```
参数有：
- `event`：要触发的事件，实际
- `args`：传给回调的参数列表
- `kwargs`：传给回调的关键字列表

## 创建新的回调事件
前面已经介绍了很多内置的回调事件，比如`on_init_end`、`on_training_run_start`等，这些事件触发的位置都在`Trainer`中已经定义好了。
也可以创建新的回调事件，比如：
```python
class VerifyBatchCallback(TrainerCallback):
    def verify_train_batch(self, trainer, xb, yb):
        assert xb.shape[0] == trainer.run_config["train_per_device_batch_size"]
        assert xb.shape[1] == 1
        assert xb.shape[2] == 28
        assert xb.shape[3] == 28
        assert yb.shape[0] == trainer.run_config["train_per_device_batch_size"]
```
然后在训练过程中进行触发（最好就是子类化原来的`Trainer`）：
```python
class TrainerWithCustomCallbackEvent(Trainer):
    def calculate_train_batch_loss(self, batch) -> dict:
        xb, yb = batch
        self.callback_handler.call_event(
            "verify_train_batch", trainer=self, xb=xb, yb=yb
        )
        return super().calculate_train_batch_loss(batch)
```
这样就实现了很大的灵活性。