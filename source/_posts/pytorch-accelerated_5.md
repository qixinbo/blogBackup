---
title: 轻量级PyTorch通用训练模板pytorch-accelerated解析：5 -- Trainer运行及案例赏析
tags: [PyTorch]
categories: machine learning 
date: 2022-6-26
---

# 简介
首先对`pytorch-accelerated`的核心类`Trainer`进行逐行代码的注释理解，然后再以官方的几个例子进行注解说明。

# Trainer逐行代码注解

```python
# Copyright © 2021 Chris Hughes
import math
import os
from collections import Callable
from enum import Enum
from functools import partial
from typing import Iterable

import torch
from accelerate import Accelerator, DistributedType
from torch.utils.data import DataLoader
# 导入内置的回调函数
from pytorch_accelerated.callbacks import (
    CallbackHandler,
    LogMetricsCallback,
    PrintProgressCallback,
    TerminateOnNaNCallback,
    StopTrainingError,
    ProgressBarCallback,
    MoveModulesToDeviceCallback,
)
from pytorch_accelerated.run_config import TrainerRunConfig
from pytorch_accelerated.tracking import RunHistory, InMemoryRunHistory, LossTracker

# 默认使用的回调函数有如下5个：
DEFAULT_CALLBACKS = (
    # 该回调在训练或评估开始时，将所有属于`torch.nn.Module`的实例（除了网络模型）的`trainer`属性移动到相应的设备上。
    MoveModulesToDeviceCallback,
    # 该回调在训练时监测是否出现'NaN'损失值，从而及时终止训练
    TerminateOnNaNCallback,
    # 打印进度
    PrintProgressCallback,
    # 进度条
    ProgressBarCallback,
    # 指标记录
    LogMetricsCallback,
)

# 一些学习率调度器需要一些信息，比如在训练运行期间发生的总步数。
# 由于在创建训练数据加载器之前无法获得这些信息（它们是`Trainer.train`中产生的），在这种情况下可以使用一个占位符值，比如：
# from functools import Partial

# from pytorch_accelerated import TrainerPlaceholderValues
# from torch.optim.lr_scheduler import OneCycleLR

# create_scheduler_fn = partial(
#             OneCycleLR,
#             max_lr=config.lr,
#             epochs=TrainerPlaceholderValues.NUM_EPOCHS,
#             steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
#         )
# 这些占位符将由trainer在训练期间使用正确的数值替换。

# 具体实现原理如下。

# 首先TrainerPlaceholderValues派生自Enum，关于该类的用法仔细阅读如下两篇：
# https://docs.python.org/3/library/enum.html
# https://www.pythontutorial.net/python-oop/python-enum-class/

class TrainerPlaceholderValues(Enum):
    # 这些枚举数值实际会调用trainer中的变量及配置
    NUM_EPOCHS = "trainer.run_config.num_epochs"
    NUM_UPDATE_STEPS_PER_EPOCH = "trainer.run_config.num_update_steps_per_epoch"
    TRAIN_DATALOADER_LEN = "len(trainer._train_dataloader)"
    EVAL_DATALOADER_LEN = "len(trainer._eval_dataloader)"

    # 类方法，教程参考：
    # https://www.jianshu.com/p/87608d92fafe
    @classmethod
    def placeholder_set(cls):
        return {placeholder.name for placeholder in cls}

    @staticmethod
    def __create_new_enum(original_enum, other, operation):
        enum_members = {k: v.value for k, v in original_enum._member_map_.items()}
        enum_members[
            original_enum.name
        ] = f"{enum_members[original_enum.name]}{operation}{other}"
        new_enum = Enum("TrainerPlaceholderValues", enum_members)
        return new_enum._member_map_[original_enum.name]

    def __mul__(self, other):
        return self.__create_new_enum(self, other, "*")

    def __add__(self, other):
        return self.__create_new_enum(self, other, "+")

    def __sub__(self, other):
        raise NotImplemented(
            "Subtraction is not supported, please re-write the expression in terms of addition"
        )

# 如果实例instance是一个偏函数对象，且包含了关键字，将替换它们，返回一个新的偏函数对象
def replace_trainer_placeholder_values(trainer, instance):
    if isinstance(instance, partial):
        placeholders = TrainerPlaceholderValues.placeholder_set()
        keywords = list(instance.keywords.items())

        new_keywords = {}

        for keyword, value in keywords:
            if hasattr(value, "name"):
                # 如果value有name属性，且在TrainerPlaceholderValues的占位符集合中
                if value.name in placeholders:
                    # 则马上计算该占位符（即枚举值）的表达式，教程见：
                    # https://www.runoob.com/python/python-func-eval.html
                    new_keywords[keyword] = eval(value.value)
                else:
                    new_keywords[keyword] = value
            else:
                new_keywords[keyword] = value

        instance = partial(instance.func, *instance.args, **new_keywords)

    return instance


class Trainer:
    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        callbacks=DEFAULT_CALLBACKS,
        run_history=None,
    ):
        # 传入模型，后面会被替换成被accelerate封装后的模型
        self.model = model
        # 传入损失函数
        self.loss_func = loss_func
        # 传入优化器，后面会被替换成被accelerate封装后的优化器
        self.optimizer = optimizer
        # 将回调事件列表作为参数传入“回调函数句柄”
        # 该句柄的call_event()方法有如下形式：
        # def call_event(self, event, *args, **kwargs):
        # 即传入一个event事件及参数列表，然后该方法会判断该event在哪些回调函数中存在（即存在该成员函数或成员变量）
        self.callback_handler = CallbackHandler(
            callbacks,
        )
        # 传入运行历史，该运行历史应该是RunHistory类型
        # 如果是None的话，就使用默认实现的InMemoryRunHistory()
        # 也可以自己手写基于基类RunHistory的实现
        self.run_history: RunHistory = (
            run_history if run_history is not None else InMemoryRunHistory()
        )
        # 创建一个accelerate.Accelerator的实例，用于管理训练过程
        self._accelerator = self._create_accelerator()
        # 创建一个损失追踪器
        self._loss_tracker = LossTracker()
        # 下面是一些内部变量，它们的值会在训练过程中被设置
        self.create_scheduler_fn = None
        self.scheduler = None
        self.collate_fn = None
        self.train_dataset = None
        self.eval_dataset = None
        self._train_dataloader = None
        self._train_dl_kwargs = None
        self._eval_dl_kwargs = None
        self._eval_dataloader = None
        self.run_config: TrainerRunConfig = None

        # 初始化Trainer的末尾会调用一下`on_init_end`事件
        # 目前来看这几个内置的回调函数都没有该属性
        self.callback_handler.call_event("on_init_end", self)

    def _create_accelerator(self):
        """
        Create an instance of :class:`accelerate.Accelerator` which will be used to manage training.
        :return:
        """
        return Accelerator()


    # 创建训练集的dataloader
    def create_train_dataloader(
        self, batch_size: int, train_dl_kwargs: dict = None
    ) -> Iterable:
        # 首先获得默认的训练集dataloader的参数，包括"shuffle"、"pin_memory"、"batch_size"和"num_workers"配置
        default_train_dl_kwargs = self.get_default_train_dl_kwargs(batch_size)

        # 如果明确指定了参数字典，则对默认的字典进行更新，用 update 更新字典 a，会有两种情况：
        # （1）有相同的键时：会使用最新的字典 b 中 该 key 对应的 value 值。
        # （2）有新的键时：会直接把字典 b 中的 key、value 加入到 a 中。
        if train_dl_kwargs is not None:
            default_train_dl_kwargs.update(train_dl_kwargs)

        self._train_dl_kwargs = default_train_dl_kwargs

        # 最终是调用torch原生的DataLoader来创建数据加载器
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **self._train_dl_kwargs,
        )

    def create_eval_dataloader(
        self, batch_size: int, eval_dl_kwargs: dict = None
    ) -> Iterable:
        # 首先获得默认的验证集dataloader的参数，包括"shuffle"、"pin_memory"、"batch_size"和"num_workers"配置
        # 与训练集不同的是，验证集的shuffle是False
        default_eval_dl_kwargs = self.get_default_eval_dl_kwargs(batch_size)

        if eval_dl_kwargs is not None:
            default_eval_dl_kwargs.update(eval_dl_kwargs)

        self._eval_dl_kwargs = default_eval_dl_kwargs

        return DataLoader(
            dataset=self.eval_dataset,
            collate_fn=self.collate_fn,
            **self._eval_dl_kwargs,
        )

    # 基于之前传递给Trainer的工厂函数创建一个学习率调度器
    def create_scheduler(self):
        # 该工厂函数是个偏函数对象，它里面的关键词参数如果用到了占位符，会被实时结果所更新替代
        scheduler_type = replace_trainer_placeholder_values(
            self, self.create_scheduler_fn
        )
        # 该工厂函数再接收优化器参数，从而返回调度器实例
        return scheduler_type(self.optimizer)

    def training_run_start(self):
        """
        This method is called at the start of a training run.
        """
        pass

    # 每一训练epoch开始时的动作
    def train_epoch_start(self):
        # 默认行为是将模型设置成train模式，教程见：
        # https://zhuanlan.zhihu.com/p/494060986
        self.model.train()

    # 计算训练时的损失
    def calculate_train_batch_loss(self, batch) -> dict:
        """
        Calculates the training loss and return this along with the batch size and model outputs.
        Any additional values returned will be available in the :meth:`~callbacks.TrainerCallback.on_train_step_end` callback method.

        :param batch: the output of the train dataloader
        :return: A dictionary containing the training loss, model outputs and batch size. Can include any keys, but must include the keys 'loss', 'model_outputs' and 'batch_size'
        """
        # 获得x和y
        xb, yb = batch[0], batch[1]
        # 将x输入模型，获得预测值
        model_outputs = self.model(xb)
        # 计算损失值
        loss = self.loss_func(model_outputs, yb)

        # 返回值包括损失、模型输出值和batch size
        return {
            "loss": loss,
            "model_outputs": model_outputs,
            "batch_size": yb.size(0),
        }

    def backward_step(self, loss):
        """
        Use the accelerator to perform the backward pass on the calculated value of the loss returned by :meth:`~Trainer.calculate_train_batch_loss`.
        If gradient accumulation is enabled, this loss has been scaled by 1 / accumulation steps.

        :param loss: The loss tensor returned by :meth:`~Trainer.calculate_train_batch_loss`.
        """
        self._accelerator.backward(loss)

    def optimizer_step(self):
        """
        Performs a single optimization step and updates the parameters which have been passed to ``self.optimizer``.
        """
        self.optimizer.step()

    def scheduler_step(self):
        """
        Performs a single scheduler step if ``self.scheduler`` has been assigned.

        """
        if self.scheduler is not None:
            self.scheduler.step()

    def optimizer_zero_grad(self):
        """
        Sets the gradients of all optimized ``torch.Tensor`` s to zero.
        """
        self.optimizer.zero_grad()

    def train_epoch_end(self):
        """
        This method is called at the end of each training epoch.
        """
        pass

    def eval_epoch_start(self):
        """
        This method is called at the start of an evaluation epoch.

        The default behaviour of this method is to call ``self.model.eval()``
        """
        self.model.eval()

    def calculate_eval_batch_loss(self, batch) -> dict:
        """
        Calculates the evaluation loss and return this along with the batch size and model outputs.
        Any additional values returned will be available in the :meth:`~callbacks.TrainerCallback.on_eval_step_end` callback.

        :param batch: the output of the eval dataloader
        :return: A dictionary containing the evaluation loss, model outputs and batch size. Can include any keys, but must include the keys ``loss``, ``model_outputs`` and ``batch_size``
        """
        with torch.no_grad():
            xb, yb = batch[0], batch[1]
            model_outputs = self.model(xb)
            val_loss = self.loss_func(model_outputs, yb)

        return {
            "loss": val_loss,
            "model_outputs": model_outputs,
            "batch_size": yb.size(0),
        }

    def eval_epoch_end(self):
        """
        This method is called at the end of an evaluation epoch.
        """
        pass

    def training_run_epoch_end(self):
        """
        This method is called during a training run after both training and evaluation epochs have been completed.
        """
        pass

    def training_run_end(self):
        """
        This method is called at the end of a training run.
        """
        pass

    def evaluation_run_start(self):
        """
        This method is called at the start of an evaluation run.
        """
        pass

    def evaluation_run_end(self):
        """
        This method is called at the end of an evaluation run.
        """
        pass

    # train是Trainer的入口函数
    def train(
        self,
        train_dataset,
        num_epochs,
        eval_dataset=None,
        per_device_batch_size=8, # 默认每个设备上的batch size是8
        max_num_train_steps=None,
        gradient_accumulation_steps=1, # 默认梯度累加步数为1
        gradient_clip_value=None,
        create_scheduler_fn: Callable = None,
        train_dataloader_kwargs: dict = None,
        eval_dataloader_kwargs: dict = None,
        reset_run_history=True,
        collate_fn=None,
    ):
        # 传入训练集
        self.train_dataset = train_dataset
        # 传入验证集，默认为None
        self.eval_dataset = eval_dataset
        # 传入调度器，默认为None
        # 注意，这里不是传递一个学习率调度器的实例，而是传递一个能返回这样的实例的工厂函数。
        self.create_scheduler_fn = create_scheduler_fn
        # 传入数据加载器所使用的collate函数，该函数指定如何整理样本以形成一个mini-batch的样本，默认为None，即使用默认的整理方法
        # https://zhuanlan.zhihu.com/p/361830892
        self.collate_fn = collate_fn
        # 如果指定重置运行历史，则调用run_history的reset方法
        # 对于默认的InMemoryRunHistory()，具体就是做了：
        # （1）_current_epoch设为1，
        # （2）_metrics设为defaultdict(list)，这里用了defaultdict，是怕字典里没有key时报错，
        # 教程见：https://www.jianshu.com/p/bbd258f99fd3
        if reset_run_history:
            self.run_history.reset()

        # 传入batch_size及训练集数据加载器的参数，创建训练集dataloader
        # 接下来会被替换成被accelerate封装后的加载器
        self._train_dataloader = self.create_train_dataloader(
            batch_size=per_device_batch_size, train_dl_kwargs=train_dataloader_kwargs
        )

        # 如果明确指定了验证集后，则以与上面训练集dataloader同样的方式创建验证集的dataloader
        # 两者区别是验证集的默认的shuffle是False
        if self.eval_dataset is not None:
            # 接下来它也会被替换成被accelerate封装后的加载器
            self._eval_dataloader = self.create_eval_dataloader(
                batch_size=per_device_batch_size, eval_dl_kwargs=eval_dataloader_kwargs
            )

        # 将模型、优化器和dataloader放到accelerate上
        self._prepare_model_optimizer_and_dataloaders()

        # 封装运行配置
        self.run_config = self._create_run_config(
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_num_train_steps=max_num_train_steps,
            per_device_batch_size=per_device_batch_size,
            gradient_clip_value=gradient_clip_value,
        )

        # 创建调度器实例
        if self.create_scheduler_fn is not None:
            self.scheduler = self.create_scheduler()

        # 开始训练
        self._run_training()

    def evaluate(
        self,
        dataset=None,
        per_device_batch_size=8,
        dataloader_kwargs: dict = None,
        collate_fn=None,
    ):
        """
        Start an evaluation run.

        .. Note:: Starting an evaluation run will reset the :class:`Trainer`'s run history.

        .. Note:: During distributed evaluation, if the `per_device_batch_size` * the number of processes used does not exactly divide the dataset, and `drop_last=False` has not been passed as a dataloader kwarg, the dataloader will repeat from the start in processes that run out of batches. This should be taken into consideration when calculating metrics.

        :param dataset: the dataset to use during evaluation
        :param per_device_batch_size: the batch size to use per device
        :param dataloader_kwargs: a dictionary of keyword arguments to pass to the dataloader constructor, for details see :class:`torch.utils.data.DataLoader`
        :param collate_fn: the collate function to be used by the dataloader
        """
        self.eval_dataset = dataset
        self.collate_fn = collate_fn

        self.run_history.reset()

        self._train_dataloader = None
        self._eval_dataloader = self.create_eval_dataloader(
            batch_size=per_device_batch_size, eval_dl_kwargs=dataloader_kwargs
        )

        self._prepare_model_optimizer_and_dataloaders()

        if self.run_config is None:
            self.run_config = self._create_run_config(
                num_epochs=1,
                gradient_accumulation_steps=0,
                max_num_train_steps=None,
                per_device_batch_size=per_device_batch_size,
                gradient_clip_value=None,
            )

        self._run_evaluation()

    def get_default_train_dl_kwargs(self, batch_size) -> dict:
        """
        Return the default arguments that will be used by the training dataloader.

        :param batch_size: the batch size to use during training
        :return: a dictionary containing the default arguments for the training dataloader
        """
        return {
            "shuffle": True,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": batch_size,
            "num_workers": max(
                os.cpu_count() // torch.cuda.device_count()
                if torch.cuda.is_available()
                else os.cpu_count(),
                1,
            ),
        }

    def get_default_eval_dl_kwargs(self, batch_size) -> dict:
        """
        Return the default arguments that will be used by the evaluation dataloader.

        :param batch_size: the batch size to use during evaluation
        :return: a dictionary containing the default arguments for the evaluation dataloader
        """
        return {
            "shuffle": False,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": batch_size,
            "num_workers": max(
                os.cpu_count() // torch.cuda.device_count()
                if torch.cuda.is_available()
                else os.cpu_count(),
                1,
            ),
        }

    @property
    def device(self):
        """
        Use the internal instance of :class:`accelerate.Accelerator` to get the appropriate device
        :return: an instance of `torch.device`
        """
        return self._accelerator.device

    def _prepare_model_optimizer_and_dataloaders(self):
        """
        使用`accelerate.Accelerator`将模型、优化器和数据加载器包裹在任何训练所需的包装器中，并确保参数被放置在适当的设备上。
        """
        self._accelerator.free_memory()
        self._accelerator = self._create_accelerator()

        components = [self.model, self.optimizer]

        if self._train_dataloader is not None:
            components.append(self._train_dataloader)

        if self._eval_dataloader is not None:
            components.append(self._eval_dataloader)

        # 准备与训练相关的对象（优化器、模型、训练集dataloader、验证集dataloader）
        # 这使得这些东西做好训练的准备
        prepared_components = self._accelerator.prepare(*components)

        self.model = prepared_components[0]
        self.optimizer = prepared_components[1]

        # 这个if和elif会区分是训练阶段还是测试阶段
        # 训练数据加载器将在所有可用的GPU/TPU核中上进行分片，以便每个核看到训练数据集的不同部分。此外，所有进程的随机状态将在每次迭代开始时在dataloader中进行同步，以确保数据以相同的方式被打乱（如果决定使用shuffle=True或任何种类的随机采样器）。
        # 训练的实际批次大小将是所使用的设备数量乘以在脚本中设置的批次大小：例如，在4个GPU上训练，在创建训练数据加载器时设置的批次大小为16，则实际训练的批次大小为64。另外，可以在创建Accelerator时使用split_batches=True选项，在这种情况下，无论在1、2、4还是64个GPU上运行脚本，批次大小都会保持一致。
        if self._train_dataloader is not None:
            self._train_dataloader = prepared_components[2]
            if self._eval_dataloader is not None:
                self._eval_dataloader = prepared_components[3]

        elif self._eval_dataloader is not None:
            self._eval_dataloader = prepared_components[2]

    # 创建运行配置
    # 将运行配置集合在一个地方
    def _create_run_config(
        self,
        per_device_batch_size,
        num_epochs,
        gradient_accumulation_steps,
        max_num_train_steps,
        gradient_clip_value,
    ) -> TrainerRunConfig:
        # 获得train_per_device_batch_size配置
        if self._train_dl_kwargs is not None:
            # get()方法返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值。
            train_per_device_batch_size = self._train_dl_kwargs.get(
                "batch_size", per_device_batch_size
            )
        else:
            train_per_device_batch_size = per_device_batch_size

        # 获得eval_per_device_batch_size配置
        if self._eval_dl_kwargs is not None:
            eval_per_device_batch_size = self._eval_dl_kwargs.get(
                "batch_size", train_per_device_batch_size
            )
        else:
            eval_per_device_batch_size = train_per_device_batch_size

        # 获得num_update_steps_per_epoch配置
        if self._train_dataloader is not None:
            # 这个地方涉及梯度累加步数，关于梯度累加，一些参考教程见：
            # https://blog.kamino.link/2021/10/03/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E6%A2%AF%E5%BA%A6%E7%B4%AF%E5%8A%A0Pytorch%E5%AE%9E%E7%8E%B0/
            # https://www.cnblogs.com/lart/p/11628696.html
            # https://zhuanlan.zhihu.com/p/351999133

            num_update_steps_per_epoch = math.ceil(
                len(self._train_dataloader) / gradient_accumulation_steps
            )
        else:
            num_update_steps_per_epoch = 0

        # 获得max_num_train_steps配置
        # 如果未明确配置它，则根据其他参数计算
        if max_num_train_steps is None:
            max_num_train_steps = num_epochs * num_update_steps_per_epoch
        # 如果明确配置它了，则对num_epochs这个参数重新计算一下
        else:
            num_epochs = math.ceil(max_num_train_steps / num_update_steps_per_epoch)

        config = {
            "num_epochs": num_epochs,
            "train_per_device_batch_size": train_per_device_batch_size,
            "train_dl_kwargs": self._train_dl_kwargs,
            "eval_per_device_batch_size": eval_per_device_batch_size,
            "eval_dl_kwargs": self._eval_dl_kwargs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "train_total_batch_size": train_per_device_batch_size
            * self._accelerator.num_processes
            * gradient_accumulation_steps,
            "eval_total_batch_size": eval_per_device_batch_size
            * self._accelerator.num_processes,
            "num_update_steps_per_epoch": num_update_steps_per_epoch,
            "max_num_train_steps": max_num_train_steps,
            "is_local_process_zero": self._accelerator.is_local_main_process,
            "is_world_process_zero": self._accelerator.is_main_process,
            "is_distributed": True
            if self._accelerator.distributed_type != DistributedType.NO
            else False,
            "mixed_precision": self._accelerator.mixed_precision,
            "gradient_clip_value": gradient_clip_value,
        }

        # 将所有配置封装进TrainerRunConfig类型中
        return TrainerRunConfig(**config)

    # 开始训练
    def _run_training(self):
        # 开始训练时调用一次该成员函数，当前该函数是空的
        self.training_run_start()
        # 触发"on_training_run_start"这一回调事件
        # 比如MoveModulesToDeviceCallback、PrintProgressCallback这两个回调就拥有该事件属性
        # 从而能够在训练一开始就做一些事情
        self.callback_handler.call_event(
            "on_training_run_start",
            self,
        )

        # 对epoch进行循环
        for epoch in range(self.run_config.num_epochs):
            try:
                # 每一个训练epoch期间运行所做的事情
                self._run_train_epoch(self._train_dataloader)

                # 如果指定了验证集，则运行验证epoch
                # 其基本流程与train epoch类似，但少了训练过程
                if self.eval_dataset is not None:
                    self._run_eval_epoch(self._eval_dataloader)
                # 对epoch进行步进
                self.run_history._increment_epoch()
                # 每个epoch结束后调用
                self.training_run_epoch_end()
                # 每个epoch结束后触发on_training_run_epoch_end事件
                # 默认的回调中没有该事件
                # 不过该事件非常重要，在SaveBestModelCallback中有使用，用来保存最佳模型；以及在EarlyStoppingCallback中使用，用来提前终止训练
                self.callback_handler.call_event(
                    "on_training_run_epoch_end",
                    self,
                )
            except StopTrainingError as e:
                self._accelerator.print(e)
                self.callback_handler.call_event(
                    "on_stop_training_error",
                    self,
                )
                break
        # 整个训练结束后调用
        self.training_run_end()
        # 整个训练结束后触发on_training_run_end事件
        # 默认的回调中，PrintProgressCallback有该事件属性，会打印出训练结束的字符串。
        self.callback_handler.call_event(
            "on_training_run_end",
            self,
        )

    def _run_evaluation(self):
        """
        The method responsible for the orchestration of the high level steps which will be executed during an evaluation run.
        """
        self.evaluation_run_start()
        self.callback_handler.call_event(
            "on_evaluation_run_start",
            self,
        )
        try:
            self._run_eval_epoch(self._eval_dataloader, is_training=False)
        except StopTrainingError as e:
            self._accelerator.print(e)
            self.callback_handler.call_event(
                "on_stop_training_error",
                self,
            )
        self.evaluation_run_end()
        self.callback_handler.call_event(
            "on_evaluation_run_end",
            self,
        )

    # 每一个训练epoch期间运行所做的事情
    def _run_train_epoch(self, train_dl):
        # 将网络模型设置成train模式
        self.train_epoch_start()
        # 将损失追踪器重置一下，即设置当前epoch为1，指标列表为空。
        self._loss_tracker.reset()
        # 触发"on_train_epoch_start"事件
        # 默认的回调函数中有如下几个有该事件属性，比如：
        # PrintProgressCallback：每个epoch开始都输出一下当前epoch是多少
        # ProgressBarCallback：每个epoch开始时初始化一个进度条
        self.callback_handler.call_event(
            "on_train_epoch_start",
            self,
        )
        # 进入对batch的循环，对每个batch的运行称为一个step
        for step, batch in enumerate(train_dl):
            # 每一步开始之前触发on_train_step_start事件，默认的回调中没有该事件的定义
            self.callback_handler.call_event(
                "on_train_step_start",
                self,
            )

            # 判断是否达到了梯度累加的步数，或者到了数据集的最后
            perform_gradient_update = (
                (step + 1) % self.run_config.gradient_accumulation_steps == 0
            ) or (step + 1 == len(train_dl))

            # 如果没有达到梯度累加的步数
            if not perform_gradient_update:
                # 那么就在不同的进程中关闭梯度同步
                with self._accelerator.no_sync(self.model):
                    self._perform_forward_and_backward_passes(batch)
            # 如果达到梯度累加的步数，则会进行梯度同步
            else:
                self._perform_forward_and_backward_passes(batch)

            # 如果设定了梯度裁剪阈值，则进行梯度裁剪
            if self.run_config.gradient_clip_value is not None:
                self._clip_gradients()

            # 如果达到了梯度累加
            if perform_gradient_update:
                # 优化器更新参数
                self.optimizer_step()
                # 如果设定了学习率调度器，则调用调度器一次
                if (
                    self.scheduler is not None
                    and not self._accelerator.optimizer_step_was_skipped
                ):
                    self.scheduler_step()
                # 梯度清零
                self.optimizer_zero_grad()

        # 每个epoch结束后调用如下方法，当前其为空
        self.train_epoch_end()
        # 使用损失追踪器中的平均损失来更新运行历史中的指标
        self.run_history.update_metric("train_loss_epoch", self._loss_tracker.average)
        # 每个epoch结束后触发on_train_epoch_end事件
        # 默认的回调中，有如下拥有该事件属性：
        # ProgressBarCallback：用来关闭进度条
        # LogMetricsCallback：输出训练损失
        self.callback_handler.call_event(
            "on_train_epoch_end",
            self,
        )

    # 计算前向传播和反向传播
    def _perform_forward_and_backward_passes(self, batch):
        # 计算训练损失
        batch_output = self.calculate_train_batch_loss(batch)
        # 如果梯度累加步数大于1，进行损失标准化，教程参见上面的梯度累加的参考文献
        if self.run_config.gradient_accumulation_steps > 1:
            batch_output["loss"] /= self.run_config.gradient_accumulation_steps

        # 通过聚合所有进程上的损失值，更新损失追踪器，
        # 包括当前batch上的损失、总损失、已运行的总样本数、平均损失
        self._loss_tracker.update(
            self.gather(batch_output["loss"]).detach().mean().item(),
            batch_output["batch_size"],
        )

        # 在每一步结束时触发on_train_step_end事件
        # 默认的回调中ProgressBarCallback有该事件属性，做的动作是更新进度条
        self.callback_handler.call_event(
            "on_train_step_end", self, batch_output=batch_output, batch=batch
        )
        # 进行反向传播
        self.backward_step(batch_output["loss"])

    def _clip_gradients(self):
        """
        Clip the gradients of the model's parameters that fall outside of the threshold specified in :meth:`~Trainer.train`.

        By default, this clips the gradients using :meth:`accelerate.Accelerator.clip_grad_value_`
        """
        self._accelerator.clip_grad_value_(
            self.model.parameters(), clip_value=self.run_config.gradient_clip_value
        )

    def _run_eval_epoch(self, valid_dl, is_training: bool = True):
        """
        The method responsible for the behaviour of each evaluation epoch.

        :param valid_dl: the dataloader to be used during evaluation
        :param is_training: signals whether the evaluation is being run as part of a training run
        """
        self.eval_epoch_start()
        self._loss_tracker.reset()
        self.callback_handler.call_event(
            "on_eval_epoch_start",
            self,
        )

        for batch in valid_dl:
            self.callback_handler.call_event(
                "on_eval_step_start",
                self,
            )
            batch_output = self.calculate_eval_batch_loss(batch)
            self._loss_tracker.update(
                self.gather(batch_output["loss"]).detach().mean().item(),
                batch_output["batch_size"],
            )
            self.callback_handler.call_event(
                "on_eval_step_end", self, batch_output=batch_output, batch=batch
            )
        self.eval_epoch_end()
        metric_name = "eval_loss_epoch" if is_training else "evaluation_loss"
        self.run_history.update_metric(metric_name, self._loss_tracker.average)
        self.callback_handler.call_event(
            "on_eval_epoch_end",
            self,
        )

    # 对在不同进程上的tensor进行聚合
    def gather(self, tensor):
        """
        Gather the values in `tensor` across all processes and concatenate them on the first dimension. This can be
        useful to regroup the predictions from all processes when doing evaluation.

        .. Note:: This gather happens in all processes.

        :param tensor: (:obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`) The tensors to gather across all processes.
        :return: The gathered tensor(s) (:obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`). The first dimension of the result is `num_processes` multiplied by the first dimension of the input tensors.
        """
        return self._accelerator.gather(tensor)

    def print(self, *args, **kwargs):
        """
        Use in replacement of ``print()`` to only print once per node.
        """
        if self._accelerator is not None:
            self._accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def save_checkpoint(
        self, save_path, checkpoint_kwargs=None, save_optimizer=True, save_per_node=True
    ):
        """
        Save the model, optimizer and specified args as a checkpoint file.

        :param save_path: the path where to save the checkpoint, this should end in '.pt'
        :param checkpoint_kwargs: additional objects to include in the checkpoint
        :param save_optimizer: flag to indicate whether to include the optimizer in the checkpoint
        :param save_per_node: flag to indicate whether to save the checkpoint once per machine, if False, the checkpoint will only be saved from the world process zero. This is True by default.
        """
        # TODO: add save method for run history?

        checkpoint = {
            "model_state_dict": self._accelerator.unwrap_model(self.model).state_dict(),
        }

        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if checkpoint_kwargs is not None:
            checkpoint.update(checkpoint_kwargs)

        self._accelerator.wait_for_everyone()

        if save_per_node:

            self._accelerator.save(
                checkpoint,
                save_path,
            )
        else:

            if self.run_config.is_world_process_zero:
                self._accelerator.save(
                    checkpoint,
                    save_path,
                )

    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """
        Load the model and optimizer from a checkpoint file.

        :param checkpoint_path: the path of the checkpoint file to load
        :param load_optimizer: flag to indicate whether to load the optimizer if it is included in the checkpoint
        """
        self._accelerator.wait_for_everyone()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            if self.optimizer is None:
                raise ValueError(
                    "You are trying to load an optimizer from a checkpoint, but no optimizer"
                    "has been set in the Trainer. Either pass the correct optimizer instance when"
                    "creating the trainer, or specify load_optimizer=False when loading the checkpoint."
                )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class TrainerWithTimmScheduler(Trainer):
    """Subclass of the :class:`Trainer` that works with `timm schedulers <https://fastai.github.io/timmdocs/schedulers>`_ instead
    of standard PyTorch learning rate schedulers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_updates = None

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

    def eval_epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)
```

# 对PyTorch迁移学习案例的加速改造
```python
# Modifications Copyright © 2021 Chris Hughes
########################################################################
# 这个例子是PyTorch迁移学习的官方教程"Transfer Learning for Computer Vision Tutorial"（作者Sasank Chilamkurthy）的"加速"版本，原文在这里:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
########################################################################

# 命令行参数模块，参考教程见：
# https://docs.python.org/zh-cn/3/howto/argparse.html
import argparse
import os
# 偏函数，用来固定参数的默认值，参考教程见：
# https://www.liaoxuefeng.com/wiki/1016959663602400/1017454145929440
# https://zhuanlan.zhihu.com/p/47124891
from functools import partial

# torch原生的神经网络模块和优化器
from torch import nn, optim
# torch原生的学习率调度器
from torch.optim import lr_scheduler
# 使用torchvision的变换、数据集和模型
from torchvision import transforms, datasets, models

import sys
sys.path.insert(0, '../../../')

# pytorch-accelerated的训练器
from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues


def main(data_dir):
    # 数据变换
    data_transforms = {
        # 对于训练集，使用随机裁剪、翻转等数据增强和标准化
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        # 对于训练集，仅使用标准化
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # 创建数据集
    # 采用的是torchvision的datasets.ImageFolder
    # 字典推导式语法，参考教程见：https://www.runoob.com/python3/python-comprehensions.html
    # 使用的数据集是hymenoptera_data，下载地址在：
    # https://download.pytorch.org/tutorial/hymenoptera_data.zip
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    # 创建模型
    # 采用的是torchvision的models，以及下载预训练权重
    model = models.resnet18(pretrained=True)
    # 将模型的分类器修改一下，适用于本例
    model.fc = nn.Linear(model.fc.in_features, len(image_datasets["train"].classes))

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()

    # 使用torch自己的优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 对于学习率调度器，仍然采用torch原生的调度器
    # 但这里将它用在pytorch-accelerated的Trainer时要经过修改
    # Trainer是在step级别（即对batch进行循环）上调用调度器，而不是torch原生的StepLR那样在epoch级别上调用
    # 比如，torch原生的StepLR是这样调用的：
    # >>> # Assuming optimizer uses lr = 0.05 for all groups
    # >>> # lr = 0.05     if epoch < 30
    # >>> # lr = 0.005    if 30 <= epoch < 60
    # >>> # lr = 0.0005   if 60 <= epoch < 90
    # >>> # ...
    # >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # >>> for epoch in range(100):
    # >>>     train(...)
    # >>>     validate(...)
    # >>>     scheduler.step()

    # 但是，在pytorch-accelerated中，看源码的话就会发现，它的scheduler.step()是在如下地方：
    #     # 进入对batch的循环，对每个batch的运行称为一个step
    #     for step, batch in enumerate(train_dl):
    #         .....
    #         # 如果达到了梯度累加
    #         if perform_gradient_update:
    #             .......
    #             # 如果设定了学习率调度器，则调用调度器一次
    #             if (
    #                 self.scheduler is not None
    #                 and not self._accelerator.optimizer_step_was_skipped
    #             ):
    #                 self.scheduler_step()

    # 所以，原来的step_size在epoch层级假设为30个epoch，那么此时在step层级的step_size就要变为30*steps_per_epoch，才能得到同样的调用效果。（这是对于Pytorch原生的调度器，其他的调度器则要看具体情况）
    # 推测是这样调用的：对于原来的epoch层级的调用，在epoch层级进行循环，每个epoch都会调用它一次，此时step_size设为30，它内部会计数，当进行30次epoch循环后，就会更新一次；对于pytorch-accelerated的step层级的调用（假设每个epoch有5个batch，即5个step），首先是对于epoch进行循环，然后在每个epoch内部，再对step进行循环，每个step都会调用它一次，先假设step_size仍然是30，因为它内部会计数，那么当进行了6个epoch后它就会更新，这显然是错误的，所以为了达到以前的每30次epoch更新一次，step_size就不能再是30，而是变成30*5，即steps_per_epoch*原来的step_size。

    # 具体原理实现：exp_lr_schedular是一个partial实例，它会在replace_trainer_placeholder_values中被处理，它的参数，比如step_size会被TrainerPlaceholderValues中的占位符的实际计算值所赋值
    # 从而实现了实时更新。
    exp_lr_scheduler = partial(
        lr_scheduler.StepLR,
        step_size=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH * 7,
        gamma=0.1,
    )

    # 将模型、损失函数、优化器传入Trainer即可
    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
    )

    # 调用train函数，设置训练集、验证集、epoch数、batch size和学习率调度器
    trainer.train(
        train_dataset=image_datasets["train"],
        eval_dataset=image_datasets["val"],
        num_epochs=1,
        per_device_batch_size=4,
        create_scheduler_fn=exp_lr_scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    args = parser.parse_args()
    main(args.data_dir)

```

# 渐进式调整大小的案例
```python
# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a ResNet50d on the Imagenette Dataset using progressive resizing

# Note: this example requires installing the torchmetrics and timm packages
########################################################################

# 这个例子使用了渐进式大小调整progressive resizing
# 该技术的一个讲解可以参考该例子：https://www.yanxishe.com/TextTranslation/1614

import argparse
import os
# python的具名元组，见：https://www.runoob.com/note/25726
from collections import namedtuple
from functools import partial
from pathlib import Path

import torch
from timm import create_model
from torch import nn
# 使用torch原生的OneCycleLR，该调度器介绍参考：https://zhuanlan.zhihu.com/p/387162205
from torch.optim.lr_scheduler import OneCycleLR
# 额外添加精度这一指标
from torchmetrics import Accuracy
from torchvision import transforms, datasets

# 导入内置的回调事件
from pytorch_accelerated.callbacks import (
    TerminateOnNaNCallback,
    LogMetricsCallback,
    PrintProgressCallback,
    EarlyStoppingCallback,
    SaveBestModelCallback,
    TrainerCallback,
    ProgressBarCallback,
)

from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues

# 创建一个新的回调来计算精度
class AccuracyCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.accuracy = Accuracy(num_classes=num_classes)

    # 在训练触发时将精度变量放到正确的设备上
    def on_training_run_start(self, trainer, **kwargs):
        self.accuracy.to(trainer._eval_dataloader.device)

    # 在每一个验证步结束时更新精度
    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    # 在每一个验证epoch结束时更新精度指标，并重置
    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()


def create_transforms(train_image_size=224, val_image_size=224):
    # Data augmentation and normalization for training
    # Just normalization for validation
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(train_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(round(1.15 * val_image_size))),
                transforms.CenterCrop(val_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def main(data_dir):

    data_dir = Path(data_dir)
    num_classes = len(list((data_dir / "train").iterdir()))

    model = create_model("resnet50d", pretrained=False, num_classes=num_classes)

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01 / 25)

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            TerminateOnNaNCallback,
            AccuracyCallback(num_classes=num_classes),
            PrintProgressCallback,
            ProgressBarCallback,
            LogMetricsCallback,
            EarlyStoppingCallback(early_stopping_patience=2),
            SaveBestModelCallback(watch_metric="accuracy", greater_is_better=True),
        ),
    )

    EpochConfig = namedtuple(
        "EpochConfig", ["num_epochs", "train_image_size", "eval_image_size", "lr"]
    )

    # 设置不同的图像尺寸
    epoch_configs = [
        EpochConfig(num_epochs=2, train_image_size=64, eval_image_size=64, lr=0.01),
        EpochConfig(num_epochs=3, train_image_size=128, eval_image_size=128, lr=0.01),
        EpochConfig(num_epochs=6, train_image_size=224, eval_image_size=224, lr=0.001),
    ]

    # 渐进式调整图像大小
    for e_config in epoch_configs:
        trainer.print(f"Training with image size: {e_config.train_image_size}")

        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x),
                create_transforms(
                    train_image_size=e_config.train_image_size,
                    val_image_size=e_config.eval_image_size,
                )[x],
            )
            for x in ["train", "val"]
        }

        # Here we use placeholders for the number of epochs and number of steps per epoch, so that the
        # trainer can inject those values later. This is especially key for the number of update steps
        # which will change depending on whether training is distributed or not
        lr_scheduler = partial(
            OneCycleLR,
            max_lr=e_config.lr,
            epochs=TrainerPlaceholderValues.NUM_EPOCHS,
            steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        )

        trainer.train(
            train_dataset=image_datasets["train"],
            eval_dataset=image_datasets["val"],
            num_epochs=e_config.num_epochs,
            create_scheduler_fn=lr_scheduler,
            per_device_batch_size=32,
            reset_run_history=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    args = parser.parse_args()
    main(args.data_dir)
```