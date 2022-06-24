---
title: PyTorch图像模型库timm解析
tags: [PyTorch]
categories: machine learning 
date: 2022-6-25
---

参考文档：[0](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055)、[1](https://timm.fast.ai/)、[2](https://rwightman.github.io/pytorch-image-models/)

# 简介
[`PyTorch Image Models (timm)`](https://github.com/rwightman/pytorch-image-models)是[`Ross Wightman`](https://twitter.com/wightmanr)创建的深度学习库，是一个大型集合，包括了`SOTA`计算机视觉模型、神经网络层、实用函数、优化器、调度器、数据加载器、数据增强器以及训练/验证脚本等。

# 安装
```python
pip install timm
```
## 示例数据集（可选）
在演示之前，先下载一些流行的数据集作为示范。在这里，[Chris Hughes](https://medium.com/@chris.p.hughes10)使用了两个数据集：
- [牛津大学`IIIT`宠物数据集](https://www.robots.ox.ac.uk/~vgg/data/pets/)，该数据集有`37`个类别，每个类别大约有`200`张图片
- [`Imagenette`](https://github.com/fastai/imagenette)，这是`Imagenet`中`10`个容易分类的类别的一个子集。

(1)`IIIT`宠物数据集
下载并解压：
```python
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P pets
tar zxf pets/images.tar.gz -C pets
```

（2）`Imagenette`数据集
下载并解压：
```python
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -P imagenette
tar zxf imagenette/imagenette2-320.tgz -C imagenette
gzip -d imagenette/imagenette2-320.tgz
```
# 模型
`timm`最受欢迎的功能之一是其庞大且不断增长的模型架构集合。其中大部分模型包含预训练的权重——这些权重要么是在`PyTorch`中原生训练的，要么是从`Jax`和`TensorFlow`等其他库中移植的——可以轻松下载和使用。
## 列出可用模型
列出所有可用模型：
```python
import timm
timm.list_models()
```
列出所有可用的预训练模型：
```python
timm.list_models(pretrained=True)
```
通过通配符搜索特定模型：
```python
all_densenet_models = timm.list_models('*densenet*')
```
`timm`中有几百个模型，且该数字还在不断增长，如果你觉得选择困难的话，可以参考`Papers with code`上的[总结页](https://paperswithcode.com/lib/timm)，它包含了`timm`中许多模型的基准和原始论文的链接。
## 创建模型
### 常规用法
```python
import timm 
model = timm.create_model('resnet34')
```
使用`timm`创建模型非常简单。`create_model`是一个用来可以创建超过`300`个模型的工厂函数。
创建一个预训练模型，则仅需额外传递一个参数：
```python
model = timm.create_model('resnet34', pretrained=True)
```
为了进一步了解如何使用这个模型，可以访问它的配置：
```python
model.default_cfg
```
其中包含的信息有：应该用来归一化输入数据的统计数据`mean`和`std`、输出类别的数目`num_classes`和网络中分类器的名称`classifier`等信息。
也可以直接打印出整个模型的架构：
```python
print(model)
```

### 创建可变输入通道数目的图像的预训练模型
`timm`模型有一个不太为人所知、但却非常有用的特点，那就是它们能够处理具有不同通道数的输入图像，这对大多数其他库来说都是一个问题；[这里](https://timm.fast.ai/models#So-how-is-timm-able-to-load-these-weights?)给出了一个关于这个工作原理的出色解释。直观地说，`timm`通过对少于3个通道的初始卷积层的权重进行求和，或者智能地将这些权重复制到所需的通道数上，来实现这一目的。
```python
model = timm.create_model('resnet34', pretrained=True, in_chans=1)
```
值得注意的是，虽然这使我们能够使用一个预训练的模型，但输入的图像与模型训练所基于的图像有很大的不同。正因为如此，我们不应该期待同样的性能水平，在将模型用于任务之前，应该在新的数据集上对其进行微调。

## 定制化模型
除了用现有架构创建模型外，`create_model`还支持一些参数，使我们能够为特定的任务定制一个模型。
不过需要注意的是，支持的参数可能取决于底层的模型架构。
- 一些参数，如`global_pool`就是与具体模型相关，该参会决定全局池化的类型，它在类`ResNet`的模型中是有效的，但就不适用于比如`ViT`这样的模型，因为`ViT`不使用平均池化。
- 另一些参数，如丢弃率`drop_rate`和输出类别数`num_classes`就适用于大多数模型。

所以提前查看当前模型的默认架构是非常有必要的。

以之前的`resnet34`为例，看如何定制模型：
```python
model = timm.create_model('resnet34', pretrained=True)
```
其默认配置为：
```python
model.default_cfg

{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
 'num_classes': 1000,
 'input_size': (3, 224, 224),
 'pool_size': (7, 7),
 'crop_pct': 0.875,
 'interpolation': 'bilinear',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'first_conv': 'conv1',
 'classifier': 'fc',
 'architecture': 'resnet34'}
```

### 改变输出类别数量
由上面的模型配置可以看出，网络的分类器名字是`fc`。可以用它来直接访问相应的模块：
```python
model.fc

Linear(in_features=512, out_features=1000, bias=True)
```
然而，这个名字很可能会根据使用的模型架构而改变。为了给不同的模型提供一个一致的接口，`timm`模型有`get_classifier`方法，我们可以用它来获得分类器，而不需要查询模块名称：
```python
model.get_classifier()
```
由于这个模型是在`ImageNet`上预训练的，我们可以看到最后一层输出`1000`个类。可以通过`num_classes`参数来改变这一点。
创建一个自定义类别数目的模型，仅需额外传递一个参数：
```python
model = timm.create_model('resnet34', num_classes=10)
```
此时查看该模型的分类器，可以看到，`timm`已经用一个新的、未经训练的、具有所需类别数的线性层替换了最后一层；然后就可以在自己的数据集上进行微调。

如果想完全避免创建最后一层，可以将类的数量设置为`0`，这将创建一个以`Identity()`恒等函数为最后一层的模型；这对检查倒数第二层的输出很有用。

### 全局池化
依然从上面的模型配置中可以看到`pool_size`参数，表明在分类器之前由一个全局池化层。可以通过如下命令查看：
```python
model.global_pool

SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
```
可以看到，返回了一个`SelectAdaptivePool2d`实例， 这是一个由`timm`提供的自定义层，支持不同的池化和压平配置，包括：
- `avg`：平均池化
- `max`：最大池化
- `avgmax`：平均池化和最大池化的和，然后`0.5`倍缩放
- `catavgmax`：沿着特征维度将平均池化和最大池化的输出连接起来。注意，这将使特征维度增加一倍。
- `''`：不使用池化，池化层倍一个`Indentity`恒等函数所替代

通过以下代码查看一下不同池化选项的效果：
```python
pool_types = ['avg', 'max', 'avgmax', 'catavgmax', '']

for pool in pool_types:
    # 这里一定要设置num_classes=0，
    # 否则在catavgmax和''两种情形下都会报错，因为它改变了原来模型架构，无法与分类器正确连接
    # 这里设置了num_classes=0，实际就是查看倒数第二层（即全局池化层）的输出形状
    model = timm.create_model('resnet34', pretrained=True, num_classes=0, global_pool=pool)
    model.eval()
    feature_output = model(torch.randn(1, 3, 224, 224))
    print(feature_output.shape)
```

### 修改已有模型
可以通过`reset_classifier`方法来修改已有模型：
```python
model = timm.create_model('resnet34', pretrained=True)
print(f'Original pooling: {model.global_pool}')
print(f'Original classifier: {model.get_classifier()}')
print('--------------')
model.reset_classifier(10, 'max')
print(f'Modified pooling: {model.global_pool}')
print(f'Modified classifier: {model.get_classifier()}')

Original pooling: SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
Original classifier: Linear(in_features=512, out_features=1000, bias=True)
--------------
Modified pooling: SelectAdaptivePool2d (pool_type=max, flatten=Flatten(start_dim=1, end_dim=-1))
Modified classifier: Linear(in_features=512, out_features=10, bias=True)
```
### 创建新的分类器
虽然已经证明使用单一的线性层作为分类器足以取得良好的效果，但在下游任务上微调模型时，[Chris Hughes](https://medium.com/@chris.p.hughes10)发现使用一个稍大的头可以导致性能的提高。
接下来探讨一下如何进一步修改之前的`ResNet`模型。
首先，以前一样创建`ResNet`模型，指定需要`10`个输出类别。由于使用的是一个较大的头，这里使用`catavgmax`来进行池化，这样就可以提供更多的信息作为分类器的输入。
```python
model = timm.create_model('resnet34', pretrained=True, num_classes=10, global_pool='catavgmax')
```
对于该模型的已有分类器，看一下它的输入特征：
```python
num_in_features = model.get_classifier().in_features
num_in_features

1024
```
下面用一个自定义的分类器来直接替换原来的分类器：
```python
import torch.nn as nn
model.fc = nn.Sequential(
    nn.BatchNorm1d(num_in_features),
    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(in_features=512, out_features=10, bias=False)
)
```
使用一个模拟数据来测试一下新分类器的输出：
```python
model.eval()
model(torch.randn(1, 3, 224, 224)).shape

torch.Size([1, 10])
```
可以看出，结果符合预期，经过修改后的模型可以用来训练了。

## 特征提取
`timm`模型有一套统一的机制来获得各种类型的中间特征，这对于将一个架构作为下游任务的特征提取器是非常有用的。
这一部分使用宠物数据集中的图像作为一个例子。
在程序中加载`IIIT`宠物数据集：
```python
from pathlib import Path
pets_path = Path('pets/images')
pets_image_paths = list(pets_path.iterdir())
```
选取其中一张图像，并转为`PyTorch`期望的数据格式：
```python
from PIL import Image
import numpy as np

image = Image.open(pets_image_paths[1])
image = torch.as_tensor(np.array(image, dtype=np.float32)).transpose(2, 0)[None]

image.shape
torch.Size([1, 3, 500, 375])
```
使用`timm`常规用法创建一个模型（这里换成了`resnet50d`）：
```python
model = timm.create_model('resnet50d', pretrained=True)
model.default_cfg

{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
 'num_classes': 1000,
 'input_size': (3, 224, 224),
 'pool_size': (7, 7),
 'crop_pct': 0.875,
 'interpolation': 'bicubic',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'first_conv': 'conv1.0',
 'classifier': 'fc',
 'architecture': 'resnet50d'}
```
如果我们只对最终的特征图感兴趣——也就是本例中池化之前的最终卷积层的输出——可以使用`forward_features`方法来绕过全局池化和分类层：
```python
feature_output = model.forward_features(image)
```
可以对它可视化一下：
```python
import matplotlib.pyplot as plt

def visualize_feature_output(t):
    plt.imshow(feature_output[0].transpose(0, 2).sum(-1).detach().numpy())
    plt.show()

visualize_feature_output(feature_output)
```

### 多个特征输出
虽然`forward_features`方法可以方便地获得最终的特征图，但`timm`也提供了一些功能，使得可以将模型作为特征骨干，输出选定层次的特征图。
先看一个之前模型中的特征信息：
```python
model.feature_info

[{'num_chs': 64, 'reduction': 2, 'module': 'act1'},
 {'num_chs': 64, 'reduction': 4, 'module': 'layer1'},
 {'num_chs': 128, 'reduction': 8, 'module': 'layer2'},
 {'num_chs': 256, 'reduction': 16, 'module': 'layer3'},
 {'num_chs': 512, 'reduction': 32, 'module': 'layer4'}]
```
以上是常规创建的模型的输出信息。
实际上，在创建模型时，可以添加参数`features_only=True`来指定所使用模型作为特征骨干，即：
```python
model = timm.create_model('resnet50d', pretrained=True, features_only=True)

model

FeatureListNet(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
..............
```
此时生成的模型是`FeatureListNet`类型。
如下所示，可以得到更多关于返回的特征的信息，如具体的模块名称，特征的减少量和通道的数量：
```python
model.feature_info.module_name()
['act1', 'layer1', 'layer2', 'layer3', 'layer4']

model.feature_info.reduction()
[2, 4, 8, 16, 32]

model.feature_info.channels()
[64, 256, 512, 1024, 2048]
```
默认情况下，大多数模型将输出`5`层（并非所有模型都有这么多步长），第一层从`2`开始（但有些从`1`或`4`开始）。
可以使用`out_indices`和`output_stride`参数来修改特征层的索引和数量，如[文档](https://rwightman.github.io/pytorch-image-models/feature_extraction/#multi-scale-feature-maps-feature-pyramid)中所示。
将图像传入该特征提取模型中，看一下它的输出：
```python
out = model(image)

for o in out:
    print(o.shape)

torch.Size([1, 64, 250, 188])
torch.Size([1, 256, 125, 94])
torch.Size([1, 512, 63, 47])
torch.Size([1, 1024, 32, 24])
torch.Size([1, 2048, 16, 12])
```
可以看出，能返回`5`个特征图，以及形状和通道数都符合预期。
还可以具体可视化一下特征图：
```python
for o in out:
    plt.imshow(o[0].transpose(0, 2).sum(-1).detach().numpy())
    plt.show()
```

### 使用Torch FX
`TorchVision`最近发布了一个名为`FX`的新工具，它可以更容易地访问`PyTorch Module`正向传递过程中的输入的中间转换。具体是通过符号性地运行前向方法来产生一个图`graph`，其中每个节点代表一个操作。由于节点被赋予了人类可读的名称，所以很容易准确地指定我们要访问的节点。`FX`在[这篇文档](https://pytorch.org/docs/stable/fx.html#module-torch.fx)和[这篇博文](https://pytorch.org/blog/FX-feature-extraction-torchvision/)中有更详细的描述。
注意：`Chris Hughes`在撰写[本教程](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583)时，使用`FX`时，动态控制流还不能用静态图来表示。
由于`timm`中几乎所有的模型都可以用符号追踪，我们可以用`FX`来操作这些模型。
下面来探讨一下如何使用`FX`从`timm`模型中提取特征。
（1）获取节点：
```python
# 导入fx必要的包
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
# 在创建模型时指定exportable参数，使得模型可被追踪
model = timm.create_model('resnet50d', pretrained=True, exportable=True)
# 获得节点
# 因为模型分别以train和evel模式都执行一次，所以两种模式下的节点名称都会返回。
nodes, _ = get_graph_node_names(model)

nodes

['x',
 'conv1.0',
 'conv1.1',
 'conv1.2',
 'conv1.3',
 'conv1.4',
 'conv1.5',
 'conv1.6',
 'bn1',
 'act1',
 'maxpool',
 'layer1.0.conv1',
 'layer1.0.bn1',
 'layer1.0.act1',
 'layer1.0.conv2',
 'layer1.0.bn2',
 'layer1.0.act2',
 ............
```
（2）特征提取器：
```python
# 使用FX可以很容易地获得任意节点的输出
# 这里以选择layer1的第二个激活函数为例
features = {'layer1.0.act2': 'out'}

# 使用create_feature_extractor可以在这个点上切断整个模型
feature_extractor = create_feature_extractor(model, return_nodes=features)
# 切断后的模型如下
feature_extractor

ResNet(
  (conv1): Module(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Module(
    (0): Module(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
    )
  )
)
```
（3）提取特征：
```python
# 输入图像，返回特征
out = feature_extractor(image)
# 可视化一下
plt.imshow(out['out'][0].transpose(0, 2).sum(-1).detach().numpy())
```

## 模型导出
训练结束后，通常建议将模型导出为优化的格式，以便进行推理；`PyTorch`有多种导出选项可以做到这一点。由于几乎所有的`timm`模型都是可编写脚本和可追踪的，因此可以利用这些格式。

### 导出为TorchScript
`TorchScript`是一种从`PyTorch`代码中创建可序列化和可优化的模型的方法；任何`TorchScript`程序都可以从`Python`进程中保存，并在没有`Python`依赖性的进程中加载。
可以通过两种不同的方式将一个模型转换为`TorchScript`。
- 追踪：运行代码，记录发生的操作，并构造一个包含这些操作的`ScriptModule`。控制流或动态行为（如`if/else`语句）会被抹去。
- 脚本化：使用脚本编译器对`Python`源代码进行直接分析，将其转化为`TorchScript`。这保留了动态控制流，对不同大小的输入都有效。

关于`TorchScript`的更多信息可以在[该文档](https://pytorch.org/docs/stable/jit.html)和[该教程](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)中看到。
由于大多数`timm`模型是可编写脚本的，这里使用脚本来导出上面的`ResNet-D`模型。可以在创建模型时使用`scriptable`参数来使模型是`jit`可脚本化。
```python
model = timm.create_model('resnet50d', pretrained=True, scriptable=True)
model.eval()
```
在导出模型之前调用`model.eval()`是非常重要的，这样可以使模型进入推理模式，因为诸如`dropout`和`batchnorm`这样的运算符在不同的模式下表现不同。
确认一下可以脚本化模型：
```python
scripted_model = torch.jit.script(model)

scripted_model

RecursiveScriptModule(
  original_name=ResNet
  (conv1): RecursiveScriptModule(
    original_name=Sequential
    (0): RecursiveScriptModule(original_name=Conv2d)
    (1): RecursiveScriptModule(original_name=BatchNorm2d)
    (2): RecursiveScriptModule(original_name=ReLU)
    (3): RecursiveScriptModule(original_name=Conv2d)
    (4): RecursiveScriptModule(original_name=BatchNorm2d)
    (5): RecursiveScriptModule(original_name=ReLU)
    (6): RecursiveScriptModule(original_name=Conv2d)
  )
```
同时模型也能正常使用：
```python
scripted_model(torch.rand(8, 3, 224, 224)).shape

torch.Size([8, 1000])
```

### 导出为ONNX
[`Open Neural Network eXchange(ONNX)`](https://onnx.ai/)是一种表示机器学习模型的开放标准格式。
可以使用`torch.onnx`模块将`timm`模型导出到`ONNX`，使它们能够被任何支持`ONNX`的运行时`runtimes`所使用。如果调用`torch.onnx.export()`的模块不是`ScriptModule`，它首先会做相当于`torch.jit.trace()`的工作；用给定的`args`执行一次模型，并记录执行期间发生的所有操作。这意味着，如果模型是动态的，例如，根据输入数据改变行为，导出的模型将不能捕捉到这种动态行为。同样，跟踪可能只对特定的输入尺寸有效。
关于`ONNX`的更多细节可以在[该文档](https://pytorch.org/docs/master/onnx.html)中找到。
为了能够以`ONNX`格式导出一个`timm`模型，可以在创建模型时使用`exportable`参数，以确保模型是可追踪的：
```python
model = timm.create_model('resnet50d', pretrained=True, exportable=True)
model.eval()
```
然后使用`torch.onnx.export`来追踪和导出模型：
```python
x = torch.randn(2, 3, 224, 224, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,                                       # model being run
                  x,                                           # model input (or a tuple for multiple inputs)
                  "resnet50d.onnx",                            # where to save the model (can be a file or file-like object)
                  export_params=True,                          # store the trained parameter weights inside the model file
                  opset_version=10,                            # the ONNX version to export the model to
                  do_constant_folding=True,                    # whether to execute constant folding for optimization
                  input_names = ['input'],                     # the model's input names
                  output_names = ['output'],                   # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},  # variable length axes
                                'output': {0 : 'batch_size'}})
```
使用`check_model`验证一下模型是否有效：
```python
import onnx

onnx_model = onnx.load("resnet50d.onnx")
onnx.checker.check_model(onnx_model)
```
由于已经指定模型应该是可追踪的，也可以手动进行追踪，如下所示：
```python
traced_model = torch.jit.trace(model, torch.rand(8, 3, 224, 224))
traced_model(torch.rand(8, 3, 224, 224)).shape
```

# 数据增强
`timm`包括很多数据增强变换，它们可以被串联起来组成增强管道；与`TorchVision`类似，这些管道需要一个`PIL`图像作为输入。
最简单的方法是使用`create_transform`工厂函数，下面探索如何使用它。
```python
from PIL import Image
from timm.data.transforms_factory import create_transform

create_transform(224,)

Compose(
    Resize(size=256, interpolation=bilinear, max_size=None, antialias=None)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
)
```
可以看到，`create_transform`已经创建了一些基本的增强管道，包括调整大小、归一化和将图像转换为张量。
```python
create_transform(224, is_training=True)

Compose(
    RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=None)
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
)
```
正如所期望的那样，可以看到，当设置`is_training=True`时，额外的转换，如水平翻转和颜色抖动，也包括在内。这些增强方式的数值大小可以通过参数`hflip`、`vflip`和`color_jitter`来控制。
还可以看到，用于调整图像大小的方法也因是否是模型训练而不同。在验证期间使用标准的`Resize`和`CenterCrop`，而在训练期间则使用`RandomResizedCropAndInterpolation`。
通过下面的代码可以看看`RandomResizedCropAndInterpolation`具体干了什么。由于`timm`中这个变换的实现使我们能够设置不同的图像插值方法；在这里我们选择插值是`random`，即随机选择。
```python
image = Image.open(pets_image_paths[0])

from timm.data.transforms import RandomResizedCropAndInterpolation
tfm = RandomResizedCropAndInterpolation(size=350, interpolation='random')


import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 4, figsize=(10, 5))
for idx, im in enumerate([tfm(image) for i in range(4)]):
    ax[0, idx].imshow(im)   
for idx, im in enumerate([tfm(image) for i in range(4)]):
    ax[1, idx].imshow(im)
fig.tight_layout()
plt.show()
```
多次运行该转换，可以观察到对图像进行了不同的剪裁。虽然这在训练过程中是有益的，但在评估过程中可能会增加任务的难度。根据图片的类型，这种类型的转换可能会导致图片的主体被裁剪掉。如果这种情况不常发生，这应该不是一个大问题，可以通过调整比例参数来避免这种情况。
```python
tfm = RandomResizedCropAndInterpolation(size=224, scale=(0.8, 1))
```

## RandAugment
当开始一个新的任务时，可能很难知道要使用哪些增强，以及以何种顺序使用；由于现在有大量的增强，组合的数量是巨大的。
通常，一个好的开始是使用一个在其他任务上表现出良好性能的增强管道。`RandAugment`就是这样一个策略，它是一种自动化的数据增强方法，从一组增强中统一采样操作——如均衡化、旋转、过曝、颜色抖动、海报化、改变对比度、改变亮度、改变锐度、剪切和平移——并按顺序应用其中的一些；更多信息请参见[原始论文](https://arxiv.org/abs/1909.13719)。
然而，在`timm`中提供的实现有几个关键的区别，这些区别由`timm`的创造者`Ross Wightman`在[`ResNets Strike Back`](https://arxiv.org/pdf/2110.00476v1.pdf)论文的附录中做了最好的描述，将其转述如下：
> 原始的`RandAugment`规范有两个超参数，即`M`和`N`；其中`M`是变换幅度，`N`是每幅图像统一采样和应用的变换数量。`RandAugment`的目标是，`M`和`N`都是人类可以解释的。
> 然而，[在最初的实施中]M的情况最终并非如此。一些增强随着数值变大却是倒退的，或者在范围内不是单调增加的，因此增加`M`并不能增加所有增强的效果。
> `timm`的实现试图通过增加一个`increasing`模式（默认启用）来改善这种情况，在这种模式下，所有的增强的效果都会随着幅度的增加而增加。
> 此外，`timm`增加了一个`MSTD`参数，它在每个变换的`M`值中增加了具有指定标准偏差的高斯噪声。如果`MSTD`被设置为`'-inf'`，则每次变换时，`M`会从`0-M`中均匀地取样。
> `timm`的`RandAugment`会注意减少对图像平均值的影响，归一化参数可以作为一个参数传递，这样所有可能引入边界像素的增强可以使用指定的平均值，而不是像其他实现那样默认为`0`或一个硬编码的元组。
> 最后，默认情况下不包括`Cutout`，以支持单独使用`timm`的随机擦除实现，这对平均数和标准偏差的影响较小。

随机擦除的实现可以查看[该文章](https://timm.fast.ai/RandomErase)。
现在了解了什么是`RandAugment`，再看看如何在增强管道中使用它。
在`timm`中，通过使用配置字符串来定义`RandAugment`策略的参数；它由多个部分组成，以破折号（`-`）分隔：第一个部分定义了`RandAugment`的具体变体（目前只支持`Rand`），其余部分可以按任何顺序排列，它们是：
- `m`：整型，增强的强度
- `n`：整型，每张图像选择的变换的数目，可选，默认设置为`2`
- `mstd`：浮点型，施加的幅度噪声的标准差
- `mmax`：整型，设置幅度的上限，默认为`10`
- `w`：整型，概率权重指数（影响操作选择的一组权重的指数）
- `inc`：布尔型，是否使用随幅度增加而增加的增强，这是可选的，默认为`0`

比如：
- `rand-m9-n3-mstd0.5`：幅度为`9`、每张图像有`3`个增强操作、噪声标准差为`0.5`的随机增强
- `rand-mstd1-w0`：噪声标准差`1.0`、概率权重指数`0`、默认强度最大值为`10`、每张图像有`2`个增强操作

向`create_transform`传递一个配置字符串，如下可以看到这是由`RandAugment`对象处理，而且可以看到所有可用的操作的名称：
```python
create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')

Compose(
    RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandAugment(n=2, ops=
    AugmentOp(name=AutoContrast, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Equalize, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Invert, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Rotate, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Posterize, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Solarize, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=SolarizeAdd, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Color, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Contrast, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Brightness, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Sharpness, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=ShearX, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=ShearY, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=TranslateXRel, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=TranslateYRel, p=0.5, m=9, mstd=0.5))
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
)
```
还可以直接通过使用`rand_augment_transform`函数来创建这个`RandAugment`对象：
```python
from timm.data.auto_augment import rand_augment_transform

tfm = rand_augment_transform(
    config_str='rand-m9-mstd0.5', 
    hparams={'img_mean': (124, 116, 104)}
)
tfm

RandAugment(n=2, ops=
    AugmentOp(name=AutoContrast, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Equalize, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Invert, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Rotate, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Posterize, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Solarize, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=SolarizeAdd, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Color, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Contrast, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Brightness, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=Sharpness, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=ShearX, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=ShearY, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=TranslateXRel, p=0.5, m=9, mstd=0.5)
    AugmentOp(name=TranslateYRel, p=0.5, m=9, mstd=0.5))
```
可以将该增强策略应用到图像上，看看其效果：
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 4, figsize=(10, 5))

for idx, im in enumerate([tfm(image) for i in range(4)]):
    ax[0, idx].imshow(im)
for idx, im in enumerate([tfm(image) for i in range(4)]):
    ax[1, idx].imshow(im)

fig.tight_layout()
plt.show()
```

## CutMix和Mixup
`timm`使用它的`Mixup`类为[`CutMix`](https://arxiv.org/abs/1905.04899)和[`Mixup`](https://arxiv.org/abs/1710.09412)增强功能提供了一个灵活的实现，它可以处理这两种增强功能并提供在它们之间切换的选项。
通过使用`Mixup`，可以从各种不同的混合策略中进行选择：
- `batch`：在每个批次上进行`CutMix`与`Mixup`的选择、`lambda`和`CutMix`区域采样
- `pair`：在一个批次内的取样对上进行混合、`lambda`和区域取样。
- `elem`：在批次内的每个图像上进行混合、`lambda`和区域取样。
- `half`：与`elementwise`相同，但每个混合对中的一个被丢弃，这样每个样本在每个`epoch`中被看到一次

下面看一下具体是怎样工作的。
首先得需要创建一个数据加载器、迭代器，然后才能将这些增强施加到`batch`上。
```python
from timm.data import ImageDataset
from torch.utils.data import DataLoader

def create_dataloader_iterator():
    dataset = ImageDataset('pets/images', transform=create_transform(224))
    dl = iter(DataLoader(dataset, batch_size=2))
    return dl

dataloader = create_dataloader_iterator()
inputs, classes = next(dataloader)
```
这里再创建一个可视化函数：
```python
# Taken from timmdocs https://fastai.github.io/timmdocs/mixup_cutmix
import numpy as np
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

import torchvision
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[x.item() for x in classes])
```
下面创建`Mixup`变换，其支持如下参数：
- `mixup_alpha`：浮点型，`mixup`的`alpha`值，如果大于`0`，`mixup`将被激活（默认为`1`）
- `cutmix_alpha`：浮点型，`cutmix`的`alpha`值，如果大于`0`，则`cutmix`激活（默认是`0`）。
- `cutmix_minmax`：`List[float])`型，`cutmix`的最小/最大图像比例，如果不是`None`，`cutmix`将被激活并使用这个与`alpha`的比值。
- `prob`：`float`型， 每个批次或元素应用`mixup`或`cutmix`的概率（默认是`1`）。
- `switch_prob`：`float`型，当两者都激活时，切换到`cutmix`而不是`mixup`的概率（默认是`0.5`）。
- `mode`：`str`型， 如何应用`mixup/cutmix`参数（默认是`batch`）
- `label_smoothing`：浮点型，应用于混合目标张量的标签平滑量（默认是`0.1`）
- `num_classes`：`int`型，目标变量的类别数量。

创建一个`Mixup`变换：
```python
from timm.data.mixup import Mixup

mixup_args = {
    'mixup_alpha': 1.,
    'cutmix_alpha': 1.,
    'prob': 1,
    'switch_prob': 0.5,
    'mode': 'batch',
    'label_smoothing': 0.1,
    'num_classes': 2}
mixup_fn = Mixup(**mixup_args)
```
由于`mixup`和`cutmix`是在一批次图像上进行的，可以在应用增强之前将这批图像放在`GPU`上，以加快进度：
```python
mixed_inputs, mixed_classes = mixup_fn(inputs.to(torch.device('cuda:0')), classes.to(torch.device('cuda:0')))
out = torchvision.utils.make_grid(mixed_inputs)
imshow(out, title=mixed_classes)
```

# 数据集
`timm`提供了许多有用的工具来处理不同类型的数据集。最简单的入门方法是使用`create_dataset`函数，它将为我们创建一个合适的数据集。
`create_dataset`需要有两个参数：
- `name`：要加载的数据集的名称
- `root`：数据集在本地文件系统中的根文件夹。

也可以有额外的关键字参数用于指定选项，如是否要加载训练集或验证集。
还可以使用`create_dataset`来加载来自不同地方的数据：
- [`TorchVision`](https://pytorch.org/vision/main/datasets.html)数据集
- [`TensorFlow`](https://www.tensorflow.org/datasets)数据集
- 存储在本地文件夹中的数据集

## 加载TorchVision数据集
要加载`TorchVision`包含的数据集，只需在希望加载的数据集的名称前指定前缀`torch/`。如果数据在文件系统中不存在，可以通过设置`download=True`来下载这些数据。此外，还可以使用`split`参数来指定加载训练数据集。
```python
from timm.data import create_dataset
ds = create_dataset('torch/cifar10', 'cifar10', download=True, split='train')
```

## 加载TensorFlow数据集
`timm`还可以使得从`TensorFlow`数据集中下载和使用数据集；同时封装了底层的`tfds`对象。
当加载`TensorFlow`数据集时，在数据集的名称前加上`tfds/`。此时建议设置几个额外的参数，这些参数对于本地或`TorchVision`数据集来说是不需要的。
- `batch_size`：这是用来确保在分布式训练过程中，样本总数划分到所有节点上能整除批处理大小。
- `is_training`：如果设置了，数据集将被打乱。注意，这与设置`split`是不同的。

虽然这个封装从`TFDS`数据集中返回解压缩的图像示例，但需要的任何增强和批处理仍然由`PyTorch`处理。
```python
ds = create_dataset('tfds/beans', 'beans', download=True, split='train[:10%]', batch_size=2, is_training=True)
```

## 加载本地数据
也可以从本地文件夹加载数据，在这种情况下，只需使用一个空字符串（`''`）作为数据集名称。
除了能够从`ImageNet`风格的文件夹层次中加载数据外，`create_dataset`还可以让我们从一个或多个`tar`档案中提取数据；可以用它来避免解开档案的麻烦。
作为一个例子，可以在`Imagenette`数据集上试试这个方法。
此外，到目前为止，一直在加载原始图像，所以这里也使用变换参数来应用一些变换：
```python
ds = create_dataset(name='', root='imagenette/imagenette2-320.tar', transform=create_transform(224))
```

## ImageDataset类
如上所述，`create_dataset`函数为处理不同类型的数据提供了很多选择。`timm`之所以能够提供这样的灵活性，是通过尽可能地使用`TorchVision`中提供的现有数据集类，以及提供一些额外的实现——`ImageDataset`和`IterableImageDataset`，它们可用于广泛的场景。
从本质上讲，`create_dataset`通过选择一个合适的类为我们简化了这个过程，但有时我们可能希望直接与底层组件一起工作。
`Chris Hughes`最常使用的实现是`ImageDataset`，它类似于`torchvision.datasets.ImageFolder`，但有一些附加功能。
下面探讨一下如何使用它来加载之前解压缩的`imagenette`数据集：
```python
from timm.data import ImageDataset
imagenette_ds = ImageDataset('imagenette/imagenette2-320/train')
```
`ImageDataset`的灵活性的关键在于，它索引和加载样本的方式被抽象成一个解析器对象`parser`。
`timm`中包含了多个解析器，包括从文件夹、`tar`文件和`tensorflow`数据集读取图像的解析器。解析器可以作为一个参数传递给数据集，可以直接访问解析器。
```python
imagenette_ds.parser

<timm.data.parsers.parser_image_folder.ParserImageFolder at 0x7f66e8146ee0>
```
可以看到，默认的解析器是`ParserImageFolder`的一个实例。解析器还包含有用的信息，比如类别查找，如下所示：
```python
imagenette_ds.parser.class_to_idx

{'n01440764': 0,
 'n02102040': 1,
 'n02979186': 2,
 'n03000684': 3,
 'n03028079': 4,
 'n03394916': 5,
 'n03417042': 6,
 'n03425413': 7,
 'n03445777': 8,
 'n03888257': 9}
```
### 手动选择解析器——以tar包为例
因此，除了选择一个合适的数据集类之外，`create_dataset`还负责选择正确的解析器。
再次考虑压缩的`Imagenette`数据集，可以通过手动选择`ParserImageInTarparser`并覆盖`ImageDataset`的默认解析器来实现同样的结果：
```python
from timm.data.parsers.parser_image_in_tar import ParserImageInTar

data_path = 'imagenette'
ds = ImageDataset(data_path, parser=ParserImageInTar(data_path))
```
### 自定义解析器——以pets数据集为例
遗憾的是，数据集的结构并不总是像`ImageNet`那样；也就是说，具有以下结构：
```python
root/class_1/xx1.jpg
root/class_1/xx2.jpg
root/class_2/xx1.jpg
root/class_2/xx2.jpg
```
对于这些数据集，`ImageDataset`不会开箱即用。虽然我们总是可以实现一个自定义的数据集来处理这个问题，但这可能是一个挑战，取决于数据的存储方式。另一个选择是编写一个与`ImageDataset`配合使用的自定义解析器。
作为一个例子，考虑前面牛津大学的宠物数据集，其中所有的图片都位于一个文件夹中，而类的名称——在这种情况下是每个品种的名称——包含在文件名中：
```python
ls pets/images/

Abyssinian_100.jpg*                 keeshond_186.jpg*
Abyssinian_100.mat                  keeshond_187.jpg*
Abyssinian_101.jpg*                 keeshond_188.jpg*
Abyssinian_101.mat                  keeshond_189.jpg*
Abyssinian_102.jpg*                 keeshond_18.jpg*
Abyssinian_102.mat                  keeshond_190.jpg*
Abyssinian_103.jpg*                 keeshond_191.jpg*
Abyssinian_104.jpg*                 keeshond_192.jpg*
Abyssinian_105.jpg*                 keeshond_193.jpg*
Abyssinian_106.jpg*                 keeshond_194.jpg*
................
```
在这种情况下，由于我们仍然是从本地文件系统加载图片，所以只需对`ParserImageFolder`稍作调整。
先看看`ParserImageFolder`是如何实现的，以获得启发：
```python
??timm.data.parsers.parser_image_folder.ParserImageFolder

class ParserImageFolder(Parser):
    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
```
可以看到，`ParserImageFolder`做了几件事：
- 为类别创建一个映射`class_map`
- 实现`__len__`以返回样本的数量
- 实现`__filename`来返回样本的文件名，通过选项来决定它应该是绝对路径还是相对路径
- 实现`__getitem__`以返回样本和目标。

现在理解了必须实现的方法，可以在此基础上创建自定义的实现。此处使用了标准库中的`pathlib`来提取类别名并处理路径（可能比`os`更容易操作）：
```python
from pathlib import Path
from timm.data.parsers.parser import Parser

class ParserImageName(Parser):
    def __init__(self, root, class_to_idx=None):
        super().__init__()

        self.root = Path(root)
        self.samples = list(self.root.glob("*.jpg"))

        if class_to_idx:
            self.class_to_idx = class_to_idx
        else:
            classes = sorted(
                set([self.__extract_label_from_path(p) for p in self.samples]),
                key=lambda s: s.lower(),
            )
            self.class_to_idx = {c: idx for idx, c in enumerate(classes)}

    def __extract_label_from_path(self, path):
        return "_".join(path.parts[-1].split("_")[0:-1])

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.class_to_idx[self.__extract_label_from_path(path)]
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = filename.parts[-1]
        elif not absolute:
            filename = filename.absolute()
        return filename
```
现在就可以把解析器的一个实例传递给`ImageDataset`，应该能使它正确地加载宠物数据集：
```python
data_path = Path('pets/images')
ds = ImageDataset(str(data_path), parser=ParserImageName(data_path))
ds[0]

(<PIL.Image.Image image mode=RGB size=500x332>, 9)
```
此外，与默认的解析器一样，可以查看类别与索引之间的映射：
```python
ds.parser.class_to_idx

{'Abyssinian': 0,
 'american_bulldog': 1,
 'american_pit_bull_terrier': 2,
 'basset_hound': 3,
 'beagle': 4,
 'Bengal': 5,
 'Birman': 6,
 'Bombay': 7,
 'boxer': 8,
 'British_Shorthair': 9,
 'chihuahua': 10,
 'Egyptian_Mau': 11,
 'english_cocker_spaniel': 12,
 'english_setter': 13,
 'german_shorthaired': 14,
 'great_pyrenees': 15,
 'havanese': 16,
 'japanese_chin': 17,
 'keeshond': 18,
 'leonberger': 19,
 'Maine_Coon': 20,
 'miniature_pinscher': 21,
 'newfoundland': 22,
 'Persian': 23,
 'pomeranian': 24,
 'pug': 25,
 'Ragdoll': 26,
 'Russian_Blue': 27,
 'saint_bernard': 28,
 'samoyed': 29,
 'scottish_terrier': 30,
 'shiba_inu': 31,
 'Siamese': 32,
 'Sphynx': 33,
 'staffordshire_bull_terrier': 34,
 'wheaten_terrier': 35,
 'yorkshire_terrier': 36}
```

# 优化器
`timm`具有大量的优化器，其中一些是`PyTorch`所不具备的。除了使人们能够方便地使用`SGD`、`Adam`和`AdamW`等熟悉的优化器外，还有一些值得注意的优化器有：
- `AdamP`：见[该论文](https://arxiv.org/abs/2006.08217)
- `RMSPropTF`：基于原始`TensorFlow`实现的`RMSProp`的实现，以及[这里](https://github.com/pytorch/pytorch/issues/23796)讨论的其他小的调整。根据`Chris Hughes`的经验，这通常会产生比`PyTorch`版本更稳定的训练效果。
- `LAMB`：来自`Apex`的`FusedLAMB`优化器的纯`pytorch`变体，在使用`PyTorch XLA`时，它与`TPU`兼容。
- `AdaBelief`：见[该论文](https://arxiv.org/abs/2010.07468)。关于设置超参数的指导可在[此](https://github.com/juntang-zhuang/Adabelief-Optimizer#quick-guide)获得。
- `MADGRAD`：见[该论文](https://arxiv.org/abs/2101.11075)
- `AdaHessian`：自适应二阶优化器，见[该论文](https://arxiv.org/abs/2006.00719)。

`timm`中的优化器支持与`torch.optim`中的优化器相同的接口，在大多数情况下，可以简单地放入训练脚本中，不需要做任何改动。
要查看`timm`实现的所有优化器，可以查看`timm.opt`模块：
```python
import inspect
import timm.optim

[cls_name for cls_name, cls_obj in inspect.getmembers(timm.optim) if inspect.isclass(cls_obj) if cls_name !='Lookahead']

['AdaBelief',
 'Adafactor',
 'Adahessian',
 'AdamP',
 'AdamW',
 'Lamb',
 'Lars',
 'MADGRAD',
 'Nadam',
 'NvNovoGrad',
 'RAdam',
 'RMSpropTF',
 'SGDP']
```
创建一个优化器的最简单方法是使用`create_optimizer_v2`工厂函数，该函数期望得到以下信息：
- 一个模型，或一组参数
- 优化器的名称
- 任何要传递给优化器的参数

可以使用这个函数来创建基于`timm`的优化器，以及来自`torch.optimizer`的优化器和来自`Apex`的[融合优化器](https://nvidia.github.io/apex/optimizers.html)（如果已安装）的任意的优化器。

看一下一些例子。
```python
model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.Flatten(0, 1)
)
optimizer = timm.optim.create_optimizer_v2(model, opt='sgd', lr=0.01, momentum=0.8); 

optimizer, type(optimizer)

(SGD (
 Parameter Group 0
     dampening: 0
     lr: 0.01
     momentum: 0.8
     nesterov: True
     weight_decay: 0.0
 ),
 torch.optim.sgd.SGD)
```
可以看到，由于`timm`不包含`SGD`的实现，上述代码使用`torch.optim`的实现来创建了优化器。
再试着创建一个在`timm`中实现的优化器：
```python
optimizer = timm.optim.create_optimizer_v2(model, 
                                           opt='lamb',
                                           lr=0.01,
                                           weight_decay=0.01)
optimizer, type(optimizer)

(Lamb (
 Parameter Group 0
     always_adapt: False
     betas: (0.9, 0.999)
     bias_correction: True
     eps: 1e-06
     grad_averaging: True
     lr: 0.01
     max_grad_norm: 1.0
     trust_clip: False
     weight_decay: 0.0
 
 Parameter Group 1
     always_adapt: False
     betas: (0.9, 0.999)
     bias_correction: True
     eps: 1e-06
     grad_averaging: True
     lr: 0.01
     max_grad_norm: 1.0
     trust_clip: False
     weight_decay: 0.01
 ),
 timm.optim.lamb.Lamb)
```
当然，如果不愿意使用`create_optimizer_v2`，所有这些优化器都可以用常规的方式创建。
```python
optimizer = timm.optim.RMSpropTF(model.parameters(), lr=0.01)
```
## 应用案例
大部分的优化器用法如下：
```python
# replace 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# with
optimizer = timm.optim.AdamP(model.parameters(), lr=0.01)

for epoch in num_epochs:
    for batch in training_dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
截至当前，唯一的例外是二阶`Adahessian`优化器，它在执行反向传播步骤时需要一个小的调整；类似的调整可能需要用于未来可能添加的其他二阶优化器。即：
```python
optimizer = timm.optim.Adahessian(model.parameters(), lr=0.01)

is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order # True

for epoch in num_epochs:
    for batch in training_dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward(create_graph=second_order)
        optimizer.step()
        optimizer.zero_grad()
```
## Lookahead
`timm`也使我们能够将`lookahead`算法应用于优化器；参考资料比如[这个视频](https://www.youtube.com/watch?v=TxGxiDK0Ccc)。`Lookahead`可以提高学习的稳定性并降低其内部优化器的方差，其计算和内存成本可以忽略不计。
可以通过在优化器名称前加上`lookahead_`来将`Lookahead`应用到优化器中：
```python
optimizer = timm.optim.create_optimizer_v2(model.parameters(), opt='lookahead_adam', lr=0.01)
```
或由`timm`的`Lookahead`类中的优化器实例进行包装：
```python
timm.optim.Lookahead(optimizer, alpha=0.5, k=6)
```
当使用`Lookahead`时，需要更新训练脚本，加入以下一行，以更新慢的权重：
```python
optimizer.sync_lookahead()
```
一个例子如下：
```python
optimizer = timm.optim.AdamP(model.parameters(), lr=0.01)
optimizer = timm.optim.Lookahead(optimizer)

for epoch in num_epochs:
    for batch in training_dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    optimizer.sync_lookahead()
```

# 调度器
`timm`包含以下调度器
- `StepLRScheduler`：学习率每`n`步衰减；类似于`torch.optim.lr_scheduler.StepLR`
- `MultiStepLRScheduler`：一个支持多个目标里程碑的步进调度器，在这些里程碑上降低学习率；类似于`torch.optim.lr_scheduler.MultiStepLR`
- `PlateauLRScheduler`：在每次指定的指标出现高原期时，以指定的系数降低学习率；类似于` `torch.optim.lr_scheduler.ReduceLROnPlateau`
- `CosineLRScheduler`：具有重启功能的余弦衰减调度器；类似于`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
- `TanhLRScheduler`：带重启的双曲正切衰变调度器
- `PolyLRScheduler`：多项式衰变调度器。

虽然许多在`timm`中实现的调度器在`PyTorch`中也有对应的调度器，但`timm`版本通常有不同的默认超参数，并提供额外的选项和灵活性；所有`timm`调度器都有预热`epochs`，以及在调度中添加随机噪声的选项。此外，`CosineLRScheduler`和`PolyLRScheduler`支持被称为`k-decay`的衰减选项。

## 应用案例
在研究这些调度器提供的一些选项之前，首先探讨一下如何在自定义训练脚本中使用`timm`的调度器。
与`PyTorch`中包含的调度器不同，在每个`epoch`中更新两次`timm`调度器是最佳实践。
- `.step_update`方法应该在每次优化器更新后被调用，并给出下一次更新的索引；这就是`PyTorch`调度器调用`.step`的地方
- `.step`方法应该在每个`epoch`结束时被调用，并标明下一个`epoch`的索引。

通过明确提供更新次数和`epoch`索引，这使得`timm`调度器能够消除在`PyTorch`调度器中观察到的混乱的 `last_epoch`和`-1`行为。一个例子如下：
```python
training_epochs = 300
cooldown_epochs = 10
num_epochs = training_epochs + cooldown_epochs

optimizer = timm.optim.AdamP(my_model.parameters(), lr=0.01)
scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=training_epochs)

for epoch in range(num_epochs):

    num_steps_per_epoch = len(train_dataloader)
    num_updates = epoch * num_steps_per_epoch

    for batch in training_dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step_update(num_updates=num_updates)

        optimizer.zero_grad()

    scheduler.step(epoch + 1)
```

## 调节学习率调度器
为了展示`timm`提供的一些选项，探索一些可用的超参数，以及修改这些参数对学习率调度的影响。
在这里，将专注于`CosineLRScheduler`，因为这是`timm`训练脚本中默认使用的调度器。然而，如上所述，添加预热和噪声等功能存在于上述所有的调度器中。
```python
scheduler = timm.scheduler.CosineLRScheduler(optimizer,
                                            t_initial=num_epoch_repeat*num_steps_per_epoch,
                                            lr_min=1e-6,
                                            cycle_limit=num_epoch_repeat+1,
                                            t_in_epochs=False)
```

# 指数滑动平均模型
在训练一个模型时，通过对整个训练过程中观察到的参数进行移动平均来设置模型的权重值，而不是使用最后一次增量更新后得到的参数，这样做是有益的。在实践中，这通常是通过维护`EMA`模型来实现的，`EMA`模型是我们正在训练的模型的一个副本。然而，我们不是在每个更新步骤后更新这个模型的所有参数，而是使用现有参数值和更新值的线性组合来设置这些参数。
为了理解为什么这可能是有益的，让我们考虑这样的情况：我们的模型，在训练的早期阶段，在一批数据上表现得特别差。这可能会导致对参数进行大量更新，过度补偿所获得的高损失，这对接下来的批次是不利的。通过只纳入最新参数的一小部分，大的更新将被 "平滑"，对模型的权重产生较小的整体影响。
有时，这些平均的参数在评估过程中有时会产生明显更好的结果，这种技术已经被用于流行模型的一些训练方案中，如训练`MNASNet`、`MobileNet-V3`和`EfficientNet`。使用`timm`中实现的`ModelEmaV2`模块，可以复制这种行为，并将同样的做法应用于自己的训练脚本。
（具体技术细节不再详述）