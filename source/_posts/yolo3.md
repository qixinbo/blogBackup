---
title: YOLO系列算法原理及极简代码解析
tags: [YOLO]
categories: machine learning 
date: 2021-9-25
---

# 介绍
物体检测的两个步骤可以概括为：
步骤一：检测目标位置（生成矩形框）
步骤二：对目标物体进行分类
物体检测主流的算法框架大致分为one-stage与two-stage。two-stage算法代表有R-CNN系列，one-stage算法代表有Yolo系列。two-stage算法将步骤一与步骤二分开执行，输入图像先经过候选框生成网络（例如faster rcnn中的RPN网络），再经过分类网络；one-stage算法将步骤一与步骤二同时执行，输入图像只经过一个网络，生成的结果中同时包含位置与类别信息。two-stage与one-stage相比，精度高，但是计算量更大，所以运算较慢。


# YOLO
## YOLOv1
[【论文解读】Yolo三部曲解读——Yolov1](https://zhuanlan.zhihu.com/p/70387154)
YOLOv1的网络架构如下图：
![yolov1](https://user-images.githubusercontent.com/6218739/133874079-e9c89d7b-f8f3-4078-8d9a-30e9f0451df3.png)
直接上结构图，输入图像大小为448乘448，经过若干个卷积层与池化层，变为7乘7乘1024张量（图一中倒数第三个立方体），最后经过两层全连接层，输出张量维度为7乘7乘30，这就是Yolo v1的整个神经网络结构，和一般的卷积物体分类网络没有太多区别，最大的不同就是：分类网络最后的全连接层，一般连接于一个一维向量，向量的不同位代表不同类别，而这里的输出向量是一个三维的张量（7乘7乘30）。上图中Yolo的backbone网络结构，受启发于GoogLeNet，也是v2、v3中Darknet的先锋。本质上来说没有什么特别，没有使用BN层，用了一层Dropout。除了最后一层的输出使用了线性激活函数，其他层全部使用Leaky Relu激活函数。网络结构没有特别的东西，不再赘述。

输出张量维度的意义：
（1）7乘7的含义
7乘7是指图片被分成了7乘7个格子，如下图：
![grid-yolov1](https://user-images.githubusercontent.com/6218739/133877539-5d2fe916-a712-4676-9a5e-06a3657a27cf.png)
在Yolo中，如果一个物体的中心点，落在了某个格子中，那么这个格子将负责预测这个物体。而那些没有物体中心点落进来的格子，则不负责预测任何物体。这个设定就好比该网络在一开始，就将整个图片上的预测任务进行了分工，一共设定7乘7个按照方阵列队的检测人员，每个人员负责检测一个物体，大家的分工界线，就是看被检测物体的中心点落在谁的格子里。当然，是7乘7还是9乘9，是上图中的参数S，可以自己修改，精度和性能会随之有些变化。
（2）30的含义
刚才设定了49个检测人员，那么每个人员负责检测的内容，就是这里的30（注意，30是张量最后一维的长度）。在Yolo v1论文中，30是由$(4+1) \times 2 +20$得到的。其中$4+1$是矩形框的中心点坐标(x,y)、长宽(w,h)以及是否属于被检测物体的置信度c；2是一个格子共回归两个矩形框，每个矩形框分别产生5个预测值（每个格子预测矩形框个数，是可调超参数；论文中选择了2个框，当然也可以只预测1个框，具体预测几个矩形框，无非是在计算量和精度之间取一个权衡。如果只预测一个矩形框，计算量会小很多，但是如果训练数据都是小物体，那么网络学习到的框，也会普遍比较小，测试时如果物体较大，那么预测效果就会不理想；如果每个格子多预测几个矩形框，如上文中讲到的，每个矩形框的学习目标会有所分工，有些学习小物体特征，有些学习大物体特征等；在Yolov2、v3中，这个数目都有一定的调整。）；20代表预测20个类别。这里有几点需要注意：1. 每个方格（grid） 产生2个预测框，2也是参数，可以调，但是一旦设定为2以后，那么每个方格只产生两个矩形框，最后选定置信度更大的矩形框作为输出，也就是最终每个方格只输出一个预测矩形框。2. 每个方格只能预测一个物体。虽然可以通过调整参数，产生不同的矩形框，但这只能提高矩形框的精度。所以当有很多个物体的中心点落在了同一个格子里，该格子只能预测一个物体。也就是格子数为7乘7时，该网络最多预测49个物体。
如上述原文中提及，在强行施加了格点限制以后，每个格点只能输出一个预测结果，所以该算法最大的不足，就是对一些邻近小物体的识别效果不是太好，例如成群结队的小鸟。

损失函数：
![loss-yolov1](https://user-images.githubusercontent.com/6218739/133877787-e073a763-b371-4908-8456-60f6329974b7.png)
论文中Loss函数，密密麻麻的公式初看可能比较难懂。其实论文中给出了比较详细的解释。所有的损失都是使用平方和误差公式。
（1）预测框的中心点(x,y)。造成的损失是上图中的第一行。其中$\mathbb{I}\_{ij}^{obj}$为控制函数，在标签中包含物体的那些格点处，该值为 1 ；若格点不含有物体，该值为 0。也就是只对那些有真实物体所属的格点进行损失计算，若该格点不包含物体，那么预测数值不对损失函数造成影响。（x,y）数值与标签用简单的平方和误差。
（2）预测框的宽高。造成的损失是上图的第二行。$\mathbb{I}\_{ij}^{obj}$的含义一样，也是使得只有真实物体所属的格点才会造成损失。这里对在损失函数中的处理分别取了根号，原因在于，如果不取根号，损失函数往往更倾向于调整尺寸比较大的预测框。例如，20个像素点的偏差，对于800乘600的预测框几乎没有影响，此时的IOU数值还是很大，但是对于30乘40的预测框影响就很大。取根号是为了尽可能的消除大尺寸框与小尺寸框之间的差异。
（3）第三行与第四行，都是预测框的置信度C。当该格点不含有物体时，该置信度的标签为0；若含有物体时，该置信度的标签为预测框与真实物体框的IOU数值（IOU计算公式为：两个框交集的面积除以并集的面积）。
（4）第五行为物体类别概率P，对应的类别位置，该标签数值为1，其余位置为0，与分类网络相同。
此时再来看$\lambda\_{coord}$与$\lambda\_{noobj}$，Yolo面临的物体检测问题，是一个典型的类别数目不均衡的问题。其中49个格点，含有物体的格点往往只有3、4个，其余全是不含有物体的格点。此时如果不采取点措施，那么物体检测的mAP不会太高，因为模型更倾向于不含有物体的格点。$\lambda\_{coord}$与$\lambda\_{noobj}$的作用，就是让含有物体的格点，在损失函数中的权重更大，让模型更加“重视”含有物体的格点所造成的损失。在论文中， 取值分别为5与0.5。

一些技巧：
（1）回归offset代替直接回归坐标
不直接回归中心点坐标数值，而是回归相对于格点左上角坐标的位移值。例如，第一个格点中物体坐标为$(2.3, 3.6)$，另一个格点中的物体坐标为$(5.4, 6.3)$，这四个数值让神经网络暴力回归，有一定难度。所以这里的offset是指，既然格点已知，那么物体中心点的坐标一定在格点正方形里，相对于格点左上角的位移值一定在区间$[0, 1)$中。让神经网络去预测$(0.3, 0.6)$与$(0.4, 0.3)$会更加容易，在使用时，加上格点左上角坐标$(2, 3)$、$(5, 6)$即可。

（2）同一格点的不同预测框有不同作用
前文中提到，每个格点预测两个或多个矩形框。此时假设每个格点预测两个矩形框。那么在训练时，见到一个真实物体，我们是希望两个框都去逼近这个物体的真实矩形框，还是只用一个去逼近？或许通常来想，让两个人一起去做同一件事，比一个人做一件事成功率要高，所以可能会让两个框都去逼近这个真实物体。但是作者没有这样做，在损失函数计算中，只对和真实物体最接近的框计算损失，其余框不进行修正。这样操作之后作者发现，一个格点的两个框在尺寸、长宽比、或者某些类别上逐渐有所分工，总体的召回率有所提升
（3）使用非极大抑制生成预测框
通常来说，在预测的时候，格点与格点并不会冲突，但是在预测一些大物体或者邻近物体时，会有多个格点预测了同一个物体。此时采用非极大抑制技巧，过滤掉一些重叠的矩形框。不过此时mAP提升并没有像在RCNN或DPM中那样显著提升。
（4）推理时将类别预测最大值乘以预测框最大值作为输出置信度
在推理时，使用物体的类别预测最大值p乘以预测框的最大值c，作为输出预测物体的置信度。这样也可以过滤掉一些大部分重叠的矩形框。输出检测物体的置信度，同时考虑了矩形框与类别，满足阈值的输出更加可信。

## YOLOv2
[【论文解读】Yolo三部曲解读——Yolov2](https://zhuanlan.zhihu.com/p/74540100)
Yolov2论文标题就是更好，更快，更强。Yolov1发表之后，计算机视觉领域出现了很多trick，例如批归一化、多尺度训练，v2也尝试借鉴了R-CNN体系中的anchor box，所有的改进提升，下面逐一介绍。
1. Batch Normalization（批归一化）
检测系列的网络结构中，BN逐渐变成了标配。在Yolo的每个卷积层中加入BN之后，mAP提升了2%，并且去除了Dropout。
2. High Resolution Classifier（分类网络高分辨率预训练）
在Yolov1中，网络的backbone部分会在ImageNet数据集上进行预训练，训练时网络输入图像的分辨率为224乘224。在v2中，将分类网络在输入图片分辨率为448乘448的ImageNet数据集上训练10个epoch，再使用检测数据集（例如coco）进行微调。高分辨率预训练使mAP提高了大约4%。
3. Convolutional With Anchor Boxes（Anchor Box替换全连接层）
第一篇解读v1时提到，每个格点预测两个矩形框，在计算loss时，只让与ground truth最接近的框产生loss数值，而另一个框不做修正。这样规定之后，作者发现两个框在物体的大小、长宽比、类别上逐渐有了分工。在v2中，神经网络不对预测矩形框的宽高的绝对值进行预测，而是预测与Anchor框的偏差（offset），每个格点指定n个Anchor框。在训练时，最接近ground truth的框产生loss，其余框不产生loss。在引入Anchor Box操作后，mAP由69.5下降至69.2，原因在于，每个格点预测的物体变多之后，召回率大幅上升，准确率有所下降，总体mAP略有下降。
v2中移除了v1最后的两层全连接层，全连接层计算量大，耗时久。文中没有详细描述全连接层的替换方案，这里笔者猜测是利用1乘1的卷积层代替（欢迎指正），具体的网络结构原文中没有提及，官方代码也被yolo v3替代了。v2主要是各种trick引入后的效果验证，建议不必纠结于v2的网络结构。
4. Dimension Clusters（Anchor Box的宽高由聚类产生）
这里算是作者的一个创新点。Faster R-CNN中的九个Anchor Box的宽高是事先设定好的比例大小，一共设定三个面积大小的矩形框，每个矩形框有三个宽高比：1:1，2:1，1:2，总共九个框。而在v2中，Anchor Box的宽高不经过人为获得，而是将训练数据集中的矩形框全部拿出来，用kmeans聚类得到先验框的宽和高。例如使用5个Anchor Box, 那么kmeans聚类的类别中心个数设置为5。
加入了聚类操作之后，引入Anchor Box之后，mAP上升。
需要强调的是，聚类必须要定义聚类点（矩形框）之间的距离函数，文中使用（1-IOU）数值作为两个矩形框的的距离函数，这里的运用也是非常的巧妙。
5. Direct location prediction（绝对位置预测）
Yolo中的位置预测方法很清晰，就是相对于左上角的格点坐标预测偏移量。这里的Direct具体含义，应该是和其他算法框架对比后得到的。比如其他流行的位置预测公式是先预测一个系数，系数又需要与先验框的宽高相乘才能得到相较于参考点的位置偏移，而在yolov2中，系数通过一个激活函数直接产生偏移位置数值，与矩形框的宽高独立开，变得更加直接。
6. Fine-Grained Features（细粒度特征）
在26乘26的特征图，经过卷积层等，变为13乘13的特征图后，作者认为损失了很多细粒度的特征，导致小尺寸物体的识别效果不佳，所以在此加入了passthrough层。passthrough层就是将26乘26乘1的特征图，变成13乘13乘4的特征图，在这一次操作中不损失细粒度特征。
7. Multi-Scale Training（多尺寸训练）
很关键的一点是，Yolo v2中只有卷积层与池化层，所以对于网络的输入大小，并没有限制，整个网络的降采样倍数为32，只要输入的特征图尺寸为32的倍数即可，如果网络中有全连接层，就不是这样了。所以Yolo v2可以使用不同尺寸的输入图片训练。
作者使用的训练方法是，在每10个batch之后，就将图片resize成{320, 352, ..., 608}中的一种。不同的输入，最后产生的格点数不同，比如输入图片是320乘320，那么输出格点是10乘10，如果每个格点的先验框个数设置为5，那么总共输出500个预测结果；如果输入图片大小是608乘608，输出格点就是19乘19，共1805个预测结果。
在引入了多尺寸训练方法后，迫使卷积核学习不同比例大小尺寸的特征。当输入设置为544乘544甚至更大，Yolo v2的mAP已经超过了其他的物体检测算法。

## YOLOv3
[【论文解读】Yolo三部曲解读——Yolov3](https://zhuanlan.zhihu.com/p/76802514)
![yolov3](https://user-images.githubusercontent.com/6218739/133880395-2f580763-b4fc-4aea-b36e-2d745834a1da.png)
Yolov3使用Darknet-53作为整个网络的分类骨干部分（见上图虚线部分）。
Darknet-53的架构如下图：
![darknet53](https://user-images.githubusercontent.com/6218739/133880487-984b7259-6b63-42e4-8d6a-7ae420fb4591.png)
backbone部分由Yolov2时期的Darknet-19进化至Darknet-53，加深了网络层数，引入了Resnet中的跨层加和操作。Darknet-53处理速度每秒78张图，比Darknet-19慢不少，但是比同精度的ResNet快很多。Yolov3依然保持了高性能。

网络结构解析：
1. Yolov3中，只有卷积层，通过调节卷积步长控制输出特征图的尺寸。所以对于输入图片尺寸没有特别限制。
2. Yolov3借鉴了金字塔特征图思想，小尺寸特征图用于检测大尺寸物体，而大尺寸特征图检测小尺寸物体。特征图的输出维度为$N \times N \times [3 \times (4+1+80)]$， $N \times N$为输出特征图格点数，一共3个Anchor框，每个框有4维预测框数值和1维预测框置信度，80维物体类别数。
3. Yolov3总共输出3个特征图，第一个特征图下采样32倍，第二个特征图下采样16倍，第三个下采样8倍。输入图像经过Darknet-53（无全连接层），再经过Yoloblock生成的特征图被当作两用，第一用为经过3乘3卷积层、1乘1卷积之后生成特征图一，第二用为经过1乘1卷积层加上采样层，与Darnet-53网络的中间层输出结果进行拼接，产生特征图二。同样的循环之后产生特征图三。
4. concat操作与加和操作的区别：加和操作来源于ResNet思想，将输入的特征图，与输出特征图对应维度进行相加，即$y=f(x)+x$；而concat操作源于DenseNet网络的设计思路，将特征图按照通道维度直接进行拼接，例如8乘8乘16的特征图与8乘8乘16的特征图拼接后生成8乘8乘32的特征图。
5. 上采样层(upsample)：作用是将小尺寸特征图通过插值等方法，生成大尺寸图像。例如使用最近邻插值算法，将8乘8的图像变换为16乘16。上采样层不改变特征图的通道数。

Yolo的整个网络，吸取了Resnet、Densenet、FPN的精髓，可以说是融合了目标检测当前业界最有效的全部技巧。

YOLOv3与YOLOv2和YOLOv1相比最大的改善就是对boundingbox进行了跨尺度预测(Prediction Across Scales)，提高YOLO模型对不同尺度对象的预测精度。
![yolov3-output](https://user-images.githubusercontent.com/6218739/133882919-4d3ede76-ff3d-4435-a4c0-a577c181bd2d.png)
[YOLO_v3论文解读](https://zhuanlan.zhihu.com/p/75811997)

结合上图看，卷积网络在79层后，经过下方几个黄色的卷积层得到一种尺度的检测结果。相比输入图像，这里用于检测的特征图有32倍的下采样。比如输入是416乘416的话，这里的特征图就是13乘13了。由于下采样倍数高，这里特征图的感受野比较大，因此适合检测图像中尺寸比较大的对象。
为了实现细粒度的检测，第79层的特征图又开始作上采样（从79层往右开始上采样卷积），然后与第61层特征图拼接（Concatenation），这样得到第91层较细粒度的特征图，同样经过几个卷积层后得到相对输入图像16倍下采样的特征图。它具有中等尺度的感受野，适合检测中等尺度的对象。
最后，第91层特征图再次上采样，并与第36层特征图拼接（Concatenation），最后得到相对输入图像8倍下采样的特征图。它的感受野最小，适合检测小尺寸的对象。
随着输出的特征图的数量和尺度的变化，先验框的尺寸也需要相应的调整。YOLO2已经开始采用K-means聚类得到先验框的尺寸，YOLO3延续了这种方法，为每种下采样尺度设定3种先验框，总共聚类出9种尺寸的先验框。在COCO数据集这9个先验框是：
$$
(10 \times 13)，(16 \times 30)，(33 \times 23)，(30 \times 61)，(62 \times 45)，(59 \times 119)，(116 \times 90)，(156 \times 198)，(373 \times 326)
$$
分配上，在最小的13乘13特征图上（有最大的感受野）应用较大的先验框$(116 \times 90)，(156 \times 198)，(373 \times 326)$，适合检测较大的对象。中等的26乘26特征图上（中等感受野）应用中等的先验框$(30 \times 61)，(62 \times 45)，(59 \times 119)$，适合检测中等大小的对象。较大的52乘52特征图上（较小的感受野）应用较小的先验框$(10 \times 13)，(16 \times 30)，(33 \times 23)$，适合检测较小的对象。

YOLOv3前向解码过程：
根据不同的输入尺寸，会得到不同大小的输出特征图，以图二中输入图片$256 \times 256 \times 3$为例，输出的特征图为$8 \times 8 \times 255$、$16 \times 16 \times 255$、$32 \times 32 \times 255$。在Yolov3的设计中，每个特征图的每个格子中，都配置3个不同的先验框（就是下面的锚框），所以最后三个特征图，这里暂且reshape为$8 \times 8 \times 3 \times 85$、$16 \times 16 \times 3 \times 85$、$32 \times 32 \times 3 \times 85$，这样更容易理解，在代码中也是reshape成这样之后更容易操作。
三张特征图就是整个Yolo输出的检测结果，检测框位置（4维）、检测置信度（1维）、类别（80维）都在其中，加起来正好是85维。特征图最后的维度85，代表的就是这些信息，而特征图其他维度$N \times N \times 3$，$N \times N$代表了检测框的参考位置信息，3是3个不同尺度的先验框。

三个特征图一共可以解码出 $8 × 8 × 3 + 16 × 16 × 3 + 32 × 32 × 3 = 4032$ 个box以及相应的类别、置信度。这4032个box，在训练和推理时，使用方法不一样：
1. 训练时4032个box全部送入打标签函数，进行后一步的标签以及损失函数的计算。
2. 推理时，选取一个置信度阈值，过滤掉低阈值box，再经过nms（非极大值抑制），就可以输出整个网络的预测结果了。

YOLOv3训练策略（反向过程）：
1. 预测框一共分为三种情况：正例（positive）、负例（negative）、忽略样例（ignore）。
2. 正例：任取一个ground truth，与4032个框全部计算IOU，IOU最大的预测框，即为正例。并且一个预测框，只能分配给一个ground truth。例如第一个ground truth已经匹配了一个正例检测框，那么下一个ground truth，就在余下的4031个检测框中，寻找IOU最大的检测框作为正例。ground truth的先后顺序可忽略。正例产生置信度loss、检测框loss、类别loss。预测框为对应的ground truth box标签；类别标签对应类别为1，其余为0；置信度标签为1。
3. 忽略样例：正例除外，与任意一个ground truth的IOU大于阈值（论文中使用0.5），则为忽略样例。忽略样例不产生任何loss。
4. 负例：正例除外（与ground truth计算后IOU最大的检测框，但是IOU小于阈值，仍为正例），与全部ground truth的IOU都小于阈值（0.5），则为负例。负例只有置信度产生loss，置信度标签为0。


# YOLOv3源码
从头实现YOLOv3的源码见：
[YOLOv3 in PyTorch](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3)
该源码的视频讲解见：
[YOLOv3 from Scratch](https://www.bilibili.com/video/BV1bo4y1X78v?spm_id_from=333.999.0.0)
## 模型架构
整个YOLOv3的模型架构如下配置：
```python
# 对于里面的元素
# 如果是元组，代表：(输出通道, 卷积核尺寸, 步长)
# YOLOv3中所有的卷积块（注意是卷积块，它由卷积层+批标准化层+LeakyReLU层构成）都是相同的，在下面的代码中用CNNBlock类实现
# 如果是列表，"B"代表残差块Residual Block，后面的次数代表重复次数，在下面用ResidualBlock类实现
# 如果是字符，那么"S"代表Scale不同尺度预测块，在此处计算损失，在下面用ScalePrediction类实现
# "U"代表Upsampling上采样，且与上一层进行连接，生成新的尺度预测
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # 到这里就是Darknet-53 backbone，53是全部卷积层的个数，它会在imagenet上进行预训练
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
```
首先看一下CNN卷积块的实现：
```python
class CNNBlock(nn.Module):
    # 此处会加上一个BN层的开关，如果关了BN层，就相当于是只有卷积层，而不是卷积块
   # 不加BN和ReLU层的纯卷积层是会在尺度预测的地方用到，即网络末端的卷积是纯卷积层
   # 同时使用kwargs参数接收其他参数，比如kerneal size，stride，padding等参数
   # 在整个网络中图像的宽高变化即维度压缩，是通过卷积块的stride参数来实现的
   # 由下面的分析可知，在残差块中图像宽高不变，但两个残差块中间的卷积块的stride为2，此时会对图像的宽高进行压缩减半
   # 整个网络中压缩最厉害的分支是一共压缩了5次，即压缩了32倍，另外两支分别压缩了16倍和8倍
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
       # 如果使用了BN，那么偏置这个参数就没必要了，所以此处会根据BN层的有无进行偏置bias参数的开关
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)
```
再看一下残差块的实现：
```python
class ResidualBlock(nn.Module):
    # 这里给出了是否使用残差连接的开关，在darknet-53部分都是打开残差连接，但到了尺度预测部分，该残差块是关闭了残差连接
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        # 整个残差块是一个ModuleList
        self.layers = nn.ModuleList()
       # 根据重复次数进行循环
        for repeat in range(num_repeats):
            self.layers += [
                # 每个残差块中的第一个卷积块都是通道数减半，卷积核尺寸为1，步长是默认的1，填充是默认的0，因此图像的输入和输出宽高不变
            # 第二个卷积块通道数变为两倍，卷积核尺寸为3，填充为1，步长仍是默认的1，因此图像的输入和输出宽高不变
            # 所以，总的来说，经过一个残差块后，图像的通道数、宽和高都不会变
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # 如果启用残差连接，那么就直接将x和经过处理后的x相加
            if self.use_residual:
                x = x + layer(x)
            # 如果没有启用残差连接，那么就直接处理x，不管作为输入的x
            else:
                x = layer(x)

        return x
```

再来看不同尺度预测的类实现：
```python
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            # 在每个尺度预测块中，先用一个卷积块将通道数加倍，同时通过设置卷积核尺寸为3，填充为1，步长是默认的1，来保持宽高不变
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            # 然后将得到的特征图通过一个卷积块转化为最终想要的向量的模样
         # 3指的是对于对于每一个grid cell，都有3个anchor boxes
         # 对于每一个anchor box，都需要有num_classes+5个元素，前面是类别数目，比如20，5是包含了x, y, w, h和置信度
         # 注意这里不使用BN层，同时卷积核为1，步长是默认的1，填充是默认的0，因此宽高不变
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            # 对x进行预测后，需要对结果进行reshape，形状依次为batch size、3、类别数+5、特征图宽度、特征图高度
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            # 再交换一下维度，把宽、高提到(类别数+5)的前面
         # 比如某一个尺度预测后，得到的向量形状为N x 3 x 13 x 13 x (5+num_classes)，grid cell就是13x13大小
            .permute(0, 1, 3, 4, 2)
        )
```

最后看整个模型的架构，即将上面的组件组合起来：
```python
class YOLOv3(nn.Module):
    # 输入通道默认为3， 类别数默认为80
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        # 将所有模型组件都放在ModuleList中
        layers = nn.ModuleList()
        in_channels = self.in_channels

      # 开始解析上面的config配置
        for module in config:
            # 如果元素是个元组，代表它是个卷积块
            if isinstance(module, tuple):
                # 取出卷积块的相应配置
                out_channels, kernel_size, stride = module
                # 往整个网络里添加卷积块
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        # 如果卷积核为3，则填充为1，否则就填充为0，这样是为了当卷积核为3、步长为2时，填充设为1，此时宽高减半
                  # Pytorch默认卷积层的尺寸计算是向下取整，即(k+2*1-3)/2+1=k/2+floor(-0.5)+1=k/2-1+1=k/2
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                # 更新通道数
                in_channels = out_channels

         # 如果元素是个列表，代表是残差块
            elif isinstance(module, list):
                # 取出残差块的相应配置
                num_repeats = module[1]
                # 往整个网络里添加残差块
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

         # 如果元素是个字符，那么就进入尺度预测模块
            elif isinstance(module, str):
                # 如果是S，代表要进行在某一尺度上的预测了
                if module == "S":
                    layers += [
                        # 下面这三块的网络架构参考下面那张YOLOv3的架构图
                  # 原码中残差块只重复了1次，为了与下面架构图中的YoloBlock相对应，这里改为重复2次，影响不大，因为在残差块中不改变图像大小
                  # 同时注意此时残差块关闭了残差连接
                        ResidualBlock(in_channels, use_residual=False, num_repeats=2),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    # 更新一下通道数
                    in_channels = in_channels // 2

             # 如果是U，则进入上采样
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    # 通道数变为3倍，原因是这个地方进行了通道连接concatenation操作
               # 特别注意的是，不要在这个地方推导图像在整个模型中的处理过程，因为此时会发现前后通道数是不符的，因为通道一下从256跳到了768
               # 这个地方不是forward函数，并不是真正的数据处理过程，可以理解成这个地方仅是模型架构定义
                    in_channels = in_channels * 3

        return layers

    def forward(self, x):
        # 每一个尺度下都有一个output，这里用一个列表来承载三个output
        outputs = []
        # 存放进入不同预测分支的中间计算结果
        route_connections = []
        # 对网络中的每一层进行遍历
        for layer in self.layers:
            # 如果是尺度预测层，表示进入某一尺度的预测阶段，即进入某一个预测分支
            if isinstance(layer, ScalePrediction):
                # 将预测结果添加进outputs中，注意这个地方是对x的一个分叉计算
            # 即x在这里走了两条路，一条路是进入尺度预测模块进行计算，另一条路是继续呆在主分支中，用于后续计算
                outputs.append(layer(x))
                # 返回到主分支中
                continue

         # 对常规的网络层进行计算，包含卷积块和重复次数不为8的残差块
            x = layer(x)

         # 对于重复次数为8的残差块，由架构图可知，都是在这里进入不同的尺度预测分支
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                # 将需要进入某分支的结果存放起来
                route_connections.append(x)

         # 如果是遇到上采样模块
            elif isinstance(layer, nn.Upsample):
                # 就会将当前x与存放中间结果的route中的最后一个中间结果进行连接concatenation
            # 这个地方会将通道数变为3倍，因为上采样后的图像为n_channel，原主分支中的图像为2*n_channel，连接后就变为3*n_channel
                x = torch.cat([x, route_connections[-1]], dim=1)
                # 用完最后一个元素就把它丢了，这样就能在下一次取到上一个存储的中间结果
                route_connections.pop()

      # 最终outputs里是存放了三个尺度的预测模型
        return outputs
```
YOLOv3架构图重新贴一下：
![yolov3](https://user-images.githubusercontent.com/6218739/133880395-2f580763-b4fc-4aea-b36e-2d745834a1da.png)

## 数据集
作者提供了YOLO格式的PASCAL VOC和MS COCO数据集的下载，分别在下面链接：
[Pascal voc dataset used in YOLOv3 video](https://www.kaggle.com/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video)
[MS-COCO-YOLOv3](https://www.kaggle.com/dataset/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e)
关于数据集的格式可以参见下面的介绍：
[Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

```python
class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file, # csv文件路径
        img_dir, # 图像文件路径
        label_dir, # 标签文件路径
        anchors, # 九个锚框
        image_size=416, # 图像尺寸
        S=[13, 26, 52], # 三个特征图大小
        C=20, # 类别数
        transform=None, # 图像变换
    ):
        self.annotations = pd.read_csv(csv_file) # 图像和标签成对出现
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # 将三个尺度下的三个锚框连起来，注意两个list相加就是join的效果
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5 # 这个阈值用来区分忽略样例和负例，详见上面的解析

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # 1就是代表第二列，即标签列
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # 得到的边界框格式为(x, y, w, h, 类别)

        # 读入图像文件
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # 图像增强变换
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # 最终目标是在三个特征图尺度上，每个尺度的每个格点上都有(self.num_anchors // 3)个预测框，每个预测框上都有6个分量，即[置信度标签, x, y, w, h, 类别]
        # 置信度标签为1，代表正例；标签为-1，代表忽略样例；标签为0，代表负例。
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        # 对该图像中的所有的边界框进行循环，目的是为了确定哪个锚框、哪个格点与其对应
        for box in bboxes: 
            # 确定与边界框对应的锚框是通过计算两者之间的IoU
            # box的第2、3元素就是宽度和高度
            # 这里的IoU计算相当于将锚框和边界框的中心放在一块，然后根据它们的宽高来计算
            # 即为了确定哪一种形状的锚框与该边界框最相近
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            # 找出重合度最大的即最好的锚框
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            # 取出该边界框的x, y, w, h
            x, y, width, height, class_label = box
            # 下面这个列表初始化为False，但最终目的是变为True，即保证在每个尺度下该边界框都有对应的锚框
            has_anchor = [False] * 3  # each scale should have one anchor

            # 因为一共有9个锚框，但分布在3个尺度下，下面就是将具体的锚框与它所属的尺度对应起来，即找到在每个尺度下的最好的锚框是哪个，将其判断为正例，其他不好的锚框进一步判断为负例还是忽略样例
            # 即对所有的锚框都会做判断，正例的锚框就会计算置信度loss、检测框loss、类别loss，负例只会计算置信度loss，忽略样例则什么loss都不计算
            # 先从9个锚框中重合度最大的锚框开始进行循环
            for anchor_idx in anchor_indices:
                # 根据锚框的索引和每个尺度下拥有的锚框数量，就可以确定锚框所在的尺度
                # 比如如果锚框索引为8，且每个尺度下有3个锚框，那么scale_idx就是2，即第3个尺度，因此scale_idx就是该锚框所属的尺度的索引
                scale_idx = anchor_idx // self.num_anchors_per_scale
                # 上述锚框索引为8，指的是该锚框在所有锚框中的索引，下面就是计算该锚框在该尺度下的索引，即anchor_on_scale就是2，也就是该尺度下该锚框是第3个
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # 获取该尺度下的grid cell的个数，即格点个数
                S = self.S[scale_idx]
                # 提醒一下：边界框的x和y坐标是其中心相对于整张图像的位置
                # 下面就是计算边界框属于图像中的哪个格点
                # 比如假设整个图像宽为W，那么边界框绝对位置就在W*x，而每个格点的宽度为W/S，那么在哪个格点就是W*x/(W/S)=S*x
                # 这里需要注意的是x是宽，但在矩阵中是列，即j,
                # 而y是高，在矩阵中是行，即i
                i, j = int(S * y), int(S * x)  # which cell
                # 下面这一行就是对于这一边界框，取出某一尺度下的锚框、格点及置信度（0元素就是代表是一个物体的可能性即置信度）
                # 刚开始anchor_taken都是0，表明在该尺度的该锚框没有被取走或说没有被判断，非0的话又有两种，1是代表是正例，-1代表是忽略样例
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                # 如果该尺度下（或称该格点）的该锚框没有被判断，且该尺度上之前没有确定锚框（即还没有出现正例）
                if not anchor_taken and not has_anchor[scale_idx]:
                    # 就把第0个元素，即是一个物体的置信度置为1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # 计算边界框的中心在格点中的相对位置
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    # 计算边界框的宽高相对于格点的大小
                    # 仍然假设整张图像宽为W，边界框的绝对宽度就是W*width，那么它相对于格点的大小就是W*width/(W/S)=width*S
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    # 然后把上面的边界框相对于格点的相对位置和相对大小信息都存储到targets相应元素中，与具体的尺度、锚框和格点进行匹配。
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    # 将物体类别也存储到相应元素
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    # 将该尺度下是否确定了锚框置为True，即该锚框为正例
                    has_anchor[scale_idx] = True

                # 如果已经出现了正例，但该锚框还没有被判断，即anchor_taken=0
                # 此时再判断该锚框与边界框的IoU是否大于阈值，如果大于阈值，且因为其不是正例，那么就将其置信度标签置为-1，即它为忽略样例，不参与损失计算
                # 这种情况出现在在该尺度上（或称在该格点上），有多个锚框都能与边界框吻合较好，但只取最好的那个
                # 但如果IoU小于阈值，那么其置信度标签仍为0，代表负例，会产生置信度loss，但不会产生其他类型的损失
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)

```

## 损失计算
YOLOv3中的损失有三种，一种是xywh带来的误差，即检测框loss；一种是置信度带来的误差，即是否是个物体obj带来的loss，称为置信度loss；一种是类别带来的误差，称为类别loss。
```python
class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # 均方差损失计算
        self.bce = nn.BCEWithLogitsLoss() # 加了Sigmoid的二进制交叉熵损失
        self.entropy = nn.CrossEntropyLoss() # 交叉熵损失
        self.sigmoid = nn.Sigmoid() 

        # 确定损失函数中的各个权重常数，用来控制不同loss之间的比例
        self.lambda_class = 1 # 类别损失权重常数
        self.lambda_noobj = 10 # 负例损失权重常数
        self.lambda_obj = 1 # 正例损失权重常数
        self.lambda_box = 10 # 检测框损失权重常数

    def forward(self, predictions, target, anchors):
        # 判断每个尺度上的格点上是否是物体，即正例还是负例
        # 1为正例，0为负例，-1则为忽略样例
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   负例造成的置信度损失    #
        # ======================= #

        # 使用二进制交叉熵计算损失
        no_object_loss = self.bce(
            # 取出置信度数值，即第0个元素
            # 这里使用0:1的形式，而不是直接使用0来取得元素，是为了保持维度不变
            # [noobj] 是使用了numpy的布尔索引，从而取出那些负例
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   正例造成的置信度损失 #
        # ==================== #

        # 这个地方正例损失按上面的解析应该使用一个简单的bce即可，同时置信度标签在yolov3中是1和0二分类，而这里原作者使用的是IoU来作为置信度标签，即如下形式：
        object_loss = self.bce(
            (predictions[..., 0:1][obj]), (target[..., 0:1][obj]),
        )

        # # 原来的代码中是如下形式，这里先不仔细研究异同了
        # anchors = anchors.reshape(1, 3, 1, 1, 2)
        # box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #  检测框损失               #
        # ======================== #

        # 这个地方涉及检测框的解码部分
        # 注意只取出正例造成的损失
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x, y坐标
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   类别损失   #
        # ================== #

        # 计算交叉熵损失，注意只取出正例造成的损失
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
```

## 超参数配置文件
此处就是将超参数配置都摘出来放在一个统一的配置文件中。
```python
DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20 #类别数
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

# 通过在训练集上kmeans聚类得到的锚框的大小
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
```

## 训练函数
```python
def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # 显示进度条
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        # 三个不同尺度下的目标
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            # 前面的损失函数需要在三个尺度下都要计算一遍
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    # 定义模型
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    # 优化器
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # 损失函数
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # 数据加载器
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    # 可以加载已训练好的模型
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # 开始迭代训练
    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            # 得到预测框和真实的边界框的对比
            # 因为对于一张图像，在三个尺度上会有多个预测框与之吻合挺好，这里使用了NMS非极大值抑制来选择出最好的一个预测框
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            # 计算mAP
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()
```




