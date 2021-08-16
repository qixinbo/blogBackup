---
title: 一键深度学习：将常用深度学习算法集成在ImagePy软件
tags: [Machine Learning]
categories: machine learning
date: 2021-8-16
---

# 起因
日常工作中，调用深度学习算法通常需要在命令行中进行，该过程通常涉及复杂的流程，比如修改配置文件、指定文件路径、打开命令行调用算法运行。此时如果能有一个图形界面软件实现“一键调用”，就会极大地节省工作量，提高工作效率，避免来来回回地反复修改文件、执行命令等。

最近新写了一个库，就是把常用的深度学习算法都集成在了ImagePy中，这样用户和开发者就能直接在ImagePy中愉快地“玩”算法了。


# OneButtonDeepLearning
该仓库在[这里https://github.com/qixinbo/OneButtonDeepLearning](https://github.com/qixinbo/OneButtonDeepLearning)。
宗旨就是：让深度学习算法触手可及、一键调用，不必每次在命令行进行复杂配置。

# 使用方法
只需将要使用的模型文件夹复制到`imagepy/plugins`文件夹下，再次启动ImagePy后即可在菜单栏看到该算法。

## 配置环境
如果运行深度学习算法的环境没有事先搭建好，那么在模型的`menus` 下都有一个配置文件，直接运行:
~~~~
pip install -r requirements.txt
~~~~ 
即可下载相应的依赖包。

# 当前模型

## 光学字符识别OCR
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice.

![ocr-demo](https://raw.githubusercontent.com/qixinbo/OneButtonDeepLearning/main/OCR/menus/OCR/demo.png)

## 目标检测YOLOv5
[YOLOv5](https://github.com/ultralytics/yolov5) is a family of compound-scaled object detection models trained on the COCO dataset.

![yolov5-demo](https://raw.githubusercontent.com/qixinbo/OneButtonDeepLearning/main/YOLOv5/menus/YOLOv5/demo.png)

## 人脸识别
[InsightFace](https://github.com/deepinsight/insightface) is an open source 2D&3D deep face analysis toolbox, and efficiently implements a rich variety of state of the art algorithms of face recognition, face detection and face alignment, which optimized for both training and deployment.

![face-demo](https://raw.githubusercontent.com/qixinbo/OneButtonDeepLearning/main/FaceAnalysis/menus/Face/demo.png)

## 胞状物体分割Cellpose
[Cellpose](https://github.com/MouseLand/cellpose) is a generalist algorithm for cell and nucleus segmentation.

![cellpose-demo](https://raw.githubusercontent.com/qixinbo/OneButtonDeepLearning/main/Cellpose/menus/Cellpose/demo.png)

## 胞状物体分割BulkSeg
[BulkSeg](https://github.com/qixinbo/BulkSeg) which is inspired by Cellpose, is a fast and generalist algorithm for segmenting bulk-like objects.
![bulkseg-demo](https://raw.githubusercontent.com/qixinbo/OneButtonDeepLearning/main/BulkSeg/menus/BulkSeg/demo.png)

## 物体分割DeepLab
[DeepLab](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py) is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.
![deeplab-demo](https://raw.githubusercontent.com/qixinbo/OneButtonDeepLearning/main/DeepLab/menus/DeepLab/demo.png)

# 计划
下一步计划添加的模型有：
- 图像生成
- 风格迁移