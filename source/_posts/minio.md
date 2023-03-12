---
title: MinIO简明教程
tags: [Web Crawler]
categories: data analysis
date: 2023-3-12
---

（以下内容全部来自ChatGPT）
# 介绍
Minio是一个开源的对象存储服务器，它是一个轻量级的替代品，可以用来存储和管理大量的非结构化数据，例如图像、视频、文本等。它支持S3 API，允许您使用S3兼容的客户端工具和库将文件上传到Minio服务器中。
Minio的主要特点包括：
1.  高性能：Minio是一个高性能的对象存储服务器，它可以处理大量的并发请求，并提供快速的读写操作。
2.  分布式：Minio支持分布式部署，可以在多个节点上运行，以提供更高的可用性和可扩展性。
3.  可扩展性：Minio可以轻松地扩展到数百甚至数千台服务器，以满足不断增长的数据存储需求。
4.  安全性：Minio提供了强大的安全功能，包括SSL/TLS加密、访问控制、身份验证等。
5.  开放性：Minio是一个开放源代码项目，它使用Apache许可证2.0发布，因此可以自由使用、修改和分发。

Minio可以在本地或云中运行，可以与Amazon S3、Azure Blob Storage等其他对象存储服务进行交互。它还提供了丰富的API和SDK，可以轻松地集成到各种应用程序和系统中。
总之，Minio是一个快速、可扩展、安全和开放的对象存储服务器，是处理大量非结构化数据的理想选择。

# Minio对象属性
Minio是一个对象存储服务器，不是一个数据库。因此，它不是一个关系型数据库或非关系型数据库，而是一个面向对象的存储系统。在Minio中，每个对象都是一个二进制文件，可以包含任何类型的数据。

虽然Minio不是一个数据库，但是它允许您在对象中存储元数据，这些元数据可以用来描述对象的属性。以下是一些常见的Minio对象属性：
1.  object name（对象名称）：对象的唯一名称。
2.  bucket name（存储桶名称）：对象所在的存储桶名称。
3.  content type（内容类型）：对象的MIME类型，例如'image/jpeg'或'application/pdf'。
4.  content length（内容长度）：对象的大小，以字节为单位。
5.  last modified（最后修改时间）：对象的最后修改时间。
6.  user-defined metadata（用户定义的元数据）：对象的自定义元数据，例如作者、创建日期等。
7.  etag（实体标签）：对象的唯一标识符，通常是一个哈希值。

通过使用这些属性，您可以在Minio中存储和检索任意类型的数据，并对其进行元数据标记，以便更好地管理和组织您的存储数据。

# 基本用法
## 安装Minio
您可以从Minio的官方网站上下载适合您操作系统的Minio二进制文件。安装完成后，您可以在命令行中运行Minio服务器。

## 启动Minio
在命令行中运行以下命令启动Minio服务器：
```
minio server /path/to/data
```

其中，`/path/to/data`是您要存储数据的目录路径。

## 访问Minio控制台
在浏览器中访问`http://localhost:9000`，您将看到Minio的控制台。在控制台中，您可以创建和管理存储桶、上传和下载文件等。

## 使用S3兼容客户端
您可以使用S3兼容的客户端工具和库将文件上传到Minio服务器中。例如，您可以使用AWS CLI命令行工具：
```
aws s3 cp /path/to/local/file s3://mybucket/myfile
```
其中，`mybucket`是您要上传文件的存储桶名称，`myfile`是您要上传文件的文件名。

## 使用Minio SDK
Minio还提供了SDK，您可以使用它来编写自己的应用程序来与Minio服务器交互。例如，您可以使用Minio SDK for Python：
```python
from minio import Minio
from minio.error import ResponseError

# Initialize Minio client
client = Minio('localhost:9000',
               access_key='ACCESS_KEY',
               secret_key='SECRET_KEY',
               secure=False)

# Upload file to bucket
try:
    client.fput_object('mybucket', 'myfile', '/path/to/local/file')
except ResponseError as err:
    print(err)
```
其中，`ACCESS_KEY`和`SECRET_KEY`是您在Minio服务器上创建的访问密钥。
这些是使用Minio的一些基本教程。您可以在Minio的官方文档中找到更多信息和教程。

# python接口
1.  安装Minio SDK for Python：首先，您需要安装Minio SDK for Python。您可以使用pip包管理器安装它：
```
pip install minio
```
2.  初始化Minio客户端：使用Minio SDK for Python，您可以轻松地初始化Minio客户端。例如：
```python
from minio import Minio

# Initialize Minio client
client = Minio('localhost:9000',
               access_key='ACCESS_KEY',
               secret_key='SECRET_KEY',
               secure=False)
```

其中，`localhost:9000`是Minio服务器的地址和端口，`ACCESS_KEY`和`SECRET_KEY`是您在Minio服务器上创建的访问密钥。
3.  创建存储桶：使用Minio客户端，您可以轻松地创建存储桶。例如：
```python
# Create bucket
try:
    client.make_bucket('mybucket')
except BucketAlreadyOwnedByYou as err:
    pass
except BucketAlreadyExists as err:
    pass
except ResponseError as err:
    print(err)
```
其中，`mybucket`是您要创建的存储桶名称。
4.  上传文件：使用Minio客户端，您可以轻松地上传文件到Minio服务器中。例如：
```python
# Upload file to bucket
try:
    client.fput_object('mybucket', 'myfile', '/path/to/local/file')
except ResponseError as err:
    print(err)
```

其中，`mybucket`是您要上传文件的存储桶名称，`myfile`是您要上传文件的文件名，`/path/to/local/file`是本地文件的路径。
5.  下载文件：使用Minio客户端，您可以轻松地从Minio服务器中下载文件。例如：
```python
# Download file from bucket
try:
    client.fget_object('mybucket', 'myfile', '/path/to/local/file')
except ResponseError as err:
    print(err)
```

其中，`mybucket`是您要下载文件的存储桶名称，`myfile`是您要下载的文件名，`/path/to/local/file`是本地文件的路径。

这些是使用Python编写与Minio服务器交互的基本教程。您可以在Minio的官方文档中找到更多信息和教程。

# 加入新硬盘
（未验证，谨慎使用）
要将新硬盘添加到Minio中，您可以按照以下步骤进行操作：
1.  挂载新硬盘：将新硬盘挂载到Minio服务器中。您需要确保新硬盘已经格式化，并且已经挂载到Minio服务器的文件系统中。
2.  创建新存储桶：使用Minio控制台或`mc`命令行工具，创建一个新的存储桶，用于存储新硬盘上的对象数据。
3.  启动Minio服务器：在命令行中启动Minio服务器，并将新存储桶指定为新硬盘的默认存储位置。使用以下命令启动Minio服务器：
```
minio server /path/to/data1 /path/to/data2
```
其中，`/path/to/data1`是现有硬盘的数据目录路径，`/path/to/data2`是新硬盘的数据目录路径。您可以在启动Minio服务器时指定多个数据目录路径，以将对象数据分布在多个硬盘上。
4.  确认扩容成功：您可以使用Minio控制台或`mc`命令行工具来确认扩容是否成功。检查新存储桶是否已经创建，并且新硬盘上的对象数据是否已经复制到新存储桶中。

重复以上步骤，您可以轻松地将多个硬盘添加到Minio中，以扩展存储容量。

其他教程：
https://blog.csdn.net/qq_35036073/article/details/108262407
https://www.cnblogs.com/liugp/p/16560313.html
