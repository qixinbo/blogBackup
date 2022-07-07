---
title: 用Docker打包FastAPI程序
tags: [FastAPI, Docker]
categories: coding 
date: 2022-7-5
---

# 简介
这一篇看看如何将使用`FastAPI`编写的程序打包成`docker`镜像。
参考教程见[这里](https://fastapi.tiangolo.com/deployment/docker/)。
# 安装依赖包
一般是使用`requirements.txt`来管理所依赖的包的名字及版本。
该文件的内容形如：
```python
fastapi>=0.68.0,<0.69.0
pydantic>=1.8.0,<2.0.0
uvicorn>=0.15.0,<0.16.0
```
可以在程序编写过程中手动指定以上内容。
也可以在代码完成后，使用如下命令自动生成：
```python
pip freeze > requirements.txt
```
有了上述`requirements.txt`文件后，则可以进行安装：
```python
pip install -r requirements.txt
```

# 创建FastAPI项目
（1）创建`app`文件夹，然后进入该目录；
（2）创建一个空的文件`__init__.py`；
（3）创建`main.py`，内容为：
```python
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```

# 创建Dockerfile文件
在当前目录下，创建`Dockerfile`文件，内容为：
```python
# 指定基础镜像
FROM python:3.9

# 指定镜像内的工作目录
WORKDIR /code

# 从本地源路径中复制requirements文件到目标路径中，注意这里仅复制了这个文件，是为了使得接着构建的镜像能够利用缓存；因为依赖包通常不频繁发生改动，所以先把它构建了。
COPY ./requirements.txt /code/requirements.txt

# 执行命令，这里是安装依赖包。
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 再复制其余的源文件，这些源文件因为是业务逻辑，所以变动的可能性很大，放在最上面构建，而不是放在最开头构建，能够有效地避免这一部分带来的变化，从而利用缓存，节省构建时间。
COPY ./app /code/app

# 容器启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

关于`Dockerfile`编写的详细教程，可以参考该[教程](https://yeasy.gitbook.io/docker_practice/image/build)。

# 构建Docker镜像
当前目录的文件结构如下：
```sh
.
├── app
│   ├── __init__.py
│   └── main.py
├── Dockerfile
└── requirements.txt
```
在该目录下执行：
```sh
docker build -t myimage .
```

# 启动Docker容器
```python
docker run -d --name mycontainer -p 80:80 myimage
```

# 测试访问
访问` http://127.0.0.1/items/5?q=somequery`（或者使用具体的IP地址），则能正确返回结果。
交互式文档`http://127.0.0.1/docs`和`http://127.0.0.1/redoc`也能正常访问。

# 分享镜像
搭建好镜像后，可以通过以下几种方式分享：
（1）上传到docker hub中央库或私有中央库；
（2）传输最原始的文件结构，由对方自己`build`，这种是最轻便的；
（3）通过`docker save`成`tar`包，可以适用于对方无法联网的情形。