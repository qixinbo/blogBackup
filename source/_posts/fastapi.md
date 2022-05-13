---
title: FastAPI用户指南
tags: [FastAPI]
categories: coding 
date: 2022-5-8
---

参考文献：
[FastAPI官方文档](https://fastapi.tiangolo.com/)
[中文翻译](https://fastapi.tiangolo.com/zh/)
（注意，当前2022年5月8日的中文翻译有一些错误）

# 介绍
FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架，使用 Python 3.6+ 并基于标准的 Python 类型提示。
FastAPI 站在以下巨人的肩膀之上：
- [Starlette](https://www.starlette.io/)负责 web 部分。
- [Pydantic](https://pydantic-docs.helpmanual.io/)负责数据部分。


# 安装
```python
pip install fastapi
```
还需要一个 ASGI 服务器，生产环境可以使用[Uvicorn](https://www.uvicorn.org/)或者[Hypercorn](https://gitlab.com/pgjones/hypercorn)。
```python
pip install uvicorn[standard]
```
# 示例1
## 源文件
创建一个`main.py`文件并写入以下内容:
```python
from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
```

## 运行
```python
uvicorn main:app --reload
```
uvicorn main:app 命令含义如下:
- `main`：main.py 文件（一个 Python "模块"）。
- `app`：在 main.py 文件中通过 app = FastAPI() 创建的对象。
- `--reload`：让服务器在更新代码后重新启动。仅在开发时使用该选项。

## 查看
使用浏览器访问[http://127.0.0.1:8000/items/5?q=somequery](http://127.0.0.1:8000/items/5?q=somequery)。
将会看到如下 JSON 响应：
```python
{"item_id":5,"q":"somequery"}
```
这里已经创建了一个具有以下功能的 API：
- 通过路径 `/` 和 `/items/{item_id}` 接受 HTTP 请求。
- 以上路径 都接受 GET 操作（也被称为 HTTP 方法）。
- `/items/{item_id}` 路径 有一个路径参数 `item_id` 并且应该为`int`类型。
- `/items/{item_id}` 路径 有一个可选的`str`类型的查询参数 `q`。


## 交互式API文档
现在访问[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。
会看到自动生成的交互式 API 文档（由 [Swagger UI](https://github.com/swagger-api/swagger-ui)生成）。

## 可选的API文档
访问[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)。
会看到另一个自动生成的文档（由[ReDoc](http://127.0.0.1:8000/redoc)生成）。
Redoc也很有用，尤其有时它检测到的请求体格式要比Swagger的准确。

# 示例2
现在修改`main.py`文件来从`PUT`请求中接收请求体。
以及借助`Pydantic`来使用标准的 Python 类型声明请求体。
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
```
请求体是一个JSON格式的数据体。

对于上面的代码，FastAPI 将会：
- 校验`GET`和`PUT`请求的路径中是否含有`item_id`
- 校验`GET`和`PUT`请求中的`item_id`是否为`int`类型：如果不是，客户端将会收到清晰有用的错误信息。
- 检查`GET`请求中是否有命名为`q`的可选查询参数：因为`q`被声明为`= None`，所以它是可选的；如果没有`None`它将会是必需的 (如`PUT`例子中的请求体)。
- 对于访问`/items/{item_id}`的`PUT`请求，将请求体读取为 JSON 并：检查是否有必需属性`name`并且值为`str`类型 、检查是否有必需属性`price`并且值为 `float`类型、检查是否有可选属性`is_offer`， 如果有的话值应该为`bool`类型；以上过程对于多层嵌套的 JSON 对象同样也会执行。
- 自动对 JSON 进行转换或转换成 JSON。
- 通过 OpenAPI 文档来记录所有内容，可被用于：交互式文档系统和许多编程语言的客户端代码自动生成系统
- 直接提供 2 种交互式文档 web 界面。


# Python类型提示
Python 3.6+ 版本加入了对"类型提示"的支持。
这些"类型提示"是一种新的语法（在 Python 3.6 版本加入）用来声明一个变量的类型。
类型提示是如下这样的：
```python
first_name:str,last_name:str
```
这和声明默认值是不同的，例如：
```python
first_name="john",last_name="doe"
```
我们用的是冒号（:），不是等号（=）。
而且添加类型提示一般不会改变原来的运行结果。

## 简单类型
不只是 `str`，你能够声明所有的标准 Python 类型。
比如以下类型：`int`、`float`、`bool`、`bytes`。
## 嵌套类型
有些容器数据结构可以包含其他的值，比如 `dict`、`list`、`set` 和 `tuple`。它们内部的值也会拥有自己的类型。
你可以使用 Python 的 `typing` 标准库来声明这些类型以及子类型，它专门用来支持这些类型提示。
（注意，`typing`支持3.6版本以上的所有python版本，如果是3.9以上的版本，甚至不需要`typing`即可实现这些容器结构的类型声明）
### 列表
例如，让我们来定义一个由 `str` 组成的 `list` 变量。
从 `typing` 模块导入 `List`（注意是大写的 `L`）：
```python
from typing import List

def process_items(items: List[str]):
    for item in items:
        print(item)
```
同样以冒号（:）来声明这个变量。输入 `List` 作为类型。
由于列表是带有"子类型"的类型，所以把子类型`str`放在方括号中。这表示："变量 `items` 是一个 `list`，并且这个列表里的每一个元素都是`str`"。

### 元组和集合
声明`tuple` 和`set`的方法也是一样的：
```python
from typing import Set, Tuple

def process_items(items_t: Tuple[int, int, str], items_s: Set[bytes]):
    return items_t, items_s
```
这表示：
（1）变量`items_t` 是一个`tuple`，其有三个元素，依次是`int`类型、`init`类型和`str`类型。
（2）变量`items_s`是一个`set`，其中的每个元素都是`bytes`类型。

### 字典
定义 `dict` 时，需要传入两个子类型，用逗号进行分隔。
第一个子类型声明 `dict` 的所有键，第二个子类型声明 `dict` 的所有值。
```python
from typing import Dict

def process_items(prices: Dict[str, float]):
    for item_name, item_price in prices.items():
        print(item_name)
        print(item_price)
```
这表示，变量 prices 是一个 dict：这个 dict 的所有键为 `str` 类型（可以看作是字典内每个元素的名称），这个 dict 的所有值为 `float` 类型（可以看作是字典内每个元素的价格）。

## 类作为类型
也可以将类声明为变量的类型。
假设你有一个名为 `Person` 的类，拥有 `name`属性，就可以如下这样将类声明为类型：
```python
class Person:
    def __init__(self, name: str):
        self.name = name

def get_person_name(one_person: Person):
    return one_person.name
```

## Pydantic模型
Pydantic 是一个用来用来执行数据校验的 Python 库。
你可以将数据的"结构"声明为带属性的类，然后每个属性都拥有类型。
接着可以用一些值来创建这个类的实例，这些值会被校验，并被转换为适当的类型（在需要的情况下），返回一个包含所有数据的对象。
一个例子如下：
```python
from datetime import datetime
from typing import List, Optional

# 导入pydantic的BaseModel
from pydantic import BaseModel

# 继承BaseModel，形成一个带属性的类，每个属性都可以声明类型，且有默认值
class User(BaseModel):
    id: int
    name = "John Doe"
    signup_ts: Optional[datetime] = None
    friends: List[int] = []


external_data = {
    "id": "123",
    "signup_ts": "2017-06-01 12:22",
    "friends": [1, "2", b"3"],
}
user = User(**external_data)
print(user)
# > User id=123 name='John Doe' signup_ts=datetime.datetime(2017, 6, 1, 12, 22) friends=[1, 2, 3]
print(user.id)
# > 123
```

## FastAPI 中的类型提示
FastAPI 利用这些类型提示来做下面几件事。
使用 FastAPI 时用类型提示声明参数可以获得：
（1）编辑器支持，
（2）类型检查。
并且 FastAPI 还会用这些类型声明来：
（1）定义参数要求：声明对请求路径参数、查询参数、请求头、请求体、依赖等的要求。
（2）转换数据：将来自请求的数据转换为需要的类型。
（3）校验数据： 对于每一个请求：当数据校验失败时自动生成错误信息返回给客户端。
（4）使用 OpenAPI 记录 API，然后用于自动生成交互式文档的用户界面。


# 并发和异步/等待
如果使用的第三方库说了使用`await`来调用，例如：
```python
results = await some_library()
```
那么就用`async def`声明路径操作函数，如下所示：
```python
@app.get('/')
async def read_results():
    results = await some_library()
    return results
```
如果正在使用一个第三方库来与某些东西（数据库、API、文件系统等）进行通信，并且它不支持使用`await`（目前大多数数据库都是这种情况），那么就只需`def`声明路径操作，例如：
```python
@app.get('/')
def results():
    results = some_library()
    return results
```
如果你的应用程序（以某种方式）不必与其他任何东西通信并等待它响应，请使用`async def`。
如果你是啥都不知道，就直接使用普通`def`。

注意：可以根据需要混合使用`def`和`async def`。FastAPI 会对它们做正确的事情。
无论如何，在上述任何情况下，FastAPI 仍将以异步方式工作并且非常快。
但是按照上面的步骤，它将能够进行一些性能优化。


# 用户指南
## OpenAPI
FastAPI 使用OpenAPI标准将所有 API 转换成模式schema。
### 模式Schema
模式是对事物的一种定义或描述。它并非具体的实现代码，而只是抽象的描述。
### API模式
此时所指的API模式就是API的规范，OpenAPI 就是一种规定如何定义 API 模式的规范。
 OpenAPI定义的模式包括API 路径，以及它们可能使用的参数等等。
### 数据模式
模式这个术语也可能指的是某些数据比如 JSON 的结构。
在这种情况下，它可以表示 JSON 的属性及其具有的数据类型，等等。
### OpenAPI 和 JSON Schema
OpenAPI定义了 API 模式。该模式中包含了你API 发送和接收的数据的定义（或称为数据模式），这些定义通过 JSON Schema 这一JSON 数据模式标准所生成。
### 查看 openapi.json
如果你对原始的 OpenAPI 模式长什么样子感到好奇，FastAPI自动生成了它，它就是一个json文件，可以通过[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)访问。

### OpenAPI的用途
驱动 FastAPI 内置的 2 个交互式文档系统的正是 OpenAPI 模式。
并且还有数十种替代方案，它们全部都基于 OpenAPI。你可以轻松地将这些替代方案中的任何一种添加到使用 FastAPI 构建的应用程序中。
你还可以使用它自动生成与你的 API 进行通信的客户端代码。例如 web 前端，移动端或物联网嵌入程序。

## 路径参数
这里的路径指的是 URL 中从第一个 `/` 起的后半部分。
路径也通常被称为端点或路由。
开发 API 时，「路径」是用来分离「关注点」和「资源」的主要手段。
举例：
```python
@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}
```
路径参数 `item_id` 的值将作为参数 `item_id` 传递给你的函数。

### 有类型的路径参数
可以使用标准的 Python 类型标注为函数中的路径参数声明类型。
比如：
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int): # 声明item_id为int型
    return {"item_id": item_id}
```

### 数据转换
如果你运行示例并打开浏览器访问[http://127.0.0.1:8000/items/3](http://127.0.0.1:8000/items/3)，那么将得到如下返回：
```python
{"item_id":3}
```
注意函数接收（并返回）的值为 3，是一个 Python `int` 值，而不是字符串 `"3"`。
所以，FastAPI 通过上面的类型声明提供了对请求的自动"解析"。

### 数据校验
但如果你通过浏览器访问[http://127.0.0.1:8000/items/foo](http://127.0.0.1:8000/items/foo)，你会看到一个清晰可读的 HTTP 错误：
```python
{
    "detail": [
        {
            "loc": [
                "path",
                "item_id"
            ],
            "msg": "value is not a valid integer",
            "type": "type_error.integer"
        }
    ]
}
```
所以，通过同样的 Python 类型声明，FastAPI 提供了数据校验功能。
注意上面的错误同样清楚地指出了校验未通过的具体原因。
在开发和调试与你的 API 进行交互的代码时，这非常有用。

### 路径的顺序
在创建路径操作时，会发现有些情况下路径是固定的。
比如 `/users/me`，我们假设它用来获取关于当前用户的数据，然后，还可以使用路径 `/users/{user_id}` 来通过用户 ID 获取关于特定用户的数据。
由于路径操作是按顺序依次运行的，需要确保路径 `/users/me` 声明在路径 `/users/{user_id}`之前。否则，`/users/{user_id}` 的路径还将与 `/users/me` 相匹配，"认为"自己正在接收一个值为 `"me"` 的 `user_id` 参数。 

### 预设值
如果有一个接收路径参数的路径操作，但希望预先设定可能的有效参数值，则可以使用标准的 Python Enum 类型。
比如：
```python
from enum import Enum

from fastapi import FastAPI

# 导入 Enum 并创建一个继承自 str 和 Enum 的子类。
# 通过从 str 继承，API 文档将能够知道这些值必须为 string 类型并且能够正确地展示出来。
class ModelName(str, Enum):
    # 然后创建具有固定值的类属性，这些固定值将是可用的有效值：
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

@app.get("/models/{model_name}")
# 使用你定义的枚举类（ModelName）创建一个带有类型标注的路径参数：
async def get_model(model_name: ModelName):
    # 路径参数的值是一个枚举成员
    # 可以将它与你创建的枚举类 ModelName 中的枚举成员进行比较
    if model_name == ModelName.alexnet:
        # 可以返回枚举成员，即使嵌套在 JSON 结构中（例如一个 dict 中）。
        # 在返回给客户端之前，它们将被转换为对应的值：
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    # 可以使用 model_name.value 或通常来说 your_enum_member.value 来获取实际的值（在这个例子中它是一个字符串str）
    # 也可以通过 ModelName.lenet.value 来获取值 "lenet"。
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}
```

### 包含路径的路径参数
假设有一个路径操作，它的路径为 `/files/{file_path}`。
但是你需要 `file_path` 自身也包含路径，比如 `home/johndoe/myfile.txt`。
因此，该文件的URL将类似于这样：`/files/home/johndoe/myfile.txt`。

OpenAPI 不支持这样的声明路径参数以在其内部包含路径，因为这可能会导致难以测试和定义的情况出现。
不过，你仍然可以通过 Starlette 的一个内部工具在 FastAPI 中实现它。
而且文档依旧可以使用，但是不会添加任何文档，来说明该参数应包含路径。
可以使用直接来自 Starlette 的选项来声明一个包含路径的路径参数：
```python
/files/{file_path:path}
```
在这种情况下，参数的名称为 `file_path`，结尾部分的 `:path` 说明该参数应匹配任意的路径。
因此，可以这样使用它：
```python
from fastapi import FastAPI

app = FastAPI()

# 你可能会需要参数包含 /home/johndoe/myfile.txt，以斜杠（/）开头。
# 在这种情况下，URL 将会是 /files//home/johndoe/myfile.txt，在files 和 home 之间有一个双斜杠（//）。
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
```

### 额外校验
可以使用Path为路径参数声明一些元数据及进行数值校验。
（1）声明元数据
```python
from typing import Optional

# 从 fastapi 导入 Path（有关Query的用法参见下节的查询参数的校验）
from fastapi import FastAPI, Path, Query

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
    # 声明路径参数 item_id的 title 元数据值
    # 路径参数总是必需的，因为它必须是路径的一部分。所以，你应该在声明时使用 ... 将其标记为必需参数。
    # 然而，即使你使用 None 声明路径参数或设置一个其他默认值也不会有任何影响，它依然会是必需参数。
    item_id: int = Path(..., title="The ID of the item to get"),
    q: Optional[str] = Query(None, alias="item-query"),
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results
```
（2）按需对参数排序
（2.1）FastAPI通过参数的名称、类型和默认值声明（Query、Path 等）来检测参数，而不在乎参数的顺序。
比如：
```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
    q: str, item_id: int = Path(..., title="The ID of the item to get")
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results
```
（2.2）另外一种排序方式
如果声明查询参数`q`时既不想使用Query，也不想使用默认值，与此同时，使用 Path 声明路径参数 item_id，并使它们的顺序与上面不同，Python 对此有一些特殊的语法。即传递 `*`星号 作为函数的第一个参数。
```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
    # Python 不会对该 * 做任何事情，但是它将知道之后的所有参数都应作为关键字参数（键值对），也被称为 kwargs，来调用。即使它们没有默认值。
    *, item_id: int = Path(..., title="The ID of the item to get"), q: str
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results
```
（3）数值校验
使用 Query 和 Path（以及你将在后面看到的其他类）可以声明字符串约束，但也可以声明数值约束。
```python
from fastapi import FastAPI, Path, Query

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
    # 添加 ge=1 后，item_id 将必须是一个大于（greater than）或等于（equal）1 的整数。
    # gt：大于（greater than）
    # le：小于等于（less than or equal）
    *, 
    item_id: int = Path(..., title="The ID of the item to get", ge=1, le=1000), 
    q: str,
    # 数值校验同样适用于 float 值。
    size: float = Query(..., gt=0, lt=10.5)
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results
```


## 查询参数
声明不属于路径参数的其他函数参数时，它们将被自动解释为"查询字符串"参数。
```python
from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```
查询字符串是键值对的集合，这些键值对位于 URL 的 `?` 之后，并以 `&` 符号分隔。
比如以下URL：
```python
http://127.0.0.1:8000/items/?skip=0&limit=10
```
查询参数为：
（1）skip：对应的值为 0
（2）limit：对应的值为 10
由于它们是 URL 的一部分，因此它们的"原始值"是字符串。
但是，当你为它们声明了 Python 类型（在上面的示例中为 `int`）时，它们将转换为该类型并针对该类型进行校验。
应用于路径参数的所有相同过程也适用于查询参数，包括编辑器支持、数据"解析"、数据校验、自动生成文档。
### 默认值
由于查询参数不是路径的固定部分，因此它们可以是可选的，并且可以有默认值。
在上面的示例中，它们具有 `skip=0` 和 `limit=10` 的默认值。
### 可选参数
通过同样的方式，你可以将它们的默认值设置为 None 来声明可选查询参数：
```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
```
### 多个路径和查询参数
可以同时声明多个路径参数和查询参数，FastAPI 能够识别它们。
而且不需要以任何特定的顺序来声明。
它们将通过名称被检测到：
```python
from typing import Optional

from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: Optional[str] = None, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item
```

### 必需查询参数
当你为非路径参数（目前而言，我们所知道的仅有查询参数）声明了默认值时，则该参数不是必需的。
如果你不想添加一个特定的值，而只是想使该参数成为可选的，则将默认值设置为 None。
但当你想让一个查询参数成为必需的，不声明任何默认值就可以：
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):
    item = {"item_id": item_id, "needy": needy}
    return item
```

### 额外校验
FastAPI 允许你为参数声明额外的信息和校验，方法是使用Query。
```python
from typing import Optional

# 从 fastapi 导入 Query
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
# 添加约束条件：即使 q 是可选的，但只要提供了该参数，则该参数值不能超过50个字符的长度。
# 方法就是将 Query 用作查询参数的默认值，并将它的 max_length 参数设置为 50
# 必须用 Query(None) 替换默认值 None，Query 的第一个参数同样也是用于定义默认值。
# max_length 参数将会校验数据，在数据无效时展示清晰的错误信息，并在 OpenAPI 模式的路径操作中记录该参数。
async def read_items(q: Optional[str] = Query(None, max_length=50)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

# 添加 min_length 参数等更多校验，以及正则表达式
# @app.get("/items/")
# async def read_items(
#     q: Optional[str] = Query(None, min_length=3, max_length=50, regex="^fixedquery$")
# ):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q": q})
#     return results
```

以上参数`q`默认值是`None`，所以它是可选的。
一般情形下不声明默认值就表明`q`是必需参数，但此时正在用 Query 声明它，因此，需要点特殊写法。
当在使用 Query 且需要声明一个值是必需的时，可以将 `...` 用作第一个参数值：
```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(q: str = Query(..., min_length=3)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```
这将使 FastAPI 知道此查询参数是必需的。

### 查询参数列表或多个值
要声明类型为 list 的查询参数，需要显式地使用 Query，否则该参数将被解释为请求体。
即当你使用 Query 显式地定义查询参数时，还可以声明它去接收一组值，或换句话来说，接收多个值。
```python
from typing import List, Optional

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(q: Optional[List[str]] = Query(None)):
    query_items = {"q": q}
    return query_items
```
然后输入网址为:[http://127.0.0.1:8000/items/?q=foo&q=bar](http://127.0.0.1:8000/items/?q=foo&q=bar)
那么会在路径操作函数的函数参数 `q` 中以一个 Python list 的形式接收到查询参数 `q` 的多个值（`foo` 和 `bar`）。
也可以对多个值配置默认值：
```python
from typing import List

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(q: List[str] = Query(["foo", "bar"])):
    query_items = {"q": q}
    return query_items
```

### Query的更多用法
（1）声明更多元数据
你可以添加更多有关该参数的信息，比如增加title和description。
这些信息将包含在生成的 OpenAPI 模式中，并由文档用户界面和外部工具所使用。
```python
from typing import Optional

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(
        None,
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
        min_length=3,
    )
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```
（2）别名参数
比如：
```python
from typing import Optional

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(q: Optional[str] = Query(None, alias="item-query")):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```
（3）弃用参数
现在假设你不再喜欢此参数。
你不得不将其保留一段时间，因为有些客户端正在使用它，但你希望文档清楚地将其展示为已弃用。
```python
from typing import Optional

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(
        None,
        alias="item-query",
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
        min_length=3,
        max_length=50,
        regex="^fixedquery$",
        # 配置该参数deprecated=True
        deprecated=True,
    )
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

## 请求体
当需要将数据从客户端（例如浏览器）发送给 API 时，需要将其作为“请求体”发送。
请求体request body是客户端发送给 API 的数据。响应体response body是 API 发送给客户端的数据。
你的 API 几乎总是要发送响应体。但是客户端并不总是需要发送请求体。
FastAPI使用 Pydantic 模型来声明请求体。
注意：不能使用 GET 操作（HTTP 方法）发送请求体。要发送数据，必须使用下列方法之一：POST（较常见）、PUT、DELETE 或 PATCH。
```python
from typing import Optional

from fastapi import FastAPI
# 从 pydantic 中导入 BaseModel
from pydantic import BaseModel

# 将你的数据模型声明为继承自 BaseModel 的类
# 使用标准的 Python 类型来声明所有属性
class Item(BaseModel):
    # 和声明查询参数时一样，当一个模型属性具有默认值时，它不是必需的。否则它是一个必需属性。将默认值设为 None 可使其成为可选属性。
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


app = FastAPI()

# 使用与声明路径和查询参数的相同方式声明请求体，即可将其添加到「路径操作」中
@app.post("/items/")
async def create_item(item: Item):
    # 在函数内部，你可以直接访问模型对象的所有属性
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```
可以看出，仅仅使用了 Python 类型声明，FastAPI 将会：
（1）将请求体作为 JSON 读取。
（2）转换为相应的类型（在需要时）。
（3）校验数据：如果数据无效，将返回一条清晰易读的错误信息，指出不正确数据的确切位置和内容。
（4）将接收的数据赋值到参数 item 中：由于已经在函数中将它声明为 Item 类型，还将获得对于所有属性及其类型的一切编辑器支持（代码补全等）。
（5）为模型生成 JSON 模式 定义，这些模式将成为生成的 OpenAPI 模式的一部分，并且被自动化文档 UI 所使用。

### 请求体 + 路径参数 + 查询参数
还可以同时声明请求体、路径参数和查询参数。
FastAPI 会识别它们中的每一个，并从正确的位置获取数据。
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

app = FastAPI()

@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: Optional[str] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result
```
函数参数将依次按如下规则进行识别：
（1）如果在路径中也声明了该参数，它将被用作路径参数。
（2）如果参数属于单一类型（比如 int、float、str、bool 等）它将被解释为查询参数。
（3）如果参数的类型被声明为一个 Pydantic 模型，它将被解释为请求体。

### 可选请求体
可以通过将默认值设置为 None 来将请求体参数声明为可选参数。
### 多个请求体参数
可以添加多个请求体参数到路径操作函数中，即使一个请求只能有一个请求体。
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


class User(BaseModel):
    username: str
    full_name: Optional[str] = None


@app.put("/items/{item_id}")
# FastAPI 将注意到该函数中有多个请求体参数（两个 Pydantic 模型参数）。
# 它将使用参数名称作为请求体中的键（字段名称），并期望一个类似于以下内容的请求体
# {
#     "item": {
#         "name": "Foo",
#         "description": "The pretender",
#         "price": 42.0,
#         "tax": 3.2
#     },
#     "user": {
#         "username": "dave",
#         "full_name": "Dave Grohl"
#     }
# }
async def update_item(item_id: int, item: Item, user: User):
    results = {"item_id": item_id, "item": item, "user": user}
    return results
```
### 请求体中的单一值
与使用 Query 和 Path 为查询参数和路径参数定义额外数据的方式相同，FastAPI 提供了一个同等的 Body。
```python
from typing import Optional

from fastapi import Body, FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


class User(BaseModel):
    username: str
    full_name: Optional[str] = None


@app.put("/items/{item_id}")
async def update_item(
    # 例如，为了扩展先前的模型，你可能决定除了 item 和 user 之外，还想在同一请求体中具有另一个键 importance。
    # 如果就按原样声明它，因为它是一个单一值，FastAPI 将假定它是一个查询参数。
    # 但是可以使用 Body 指示 FastAPI 将其作为请求体的另一个键进行处理。
    # Body 同样具有与 Query、Path 以及其他后面将看到的类完全相同的额外校验和元数据参数。
    item_id: int, item: Item, user: User, importance: int = Body(...)
):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results
```

### 单个请求体参数嵌入一个键中
```python
from typing import Optional

from fastapi import Body, FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.put("/items/{item_id}")
async def update_item(
    item_id: int, 
    # 假设只有一个来自 Pydantic 模型 Item 的请求体参数 item。
    # 默认情况下，FastAPI 将直接期望这样的请求体。
    # 但是，如果希望它拥有 item 键，就像在声明额外的请求体参数时所做的那样
    # 则可以使用一个特殊的 Body 参数 embed。
    # 在这种情况下，FastAPI 将期望像这样的请求体：
    # {
    #     "item": {
    #         "name": "Foo",
    #         "description": "The pretender",
    #         "price": 42.0,
    #         "tax": 3.2
    #     }
    # }
    # 而不是：
    # {
    #     "name": "Foo",
    #     "description": "The pretender",
    #     "price": 42.0,
    #     "tax": 3.2
    # }
    item: Item = Body(..., embed=True)):
    results = {"item_id": item_id, "item": item}
    return results
```

### 请求体中的字段校验
与使用 Query、Path 和 Body 在路径操作函数中声明额外的校验和元数据的方式相同，可以使用 Pydantic 的 Field 在 Pydantic 模型内部声明校验和元数据。
```python
from typing import Optional

from fastapi import Body, FastAPI
# 注意，Field 是直接从 pydantic 导入的，而不是像其他的（Query，Path，Body 等）都从 fastapi 导入。
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str
    # 对模型属性使用 Field
    # Field 的工作方式和 Query、Path 和 Body 相同，包括它们的参数等等也完全相同。
    description: Optional[str] = Field(
        None, title="The description of the item", max_length=300
    )
    price: float = Field(..., gt=0, description="The price must be greater than zero")
    tax: Optional[float] = None

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item = Body(..., embed=True)):
    results = {"item_id": item_id, "item": item}
    return results
```

### 嵌套模型
（1）普通python类型作为嵌套
使用 FastAPI，你可以定义、校验、记录文档并使用任意深度嵌套的模型（归功于Pydantic）。
```python
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    # 将一个属性定义为拥有子元素的类型，比如list
    # 这将使 tags 成为一个由元素组成的列表。不过它没有声明每个元素的类型。
    # tags: list = []
    # 但是 Python 有一种特定的方法来声明具有子类型的列表：
    tags: List[str] = []


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results
```
但是随后我们考虑了一下，意识到标签不应该重复，它们很大可能会是唯一的字符串。
Python 具有一种特殊的数据类型来保存一组唯一的元素，即 set。
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results
```
这样，即使你收到带有重复数据的请求，这些数据也会被转换为一组唯一项。
而且，每当你输出该数据时，即使源数据有重复，它们也将作为一组唯一项输出。
并且还会被相应地标注 / 记录文档。

（2）Pydantic模型作为嵌套
Pydantic 模型的每个属性都具有类型。
但是这个类型本身可以是另一个 Pydantic 模型。
因此，你可以声明拥有特定属性名称、类型和校验的深度嵌套的 JSON 对象。
上述这些都可以任意的嵌套。
比如：
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Image(BaseModel):
    url: str
    name: str


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = []
    # 将子模型用作类型
    # 这意味着 FastAPI 将期望类似于以下内容的请求体：
    # {
    #     "name": "Foo",
    #     "description": "The pretender",
    #     "price": 42.0,
    #     "tax": 3.2,
    #     "tags": ["rock", "metal", "bar"],
    #     "image": {
    #         "url": "http://example.com/baz.jpg",
    #         "name": "The Foo live"
    #     }
    # }

    image: Optional[Image] = None


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results
```
（3）特殊类型及其校验
除了普通的单一值类型（如 str、int、float 等）外，还可以使用其他的更复杂的单一值类型。
要了解所有的可用选项，请查看关于 [Pydantic字段类型](https://pydantic-docs.helpmanual.io/usage/types/) 的文档。
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()


class Image(BaseModel):
    # 例如，在 Image 模型中我们有一个 url 字段，我们可以把它声明为 Pydantic 的 HttpUrl，而不是 str。
   # 该字符串将被检查是否为有效的 URL，并在 JSON Schema / OpenAPI 文档中进行记录。
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()
    image: Optional[Image] = None


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results
```
（4）深度嵌套模型
可以定义任意深度的嵌套模型：
```python
from typing import List, Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()


class Image(BaseModel):
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()
    images: Optional[List[Image]] = None


class Offer(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    items: List[Item]


@app.post("/offers/")
async def create_offer(offer: Offer):
    return offer
```

### 纯列表请求体
如果你期望的 JSON 请求体的最外层是一个 JSON `array`（即 Python `list`），则可以在路径操作函数的参数中声明此类型。
```python
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()


class Image(BaseModel):
    url: HttpUrl
    name: str


@app.post("/images/multiple/")
async def create_multiple_images(images: List[Image]):
    return images
```

### 任意dict构成的请求体
也可以将请求体声明为使用某类型的key和其他类型的value的 dict。
无需事先知道有效的字段/属性（比如使用 Pydantic 模型的场景）是什么。
如果你想接收一些尚且未知的键，这将很有用。
还有一些奇葩的场景，如下：
```python
from typing import Dict

from fastapi import FastAPI

app = FastAPI()

@app.post("/index-weights/")
# 当你想要接收其他类型的键（键的类型通常都是str）时，例如 int。
# 请记住 JSON 仅支持将 str 作为键。
# 但是 Pydantic 具有自动转换数据的功能。
# 这意味着，即使你的 API 客户端只能将字符串作为键发送，只要这些字符串内容仅包含整数，Pydantic 就会对其进行转换并校验。
# 然后你接收的名为 weights 的 dict 实际上将具有 int 类型的键和 float 类型的值。
async def create_index_weights(weights: Dict[int, float]):
    return weights
```

### 声明请求体的示例数据
可以声明你想接收的数据的示例模样。
有几种方法可以做到：
（1）Pydantic schema_extra
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

    # 可以使用 Config 和 schema_extra 为Pydantic模型声明一个示例
    class Config:
        schema_extra = {
            "example": {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            }
        }


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results
```
（2）Field 的附加参数
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


class Item(BaseModel):
    # 在 Field, Path, Query, Body 和其他你之后将会看到的工厂函数中，
    # 可以通过给工厂函数传递其他的任意参数来给JSON模式声明额外信息，比如增加 example。
    # 请记住，传递的那些额外参数不会添加任何验证，只会添加注释，用于文档的目的。
    name: str = Field(..., example="Foo")
    description: Optional[str] = Field(None, example="A very nice Item")
    price: float = Field(..., example=35.4)
    tax: Optional[float] = Field(None, example=3.2)


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results
```
（3）Body的额外参数
```python
from typing import Optional

from fastapi import Body, FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item = Body(
        ...,
        example={
            "name": "Foo",
            "description": "A very nice Item",
            "price": 35.4,
            "tax": 3.2,
        },
    ),
):
    results = {"item_id": item_id, "item": item}
    return results
```

### 更新数据
（1）用`PUT`更新数据
```python
from typing import List, Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    tax: float = 10.5
    tags: List[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    return items[item_id]

# PUT 用于接收替换现有数据的数据。
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    # 把输入数据转换为以 JSON 格式存储的数据（比如，使用 NoSQL 数据库时），可以使用 jsonable_encoder。例如，把 datetime 转换为 str。
    update_item_encoded = jsonable_encoder(item)
    items[item_id] = update_item_encoded
    return update_item_encoded
```
当使用如下请求体：
```python
{
  "name": "bar111",
  "description": "string",
  "price": 0
}
```
用`PUT`更新`bar`时，因为上述数据未包含已存储的属性 `"tax": 20.2`，新的输入模型会把 `"tax": 10.5` 作为默认值。
因此，本次操作把 `tax` 的值「更新」为 `10.5`。
（2）用`PATCH`进行部分更新
HTTP PATCH 操作用于更新 部分 数据。
即，只发送要更新的数据，其余数据保持不变。
PATCH 没有 PUT 知名，也怎么不常用。
很多人甚至只用 PUT 实现部分更新。
FastAPI 对此没有任何限制，可以随意互换使用这两种操作。
但本指南也会分别介绍这两种操作各自的用途。
仍然以上述请求体为例：
```python
from typing import List, Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    tax: float = 10.5
    tags: List[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    return items[item_id]


@app.patch("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    stored_item_data = items[item_id]
    stored_item_model = Item(**stored_item_data)
    # 更新部分数据时，可以在 Pydantic 模型的 `.dict()` 中使用 `exclude_unset` 参数。
    # 如下代码生成的 dict 只包含创建 item 模型时显式设置的数据，而不包括默认值。
    # 即：update_data =  {'name': 'bar111', 'description': 'string', 'price': 0.0}
    # 而不是
    # update_data =  {'name': 'bar111', 'description': 'string', 'price': 0.0, 'tax': 10.5, 'tags': []}
    update_data = item.dict(exclude_unset=True)
    # 接下来，用 .copy() 为已有模型创建调用 update 参数的副本，该参数为包含更新数据的 dict。
    updated_item = stored_item_model.copy(update=update_data)
    items[item_id] = jsonable_encoder(updated_item)
    return updated_item
```
实际上，HTTP `PUT` 也可以完成相同的操作。 但本节以 `PATCH` 为例的原因是，该操作就是为了这种用例创建的。

## 其他数据类型
到目前为止，一直在使用常见的数据类型，如:`int`、`float`、`str`、`bool`，但是也可以使用更复杂的数据类型。
在这些复杂数据类型上，也能有如下功能：编辑器支持、传入请求的数据转换、响应数据的转换、数据验证、自动补全和文档。
一些常用的复杂数据类型:
（1）`UUID`:
一种标准的 "通用唯一标识符" ，在许多数据库和系统中用作ID。 
在请求和响应中将以 `str` 表示。
（2）`datetime.datetime`:
日期时间。
在请求和响应中将表示为 ISO 8601 格式的 str ，比如: `2008-09-15T15:53:00+05:00`.
（3）`datetime.date`:
日期。
在请求和响应中将表示为 ISO 8601 格式的 str ，比如: `2008-09-15`.
（4）`datetime.time`:
时间。
在请求和响应中将表示为 ISO 8601 格式的 str ，比如: `14:23:55.003`.
（5）`datetime.timedelta`:
时间间隔。
在请求和响应中将表示为 float 代表总秒数。
Pydantic 也允许将其表示为 "ISO 8601 时间差异编码"。
（6）`frozenset`:
在请求和响应中，作为 `set` 对待：
在请求中，列表将被读取，消除重复，并将其转换为一个 `set`。
在响应中 `set` 将被转换为 `list` 。
产生的schema将指定哪些`set` 的值是唯一的 (使用 JSON Schema的 `uniqueItems`)。
（7）bytes:
标准的 Python `bytes`。
在请求和响应中被当作 `str` 处理。
生成的schema将指定这个 `str` 是 `binary` "格式"。
（8）Decimal:
标准的 Python `Decimal`。
在请求和响应中被当做 `float` 一样处理。
（9）可以在这里检查所有有效的pydantic数据类型: [Pydantic data types](https://pydantic-docs.helpmanual.io/usage/types/)。

## Cookie参数
可以像定义 `Query` 参数和 `Path` 参数一样来定义 `Cookie` 参数。
```python
from typing import Optional
# 导入 Cookie
from fastapi import Cookie, FastAPI

app = FastAPI()

@app.get("/items/")
# 需要使用 Cookie 来声明 cookie 参数，否则参数将会被解释为查询参数。
# 声明 Cookie 参数的结构与声明 Query 参数和 Path 参数时相同。
# 第一个值是参数的默认值，同时也可以传递所有验证参数或注释参数，来校验参数
# Cookie 、Path 、Query是兄弟类，它们都继承自公共的 Param 类
# 但请记住，从 fastapi 导入的 Query、Path、Cookie 或其他参数声明函数，这些实际上是返回特殊类的函数。
async def read_items(ads_id: Optional[str] = Cookie(None)):
    return {"ads_id": ads_id}
```

## Header参数
可以使用定义 `Query`, `Path` 和 `Cookie` 参数一样的方法定义 `Header` 参数。
```python
from typing import Optional
# 导入 Header
from fastapi import FastAPI, Header

app = FastAPI()

@app.get("/items/")
# 为了声明headers， 需要使用Header, 否则参数将被解释为查询参数。
# 使用和Path, Query and Cookie 一样的结构定义 header 参数
# 第一个值是默认值，你可以传递所有的额外验证或注释参数
async def read_items(user_agent: Optional[str] = Header(None)):
    return {"User-Agent": user_agent}
```
### 自动转换
Header 在 Path、 Query 和 Cookie 提供的功能之上有一点额外的功能。
大多数标准的headers用 "连字符" 分隔，也称为 "减号" (-)。
但是像 `user-agent` 这样的变量在Python中是无效的。
因此, 默认情况下, Header 将把参数名称的字符从下划线 (_) 转换为连字符 (-) 来提取并记录 headers.
同时，HTTP headers 是大小写不敏感的，因此，因此可以使用标准Python样式(也称为 "`snake_case`")声明它们。
因此，可以像通常在Python代码中那样使用 `user_agent` ，而不需要将首字母大写为 `User_Agent` 或类似的东西。
如果出于某些原因，需要禁用下划线到连字符的自动转换，设置Header的参数 `convert_underscores` 为 `False`（注意，一些HTTP代理和服务器不允许使用带有下划线的headers。）。
### 重复的headers
有可能收到重复的headers。这意味着，相同的header具有多个值。
```python
from typing import List, Optional

from fastapi import FastAPI, Header

app = FastAPI()


@app.get("/items/")
# 可以在类型声明中使用一个list来定义这些情况。
async def read_items(x_token: Optional[List[str]] = Header(None)):
    return {"X-Token values": x_token}
```

## 响应模型
可以在任意的路径操作中使用 `response_model` 参数来声明用于响应的模型。
```python
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: List[str] = []


@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item
```
注意，`response_model`是「装饰器」方法（`get`，`post` 等）的一个参数。不像之前的所有参数和请求体，它不属于路径操作函数。
它接收的类型与为 Pydantic 模型属性所声明的类型相同，因此它可以是一个 Pydantic 模型，但也可以是一个由 Pydantic 模型组成的 list，例如 `List[Item]`。
FastAPI 将使用此 `response_model` 来：
（1）将输出数据转换为其声明的类型。
（2）校验数据。
（3）在 OpenAPI 的路径操作中为响应添加一个 JSON Schema。
（4）并在自动生成文档系统中使用。
但最重要的是：
会将输出数据限制在该模型定义内。这一点非常重要。
（响应模型在参数中被声明，而不是作为函数返回类型标注，这是因为路径函数可能不会真正返回该响应模型，而是返回一个 dict、数据库对象或其他模型，然后再使用 `response_model` 来执行字段约束和序列化。）
比如：
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app = FastAPI()

# 创建一个有明文密码的输入模型
class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: Optional[str] = None

# 一个没有明文密码的输出模型
class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


# 将 response_model 声明为了不包含密码的 UserOut 模型
@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    # 即便我们的路径操作函数将会返回包含密码的相同输入用户
    # FastAPI 将会负责过滤掉未在输出模型中声明的所有数据（使用 Pydantic）。
    return user
```

### 默认值
```python
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    # 响应模型可以具有默认值
    # 但如果它们并没有存储实际的值，你可能想从结果中忽略它们的默认值。
    description: Optional[str] = None
    price: float
    tax: float = 10.5
    tags: List[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}

# 可以设置路径操作装饰器的 response_model_exclude_unset=True 参数
# 这样响应中将不会包含那些默认值，而是仅有实际设置的值，比如foo这个id
# 如果你的数据在具有默认值的模型字段中有实际的值，比如bar这个id，这些值将包含在响应中。
# 如果数据具有与默认值相同的值，例如 ID 为 baz 的项，它们将包含在 JSON 响应中。
@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return items[item_id]
```

### 多个模型
从前面的示例继续，拥有多个相关的模型是很常见的。
对用户模型来说尤其如此，因为：
（1）输入模型需要拥有密码属性。
（2）输出模型不应该包含密码。
（3）数据库模型很可能需要保存密码的哈希值。
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app = FastAPI()


class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: Optional[str] = None


class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserInDB(BaseModel):
    username: str
    hashed_password: str
    email: EmailStr
    full_name: Optional[str] = None


def fake_password_hasher(raw_password: str):
    return "supersecret" + raw_password


def fake_save_user(user_in: UserIn):
    hashed_password = fake_password_hasher(user_in.password)
    # user_in 是一个 UserIn 类的 Pydantic 模型.
    # Pydantic 模型具有 .dict（） 方法，该方法返回一个拥有模型数据的 dict，暂时命名为user_dict。
    # 如果将该dict以 **user_dict 形式传递给一个函数（或类），Python将对其进行「解包」。它会将 user_dict 的键和值作为关键字参数直接传递。
    # 这样就获得了一个来自于其他 Pydantic 模型中的数据的 Pydantic 模型。
    user_in_db = UserInDB(**user_in.dict(), hashed_password=hashed_password)
    # 如下效果就是：
    # user_in_db = UserInDB(
    #     username = user_dict["username"],
    #     password = user_dict["password"],
    #     email = user_dict["email"],
    #     full_name = user_dict["full_name"],
    #     hashed_password = hashed_password,
    # )

    print("User saved! ..not really")
    return user_in_db


@app.post("/user/", response_model=UserOut)
async def create_user(user_in: UserIn):
    user_saved = fake_save_user(user_in)
    return user_saved
```

（1）减少重复
减少代码重复是 FastAPI 的核心思想之一。
因为代码重复会增加出现 bug、安全性问题、代码失步问题（当你在一个位置更新了代码但没有在其他位置更新）等的可能性。
上面的这些模型都共享了大量数据，并拥有重复的属性名称和类型。
我们可以声明一个 `UserBase` 模型作为其他模型的基类。然后可以创建继承该模型属性（类型声明，校验等）的子类。
所有的数据转换、校验、文档生成等仍将正常运行。
这样，可以仅声明模型之间的差异部分（具有明文的 `password`、具有 `hashed_password` 以及不包括密码）。
```python
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app = FastAPI()


class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserIn(UserBase):
    password: str


class UserOut(UserBase):
    pass


class UserInDB(UserBase):
    hashed_password: str


def fake_password_hasher(raw_password: str):
    return "supersecret" + raw_password


def fake_save_user(user_in: UserIn):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.dict(), hashed_password=hashed_password)
    print("User saved! ..not really")
    return user_in_db


@app.post("/user/", response_model=UserOut)
async def create_user(user_in: UserIn):
    user_saved = fake_save_user(user_in)
    return user_saved
```
（2）Union 或者 anyOf
以将一个响应声明为两种类型的 `Union`，这意味着该响应将是两种类型中的任何一种。
这将在 OpenAPI 中使用 `anyOf` 进行定义。
```python
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class BaseItem(BaseModel):
    description: str
    type: str


class CarItem(BaseItem):
    type = "car"


class PlaneItem(BaseItem):
    type = "plane"
    size: int


items = {
    "item1": {"description": "All my friends drive a low rider", "type": "car"},
    "item2": {
        "description": "Music is my aeroplane, it's my aeroplane",
        "type": "plane",
        "size": 5,
    },
}

# 定义一个 Union 类型时，首先包括最详细的类型，然后是不太详细的类型。
# 在下面的示例中，更详细的 PlaneItem 位于 Union[PlaneItem，CarItem] 中的 CarItem 之前。
@app.get("/items/{item_id}", response_model=Union[CarItem, PlaneItem])
async def read_item(item_id: str):
    return items[item_id]
```

（3）模型列表
可以用同样的方式声明由对象列表构成的响应。
```python
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: str


items = [
    {"name": "Foo", "description": "There comes my hero"},
    {"name": "Red", "description": "It's my aeroplane"},
]


@app.get("/items/", response_model=List[Item])
async def read_items():
    return items
```
（4）任意dict构成的响应
还可以使用一个任意的普通 dict 声明响应，仅声明键和值的类型，而不使用 Pydantic 模型。
如果你事先不知道有效的字段/属性名称（对于 Pydantic 模型是必需的），这将很有用。
```python
from typing import Dict

from fastapi import FastAPI

app = FastAPI()


@app.get("/keyword-weights/", response_model=Dict[str, float])
async def read_keyword_weights():
    return {"foo": 2.3, "bar": 3.4}
```

## 响应状态码
与指定响应模型的方式相同，也可以在以下任意的路径操作中使用 `status_code` 参数来声明用于响应的 HTTP 状态码。
```python
from fastapi import FastAPI

app = FastAPI()

# status_code 参数接收一个表示 HTTP 状态码的数字。
# status_code 也能够接收一个 IntEnum 类型，比如 Python 的 http.HTTPStatus。
@app.post("/items/", status_code=201)
async def create_item(name: str):
    return {"name": name}
```
注意，`status_code` 是「装饰器」方法（`get`，`post` 等）的一个参数。不像之前的所有参数和请求体，它不属于路径操作函数。
在 HTTP 协议中，将发送 3 位数的数字状态码作为响应的一部分。
这些状态码有一个识别它们的关联名称，但是重要的还是数字。
（1）100 及以上状态码用于「消息」响应。很少直接使用它们。具有这些状态代码的响应不能带有响应体。
（2）200 及以上状态码用于「成功」响应。这些是最常使用的。
- 200 是默认状态代码，它表示一切「正常」。
- 201表示「已创建」。它通常在数据库中创建了一条新记录后使用。
- 204表示「无内容」。此响应在没有内容返回给客户端时使用，因此该响应不能包含响应体。

    
（3）300 及以上状态码用于「重定向」。具有这些状态码的响应可能有或者可能没有响应体，但 304「未修改」是个例外，该响应不得含有响应体。
（4）400 及以上状态码用于「客户端错误」响应。这些可能是第二常用的类型。
- 404，用于「未找到」响应。
- 对于来自客户端的一般错误，可以只使用 400。

（5）500 及以上状态码用于服务器端错误。几乎永远不会直接使用它们。当你的应用程序代码或服务器中的某些部分出现问题时，它将自动返回这些状态代码之一。
要了解有关每个状态代码以及适用场景的更多信息，请查看[MDN 关于 HTTP 状态码的文档](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)。

## 表单数据
接收的不是 JSON，而是表单字段时，要使用 Form。
要使用表单，需预先安装`python-multipart`：
```python
pip install python-multipart
```
例如：
```python
# 从 fastapi 导入 Form
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/login/")
# 创建表单（Form）参数的方式与 Body 和 Query 一样
# OAuth2 规范的 "密码流" 模式规定要通过表单字段发送 username 和 password。
# 该规范要求字段必须命名为 username 和 password，并通过表单字段发送，不能用 JSON。
# 使用 Form 可以声明与 Body （及 Query、Path、Cookie）相同的元数据和验证。
# 声明表单体要显式使用 Form ，否则，FastAPI 会把该参数当作查询参数或请求体（JSON）参数。
async def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
```

与 JSON 不同，HTML 表单（`<form></form>`）向服务器发送数据通常使用「特殊」的编码。
FastAPI 要确保从正确的位置读取数据，而不是读取 JSON。
表单数据的「媒体类型」编码一般为 `application/x-www-form-urlencoded`。
但包含文件的表单编码为 `multipart/form-data`。文件处理详见下节。

可在一个路径操作中声明多个 Form 参数，但不能同时声明要接收 JSON 的 Body 字段。因为此时请求体的编码是 `application/x-www-form-urlencoded`，不是 `application/json`。
这不是 FastAPI 的问题，而是 HTTP 协议的规定。

## 请求文件
`File` 用于定义客户端的上传文件。
因为上传文件以「表单数据」形式发送。
所以接收上传文件，要预先安装 `python-multipart`。
有两种文件请求方式：
```python
# 从 fastapi 导入 File 和 UploadFile
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/files/")
# 创建文件（File）参数的方式与 Body 和 Form 一样
# 声明文件体必须使用 File，否则，FastAPI 会把该参数当作查询参数或请求体（JSON）参数。

# 如果把路径操作函数参数的类型声明为 bytes，FastAPI 将以 bytes 形式读取和接收文件内容。
# 这种方式把文件的所有内容都存储在内存里，适用于小型文件。
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
# 不过，很多情况下，UploadFile 更好用。
# 定义 File 参数时使用 UploadFile
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
```
UploadFile 与 bytes 相比有更多优势：
（1）使用 spooled 文件：存储在内存的文件超出最大上限时，FastAPI 会把文件存入磁盘；
（2）这种方式更适于处理图像、视频、二进制文件等大型文件，好处是不会占用所有内存；
（3）可获取上传文件的元数据；
（4）自带 [file-like](https://docs.python.org/zh-cn/3/glossary.html#term-file-like-object) `async` 接口；
（5）它暴露了一个 Python `SpooledTemporaryFile` 对象，可直接传递给其他想要`file-like`对象的库。

UploadFile 的属性如下：
（1）`filename`：上传文件的文件名字符串（str），例如` myimage.jpg`；
（2）`content_type`：内容的类型（MIME 类型 / 媒体类型）字符串（str），例如`image/jpeg`；
（3）`file`： `SpooledTemporaryFile`（一个` file-like` 对象）。该对象可直接传递给其他想要 `file-like` 对象的函数或库。

UploadFile 支持以下 `async` 方法，（使用内部 `SpooledTemporaryFile`）可调用如下方法。
（1）`write(data)`：把 `data` （`str` 或 `bytes`）写入文件；
（2）`read(size)`：按指定数量的字节或字符（`size (int)`）读取文件内容；
（3）`seek(offset)`：移动至文件`offset (int)` 字节处的位置；
例如，`await myfile.seek(0)`移动到文件开头；
执行 `await myfile.read()` 后，需再次读取已读取内容时，这种方法特别好用。
（4）`close()`：关闭文件。
因为上述方法都是 `async` 方法，要搭配`await`使用。
例如，在 `async` 路径操作函数 内，要用以下方式读取文件内容：
```python
contents = await myfile.read()
```
在普通 `def` 路径操作函数 内，则可以直接访问`UploadFile.file`：
```python
contents = myfile.file.read()
```

多文件上传：
```python
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.post("/files/")
# 可用同一个「表单字段」发送含多个文件的「表单数据」。
# 上传多个文件时，要声明含 bytes 或 UploadFile 的列表（List）：
async def create_files(files: List[bytes] = File(...)):
    # 接收的也是含 bytes 或 UploadFile 的列表（list）。
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(
    # 也可以声明元数据
    files: List[UploadFile] = File(..., description="Multiple files as UploadFile")):
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
```

## 请求表单和文件
FastAPI 支持同时使用 `File` 和 `Form` 定义文件和表单字段。
在同一个请求中接收数据和文件时，应同时使用 File 和 Form。
```python
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()


@app.post("/files/")
async def create_file(
    # 创建文件和表单参数的方式与 Body 和 Query 一样
    # 可在一个路径操作中声明多个 File 与 Form 参数，但不能同时声明要接收 JSON 的 Body 字段。因为此时请求体的编码为 multipart/form-data，不是 application/json。
    # 这不是 FastAPI 的问题，而是 HTTP 协议的规定。
    file: bytes = File(...), fileb: UploadFile = File(...), token: str = Form(...)
):
    return {
        "file_size": len(file),
        "token": token,
        "fileb_content_type": fileb.content_type,
    }
```

## 处理错误
某些情况下，需要向客户端返回错误提示。
这里所谓的客户端包括前端浏览器、其他应用程序、物联网设备等。
需要向客户端返回错误提示的场景主要如下：
（1）客户端没有执行操作的权限
（2）客户端没有访问资源的权限
（3）客户端要访问的项目不存在
等等 ...
遇到这些情况时，通常要返回 4XX（400 至 499）HTTP 状态码。
### 使用HTTPException
向客户端返回 HTTP 错误响应，可以使用 `HTTPException`。
HTTPException 是一个常规 Python 异常，包含了和 API 有关的额外数据。
```python
# 导入 HTTPException
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        # 因为是 Python 异常，所以不能 return，只能 raise。
        # 如在调用路径操作函数里的工具函数时，触发了 HTTPException，FastAPI 就不再继续执行路径操作函数中的后续代码，而是立即终止请求，并把 HTTPException 的 HTTP 错误发送至客户端。

        # 触发 HTTPException 时，可以用参数 detail 传递任何能转换为 JSON 的值，不仅限于 str。
        # 还支持传递 dict、list 等数据结构。
        # FastAPI 能自动处理这些数据，并将之转换为 JSON。
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```
有些场景下要为 HTTP 错误添加自定义响应头。例如，出于某些方面的安全需要。
一般情况下可能不会需要在代码中直接使用响应头。
但对于某些高级应用场景，还是需要添加自定义响应头：
```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}


@app.get("/items-header/{item_id}")
async def read_item_header(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "There goes my error"},
        )
    return {"item": items[item_id]}
```

### 安装自定义异常处理器
添加自定义处理器，要使用[Starlette 的异常工具](https://www.starlette.io/exceptions/)。
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 假设要触发的自定义异常叫作 UnicornException。
# 且需要 FastAPI 实现全局处理该异常。
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


app = FastAPI()

# 可以用 @app.exception_handler() 添加自定义异常控制器
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    # 接收到的错误信息清晰明了，HTTP 状态码为 418，JSON 内容如下：
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )


@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    # 请求 /unicorns/yolo 时，路径操作会触发 UnicornException。
    # 但该异常将会被 unicorn_exception_handler 处理。
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}
```
### 覆盖默认异常处理器
FastAPI 自带了一些默认异常处理器。
触发 `HTTPException` 或请求无效数据时，这些处理器返回默认的 JSON 响应结果。
不过，也可以使用自定义处理器覆盖默认异常处理器。
（这部分内容太高阶，且一般情况下使用默认异常处理器即可。跳过本部分）

## 路径操作配置
路径操作装饰器支持多种配置参数。
通过传递参数给路径操作装饰器 ，即可轻松地配置路径操作、添加元数据。
注意：以下参数应直接传递给路径操作装饰器，不能传递给路径操作函数。
### 状态码
`status_code` 用于定义路径操作响应中的 HTTP 状态码。
可以直接传递 `int` 代码， 比如 404。
如果记不住数字码的涵义，也可以用 `status` 的快捷常量，如`status.HTTP_201_CREATED`。

### tags参数
tags 参数的值是由 str 组成的 list （一般只有一个 str ），tags 用于为路径操作添加标签。
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()


@app.post("/items/", response_model=Item, tags=["items"])
async def create_item(item: Item):
    return item


@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Foo", "price": 42}]


@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "johndoe"}]
```
OpenAPI schema会自动添加标签，供 API 文档接口使用。
### summary 和 description 参数
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()


@app.post(
    "/items/",
    response_model=Item,
    # 对api的概要
    summary="Create an item",
    # 详细说明，更复杂的说明可以使用下面的文档字符串
    description="Create an item with all the information, name, description, price, tax and a set of unique tags",
)
async def create_item(item: Item):
    return item
```

### 文档字符串
描述内容比较长且占用多行时，可以在函数的 docstring 中声明路径操作的描述，FastAPI 支持从文档字符串中读取描述内容。
文档字符串支持 Markdown，能正确解析和显示 Markdown 的内容，但要注意文档字符串的缩进。
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()


@app.post("/items/", response_model=Item, summary="Create an item")
async def create_item(item: Item):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    return item
```
### 响应描述
```python
from typing import Optional, Set

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: Set[str] = set()


@app.post(
    "/items/",
    response_model=Item,
    summary="Create an item",
    # response_description 参数用于定义响应的描述说明
    # OpenAPI 规定每个路径操作都要有响应描述。
    # 如果没有定义响应描述，FastAPI 则自动生成内容为 "Successful response" 的响应描述。
    response_description="The created item",
)
async def create_item(item: Item):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    return item
```
### 弃用路径操作
`deprecated` 参数可以把路径操作标记为弃用，无需直接删除。
```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Foo", "price": 42}]


@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "johndoe"}]


@app.get("/elements/", tags=["items"], deprecated=True)
async def read_elements():
    return [{"item_id": "Foo"}]
```

## JSON兼容编码器
在某些情况下，可能需要将一个数据类型（如 Pydantic 模型）转换为与 JSON 兼容的类型（如`dict`、`list`等）。
比如想将该数据存储在数据库中。
为此，FastAPI提供了一个`jsonable_encoder()`功能。
```python
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
# 导入jsonable_encoder
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# 假设fake_db数据库只接收 JSON 兼容的数据。
# 例如，它不接收datetime对象，因为它们与 JSON 不兼容。
# 因此，datetime对象必须转换为包含ISO格式数据的str对象。
# 同样，该数据库不会接收 Pydantic 模型（具有属性的对象），只会接收dict.
fake_db = {}


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: Optional[str] = None


app = FastAPI()


@app.put("/items/{id}")
def update_item(id: str, item: Item):
    # jsonable_encoder将 Pydantic 模型转换为一个dict，并将datetime转换为str。
    # 它不会返回一个大的str，里面包含JSON格式的数据（作为字符串）。
    # 而是返回一个Python标准数据结构（例如一个dict），其中的值和子值都与 JSON 兼容。
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[id] = json_compatible_item_data
```

## 依赖项
FastAPI 提供了简单易用，但功能强大的依赖注入系统。
这个依赖系统设计的简单易用，可以让开发人员轻松地把组件集成至 FastAPI。
### 依赖注入
编程中的「依赖注入」是声明代码（本文中为路径操作函数 ）运行所需的，或要使用的「依赖」的一种方式。
然后，由系统（本文中为 FastAPI）负责执行任意需要的逻辑，为代码提供这些依赖（「注入」依赖项）。
依赖注入常用于以下场景：
（1）共享业务逻辑（复用相同的代码逻辑）
（2）共享数据库连接
（3）实现安全、验证、角色权限
等……
上述场景均可以使用依赖注入，将代码重复最小化。
依赖注入系统支持构建集成和「插件」。但实际上，FastAPI 根本不需要创建「插件」，因为使用依赖项可以声明不限数量的、可用于路径操作函数的集成与交互。
创建依赖项非常简单、直观，并且还支持导入 Python 包。毫不夸张地说，只要几行代码就可以把需要的 Python 包与 API 函数集成在一起。

### FastAPI 兼容性
依赖注入系统如此简洁的特性，让 FastAPI 可以与下列系统兼容：
- 关系型数据库
- NoSQL 数据库
- 外部支持库
- 外部 API
- 认证和鉴权系统
- API 使用监控系统
- 响应数据注入系统
等等……

### 例子
```python
from typing import Optional
# 导入 Depends
from fastapi import Depends, FastAPI

app = FastAPI()

# 创建依赖项
# 依赖项函数的形式和结构与路径操作函数一样。
# 可以把依赖项当作没有「装饰器」（即，没有 @app.get("/some-path") ）的路径操作函数。
async def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

# 声明依赖项
# 与在路径操作函数参数中使用 Body、Query 的方式相同，声明依赖项需要使用 Depends 和一个新的参数。

# 虽然，在路径操作函数的参数中使用 Depends 的方式与 Body、Query 相同，但 Depends 的工作方式略有不同。
# 这里只能传给 Depends 一个参数。
# 且该参数必须是可调用对象，比如函数。
# 该函数接收的参数和路径操作函数的参数一样。
@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# 接收到新的请求时，FastAPI 执行如下操作：
# （1）用正确的参数调用依赖项函数（「可依赖项」）
# （2）获取函数返回的结果
# （3）把函数返回的结果赋值给路径操作函数的参数
@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return commons
```
虽然，层级式依赖注入系统的定义与使用十分简单，但它却非常强大。
比如，可以定义依赖其他依赖项的依赖项。
最后，依赖项层级树构建后，依赖注入系统会处理所有依赖项及其子依赖项，并为每一步操作提供（注入）结果。

### 类作为依赖项
上面例子中依赖项的声明是个函数，它的返回值是个字典。这种方式可行，但可以更好，比如此时编辑器就没法提供很好的支持，因为它不知道字典的键和值是什么。
函数并不是声明依赖关系的唯一方法（尽管它可能更常见）。关键因素是依赖项应该是“可调用的”。
Python 中的“可调用”是 Python 可以像函数一样“调用”的任何东西，比如类`class`也是可调用的。

FastAPI 实际检查的是它是“可调用的”（函数、类或其他任何东西）和定义的参数。
如果在FastAPI 中将“可调用”作为依赖项传递，它将分析该“可调用”的参数，并以与路径操作函数的参数相同的方式处理它们。包括子依赖。
这也适用于完全没有参数的可调用对象。与没有参数的路径操作函数相同。
```python
from typing import Optional

from fastapi import Depends, FastAPI

app = FastAPI()


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# 将依赖项从上面的函数common_parameters更改为类CommonQueryParams
class CommonQueryParams:
    # __init__用于创建类实例的方法
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items/")
# 使用这个类来声明依赖
# 这将创建该类的“实例”，并且该实例将作为参数传递commons给函数。
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})
    return response
```

### 子依赖项
FastAPI 支持创建含子依赖项的依赖项。
并且，可以按需声明任意深度的子依赖项嵌套层级。
FastAPI 负责处理解析不同深度的子依赖项。
```python
from typing import Optional

from fastapi import Cookie, Depends, FastAPI

app = FastAPI()

# 创建第一层依赖项
def query_extractor(q: Optional[str] = None):
    return q

# 创建另一个依赖项函数，并同时再声明一个依赖项
def query_or_cookie_extractor(
    q: str = Depends(query_extractor), last_query: Optional[str] = Cookie(None)
):
    if not q:
        return last_query
    return q


@app.get("/items/")
async def read_query(query_or_default: str = Depends(query_or_cookie_extractor)):
    return {"q_or_cookie": query_or_default}
```
如果在同一个路径操作 多次声明了同一个依赖项，例如，多个依赖项共用一个子依赖项，FastAPI 在处理同一请求时，只调用一次该子依赖项。
FastAPI 不会为同一个请求多次调用同一个依赖项，而是把依赖项的返回值进行「缓存」，并把它传递给同一请求中所有需要使用该返回值的「依赖项」。
在高级使用场景中，如果不想使用「缓存」值，而是为需要在同一请求的每一步操作（多次）中都实际调用依赖项，可以把 Depends 的参数 `use_cache` 的值设置为 `False` :
```python
async def needy_dependency(fresh_value: str = Depends(get_value, use_cache=False)):
    return {"fresh_value": fresh_value}
```
### 路径操作装饰器中的依赖项
有时，我们并不需要在路径操作函数中使用依赖项的返回值。或者说，有些依赖项不返回值。
但仍要执行或解析该依赖项。
对于这种情况，不必在声明路径操作函数的参数时使用 Depends，而是可以在路径操作装饰器中添加一个由 `dependencies` 组成的 `list`。
```python
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()


async def verify_token(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: str = Header(...)):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

# 路径操作装饰器支持可选参数dependencies。
# 该参数的值是由 Depends() 组成的 list
# 路径操作装饰器依赖项（以下简称为“路径装饰器依赖项”）的执行或解析方式和普通依赖项一样，但就算这些依赖项会返回值，它们的值也不会传递给路径操作函数。
@app.get("/items/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]
```

### 全局依赖项
有时，我们要为整个应用添加依赖项。
通过与定义路径装饰器依赖项 类似的方式，可以把依赖项添加至整个 FastAPI 应用。
这样一来，就可以为所有路径操作应用该依赖项。
```python
from fastapi import Depends, FastAPI, Header, HTTPException


async def verify_token(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: str = Header(...)):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

# 全局依赖项
# 路径装饰器依赖项一节的思路均适用于全局依赖项
app = FastAPI(dependencies=[Depends(verify_token), Depends(verify_key)])


@app.get("/items/")
async def read_items():
    return [{"item": "Portal Gun"}, {"item": "Plumbus"}]


@app.get("/users/")
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]
```

### 有yield的依赖项
FastAPI 支持在完成后执行一些额外步骤的依赖项。
为此，请使用`yield`代替`return`，并在之后编写额外的步骤。
确保使用`yield`一次。
一个典型例子是想在发送请求时创建一个数据库对话，然后发送完成后就关闭它。
```python
# async或普通函数都可以
async def get_db():
    # 只有在yield之前和包含yield那行代码会在发送请求前执行
    db = DBSession()
    try: #使用try能收到异常
        # yield的值会注入到路径操作中，或其他依赖中
        yield db
    # 在yield之后的代码会在发送响应后再执行
    finally: # 使用finally来确保执行退出步骤，无论是否有异常。
        db.close()
```

## 安全性
有许多方法可以处理安全性、身份认证和授权等问题。
而且这通常是一个复杂而「困难」的话题。
在许多框架和系统中，仅处理安全性和身份认证就会花费大量的精力和代码（在许多情况下，可能占编写的所有代码的 50％ 或更多）。
FastAPI 提供了多种工具，可帮助你以标准的方式轻松、快速地处理安全性，而无需研究和学习所有的安全规范。
### 基本知识
（1）OAuth2
OAuth2是一个规范，它定义了几种处理身份认证和授权的方法。
它是一个相当广泛的规范，涵盖了一些复杂的使用场景。
它包括了使用「第三方」进行身份认证的方法。这就是所有带有「使用 Facebook，Google，Twitter，GitHub 登录」的系统背后所使用的机制。
有一个 OAuth 1，它与 OAuth2 完全不同，并且更为复杂，因为它直接包含了有关如何加密通信的规范。
如今它已经不是很流行，没有被广泛使用了。
OAuth2 没有指定如何加密通信，它期望你为应用程序使用 HTTPS 进行通信。
（2）OpenID Connect
OpenID Connect 是另一个基于 OAuth2 的规范。
它只是扩展了 OAuth2，并明确了一些在 OAuth2 中相对模糊的内容，以尝试使其更具互操作性。
例如，Google 登录使用 OpenID Connect（底层使用OAuth2）。
但是 Facebook 登录不支持 OpenID Connect。它具有自己的 OAuth2 风格。
（3）OpenID（非「OpenID Connect」）
还有一个「OpenID」规范。它试图解决与 OpenID Connect 相同的问题，但它不是基于 OAuth2。
因此，它是一个完整的附加系统。
如今它已经不是很流行，没有被广泛使用了。
（4）OpenAPI
OpenAPI（以前称为 Swagger）是用于构建 API 的开放规范（现已成为 Linux Foundation 的一部分）。
FastAPI 基于 OpenAPI。
这就是使多个自动交互式文档界面，代码生成等成为可能的原因。
OpenAPI 有一种定义多个安全「方案」的方法。
通过使用它们，你可以利用所有这些基于标准的工具，包括这些交互式文档系统。
OpenAPI 定义了以下安全方案：
（4.1）`apiKey`：一个特定于应用程序的密钥，可以来自：查询参数、请求头、cookie。
（4.2）`http`：标准的 HTTP 身份认证系统，包括：
- bearer: 一个值为 Bearer 加令牌字符串的 Authorization 请求头。这是从 OAuth2 继承的。
- HTTP Basic 认证方式。
- HTTP Digest，等等。

（4.3）`oauth2`：所有的 OAuth2 处理安全性的方式（称为「流程」）。 
以下几种流程适合构建 OAuth 2.0 身份认证的提供者（例如 Google，Facebook，Twitter，GitHub 等）：`implicit`、`clientCredentials`、`authorizationCode`。
但是有一个特定的「流程」可以完美地用于直接在同一应用程序中处理身份认证：`password`：接下来的几章将介绍它的示例。
（4.4）`openIdConnect`：提供了一种定义如何自动发现 OAuth2 身份认证数据的方法。此自动发现机制是 OpenID Connect 规范中定义的内容。

### 基本框架
假设在某个域中拥有后端API。并且在另一个域或同一域的不同路径中（或在移动应用程序中）有一个前端。
此时希望有一种方法让前端使用`username`和`password`与后端进行身份验证。
我们可以使用FastAPI提供的`OAuth2`构建它。
（注意，需要首先安装`python-multipart`，这是因为OAuth2使用“表单数据”来发送`username`和`password`。）
```python
from fastapi import Depends, FastAPI
# FastAPI提供了多种不同抽象级别的工具来实现安全功能。
# 在此示例中，将使用OAuth2，配合Password流和Bearer令牌。
# 具体地，使用OAuth2PasswordBearer类来做到这一点。

# bearer令牌不是唯一的选择。但它是该用例的最佳选择。
# 对于大多数用例来说，它可能是最好的，除非你是 OAuth2 专家并且确切地知道为什么有另一个选项更适合需求。
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

# 当创建OAuth2PasswordBearer类的实例时，传入tokenUrl参数。
# 此参数包含客户端（在用户浏览器中运行的前端）用于发送username和password以获取令牌的URL。
# 这里的tokenUrl="token"指的是一个相对URL，所以它相当于./token，不过该URL还尚未创建。
# 因为使用的是相对 URL，如果API位于https://example.com/，那么它将引用https://example.com/token。
# 但如果API 位于https://example.com/api/v1/，那么它就是https://example.com/api/v1/token.
# 使用相对 URL 非常重要，可以确保应用程序即使在像代理服务器后面这样的高级用例中也能正常工作。

# tokenUrl="token"不会创建该路径操作，但声明了/token这个URL将是客户端应该用来获取令牌的URL。该信息在 OpenAPI 中使用，然后在交互式 API 文档系统中使用。

# oauth2_scheme变量是OAuth2PasswordBearer的一个实例，但它也是“可调用的”。
# 因此它可被Depends使用

# 它将查看请求中的Authorization这个header，检查该值是否是Bearer以及一些令牌，并返回str类型的令牌.
# 如果它没有看到Authorization标头，或者该值没有Bearer标记，它将直接返回 401状态代码错误(UNAUTHORIZED)。
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/items/")
# 这个依赖将提供一个str赋值给token这个参数
async def read_items(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```
以上只是一个基本框架，还没有实际功能。
### 完整功能
在查看真正的具有安全功能的代码时，需要用到JWT令牌和哈希密码。
（1）JWT令牌
JWT 表示 「JSON Web Tokens」。
它是一个将 JSON 对象编码为密集且没有空格的长字符串的标准。字符串看起来像这样：
```python
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```
它没有被加密，因此任何人都可以从字符串内容中还原数据。
但它经过了签名。因此，当你收到一个由你发出的令牌时，可以校验令牌是否真的由你发出。
通过这种方式，你可以创建一个有效期为 1 周的令牌。然后当用户第二天使用令牌重新访问时，你知道该用户仍然处于登入状态。
一周后令牌将会过期，用户将不会通过认证，必须再次登录才能获得一个新令牌。而且如果用户（或第三方）试图修改令牌以篡改过期时间，你将因为签名不匹配而能够发觉。

对JWT令牌进行签名，实际就是一个密钥。要生成一个安全的随机密钥，可使用openssl。
windows版的openssl可以用别人编译好的，在[这里](https://slproweb.com/products/Win32OpenSSL.html)。
在终端中使用以下命令：
```python
openssl rand -hex 32
```

需要安装 python-jose 以在 Python 中生成和校验 JWT 令牌（Python-jose 需要一个额外的加密后端。这里推荐：pyca/cryptography。）
```python
pip install python-jose[cryptography]
```


（2）哈希密码
「哈希」的意思是：将某些内容（在本例中为密码）转换为看起来像乱码的字节序列（只是一个字符串）。
每次你传入完全相同的内容（完全相同的密码）时，你都会得到完全相同的乱码。
但是你不能从乱码转换回密码。
PassLib 是一个用于处理哈希密码的很棒的 Python 包。
它支持许多安全哈希算法以及配合算法使用的实用程序。
推荐的算法是 「Bcrypt」。
因此，安装附带 Bcrypt 的 PassLib：
```python
pip install passlib[bcrypt]
```

完整实例：
```python
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
# 将使用 FastAPI 的安全性实用工具来获取 username 和 password。
# OAuth2 规定在使用「password 流程」时，客户端/用户必须将 username 和 password 字段作为表单数据发送（因此，此处不能使用 JSON）。
# 而且规范明确了字段必须这样命名。因此 user-name 或 email 是行不通的。
# 不过不用担心，你可以在前端按照你的想法将它展示给最终用户。
# 而且你的数据库模型也可以使用你想用的任何其他名称。
# 但是对于登录路径操作，我们需要使用这些名称来与规范兼容（以具备例如使用集成的 API 文档系统的能力）。
# 具体地，导入 OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# 从 passlib 导入我们需要的工具。
from passlib.context import CryptContext

# 导入jwt相关模块
from jose import JWTError, jwt

from pydantic import BaseModel

#################################### JWT相关功能 #####################################
# 在终端中使用openssl rand -hex 32生成如下key
SECRET_KEY = "cda3c6e86b29270b741c9e1c62d052f5593921f26ae0badc4027b856f53d679f"
# 创建用于设定 JWT 令牌签名算法的变量 「ALGORITHM」，并将其设置为 "HS256"。
ALGORITHM = "HS256"
# 创建一个设置令牌过期时间的变量。
ACCESS_TOKEN_EXPIRE_MINUTES = 30 

# 定义一个将在令牌端点中用于响应的 Pydantic 模型。
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# 创建一个生成新的访问令牌的工具函数。
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

#################################### 数据库相关功能 #####################################
# 假的用户数据库
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        # 哈希密码，不明文存储，如下是明文密目"secret"的哈希，所以登录时要用secret登录
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


# 创建一个用户 Pydantic 模型
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


#################################### 哈希密码相关功能 #####################################
# 创建一个 PassLib 「上下文」。这将用于哈希和校验密码。
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 创建一个工具函数以哈希来自用户的密码。
def get_password_hash(password):
    return pwd_context.hash(password)

# 创建另一个工具函数，用于校验接收的密码是否与存储的哈希值匹配。
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

# 创建另一个工具函数用于认证并返回用户。
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


# 创建一个 get_current_user 依赖项
# get_current_user将具有一个之前所创建的同一个 oauth2_scheme 作为依赖项。
# 与之前直接在路径操作中所做的相同，新的依赖项 get_current_user 将从子依赖项 oauth2_scheme 中接收一个 str 类型的 token，具体地，就是一个JWT令牌
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 解码接收到的令牌，对其进行校验
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # JWT 的规范中有一个 sub 键，值为该令牌的主题。
        # 使用它并不是必须的，但这是放置用户标识的地方，所以在示例中使用了它。
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    # 如果令牌无效，立即返回一个 HTTP 错误。
    except JWTError:
        raise credentials_exception
    # 然后返回当前用户
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# 想要仅当此用户处于启用状态时才能获取 current_user。
# 因此，创建了一个额外的依赖项 get_current_active_user，而该依赖项又以 get_current_user 作为依赖项。
# 如果用户不存在或处于未启用状态，则这两个依赖项都将返回 HTTP 错误。
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# 在token的路径操作中通过Depends将OAuth2PasswordRequestForm作为依赖项使用
# OAuth2PasswordRequestForm 是一个类依赖项，声明了如下的请求表单：
# - username。
# - password。
# - 一个可选的 scope 字段，但实际上它是一个由空格分隔的「作用域」组成的长字符串。每个「作用域」只是一个字符串（中间没有空格）。它们通常用于声明特定的安全权限，例如：users:read 或者 users:write 是常见的例子。类依赖项 OAuth2PasswordRequestForm 的实例不会有用空格分隔的长字符串属性 scope，而是具有一个 scopes 属性，该属性将包含实际被发送的每个作用域字符串组成的列表。
# - 一个可选的 grant_type.
# - 一个可选的 client_id（该示例不需要它）。
# - 一个可选的 client_secret（该示例不需要它）。

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()): #这个是一个快捷方式
    # 使用表单字段中的 username 从（伪）数据库中获取用户数据。 
    # 如果没有这个用户，我们将返回一个错误消息，提示「用户名或密码错误」。
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 创建一个真实的 JWT 访问令牌 
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    # token 端点的响应必须是一个 JSON 对象，里面包含有：
    # （1）token_type字段：在该例中，由于我们使用的是「Bearer」令牌，因此令牌类型应为「bearer」。
    # （2）access_token字段：它是一个包含我们的访问令牌的字符串。
    return {"access_token": access_token, "token_type": "bearer"}


# 注意声明响应模型
@app.get("/users/me/", response_model=User)
# 在路径操作中使用 get_current_active_user 作为 Depends
# 注意我们将 current_user 的类型声明为 Pydantic 模型 User。
# 这将帮助我们在函数内部使用所有的代码补全和类型检查。

# 在这个端点中，只有当用户存在，身份认证通过且处于启用状态时，我们才能获得该用户
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]
```

## 中间件
可以向 FastAPI 应用添加中间件.
"中间件"是一个函数,它在每个请求被特定的路径操作处理之前,以及在每个响应返回之前工作.
- 它接收你的应用程序的每一个请求.
- 然后它可以对这个请求做一些事情或者执行任何需要的代码.
- 然后它将请求传递给应用程序的其他部分 (通过某种路径操作).
- 然后它获取应用程序生产的响应 (通过某种路径操作).
- 它可以对该响应做些什么或者执行任何需要的代码.
- 然后它返回这个响应。

```python
import time

from fastapi import FastAPI, Request

app = FastAPI()

# 要创建中间件你可以在函数的顶部使用装饰器 @app.middleware("http").
# 中间件参数接收如下参数:
- request.
- call_next函数：它将接收 request 作为参数.
（1）这个函数将 request 传递给相应的 路径操作.
（2）然后它将返回由相应的路径操作生成的 response.
（3）然后你可以在返回 response 前进一步修改它.

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # 可以 用'X-' 前缀添加专有自定义请求头.
    # 但是如果你想让浏览器中的客户端看到你的自定义请求头, 你需要把它们加到 CORS 配置 (CORS (Cross-Origin Resource Sharing)) 的 expose_headers 参数中
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## CORS跨域资源共享
CORS 或者「跨域资源共享」 指浏览器中运行的前端拥有与后端通信的 JavaScript 代码，而后端处于与前端不同的「源」的情况。
源origin是协议（`http`，`https`）、域（`myapp.com`，`localhost`，`localhost.tiangolo.com`）以及端口（`80`、`443`、`8080`）的组合。

因此，这些都是不同的源：`http://localhost`、`https://localhost`、`http://localhost:8080`。
即使它们都在 localhost 中，但是它们使用不同的协议或者端口，所以它们都是不同的「源」。
### 步骤
假设你的浏览器中有一个前端运行在`http://localhost:8080`，并且它的 JavaScript 正在尝试与运行在`http://localhost`的后端通信（因为我们没有指定端口，浏览器会采用默认的端口`80`）。
然后，浏览器会向后端发送一个 `HTTP OPTIONS` 请求，如果后端发送适当的 `headers` 来授权来自这个不同源（`http://localhost:8080`）的通信，浏览器将允许前端的 JavaScript 向后端发送请求。
为此，后端必须有一个「允许的源」列表。
在这种情况下，它必须包含`http://localhost:8080`，前端才能正常工作。
### 通配符
也可以使用 "*"（一个「通配符」）声明这个列表，表示全部都是允许的。
但这仅允许某些类型的通信，不包括所有涉及凭据的内容：像 Cookies 以及那些使用 Bearer 令牌的授权 headers 等。
因此，为了一切都能正常工作，最好显式地指定允许的源。
### 使用CORSMiddleware
可以在 FastAPI 应用中使用 CORSMiddleware 来配置它。
（1）导入 CORSMiddleware。
（2）创建一个允许的源列表（由字符串组成）。
（3）将其作为「中间件」添加到你的 FastAPI 应用中。

也可以指定后端是否允许：
（1）凭证（授权 headers，Cookies 等）。
（2）特定的 HTTP 方法（POST，PUT）或者使用通配符 "*" 允许所有方法。
（3）特定的 HTTP headers 或者使用通配符 "*" 允许所有 headers。

默认情况下，这个 CORSMiddleware 实现所使用的默认参数较为保守，所以你需要显式地启用特定的源、方法或者 headers，以便浏览器能够在跨域上下文中使用它们。
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins：一个允许跨域请求的源列表。例如 ['https://example.org', 'https://www.example.org']。可以使用 ['*'] 允许任何源。
    allow_origins=origins,
    # allow_credentials：指示跨域请求支持 cookies。默认是 False。另外，允许凭证时 allow_origins 不能设定为 ['*']，必须指定源。
    allow_credentials=True,
    # allow_methods：一个允许跨域请求的 HTTP 方法列表。默认为 ['GET']。你可以使用 ['*'] 来允许所有标准方法。
    allow_methods=["*"],
    # allow_headers：一个允许跨域请求的 HTTP 请求头列表。默认为 []。你可以使用 ['*'] 允许所有的请求头。
    # Accept、Accept-Language、Content-Language 以及 Content-Type 请求头总是允许 CORS 请求。
    allow_headers=["*"],
    # 还有其他，比如：
    # allow_origin_regex：一个正则表达式字符串，匹配的源允许跨域请求。例如 'https://.*\.example\.org'。
    # expose_headers：指示可以被浏览器访问的响应头。默认为 []。
    # max_age：设定浏览器缓存 CORS 响应的最长时间，单位是秒。默认为 600。
)


@app.get("/")
async def main():
    return {"message": "Hello World"}
```
中间件响应两种特定类型的 HTTP 请求：
（1）CORS 预检请求
这是些带有 Origin 和 Access-Control-Request-Method 请求头的 OPTIONS 请求。
在这种情况下，中间件将拦截传入的请求并进行响应，出于提供信息的目的返回一个使用了适当的 CORS headers 的 200 或 400 响应。
（2）简单请求
任何带有 Origin 请求头的请求。在这种情况下，中间件将像平常一样传递请求，但是在响应中包含适当的 CORS headers。

## SQL数据库
FastAPI不要求使用 SQL（关系）数据库。不过可以使用任何想要的关系数据库（通过SQLAlchemy实现）。
在下面示例中，将使用SQLite，因为它使用单个文件并且 Python 具有集成支持。对于大的生产应用程序，可能希望使用像PostgreSQL这样的数据库服务器。
[这里](https://github.com/tiangolo/full-stack-fastapi-postgresql)有一个带有FastAPI和PostgreSQL的官方项目生成器，全部基于Docker，包括前端和更多工具。

首先需要安装SQLAlchemy：
```python
pip install SQLAlchemy
```

文件结构：
```python
.
└── sql_app
    ├── __init__.py
    ├── crud.py
    ├── database.py
    ├── main.py
    ├── models.py
    └── schemas.py
```
（1）对于database.py文件
```python
# 导入 SQLAlchemy 部件
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 为 SQLAlchemy 创建数据库 URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

# 创建 SQLAlchemy engine
engine = create_engine(
    # connect_args仅用于SQLite. 其他数据库不需要它
    # 默认情况下，SQLite只允许一个线程与其通信，假设每个线程将处理一个独立的请求。
    # 这是为了防止意外地为不同的事物（不同的请求）共享相同的连接。
    # 但是在 FastAPI 中，使用普通函数 ( def) 多个线程可以为同一个请求与数据库交互，因此我们需要让 SQLite 知道它应该允许这个connect_args={"check_same_thread": False}
    # 此外，我们将确保每个请求获得自己的数据库连接会话，因此不需要该默认机制。
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
# 创建一个SessionLocal类
# SessionLocal类的每一个实例都是一个数据库会话。该类本身并不是数据库会话。
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 使用declarative_base()返回一个类。
# 稍后将从这个类继承来创建每个数据库模型或类（ORM 模型）：
Base = declarative_base()
```
（2）对于models.py文件
```python
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

# 从database.py文件中导入Base类
from .database import Base

# 创建Base的子类，这些类都是SQLAlchemy模型
class User(Base):
    # __tablename__属性告诉SQLAlchemy这些模型在数据库中的文件
    __tablename__ = "users"

    # 创建模型字段，每个字段在数据库中都是一列Column，也就是数据表的字段
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    # 创建数据表之间的关系
    items = relationship("Item", back_populates="owner")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="items")
```
（3）对于schemas.py文件
```python
from typing import List, Optional

# 为了不搞混SQLAlchemy模型和Pydantic模型，就用models.py存放SQLAlchemy模型，而该文件schemas.py存放Pydantic模型
from pydantic import BaseModel

# 创建ItemBase这一Pydantic模型，作为一个盛放关于item统一属性的模型
class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    owner_id: int

    # 用config属性对Pydantic模型进行配置
    class Config:
        # orm_mode参数告诉Pydantic模型读取数据，即使它不是一个dict，而是一个ORM模型（或其他任意带属性的对象）
        # 此时它会尝试通过id = data.id这样获取属性的方式来读取数据，
        # 而不是id = data['id']这样字典读key的方法
        # 这样Pydantic模型就与ORM模型可以兼容，
        # 在路径操作中也可使用response_model来声明
        orm_mode = True


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    items: List[Item] = []

    class Config:
        orm_mode = True
```

（4）对于crud.py文件
```python
# 该文件就是对数据库的操作，即增删改查

# 从SQLAlchemy.orm中导入Session，这将允许对下面的db参数进行类型声明，从而有类型检查和补全功能
from sqlalchemy.orm import Session

# 导入SQLAlchemy模型和Pydantic模型
from . import models, schemas

# 创建几个工具函数

# （1）通过id读取单个用户
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()
# （2）通过邮箱读取单个用户
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

# （3）读取多个用户
def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

# （4）创建一个用户
def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(email=user.email, hashed_password=fake_hashed_password)
    # add添加实例对象
    db.add(db_user)
    # commit提交更改
    db.commit()
    # refresh刷新以获得最新数据
    db.refresh(db_user)
    return db_user

# （5）读取多个items
def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()

# （6）创建一个item
def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int):
    # **是用来解包，并且附加另一个参数
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

（5）对于main.py文件
```python
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

# 创建数据库的数据表
# 通常，可能会使用Alembic初始化数据库（创建表等）。
# 还可以使用 Alembic 进行“迁移”（这是它的主要工作）。
# “迁移”是每当更改 SQLAlchemy 模型的结构、添加新属性等以在数据库中复制这些更改、添加新列、新表等时所需的一组步骤。
# 可以在Project Generation - Template的模板中找到一个 FastAPI 项目中的 Alembic 示例。
models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# 创建一个依赖项
def get_db():
    # 每个请求都有一个独立的会话，请求结束后就关闭它
    # 但所有会话都是同一个
    db = SessionLocal()
    try:
        # yield的用法可以参考之前部分
        yield db
    finally:
        db.close()


@app.post("/users/", response_model=schemas.User)
# 在路径操作中都使用上述依赖
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    # 返回的是SQLAlchemy模型，但是因为设置了orm_mode，就能与响应模型兼容
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/users/{user_id}/items/", response_model=schemas.Item)
def create_item_for_user(
    user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
):
    return crud.create_user_item(db=db, item=item, user_id=user_id)


@app.get("/items/", response_model=List[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items
```

运行：
```python
uvicorn sql_app.main:app --reload
```
注意得在包外面，按如上方式运行。因为文件内有相对路径导入。

查看SQLite数据，可以使用在线工具，如[https://inloop.github.io/sqlite-viewer/](https://inloop.github.io/sqlite-viewer/)。

## 大型项目的文件组织
如果正在开发一个应用程序或 Web API，很少会将所有的内容都放在一个文件中。
FastAPI 提供了一个方便的工具，可以在保持所有灵活性的同时构建你的应用程序。
如果你来自 Flask，那这将相当于 Flask 的 Blueprints。

文件结构：
```python
.
├── app                  # 「app」是一个 Python 包
│   ├── __init__.py      # 这个文件使「app」成为一个 Python 包
│   ├── main.py          # 「main」模块，例如 import app.main
│   ├── dependencies.py  # 「dependencies」模块，例如 import app.dependencies
│   └── routers          # 「routers」是一个「Python 子包」
│   │   ├── __init__.py  # 使「routers」成为一个「Python 子包」
│   │   ├── items.py     # 「items」子模块，例如 import app.routers.items
│   │   └── users.py     # 「users」子模块，例如 import app.routers.users
│   └── internal         # 「internal」是一个「Python 子包」
│       ├── __init__.py  # 使「internal」成为一个「Python 子包」
│       └── admin.py     # 「admin」子模块，例如 import app.internal.admin
```

### APIRouter路由
APIRouter可以使得对于不同对象的路径操作写在不同文件中，以使其井井有条。
（1）专门用于处理用户逻辑的文件是位于 `/app/routers/users.py` 的子模块
```python
# 专门处理用户逻辑的路由文件

# 导入 APIRouter
from fastapi import APIRouter

# 通过与 FastAPI 类相同的方式创建一个「实例」
router = APIRouter()

# 可以将 APIRouter 视为一个「迷你 FastAPI」类。
# 所有相同的选项都得到支持。
# 所有相同的 parameters、responses、dependencies、tags 等等。
@router.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]


@router.get("/users/me", tags=["users"])
async def read_user_me():
    return {"username": "fakecurrentuser"}


@router.get("/users/{username}", tags=["users"])
async def read_user(username: str):
    return {"username": username}
```
（2）专门用于处理应用程序中「项目」的路由操作：
```python
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_token_header

# 此模块中的所有路径操作都有相同的：
    # 路径 prefix：/items。
    # tags：（仅有一个 items 标签）。
    # 额外的 responses。
    # dependencies：它们都需要我们创建的 X-Token 依赖项。
# 因此，我们可以将其添加到 APIRouter 中，而不是将其添加到每个路径操作中。
router = APIRouter(
    prefix="/items",
    tags=["items"],
    dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}


@router.get("/")
async def read_items():
    return fake_items_db


@router.get("/{item_id}")
async def read_item(item_id: str):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"name": fake_items_db[item_id]["name"], "item_id": item_id}


@router.put(
    "/{item_id}",
    # 仍然可以添加更多将会应用于特定的路径操作的tags，以及一些特定于该路径操作的额外 responses
    # 最后的这个路径操作将包含标签的组合：["items"，"custom"]。
    # 并且在文档中也会有两个响应，一个用于 404，一个用于 403。
    tags=["custom"],
    responses={403: {"description": "Operation forbidden"}},
)
async def update_item(item_id: str):
    if item_id != "plumbus":
        raise HTTPException(
            status_code=403, detail="You can only update the item: plumbus"
        )
    return {"item_id": item_id, "name": "The great Plumbus"}
```

也可以在另一个 APIRouter 中包含 APIRouter，通过：
```python
router.include_router(other_router)
```
请确保在你将 `router` 包含到 FastAPI 应用程序之前进行此操作，以便 `other_router` 中的路径操作也能被包含进来。

（3）现在，假设你的组织为你提供了 `app/internal/admin.py` 文件。
它包含一个带有一些由你的组织在多个项目之间共享的管理员路径操作的 `APIRouter`。
对于此示例，它将非常简单。但是假设由于它是与组织中的其他项目所共享的，因此我们无法对其进行修改，以及直接在 APIRouter 中添加 `prefix`、`dependencies`、`tags` 等：
```python
from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def update_admin():
    return {"message": "Admin getting schwifty"}
```
但是我们仍然希望在包含 APIRouter 时设置一个自定义的 `prefix`，以便其所有路径操作以 `/admin` 开头，我们希望使用本项目已经有的 `dependencies` 保护它，并且我们希望它包含自定义的 `tags` 和 `responses`。
这些将在主体文件`main.py`中实现。


### 依赖项
我们将需要一些在应用程序的好几个地方所使用的依赖项。
因此，将它们放在 `dependencies` 模块（`app/dependencies.py`）中。
```python
from fastapi import Header, HTTPException
# 我们正在使用虚构的请求首部来简化此示例。
# 但在实际情况下，使用集成的安全性实用工具会得到更好的效果。
async def get_token_header(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def get_query_token(token: str):
    if token != "jessica":
        raise HTTPException(status_code=400, detail="No Jessica token provided")
```
所有的这些路径操作都将在自身之前计算/执行 dependencies 列表。
- 如果你还在一个具体的路径操作中声明了依赖项，它们也会被执行。
- 路由器的依赖项最先执行，然后是装饰器中的 dependencies，再然后是普通的参数依赖项。
- 你还可以添加具有 scopes 的 Security 依赖项。


### 主体
`app/main.py`模块导入并使用 FastAPI 类。
这将是你的应用程序中将所有内容联结在一起的主文件。
并且由于你的大部分逻辑现在都存在于其自己的特定模块中，因此主文件的内容将非常简单。
```python
from fastapi import Depends, FastAPI

from .dependencies import get_query_token, get_token_header
from .internal import admin
from .routers import items, users

# 甚至可以声明全局依赖项，它会和每个 APIRouter 的依赖项组合在一起：
app = FastAPI(dependencies=[Depends(get_query_token)])

# 包含来自 users 和 items 子模块的 router。
# 使用 app.include_router()，我们可以将每个 APIRouter 添加到主 FastAPI 应用程序中。
app.include_router(users.router)
app.include_router(items.router)
# 通过将以下这些参数传递给 app.include_router() 来完成所有的声明，而不必修改原始的 APIRouter
# 这样，原始的APIRouter将保持不变，因此我们仍然可以与组织中的其他项目共享相同的 app/internal/admin.py 文件。
# 但这只会影响我们应用中的 APIRouter，而不会影响使用它的任何其他代码。
# 因此，举例来说，其他项目能够以不同的身份认证方法使用相同的 APIRouter。
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_token_header)],
    responses={418: {"description": "I'm a teapot"}},
)

# 可以直接将路径操作添加到 FastAPI 应用中
@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
```

## 后台任务
可以定义在返回响应后运行的后台任务。
这对于需要在请求之后发生的操作很有用，但客户端实际上不必在接收响应之前等待操作完成。
这包括，例如：
（1）执行操作后发送的电子邮件通知： 由于连接到电子邮件服务器并发送电子邮件往往“慢”（几秒钟），可以立即返回响应并在后台发送电子邮件通知。
（2）处理数据：例如，假设您收到一个必须经过缓慢处理的文件，您可以返回“已接受”（HTTP 202）的响应并在后台处理它。
```python
# 导入BackgroundTasks
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


# 创建一个作为后台任务运行的函数。
# 它是一个可以接收参数的标准函数。
# 它可以是一个async def或普通def函数，FastAPI会知道如何正确处理它。
# 该例中任务函数将写入文件（模拟发送电子邮件的场景）。
# 并且由于写操作不使用async和await，我们用 normal 定义函数def：
def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)


@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    # 定义一个参数，其类型声明为：BackgroundTasks 
    # FastAPI将为您创建类型的对象BackgroundTasks并将其作为该参数传递。
    background_tasks: BackgroundTasks):
    # 在路径操作函数内部，使用以下方法将任务函数传递给后台任务对象
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

注意：
如果需要执行繁重的后台计算并且不一定需要它由同一进程运行（例如，不需要共享内存、变量等），可能会受益于使用其他更大的工具，例如Celery。
它们往往需要更复杂的配置，消息/作业队列管理器，如 RabbitMQ 或 Redis，但它们允许在多个进程中运行后台任务，尤其是在多个服务器中。
要查看示例，请查看Project Generators，它们都包含已配置的 Celery。
但是，如您需要从同一个FastAPI应用程序访问变量和对象，或者需要执行小型后台任务（例如发送电子邮件通知），只需使用BackgroundTasks.


### 依赖注入
`BackgroundTasks`也适用于依赖注入系统。
```python
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI

app = FastAPI()


def write_log(message: str):
    with open("log.txt", mode="a") as log:
        log.write(message)


def get_query(background_tasks: BackgroundTasks, q: Optional[str] = None):
    if q:
        message = f"found query: {q}\n"
        background_tasks.add_task(write_log, message)
    return q


@app.post("/send-notification/{email}")
async def send_notification(
    # 在此示例中，消息将在发送响应后写入文件log.txt。
    # 如果请求中有查询，它将在后台任务中写入日志。
    # 然后在路径操作函数处生成的另一个后台任务将使用email路径参数写入一条消息。
    email: str, background_tasks: BackgroundTasks, q: str = Depends(get_query)
):
    message = f"message to {email}\n"
    background_tasks.add_task(write_log, message)
    return {"message": "Message sent"}
```

## 元数据和文档URL
可以在 FastAPI 应用中自定义几个元数据配置。
```python
from fastapi import FastAPI

description = """
ChimichangApp API helps you do awesome stuff. 🚀

## Items

You can **read items**.

## Users

You will be able to:

* **Create users** (_not implemented_).
* **Read users** (_not implemented_).
"""

app = FastAPI(
    # Title：在 OpenAPI 和自动 API 文档用户界面中作为 API 的标题/名称使用。
    title="ChimichangApp",
    # Description：在 OpenAPI 和自动 API 文档用户界面中用作 API 的描述。
    description=description,
    # Version：API 版本，例如 v2 或者 2.5.0。
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Deadpoolio the Amazing",
        "url": "http://x-force.example.com/contact/",
        "email": "dp@x-force.example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/items/")
async def read_items():
    return [{"name": "Katana"}]
```

### 标签元数据
也可以使用参数 `openapi_tags`，为用于分组路径操作的不同标签添加额外的元数据。
```python
from fastapi import FastAPI

# 接受一个列表，这个列表包含每个标签对应的一个字典。
# 每个标签元数据字典的顺序也定义了在文档用户界面显示的顺序。
tags_metadata = [
    {
        # name（必要）：一个 str，它与路径操作和 APIRouter 中使用的 tags 参数有相同的标签名。
        "name": "users",
        # description：一个用于简短描述标签的 str。它支持 Markdown 并且会在文档用户界面中显示。
        "description": "Operations with users. The **login** logic is also here.",
    },
    {
        "name": "items",
        "description": "Manage items. So _fancy_ they have their own docs.",
        # externalDocs：一个描述外部文档的 dict：
            # description：用于简短描述外部文档的 str。
            # url（必要）：外部文档的 URL str。
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
]

app = FastAPI(openapi_tags=tags_metadata)

# 将 tags 参数和路径操作（以及 APIRouter）一起使用，将其分配给不同的标签
@app.get("/users/", tags=["users"])
async def get_users():
    return [{"name": "Harry"}, {"name": "Ron"}]


@app.get("/items/", tags=["items"])
async def get_items():
    return [{"name": "wand"}, {"name": "flying broom"}]
```

### 文档URL
```python
from fastapi import FastAPI
# 可以配置两个文档用户界面，包括：

#     Swagger UI：服务于 /docs。
#         可以使用参数 docs_url 设置它的 URL。
#         可以通过设置 docs_url=None 禁用它。
#     ReDoc：服务于 /redoc。
#         可以使用参数 redoc_url 设置它的 URL。
#         可以通过设置 redoc_url=None 禁用它。


# 例如，设置 Swagger UI 服务于 /documentation 并禁用 ReDoc：
app = FastAPI(docs_url="/documentation", redoc_url=None)


@app.get("/items/")
async def read_items():
    return [{"name": "Foo"}]
```

## 静态文件
可以使用`StaticFiles`来挂载静态文件。

## 测试客户端
基于Starlette，测试FastAPI应用程序变得简单而愉快。具体地，它基于Requests，因此非常熟悉和直观。
有了它，可以直接将pytest与FastAPI一起使用。
```python
from fastapi import FastAPI
# 导入TestClient
from fastapi.testclient import TestClient

app = FastAPI()


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
```
