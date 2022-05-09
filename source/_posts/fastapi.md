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