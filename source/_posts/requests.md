---
title: Requests简明教程
tags: [Web Crawler]
categories: data analysis
date: 2023-3-26
---

（以下内容全部来自ChatGPT）
# 介绍
> requests库是一个非常酷的Python库，它可以让你轻松地向其他服务器发送HTTP请求并获取响应。使用requests库，你可以像发送GET、POST、PUT和DELETE这些HTTP方法一样简单地发送HTTP请求。如果需要，你还可以添加查询字符串参数、POST数据、HTTP头等自定义HTTP请求。响应内容可以以文本、JSON、图像等格式检索，非常方便。
> 还有一个好处是requests库提供了一个内置的异常模块来处理HTTP请求和响应的错误，你不用担心出现错误的情况。如果你需要在多个请求之间保留一些参数，如Cookie和请求头，它还提供了一个Session对象。
> 如果你需要处理Web API或其他HTTP服务，使用requests库会让你感觉非常爽!

# 安装
安装非常简单：
```
pip install requests
```

# 基本用法
## 发送GET请求
使用requests发送GET请求非常简单，只需要使用requests.get()方法即可。例如：
```python
import requests

response = requests.get('https://www.baidu.com')
print(response.text)
```
上面的代码会发送一个GET请求到百度首页，并打印出响应内容。

## 发送POST请求
使用requests发送POST请求也非常简单，只需要使用requests.post()方法即可。例如：
```python
import requests

data = {'username': 'admin', 'password': '123456'}
response = requests.post('http://example.com/login', data=data)
print(response.text)
```
上面的代码会发送一个POST请求到http://example.com/login，并将data作为请求体发送过去。

## 发送带参数的请求
有时候我们需要发送带参数的请求，可以使用params参数来指定。例如：
```python
import requests

params = {'key1': 'value1', 'key2': 'value2'}
response = requests.get('http://example.com/api', params=params)
print(response.text)
```
上面的代码会发送一个GET请求到http://example.com/api，并将params作为查询字符串发送过去。

## 发送带请求头的请求
有时候我们需要发送带请求头的请求，可以使用headers参数来指定。例如：
```python
import requests

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get('https://www.baidu.com', headers=headers)
print(response.text)
```

上面的代码会发送一个GET请求到百度首页，并设置User-Agent请求头为Mozilla/5.0。

## 发送带Cookie的请求
有时候我们需要发送带Cookie的请求，可以使用cookies参数来指定。例如：
```python
import requests

cookies = {'session_id': '123456'}
response = requests.get('http://example.com', cookies=cookies)
print(response.text)
```
上面的代码会发送一个GET请求到http://example.com，并设置Cookie为session_id=123456。

## 发送带文件的请求
有时候我们需要发送带文件的请求，可以使用files参数来指定。例如：
```python
import requests

files = {'file': open('example.txt', 'rb')}
response = requests.post('http://example.com/upload', files=files)
print(response.text)
```
上面的代码会发送一个POST请求到http://example.com/upload，并将example.txt文件作为请求体发送过去。

## 发送带认证信息的请求
有时候我们需要发送带认证信息的请求，可以使用auth参数来指定。例如：
```python
import requests

auth = ('username', 'password')
response = requests.get('http://example.com', auth=auth)
print(response.text)
```
上面的代码会发送一个GET请求到http://example.com，并使用基本认证方式进行认证。

## 发送带代理的请求
有时候我们需要发送带代理的请求，可以使用proxies参数来指定。例如：
```python
import requests

proxies = {'http': 'http://127.0.0.1:8080', 'https': 'https://127.0.0.1:8080'}
response = requests.get('http://example.com', proxies=proxies)
print(response.text)
```
上面的代码会发送一个GET请求到http://example.com，并通过代理服务器127.0.0.1:8080进行访问。

## 设置超时时间
有时候我们需要设置请求超时时间，可以使用timeout参数来指定。例如：
```python
import requests

response = requests.get('http://example.com', timeout=5)
print(response.text)
```
上面的代码会发送一个GET请求到http://example.com，并设置超时时间为5秒。

以上就是requests的主要用法，使用requests可以让我们方便地发送HTTP请求并获取响应，非常适合进行Web爬虫和API开发等工作。
