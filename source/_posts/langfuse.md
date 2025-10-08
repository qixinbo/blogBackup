---
title: 开源大模型工程平台Langfuse使用体验
tags: [LLM]
categories: coding 
date: 2025-10-3
---

[Langfuse](https://langfuse.com/)是一个开源的LLM工程平台/可观测 (observability) 工具，旨在帮助开发、监控、评估、调试基于大语言模型 (LLM) 的应用。本文将从0经验开始试用下Langfuse，看看它有哪些功能及其效果。

# 安装部署
可以直接注册个[Langfuse Cloud](https://cloud.langfuse.com/auth/sign-in)账号来体验，也可以自部署，这里采用docker进行自部署：
```
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
```

顺利启动后，可以通过`http://localhost:3000`来访问。

遇到问题：
1、docker拉取镜像比较慢
解决方案：注意设置代理，或者设置国内源。
2、Postgresql端口被占用
解决方案：修改docker compose的yaml文件的端口号映射，或者停掉占用的进程。

# 环境准备
按照引导创建`Organization`和`Project`。
然后进入该`Project`，在`Settings`中创建`Secret Key`、`Public Key`和`Host`。

# 最简单示例
最简单的一个示例就是使用langfuse来监控openai模型的调用情况。
## 安装依赖
```sh
pip install langfuse openai
```
这一步是安装Langfuse的集成openai的python SDK。

## 调用示例
```python
from langfuse import observe  
from langfuse.openai import openai  
import os
  
# 配置自定义 base URL  
openai.base_url = "https://your-custom-openai-endpoint.com/v1"  
openai.api_key = "your-api-key"  

# Langfuse 环境变量  
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."  
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."  
os.environ["LANGFUSE_HOST"] = "http://localhost:3000" # 如果是使用官方云平台，则"https://cloud.langfuse.com" 
 
@observe()  
def story():  
    return openai.chat.completions.create(  
        model="gpt-4o",  
        messages=[{"role": "user", "content": "What is Langfuse?"}],  
    ).choices[0].message.content  
  
@observe()  
def main():  
    return story()  
  
main()
```
注意：上面代码把openai的apiKey以及Langfuse的两个key都写在了代码中，正规方法应该是写在本地环境变量的配置文件中，不要外泄。

## 监控看板
登录[http://localhost:3000/](http://localhost:3000/)就可以监控到该次调用的详情，包括调用的函数、输入、输出、费用等。

# 统计不同用户调用次数
以下示例代码实现了统计不同用户对于大模型的调用次数的监控：
```python
from langfuse import observe, get_client
from langfuse.openai import openai  # 注意：用 langfuse 提供的 openai wrapper

langfuse = get_client()

@observe()
def ask_llm(user_id: str, question: str):

    langfuse.update_current_trace(user_id=user_id)
    # 调用模型
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )

    answer = resp.choices[0].message.content
    return answer


if __name__ == "__main__":
    print(ask_llm("user_001", "你好，Langfuse 是什么？"))
    print(ask_llm("user_002", "如何统计调用次数？"))
    print(ask_llm("user_001", "再给我一个例子"))
```
这样就可以在langfuse的`Users`面板中查看不同用户的调用情况。

# 使用API获得监控结果
上面代码实现了对不同用户的监控，下面实现通过API获得Langfuse的监控结果：
```python
import requests
import json

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")

# 构建查询参数
query = {
    "view": "traces",  # 或 "observations"
    "dimensions": [
        {"field": "userId"}  # 按 userId 分组
    ],
    "metrics": [
        {"measure": "count", "aggregation": "count"}  # 统计调用次数
    ],
    "filters": [],
    "fromTimestamp": "2025-01-01T00:00:00Z",
    "toTimestamp": "2025-12-31T23:59:59Z"
}

# 发送请求
response = requests.get(
    f"{LANGFUSE_HOST}/api/public/metrics",
    params={"query": json.dumps(query)},
    auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
)

# 解析结果
data = response.json()
print(data)
```

