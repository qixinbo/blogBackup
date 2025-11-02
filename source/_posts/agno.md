---
title: 多智能体开发框架Agno教程
tags: [LLM]
categories: coding 
date: 2025-5-7
---

[Agno](https://docs.agno.com/introduction)是一个用于构建AI智能体（包括多模态智能体和多智能体）的开源Python框架，支持工具调用、记忆、知识检索、可观测性等特性，可用于生产环境，它一个特点是非常快，官网有它与其他框架的一个速度对比。
这里将对Agno进行下研究。

# 安装
## 创建虚拟环境
```sh
uv venv --python 3.12
source .venv/bin/activate
```
## 安装依赖
```sh
uv pip install -U agno openai anthropic mcp "fastapi[standard]" sqlalchemy
```
## 配置key
```sh
export OPENAI_API_KEY=sk-***
```

# 快速开始
创建一个`hackernews_agent.py`文件，然后粘帖：
```py
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIChat(
        id="gpt-4o",
        base_url="https://api.xxx.com/v1" # 不配置的话就默认使用OpenAI的url
        ),
    tools=[HackerNewsTools()],
    markdown=True,
)

agent.print_response("Write a report on trending startups and products.", stream=True)
```

输出结果：
```sh
┃                                                                                                         ┃
┃                                                1. Ventoy                                                ┃
┃                                                                                                         ┃
┃  • Created by: wilsonfiifi                                                                              ┃
┃  • Description: Ventoy is a tool to create bootable USB drives for ISO/WIM/IMG/VHD(x)/EFI files. It's   ┃
┃    popular for its ease of use and flexibility allowing multiple bootable files on a single USB device. ┃
┃  • Score: 159                                                                                           ┃
┃  • Discussion: Hacker News Thread                                                                       ┃
┃  • URL: github.com/ventoy/Ventoy                                                                        ┃
┃                                                                                                         ┃
┃                                          2. Affinity by Canva                                           ┃
┃                                                                                                         ┃
┃  • Created by: microflash                                                                               ┃
┃  • Description: Affinity is a product from Canva aimed at enhancing design productivity, particularly   ┃
┃    focusing on affinity-based features and UI/UX design elements.                                       ┃
┃  • Score: 8                                                                                             ┃
┃  • Discussion: Hacker News Thread                                                                       ┃
┃  • URL: affinity.studio                                                                                 ┃
┃------------------------------------------ 还有更多，略去 ------------------                             |
```

# 第一个智能体
上面的示例算是一个toy demo，下面是一个完整体的智能体。
## 创建智能体OS
创建`agno_agent.py`，然后
```py
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.tools.mcp import MCPTools

# Create the Agent
agno_agent = Agent(
    name="Agno Agent",
    model=OpenAIChat(
        id="gpt-4o",
        base_url="https://api.zhizengzeng.com/v1"),
    # Add a database to the Agent
    db=SqliteDb(db_file="agno.db"),
    # Add the Agno MCP server to the Agent
    tools=[MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")],
    # Add the previous session history to the context
    add_history_to_context=True,
    markdown=True,
)


# Create the AgentOS
agent_os = AgentOS(agents=[agno_agent])
# Get the FastAPI app for the AgentOS
app = agent_os.get_app()
```

启动这个AgentOS：
```sh
fastapi dev agno_agent.py
```
该智能体OS将会运行在`http://localhost:8000/`。

## 连接该智能体OS
Agno提供了一个连接到AgentOS的网页界面，可用于监控、管理和测试智能体系统。
打开 `os.agno.com` 并登录账户。
- 点击顶部导航栏中的“Add new OS”（添加新 OS）。
- 选择“Local”（本地），以连接运行在电脑上的本地AgentOS。
- 输入上面AgentOS的端点URL。默认是 `http://localhost:8000`。
- 给AgentOS起一个容易辨识的名称，比如“Development OS”或“Local 8000”。
- 点击“Connect”（连接）。

可以通过这个页面来与AgentOS进行对话、查看对话历史、进行评估等等。

## 使用API
上面的AgentOS提供了API，用来调用，api文档见：`http://localhost:8000/docs`。


