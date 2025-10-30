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
uv pip install -U agno openai anthropic mcp 'fastapi[standard]' sqlalchemy
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
