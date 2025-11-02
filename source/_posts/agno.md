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

# 智能体
一个 Agent（智能体）是一个由大语言模型（LLM）作为“大脑”的自治程序，它不仅能够对话，还能决策、调用工具、访问知识、记忆状态，从而执行更复杂的任务。
换句话说，它不同于传统只是“接收问题——返回答案”的聊天机器人，而是能够在运行时决定：我需要先思考／调用工具／查知识／记忆下来／然后回应。
一个 Agno Agent 一般包含以下几个关键组成部分：


| 组成部分                              | 作用                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **模型（Model）**                     | 驱动智能体“思考”的语言模型，例如 GPT‑4、Claude 等。Agent 的决策逻辑、是否调用工具、如何表达答案，都是由模型判断。                             |
| **指令/提示（Instructions / Prompts）** | 设定 Agent 的行为规范、风格、工具使用规则、输出格式。告诉模型“你是这个样子”“你要这么做”。                                            |
| **工具（Tools）**                     | Agent 可调用的外部能力，比如网络搜索、金融数据接口、数据库查询、上传下载文件等。通过工具，Agent 能“出屋子”去获取或操作外部信息。                    |
| **记忆（Memory）**                    | 用于让 Agent 在会话中或跨会话保存上下文、用户偏好、历史操作，以便更个性化或长期追踪。              |
| **知识库／检索（Knowledge / Retrieval）** | Agent 可以访问专门的知识库（例如向量数据库、PDF 文件、文档集合等）来做检索增强（RAG: Retrieval Augmented Generation）。|
| **存储／状态（Storage / Persistence）**  | 因为模型 API 本身通常是无状态的，Agent 需要持久化机制来保留会话数据、历史、工具调用记录、知识检索缓存等。                            |
| **执行与控制（Execution & Control）**    | 管理 Agent 的生命周期、工具调用时机、校验机制、守卫（guardrails）、日志监控等，以便真实环境中稳定运行。                          |


## 构建智能体
在构建高效的智能体时，建议 从简单开始 —— 只包含模型（model）、工具（tools）和指令（instructions）。
当这些基础功能运作良好后，再逐步叠加更多高级特性。
🧩 最简单的报告生成 Agent 示例：
```py
# hackernews_agent.py
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[HackerNewsTools()],
    instructions="撰写一份关于该主题的报告，只输出报告内容。",
    markdown=True,
)

agent.print_response("当前热门的初创公司和产品。", stream=True)
```
🚀 运行Agent
在开发阶段，可以使用 Agent.print_response() 方法在终端中直接输出结果。
⚠️ 注意：该方法仅适用于开发调试，在生产环境中请使用 Agent.run() 或 Agent.arun()。
```py
from typing import Iterator
from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[HackerNewsTools()],
    instructions="撰写一份关于该主题的报告，只输出报告内容。",
    markdown=True,
)

# -------- 普通运行 --------
response: RunOutput = agent.run("当前热门的初创公司和产品。")
print(response.content)

# -------- 流式输出 --------
stream: Iterator[RunOutputEvent] = agent.run("热门产品趋势", stream=True)
for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(chunk.content)

# -------- 流式输出 + 美化打印 --------
stream: Iterator[RunOutputEvent] = agent.run("热门产品趋势", stream=True)
pprint_run_response(stream, markdown=True)
```

## 运行智能体
通过调用 `Agent.run()` 或 `Agent.arun()` 来运行智能体。运行流程如下：
1. 智能体构建要发送给模型的上下文（包括系统消息、用户消息、聊天历史、用户记忆、会话状态及其他相关输入）。
2. 智能体将该上下文发送给模型。
3. 模型处理输入，返回 **一个消息** 或 **一个工具调用（tool call）**。
4. 如果模型做了工具调用，智能体会执行该工具，并将结果返回给模型。
5. 模型处理更新后的上下文，重复步骤 3–4，直到它生成一个 **无需再调用工具** 的最终消息。
6. 智能体将此最终响应返回给调用方。


### 基本执行

`Agent.run()` 方法可运行智能体，并返回一个 `RunOutput` 对象（非流式）或当 `stream=True` 时返回 `RunOutputEvent` 对象的迭代器。示例：

```python
from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[HackerNewsTools()],
    instructions="撰写一份关于该主题的报告，只输出报告内容。",
    markdown=True,
)

# 运行智能体，并将响应存为变量
response: RunOutput = agent.run("热门初创公司和产品趋势。")
# 以 markdown 格式打印响应
pprint_run_response(response, markdown=True)
```

> 也可以使用异步方式 `Agent.arun()` 来运行。参见[异步示例](https://docs.agno.com/examples/concepts/agent/async/basic)。

### 运行输入

`input` 参数为发送给智能体的输入。它可以是字符串、列表、字典、消息对象、pydantic 模型或消息列表。例如：

```python
response: RunOutput = agent.run(input="热门初创公司和产品趋势。")
```

> 若要了解如何使用结构化输入／输出，请参见 [“输入 & 输出” 文档](https://docs.agno.com/concepts/agents/input-output)。


### 运行输出
`Agent.run()`（非流式）返回一个 `RunOutput` 对象，包含以下核心属性：

* `run_id`: 本次运行的 ID。
* `agent_id`: 智能体的 ID。
* `agent_name`: 智能体名称。
* `session_id`: 会话 ID。
* `user_id`: 用户 ID。
* `content`: 响应的内容。
* `content_type`: 内容类型；若输出为结构化模型，则为该模型的类名。
* `reasoning_content`: 推理内容。
* `messages`: 发送给模型的消息列表。
* `metrics`: 本次运行的指标。

更多细节请参见 [`RunOutput` 的文档](https://docs.agno.com/reference/agents/run-response)。

---

### 流式（Streaming）

若设定 `stream=True`，`run()` 将返回一个 `RunOutputEvent` 对象的迭代器，用于逐步接收响应。例如：

```python
from typing import Iterator
from agno.agent import Agent, RunOutputEvent, RunEvent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[HackerNewsTools()],
    instructions="撰写一份关于该主题的报告，只输出报告内容。",
    markdown=True,
)

stream: Iterator[RunOutputEvent] = agent.run("热门产品趋势", stream=True)
for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(chunk.content)
```

> 若要异步流式运行，参见[异步示例](https://docs.agno.com/examples/concepts/agent/async/streaming)。

---

### 流式所有事件

默认情况下，流模式仅返回 `RunContent` 事件。
也可以通过设定 `stream_events=True` 来流式接收 **所有事件**，包括工具调用、推理步骤等。例如：

```python
response_stream: Iterator[RunOutputEvent] = agent.run(
    "热门产品趋势",
    stream=True,
    stream_events=True
)
```

---

### 处理事件

可以在收到事件时逐一处理，例如：

```python
stream = agent.run("热门产品趋势", stream=True, stream_events=True)

for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(f"内容: {chunk.content}")
    elif chunk.event == RunEvent.tool_call_started:
        print(f"工具调用启动: {chunk.tool.tool_name}")
    elif chunk.event == RunEvent.reasoning_step:
        print(f"推理步骤: {chunk.content}")
```

---

### 事件类型

下面是 `Agent.run()` 和 `Agent.arun()` 在不同配置下可能产生的事件类别：

#### 核心事件

| 事件类型                  | 描述          |
| --------------------- | ----------- |
| `RunStarted`          | 表示运行开始      |
| `RunContent`          | 模型响应文本按块返回  |
| `RunContentCompleted` | 内容流式输出完成    |
| `RunCompleted`        | 表示运行成功结束    |
| `RunError`            | 表示运行过程中发生错误 |
| `RunCancelled`        | 表示运行被取消     |

#### 控制流程事件

| 事件类型           | 描述        |
| -------------- | --------- |
| `RunPaused`    | 运行被暂停     |
| `RunContinued` | 暂停后的运行被继续 |

#### 工具相关事件

| 事件类型                | 描述            |
| ------------------- | ------------- |
| `ToolCallStarted`   | 表示工具调用开始      |
| `ToolCallCompleted` | 表示工具调用结束并返回结果 |

#### 推理相关事件

| 事件类型                 | 描述         |
| -------------------- | ---------- |
| `ReasoningStarted`   | 推理过程开始     |
| `ReasoningStep`      | 推理过程中的一个步骤 |
| `ReasoningCompleted` | 推理过程完成     |

#### 记忆相关事件

| 事件类型                    | 描述        |
| ----------------------- | --------- |
| `MemoryUpdateStarted`   | 智能体开始更新记忆 |
| `MemoryUpdateCompleted` | 智能体完成记忆更新 |

#### 会话摘要相关事件

| 事件类型                      | 描述       |
| ------------------------- | -------- |
| `SessionSummaryStarted`   | 会话摘要生成开始 |
| `SessionSummaryCompleted` | 会话摘要生成完成 |

#### 前置钩子（Pre-Hook）事件

| 事件类型               | 描述       |
| ------------------ | -------- |
| `PreHookStarted`   | 前置运行钩子开始 |
| `PreHookCompleted` | 前置钩子执行完成 |

#### 后置钩子（Post-Hook）事件

| 事件类型                | 描述       |
| ------------------- | -------- |
| `PostHookStarted`   | 后置运行钩子开始 |
| `PostHookCompleted` | 后置钩子执行完成 |

#### 解析器模型事件（Parser Model）

| 事件类型                           | 描述        |
| ------------------------------ | --------- |
| `ParserModelResponseStarted`   | 解析器模型响应开始 |
| `ParserModelResponseCompleted` | 解析器模型响应完成 |

#### 输出模型事件（Output Model）

| 事件类型                           | 描述       |
| ------------------------------ | -------- |
| `OutputModelResponseStarted`   | 输出模型响应开始 |
| `OutputModelResponseCompleted` | 输出模型响应完成 |

---

### 自定义事件

如果使用自定义工具，也可以定义自定义事件。方法如下：

```python
from dataclasses import dataclass
from agno.run.agent import CustomEvent

@dataclass
class CustomerProfileEvent(CustomEvent):
    """客户档案的自定义事件。"""
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
```

然后工具中可 `yield` 这个事件。该事件会被 Agno 内部作为普通事件处理。例如：

```python
from agno.tools import tool

@tool()
async def get_customer_profile():
    """示例：仅 yield 一个自定义事件的自定义工具。"""

    yield CustomerProfileEvent(
        customer_name="John Doe",
        customer_email="john.doe@example.com",
        customer_phone="1234567890",
    )
```

更多细节请见[完整示例文档](https://docs.agno.com/examples/concepts/agent/events/custom_events)。

---

### 指定运行用户与会话

可以通过传入 `user_id` 与 `session_id` 参数，指定当前运行关联的用户和会话。例如：

```python
agent.run(
    "讲一个关于机器人 5 秒钟的短故事",
    user_id="john@example.com",
    session_id="session_123"
)
```

更多信息请参见 [“Agent 会话” 文档](https://docs.agno.com/concepts/agents/sessions)。

---

### 传递图像／音频／视频／文件

可以通过 `images`、`audio`、`video` 或 `files` 参数向智能体传递图像、音频、视频或文件。例如：

```python
agent.run(
    "基于这张图片讲一个 5 秒钟短故事",
    images=[Image(url="https://example.com/image.jpg")]
)
```

更多详情请参见 [“多模态 Agent” 文档](https://docs.agno.com/concepts/multimodal)。

---

### 暂停与继续运行

如果运行过程中触发了 “人类在环（Human-in-the-Loop）” 的流程，智能体运行可能被暂停。这时可以调用 `Agent.continue_run()` 方法继续执行。

更多细节请参见 [“Human-in-the-Loop” 文档](https://docs.agno.com/concepts/hitl)。

---

### 取消运行

可以调用 `Agent.cancel_run()` 方法来取消当前运行。

更多详情请参见 [“取消运行” 文档](https://docs.agno.com/concepts/agents/run-cancel)。

## 调试智能体
Agno 提供了一个非常完善的 **调试模式（Debug Mode）**，它能显著提升你的开发体验，帮助你理解代理（Agent）的执行流程和中间步骤。例如：

1. 检查发送给模型的消息及其返回的响应。
2. 跟踪中间步骤并监控指标（如 token 使用量、执行时间等）。
3. 检查工具调用、错误及其结果。

---

### 启用调试模式

有三种方式可以启用调试模式：

1. 在创建Agent时设置 `debug_mode=True`，对所有运行生效。
2. 在调用 `run()` 方法时设置 `debug_mode=True`，仅对当前运行生效。
3. 设置环境变量 `AGNO_DEBUG=True`，启用全局调试模式。

示例：

```python
from agno.agent import Agent
from agno.models.openai import OpenAI
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAI(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    instructions="撰写关于该主题的报告，仅输出报告内容。",
    markdown=True,
    debug_mode=True,
    # debug_level=2,  # 取消注释可获得更详细的日志
)

# 运行代理并在终端中打印结果
agent.print_response("热门初创公司和产品趋势。")
```

💡 可以设置 `debug_level=2` 来输出更详细的调试日志。

---

### 交互式 CLI

Agno 还提供了一个内置的 **交互式命令行界面（CLI）**，可以直接在终端中与代理进行对话式测试，非常适合调试多轮交互。

示例：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAI
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAI(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,  # 将对话历史添加到上下文
    num_history_runs=3,           # 仅保留最近3轮对话
    markdown=True,
)

# 以交互式 CLI 方式运行代理
agent.cli_app(stream=True)
```

## 智能体会话
当我们调用 `Agent.run()` 时，它会创建一个**无状态（stateless）**的单次运行（run）。
但如果我们希望继续对话、实现多轮交互（multi-turn conversation），就需要用到 **“会话（Session）”**。
一个会话是由多次连续运行组成的集合。

### 基本概念

* **Session（会话）**：表示与 Agent 的一次多轮对话，包含多个连续的 `run`。每个会话由 `session_id` 标识，内部保存所有运行记录、指标和状态信息。
* **Run（运行）**：每次与 Agent 的交互（即一次用户输入与模型响应）称为一次运行，由 `run_id` 标识。
* **Messages（消息）**：表示模型与 Agent 之间传递的单条消息，是双方的通信单位。

更多细节请参考 [Session Storage（会话存储）](https://docs.agno.com/concepts/agents/storage)。

### 单轮会话示例
在下例中，Agno 自动为我们生成 `run_id` 和 `session_id`：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

response = agent.run("讲一个关于机器人的5秒短故事")
print(response.content)
print(response.run_id)
print(response.session_id)
```

### 多轮会话（Multi-turn Sessions）
每个用户都可以拥有自己的会话集，多个用户可同时与同一个 Agent 交互。

可以使用 `user_id` 和 `session_id` 来区分不同用户与会话：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="tmp/data.db")

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
)

user_1_id = "user_101"
user_2_id = "user_102"

user_1_session_id = "session_101"
user_2_session_id = "session_102"

agent.print_response("讲一个关于机器人的短故事", user_id=user_1_id, session_id=user_1_session_id)
agent.print_response("再讲一个笑话", user_id=user_1_id, session_id=user_1_session_id)

agent.print_response("告诉我关于量子物理的事情", user_id=user_2_id, session_id=user_2_session_id)
agent.print_response("光速是多少？", user_id=user_2_id, session_id=user_2_session_id)

agent.print_response("总结一下我们的对话", user_id=user_1_id, session_id=user_1_session_id)
```

---

### 在上下文中加入历史记录
可以让 Agent 自动将对话历史加入上下文，这样模型就能记住前面的信息：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.in_memory import InMemoryDb

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), db=InMemoryDb())

agent.print_response("嗨，我叫小明。很高兴认识你！")
agent.print_response("我叫什么名字？", add_history_to_context=True)
```

---

### 会话摘要（Session Summaries）
当会话内容过长时，Agent 可以生成简短摘要来概括整个对话。
设置 `enable_session_summaries=True` 即可启用：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="tmp/data.db")
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    db=db,
    enable_session_summaries=True,
)

session_id = "1001"
agent.print_response("什么是量子计算？", user_id="user_1", session_id=session_id)
agent.print_response("那大语言模型（LLM）呢？", user_id="user_1", session_id=session_id)

summary = agent.get_session_summary(session_id=session_id)
print(f"会话摘要: {summary.summary}")
```

可以通过 `SessionSummaryManager` 自定义摘要的生成方式：

```python
from agno.agent import Agent
from agno.session import SessionSummaryManager
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="agno.db")

summary_manager = SessionSummaryManager(
    model=OpenAIChat(id="gpt-4o-mini"),
    session_summary_prompt="请为以下对话创建一个简短的总结：",
)

agent = Agent(
    db=db,
    session_summary_manager=summary_manager,
    enable_session_summaries=True,
)
```

---

### 访问会话历史
启用存储（Storage）后，可以随时访问某个会话的历史记录：

```python
agent.get_messages_for_session(session_id)
agent.get_chat_history(session_id)
```

也可以自动让 Agent 在上下文中加载最近几次对话：

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    description="你是一位友好积极的智能助手。",
)
```

---

### 搜索历史会话

可以设置 Agent 搜索过去的多次会话：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
import os

os.remove("tmp/data.db")

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    user_id="user_1",
    db=SqliteDb(db_file="tmp/data.db"),
    search_session_history=True,
    num_history_sessions=2,
)
```

这将允许模型搜索最近两次会话内容。

---

### 控制会话中存储的内容

为了节省数据库空间，可设置以下参数：

* `store_media`: 是否存储图片、音频、视频、文件等
* `store_tool_messages`: 是否存储工具调用请求与结果
* `store_history_messages`: 是否存储历史消息

示例：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    db=SqliteDb(db_file="tmp/agents.db"),
    add_history_to_context=True,
    num_history_runs=5,
    store_media=False,
    store_tool_messages=False,
    store_history_messages=False,
)

agent.print_response("搜索最新的AI新闻并总结")
```


## 输入与输出
Agno 的 Agent 支持多种输入与输出形式，从最基础的字符串交互，到基于 **Pydantic 模型** 的结构化数据验证。

### 🧩 基础模式：字符串输入输出

最常见的用法是以 `str` 输入、`str` 输出：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    description="You write movie scripts.",
)

response = agent.run("Write movie script about a girl living in New York")
print(response.content)
```

> 💡 高级模式请参考：
>
> * [图片 / 音频 / 视频 / 文件作为输入](https://docs.agno.com/examples/concepts/multimodal)
> * [列表作为输入](https://docs.agno.com/examples/concepts/agent/input_and_output/input_as_list)

---

### 🏗️ 结构化输出（Structured Output）
Agno 的一个强大特性是：可以让 Agent 生成 **结构化数据（Pydantic 模型）**。
这让 Agent 能够输出固定格式的数据，适合：

* 特征提取
* 数据分类
* 模拟数据生成
* 需要**确定输出结构**的生产系统。

例如，我们创建一个 “电影脚本” Agent：

```python
from typing import List
from rich.pretty import pprint
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat

class MovieScript(BaseModel):
    setting: str = Field(..., description="电影背景设置")
    ending: str = Field(..., description="电影结尾，没有就写快乐结局")
    genre: str = Field(..., description="电影类型")
    name: str = Field(..., description="电影名")
    characters: List[str] = Field(..., description="角色名称")
    storyline: str = Field(..., description="三句话总结剧情")

structured_output_agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    description="You write movie scripts.",
    output_schema=MovieScript,
)

structured_output_agent.print_response("New York")
```

输出结果是一个 `MovieScript` 对象：

```python
MovieScript(
    setting='在繁华的纽约街头与天际线中…',
    ending='主角在帝国大厦顶端拥吻...',
    genre='Action Thriller',
    name='The NYC Chronicles',
    characters=['Isabella Grant', 'Alex Chen', ...],
    storyline='一名记者揭露巨大阴谋...'
)
```

---

#### 🧠 JSON 模式（use_json_mode）

部分模型无法直接生成结构化输出。
此时可以让 Agno 指示模型以 JSON 形式返回：

```python
agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    description="You write movie scripts.",
    output_schema=MovieScript,
    use_json_mode=True,
)
```

> ⚠️ JSON 模式比结构化模式精度稍差，但在部分模型上更稳定。

---

### ⚡ 流式结构化输出（Streaming Structured Output）

结构化输出也可以流式返回，Agno 会在事件流中生成一个结构化结果：

```python
structured_output_agent.print_response(
    "New York", stream=True, stream_events=True
)
```

---

### 📥 结构化输入（Structured Input）
Agent 的输入也可以是结构化数据（Pydantic 模型或 `TypedDict`）。

```python
from typing import List
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools
from pydantic import BaseModel, Field

class ResearchTopic(BaseModel):
    topic: str
    focus_areas: List[str]
    target_audience: str
    sources_required: int = 5

hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIChat(id="gpt-5-mini"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)

hackernews_agent.print_response(
    input=ResearchTopic(
        topic="AI",
        focus_areas=["AI", "Machine Learning"],
        target_audience="Developers",
        sources_required=5,
    )
)
```

---

#### ✅ 输入验证（input_schema）

可通过 `input_schema` 参数验证传入的输入字典是否合法：

```python
hackernews_agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    input_schema=ResearchTopic,
)

hackernews_agent.print_response(
    input={
        "topic": "AI",
        "focus_areas": ["AI", "Machine Learning"],
        "target_audience": "Developers",
        "sources_required": "5",
    }
)
```

Agno 会自动将输入校验并转化为 Pydantic 模型对象。

---

### 🔒 类型安全 Agent（Typesafe Agents）

同时设置 `input_schema` 和 `output_schema`，可以构建**端到端类型安全 Agent**。

示例：

```python
from typing import List
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools
from pydantic import BaseModel, Field
from rich.pretty import pprint

class ResearchTopic(BaseModel):
    topic: str
    sources_required: int = 5

class ResearchOutput(BaseModel):
    summary: str
    insights: List[str]
    top_stories: List[str]
    technologies: List[str]
    sources: List[str]

hn_researcher_agent = Agent(
    model=Claude(id="claude-sonnet-4-0"),
    tools=[HackerNewsTools()],
    input_schema=ResearchTopic,
    output_schema=ResearchOutput,
    instructions="Research hackernews posts for a given topic",
)

response = hn_researcher_agent.run(
    input=ResearchTopic(topic="AI", sources_required=5)
)

pprint(response.content)
```

输出：

```python
ResearchOutput(
    summary='AI development is accelerating...',
    insights=['LLMs 更高效', '开源模型崛起'],
    top_stories=['GPT-5 消息', 'Claude 新版本发布'],
    technologies=['GPT-4', 'Claude', 'Transformers'],
    sources=['https://news.ycombinator.com/item?id=123', ...]
)
```

---

### 🧩 使用解析模型（Parser Model）

可以使用一个单独的模型来解析主模型的输出：

```python
agent = Agent(
    model=Claude(id="claude-sonnet-4-20250514"),
    description="You write movie scripts.",
    output_schema=MovieScript,
    parser_model=OpenAIChat(id="gpt-5-mini"),
)
```

> 💡 优点：
>
> * 主模型负责推理；
> * 小模型负责结构化解析；
> * 提高可靠性、降低成本。

还可以通过 `parser_model_prompt` 自定义解析模型的提示词。

---

### 🔄 使用输出模型（Output Model）

当主模型擅长多模态任务（如图像分析）时，可用另一模型专门生成结构化输出：

```python
agent = Agent(
    model=Claude(id="claude-sonnet-4-20250514"),
    description="You write movie scripts.",
    output_schema=MovieScript,
    output_model=OpenAIChat(id="gpt-5-mini"),
)
```

> ✨ 一些 Gemini 模型无法同时使用工具与结构化输出，此法是一个有效解决方案。


## 上下文工程

**上下文工程**是指设计和控制发送给语言模型的信息（上下文）的过程，以此来引导模型的行为和输出。
在实践中，构建上下文可以归结为一个问题：“**哪些信息最有可能实现期望的结果？**”
在 **Agno** 中，这意味着要仔细构建系统消息（system message），其中包含Agent的描述、指令以及其他相关设定。通过精心设计这些上下文，你可以：

* 引导Agent表现出特定行为或角色；
* 限制或扩展Agent的能力；
* 确保输出结果一致、相关，并符合应用需求；
* 启用更高级的用例，例如多步推理、工具使用或结构化输出。

有效的上下文工程是一个**迭代过程**：反复优化系统消息，尝试不同的描述和指令，并利用诸如 **schemas、delegation、tool integrations** 等特性。

Agno智能体的上下文由以下部分组成：

* **System message（系统消息）**：发送给智能体的主要上下文信息，包括所有附加内容。
* **User message（用户消息）**：发送给智能体的用户输入。
* **Chat history（聊天记录）**：智能体与用户的对话历史。
* **Additional input（附加输入）**：添加到上下文中的 few-shot 示例或其他额外内容。

---

### 系统消息上下文（System message context）

以下是用于创建系统消息的一些关键参数：

1. **Description（描述）**：指导代理总体行为的描述。
2. **Instructions（指令）**：一组具体、任务导向的操作指令，用于实现目标。
3. **Expected Output（期望输出）**：描述代理预期生成的输出形式。

系统消息由代理的 `description`、`instructions` 和其他设置构建而成。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    description="You are a famous short story writer asked to write for a magazine",
    instructions=["Always write 2 sentence stories."],
    markdown=True,
    debug_mode=True,  # 设置为 True 以查看详细日志及系统消息内容
)
agent.print_response("Tell me a horror story.", stream=True)
```

该代码将生成以下系统消息：

```
You are a famous short story writer asked to write for a magazine                                                                          
<instructions>                                                                                                                             
- Always write 2 sentence stories.                                                                                                         
</instructions>                                                                                                                            
                                                                                                                                            
<additional_information>                                                                                                                   
- Use markdown to format your answer
</additional_information>
```

---

#### 系统消息参数说明

`Agent` 会创建一个默认的系统消息，可通过以下参数进行自定义：

| 参数名                                | 类型          | 默认值     | 说明                                                                                 |
| ---------------------------------- | ----------- | ------- | ---------------------------------------------------------------------------------- |
| `description`                      | `str`       | `None`  | 添加到系统消息开头的代理描述。                                                                    |
| `instructions`                     | `List[str]` | `None`  | 添加到系统提示中 `<instructions>` 标签内的指令列表。默认指令会根据 `markdown`、`expected_output` 等自动生成。     |
| `additional_context`               | `str`       | `None`  | 添加到系统消息结尾的附加上下文。                                                                   |
| `expected_output`                  | `str`       | `None`  | 期望输出描述，添加到系统消息末尾。                                                                  |
| `markdown`                         | `bool`      | `False` | 若为 True，则添加“使用 markdown 格式化输出”的指令。                                                 |
| `add_datetime_to_context`          | `bool`      | `False` | 若为 True，则在提示中添加当前日期时间，让代理具备时间感知能力。                                                 |
| `add_name_to_context`              | `bool`      | `False` | 若为 True，则将代理名称添加到上下文。                                                              |
| `add_location_to_context`          | `bool`      | `False` | 若为 True，则添加代理的地理位置，用于生成与地点相关的回复。                                                   |
| `add_session_summary_to_context`   | `bool`      | `False` | 若为 True，则将会话摘要加入上下文。详见 [sessions](/concepts/agents/sessions)。                      |
| `add_memories_to_context`          | `bool`      | `False` | 若为 True，则添加用户记忆。详见 [memory](/concepts/agents/memory)。                              |
| `add_session_state_to_context`     | `bool`      | `False` | 若为 True，则添加会话状态。详见 [state](/concepts/agents/state)。                                |
| `enable_agentic_knowledge_filters` | `bool`      | `False` | 若为 True，则允许代理选择知识过滤器。详见 [knowledge filters](/concepts/knowledge/filters/overview)。 |
| `system_message`                   | `str`       | `None`  | 直接覆盖默认系统消息。                                                                        |
| `build_context`                    | `bool`      | `True`  | 若为 False，可禁用自动构建上下文。                                                               |

详见 [Agent 参考文档](https://docs.agno.com/reference/agents/agent)。

#### 系统消息的构建方式

来看以下示例代理：

```python
from agno.agent import Agent

agent = Agent(
    name="Helpful Assistant",
    role="Assistant",
    description="You are a helpful assistant",
    instructions=["Help the user with their question"],
    additional_context="""
    Here is an example of how to answer the user's question: 
        Request: What is the capital of France?
        Response: The capital of France is Paris.
    """,
    expected_output="You should format your response with `Response: <response>`",
    markdown=True,
    add_datetime_to_context=True,
    add_location_to_context=True,
    add_name_to_context=True,
    add_session_summary_to_context=True,
    add_memories_to_context=True,
    add_session_state_to_context=True,
)
```

生成的系统消息如下：

```
You are a helpful assistant
<your_role>
Assistant
</your_role>

<instructions>
  Help the user with their question
</instructions>

<additional_information>
Use markdown to format your answers.
The current time is 2025-09-30 12:00:00.
Your approximate location is: New York, NY, USA.
Your name is: Helpful Assistant.
</additional_information>

<expected_output>
  You should format your response with `Response: <response>`
</expected_output>

Here is an example of how to answer the user's question: 
    Request: What is the capital of France?
    Response: The capital of France is Paris.

You have access to memories from previous interactions with the user that you can use:

<memories_from_previous_interactions>
- User really likes Digimon and Japan.
- User really likes Japan.
- User likes coffee.
</memories_from_previous_interactions>

Note: this information is from previous interactions and may be updated in this conversation. You should always prefer information from this conversation over the past memories.

Here is a brief summary of your previous interactions:

<summary_of_previous_interactions>
The user asked about information about Digimon and Japan.
</summary_of_previous_interactions>

Note: this information is from previous interactions and may be outdated. You should ALWAYS prefer information from this conversation over the past summary.

<session_state> ... </session_state>
```

> 💡 **提示**：
> 这个示例展示了系统消息的完整结构，以说明它的可定制性。但在实际应用中，你通常只会启用其中的一部分配置。

---

##### 附加上下文（Additional Context）

你可以通过 `additional_context` 参数在系统消息的末尾添加额外说明。

例如，下面的 `additional_context` 参数为代理添加了一条说明，告诉它可以访问特定数据库表。

```python
from textwrap import dedent
from agno.agent import Agent
from agno.models.langdb import LangDB
from agno.tools.duckdb import DuckDbTools

duckdb_tools = DuckDbTools(
    create_tables=False, export_tables=False, summarize_tables=False
)
duckdb_tools.create_table_from_path(
    path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
    table="movies",
)

agent = Agent(
    model=LangDB(id="llama3-1-70b-instruct-v1.0"),
    tools=[duckdb_tools],
    markdown=True,
    additional_context=dedent("""\
    You have access to the following tables:
    - movies: contains information about movies from IMDB.
    """),
)
agent.print_response("What is the average rating of movies?", stream=True)
```

---

##### 工具指令（Tool Instructions）

当智能体使用某个 [Toolkit](https://docs.agno.com/concepts/tools/toolkits/toolkits) 时，可以通过 `instructions` 参数将工具说明加入系统消息：

```python
from agno.agent import Agent
from agno.tools.slack import SlackTools

slack_tools = SlackTools(
    instructions=["Use `send_message` to send a message to the user.  If the user specifies a thread, use `send_message_thread` to send a message to the thread."],
    add_instructions=True,
)
agent = Agent(
    tools=[slack_tools],
)
```

这些指令会被注入到系统消息的 `<additional_information>` 标签之后。

---

##### 智能体记忆

当智能体设置了 `enable_agentic_memory=True` 时，它将具备创建或更新用户记忆的能力。
此时系统消息中会新增如下内容：

```
<updating_user_memories>
- You have access to the `update_user_memory` tool that you can use to add new memories, update existing memories, delete memories, or clear all memories.
- If the user's message includes information that should be captured as a memory, use the `update_user_memory` tool to update your memory database.
- Memories should include details that could personalize ongoing interactions with the user.
- Use this tool to add new memories or update existing memories that you identify in the conversation.
- Use this tool if the user asks to update their memory, delete a memory, or clear all memories.
- If you use the `update_user_memory` tool, remember to pass on the response to the user.
</updating_user_memories>
```

---

##### 知识过滤器

若启用了知识功能并设置了 `enable_agentic_knowledge_filters=True`，则它能自动选择合适的知识过滤器。
系统消息会新增以下说明：

```
The knowledge base contains documents with these metadata filters: [filter1, filter2, filter3].
Always use filters when the user query indicates specific metadata.

Examples:
1. If the user asks about a specific person like "Jordan Mitchell", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like user_id>': '<valid value based on the user query>'}}.
2. If the user asks about a specific document type like "contracts", you MUST use the search_knowledge_base tool with the filters parameter set to {{'document_type': 'contract'}}.
4. If the user asks about a specific location like "documents from New York", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like location>': 'New York'}}.

General Guidelines:
- Always analyze the user query to identify relevant metadata.
- Use the most specific filter(s) possible to narrow down results.
- If multiple filters are relevant, combine them in the filters parameter (e.g., {{'name': 'Jordan Mitchell', 'document_type': 'contract'}}).
- Ensure the filter keys match the valid metadata filters: [filter1, filter2, filter3].

You can use the search_knowledge_base tool to search the knowledge base and get the most relevant documents. Make sure to pass the filters as [Dict[str: Any]] to the tool. FOLLOW THIS STRUCTURE STRICTLY.
```

详细内容可参见 [知识过滤器](https://docs.agno.com/concepts/knowledge/filters/overview)。

---

#### 直接设置系统消息

可以通过 `system_message` 参数手动定义系统消息。
此时，所有其他设置将被忽略，仅使用你提供的内容。

```python
from agno.agent import Agent
agent.print_response("What is the capital of France?")

agent = Agent(system_message="Share a 2 sentence story about")
agent.print_response("Love in the year 12000.")
```

> 💡 **提示：**
> 某些模型（例如 Groq 平台上的 `llama-3.2-11b-vision-preview`）要求不包含系统消息。
> 若要移除系统消息，请设置 `build_context=False` 且 `system_message=None`。
> 注意：若设置了 `markdown=True`，仍会自动添加系统消息，因此需关闭或显式禁用。

---

### 用户消息上下文

传递给 `Agent.run()` 或 `Agent.print_response()` 的 `input` 即为用户消息。

---

#### 附加用户上下文

可以使用以下参数为用户消息添加额外上下文：

* `add_knowledge_to_context`
* `add_dependencies_to_context`

```python
from agno.agent import Agent
agent = Agent(add_knowledge_to_context=True, add_dependencies_to_context=True)
agent.print_response("What is the capital of France?", dependencies={"name": "John Doe"})
```

发送给模型的用户消息如下：

```
What is the capital of France?

Use the following references from the knowledge base if it helps:
<references>
- Reference 1
- Reference 2
</references>

<additional context>
{"name": "John Doe"}
</additional context>
```

详见 [依赖注入](https://docs.agno.com/concepts/agents/dependencies)。

---

### 聊天记录

当智能体启用数据库存储后，会自动保存会话历史（参见 [sessions](/concepts/agents/sessions)）。
可以通过 `add_history_to_context=True` 将对话历史添加到上下文中：

```python
from agno.agent.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIChat

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db = PostgresDb(db_url=db_url)

agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    db=db,
    session_id="chat_history",
    instructions="You are a helpful assistant that can answer questions about space and oceans.",
    add_history_to_context=True,
    num_history_runs=2,  # 可选：限制添加到上下文中的历史轮数
)

agent.print_response("Where is the sea of tranquility?")
agent.print_response("What was my first question?")
```

这会将之前的对话添加到上下文中，使智能体能利用先前的信息生成更连贯的回答。
详见 [sessions#session-history](/concepts/agents/sessions#session-history)。

---

### 工具调用管理

参数 `max_tool_calls_from_history` 用于限制上下文中保留的最近 `n` 次工具调用，
以控制上下文大小并降低 token 成本。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
import random

def get_weather_for_city(city: str) -> str:
    conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(conditions)
    return f"{city}: {temperature}°C, {condition}"

agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    tools=[get_weather_for_city],
    db=SqliteDb(db_file="tmp/agent.db"),
    add_history_to_context=True,
    max_tool_calls_from_history=3,  # 仅保留最近 3 次工具调用
)
agent.print_response("What's the weather in Tokyo?")
agent.print_response("What's the weather in Paris?")  
agent.print_response("What's the weather in London?")
agent.print_response("What's the weather in Berlin?")
agent.print_response("What's the weather in Mumbai?")
agent.print_response("What's the weather in Miami?")
agent.print_response("What's the weather in New York?")
agent.print_response("What's the weather in above cities?")
```

此时模型仅会看到最近 3 个城市（Mumbai、Miami、New York）的工具调用结果。

> 🔎 **说明：**
> `max_tool_calls_from_history` 仅过滤由 `num_history_runs` 加载的历史记录。
> 数据库中仍会保留完整历史。

---

### 少样本学习（Few-shot learning）与附加输入

通过 `additional_input` 参数可以在上下文中添加额外的消息（如 few-shot 示例），
这些消息会像对话历史一样参与上下文构建。

```python
from agno.agent import Agent
from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat

# Few-shot 示例
support_examples = [
    Message(role="user", content="I forgot my password and can't log in"),
    Message(role="assistant", content="""I'll help you reset your password right away...
"""),
    ...
]

agent = Agent(
    name="Customer Support Specialist",
    model=OpenAIChat(id="gpt-5-mini"),
    add_name_to_context=True,
    additional_input=support_examples,
    instructions=[
        "You are an expert customer support specialist.",
        "Always be empathetic, professional, and solution-oriented.",
        "Provide clear, actionable steps to resolve customer issues.",
        "Follow the established patterns for consistent, high-quality support.",
    ],
    markdown=True,
)
```

这让智能体能够根据少量示例学习回应风格与格式。

---

### 上下文缓存
多数模型提供商支持系统与用户消息的缓存机制，但实现方式各不相同。
通用思路是缓存**重复或静态内容**，在后续请求中重用，以减少 token 消耗。

Agno 的上下文构建逻辑天然会将最可能缓存的静态内容放在系统消息的开头。
如需进一步优化，可手动设置 `system_message`。

示例：

* [OpenAI 的提示缓存](https://platform.openai.com/docs/guides/prompt-caching)
* [Anthropic 的提示缓存](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) — [Agno 示例](/examples/models/anthropic/prompt_caching)
* [OpenRouter 的提示缓存](https://openrouter.ai/docs/features/prompt-caching)


## 依赖注入
**依赖项（Dependencies）** 是一种向智能体上下文（Agent Context）注入变量的方式。
`dependencies` 是一个字典，包含一组函数（或静态变量），这些依赖项会在智能体运行前被解析。

<Note>  
你可以使用依赖项来注入记忆、动态 few-shot 示例、检索得到的文档等。  
</Note>

---

### 基本用法

可以在智能体的 `instructions`（指令）或用户消息中引用依赖项。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    dependencies={"name": "John Doe"},
    instructions="You are a story writer. The current user is {name}."
)

agent.print_response("Write a 5 second short story about {name}")
```

<Tip>  
你既可以在 `Agent` 初始化时设置 `dependencies`，  
也可以在运行时通过 `run()` 或 `arun()` 方法传入。  
</Tip>

---

### 使用函数作为依赖项

你可以将一个可调用函数指定为依赖项。
当代理运行时，该依赖项会被自动解析并执行。

```python
import json
from textwrap import dedent
import httpx
from agno.agent import Agent
from agno.models.openai import OpenAIChat


def get_top_hackernews_stories() -> str:
    """获取并返回 HackerNews 上的热门新闻。

    Args:
        num_stories: 要获取的热门新闻数量（默认：5）
    Returns:
        JSON 字符串，包含新闻的标题、链接、评分等信息。
    """
    # 获取热门新闻
    stories = [
        {
            k: v
            for k, v in httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{id}.json"
            )
            .json()
            .items()
            if k != "kids"  # 排除评论部分
        }
        for id in httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json"
        ).json()[:num_stories]
    ]
    return json.dumps(stories, indent=4)


agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    # 每个依赖项函数会在代理运行时自动求值
    # 可以将其理解为 Agent 的“依赖注入”
    dependencies={"top_hackernews_stories": get_top_hackernews_stories},
    # 也可以手动将依赖项添加到指令中
    instructions=dedent("""\
        You are an insightful tech trend observer! 📰

        Here are the top stories on HackerNews:
        {top_hackernews_stories}\
    """),
    markdown=True,
)

# 示例使用
agent.print_response(
    "Summarize the top stories on HackerNews and identify any interesting trends.",
    stream=True,
)
```

<Check>  
依赖项会在代理运行时自动解析。  
</Check>

---

### 将依赖项添加到上下文

设置 `add_dependencies_to_context=True`，
可以将整个依赖项字典添加到用户消息中。
这样你就不必手动把依赖项插入到指令中了。

```python
import json
from textwrap import dedent
import httpx
from agno.agent import Agent
from agno.models.openai import OpenAIChat


def get_user_profile() -> str:
    """获取并返回指定用户 ID 的用户档案。

    Args:
        user_id: 要检索的用户 ID
    """

    # 从数据库中获取用户信息（此处为示例）
    user_profile = {
      "name": "John Doe",
      "experience_level": "senior",
    }

    return json.dumps(user_profile, indent=4)

agent = Agent(
    model=OpenAIChat(id="gpt-5-mini"),
    dependencies={"user_profile": get_user_profile},
    # 将整个依赖项字典添加到用户消息中
    add_dependencies_to_context=True,
    markdown=True,
)

agent.print_response(
    "Get the user profile for the user with ID 123 and tell me about their experience level.",
    stream=True,
)
# 也可以在调用 print_response 时传入依赖项
# agent.print_response(
#     "Get the user profile for the user with ID 123 and tell me about their experience level.",
#     dependencies={"user_profile": get_user_profile},
#     stream=True,
# )
```

<Note>  
这会将整个依赖项字典插入到用户消息中，位于 `<additional context>` 标签之间。  
新的用户消息看起来如下：

```
Get the user profile for the user with ID 123 and tell me about their experience level.                                                       
                                                                                                                                                 
<additional context>                                                                                                                     
{                                                                                                                                        
"user_profile": "{\n    \"name\": \"John Doe\",\n    \"experience_level\": \"senior\"\n}"                                              
}                                                                                                                                        
</additional context> 
```

</Note>

<Tip>  
你可以在以下方法中传入 `dependencies` 和 `add_dependencies_to_context` 参数：  
`run()`、`arun()`、`print_response()`、`aprint_response()`。  
</Tip>

Agno 的 “Dependencies” 机制本质上是一种 **轻量级依赖注入（Dependency Injection）**，
它允许在代理运行前动态加载变量、函数结果或外部数据（如 API 响应、数据库记录、用户信息等）。
这样可以让 LLM 代理在执行时拥有实时、个性化的上下文，而无需手动拼接 prompt。

