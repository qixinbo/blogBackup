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

---

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

---

## 多轮会话（Multi-turn Sessions）
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



