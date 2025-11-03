---
title: 多智能体开发框架Agno教程1——Teams
tags: [LLM]
categories: coding 
date: 2025-6-10
---

了解了Agno的基本运行原理，再来深入了解一下进阶的概念。
本文针对于Teams概念进行深入研究。

# 概述
一个 **Team（团队）** 是由多个智能体（或其他子团队）组成的集合，它们协作完成任务。

下面是一个简单示例：

```python
from agno.team import Team
from agno.agent import Agent

team = Team(members=[
    Agent(name="智能体 1", role="你用英文回答问题"),
    Agent(name="智能体 2", role="你用中文回答问题"),
    Team(name="团队 1", members=[Agent(name="智能体 3", role="你用法语回答问题")], role="你协调团队成员用法语回答问题"),
])
```

团队的领导者会根据成员的角色与任务性质，将任务分配给相应的成员。

与智能体类似，团队也支持以下特性：
* **模型（Model）：**
  可设置用于团队领导者（team leader）的模型，用来决定如何将任务分配给团队成员。
* **指令（Instructions）：**
  可以对团队领导者下达指令，指导其如何解决问题。
  团队成员的名称、描述和角色会自动提供给团队领导者。
* **工具（Tools）：**
  如果团队领导者需要直接使用工具，可以为团队添加工具。
* **推理（Reasoning）：**
  允许团队领导者在作出回应或分配任务前进行“思考”，并在收到成员结果后进行“分析”。
* **知识（Knowledge）：**
  如果团队需要检索信息，可以为团队添加知识库。知识库由团队领导者访问。
* **存储（Storage）：**
  团队的会话历史和状态会保存在数据库中，使团队可以从上次中断处继续对话，支持多轮、长期的交互。
* **记忆（Memory）：**
  赋予团队记忆能力，让其能够存储并回忆先前交互中的信息，从而学习用户偏好并个性化响应。


# 构建团队（Building Teams）

要构建一个高效的团队，应从简单开始 —— 只包含模型（model）、成员（members）和指令（instructions）。
当基本功能正常后，再根据需要逐步增加复杂性。

以下是一个最简单的带有专职智能体的团队示例：

```python
# 文件名：news_weather_team.py
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# 创建专职智能体
news_agent = Agent(
    id="news-agent",
    name="新闻智能体", 
    role="获取最新新闻并提供摘要",
    tools=[DuckDuckGoTools()]
)

weather_agent = Agent(
    id="weather-agent",
    name="天气智能体", 
    role="获取天气信息和预报",
    tools=[DuckDuckGoTools()]
)

# 创建团队
team = Team(
    name="新闻与天气团队",
    members=[news_agent, weather_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions="与团队成员协作，为用户提供全面的信息。根据用户请求分配任务。"
)

team.print_response("东京的最新新闻和天气怎么样？", stream=True)
```

> 💡 **提示（Tip）**
> 建议为每个团队成员明确指定 `id`、`name` 和 `role` 字段，以便团队领导者更好地识别成员。
> 其中，`id` 用于在团队内部以及领导者上下文中标识该成员。

> 📘 **注意（Note）**
> 当团队成员未指定模型时，会从其父团队继承模型。
> 如果成员显式指定了模型，则保留自身模型。
> 在嵌套团队中，成员从其**直接父级团队**继承模型。
> 若团队未指定模型，则默认使用 OpenAI 的 `gpt-4o`。
>
> 该继承规则适用于以下字段：`model`、`reasoning_model`、`parser_model`、`output_model`。
>
> 参见 [模型继承示例（model inheritance example）](https://docs.agno.com/examples/concepts/teams/basic/model_inheritance)。

---

## 运行团队（Run your Team）

运行团队时，可以使用 `Team.print_response()` 方法在终端中打印响应：

```python
team.print_response("东京的最新新闻和天气怎么样？")
```

此方法仅适用于**开发阶段**，不推荐在生产环境中使用。
在生产环境中，请使用 `Team.run()` 或异步版本 `Team.arun()`。例如：

```python
from typing import Iterator
from agno.team import Team
from agno.agent import Agent
from agno.run.team import TeamRunOutputEvent
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

news_agent = Agent(name="新闻智能体", role="获取最新新闻")
weather_agent = Agent(name="天气智能体", role="获取未来7天的天气")

team = Team(
    name="新闻与天气团队", 
    members=[news_agent, weather_agent],
    model=OpenAIChat(id="gpt-4o")
)

# 运行团队并返回响应变量
response = team.run("东京的天气怎么样？")
# 打印响应内容
print(response.content)

################ 流式响应（STREAM RESPONSE） #################
stream: Iterator[TeamRunOutputEvent] = team.run("东京的天气怎么样？", stream=True)
for chunk in stream:
    if chunk.event == "TeamRunContent":
        print(chunk.content)

################ 流式响应 + 美化打印（STREAM AND PRETTY PRINT） #################
stream: Iterator[TeamRunOutputEvent] = team.run("东京的天气怎么样？", stream=True)
pprint_run_response(stream, markdown=True)
```

---

### 修改终端显示内容

使用 `print_response` 方法时，默认只打印团队中涉及工具调用的部分（通常是任务分配信息）。
如果希望同时打印各个成员（智能体）的响应内容，可以设置参数 `show_members_responses=True`：

```python
team.print_response("东京的天气怎么样？", show_members_responses=True)
```

# 运行团队（Running Teams）
可以通过调用 `Team.run()` 或 `Team.arun()` 来运行团队。其工作流程如下：

1. 团队领导者构建要发送给模型的上下文（包括系统消息、用户消息、对话历史、用户记忆、会话状态及其他相关输入）。
2. 团队领导者将该上下文发送给模型。
3. 模型处理输入，并决定是使用 `delegate_task_to_members` 工具将任务委派给团队成员、调用其他工具，还是直接生成响应。
4. 如果发生了任务委派，团队成员会执行各自的任务，并将结果返回给团队领导者。
5. 团队领导者处理更新后的上下文，并生成最终响应。
6. 团队将该最终响应返回给调用方。

---

## 基本执行（Basic Execution）

`Team.run()` 函数运行团队并返回输出结果 —— 可以是一个 `TeamRunOutput` 对象，
也可以在启用 `stream=True` 时，返回一个由 `TeamRunOutputEvent`（以及成员智能体的 `RunOutputEvent`）组成的流。

示例如下：

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

news_agent = Agent(
    name="新闻智能体",
    model=OpenAIChat(id="gpt-4o"),
    role="获取最新新闻",
    tools=[DuckDuckGoTools()]
)
weather_agent = Agent(
    name="天气智能体",
    model=OpenAIChat(id="gpt-4o"),
    role="获取未来7天的天气",
    tools=[DuckDuckGoTools()]
)

team = Team(
    name="新闻与天气团队",
    members=[news_agent, weather_agent],
    model=OpenAIChat(id="gpt-4o")
)

# 运行团队并返回响应
response = team.run(input="东京的天气怎么样？")
# 以 Markdown 格式打印响应
pprint_run_response(response, markdown=True)
```

> 💡 **提示：**
> 你也可以使用 `Team.arun()` 异步运行团队。
> 当团队领导者在一次请求中将任务委派给多个成员时，成员会并发执行任务。

> 💡 **提示：**
> 想了解更多关于结构化输入输出（structured input/output）的信息，请参阅 [输入与输出（Input & Output）](/concepts/teams/input-output) 文档。

---

## 运行输出（Run Output）

当未启用流式（stream）模式时，`Team.run()` 函数会返回一个 `TeamRunOutput` 对象。
该对象的核心属性包括：

* `run_id`：本次运行的唯一 ID。
* `team_id`：团队 ID。
* `team_name`：团队名称。
* `session_id`：会话 ID。
* `user_id`：用户 ID。
* `content`：最终响应内容。
* `content_type`：内容类型（若为结构化输出，则为对应 Pydantic 模型的类名）。
* `reasoning_content`：推理内容。
* `messages`：发送给模型的消息列表。
* `metrics`：本次运行的指标。详情见 [团队指标（Metrics）](/concepts/teams/metrics)。
* `model`：本次运行所使用的模型。
* `member_responses`：团队成员的响应列表（若 `store_member_responses=True` 时可用）。

> 📘 **注意：**
> 未指定模型的团队成员会继承其父团队的模型。
> 这适用于：`model`、`reasoning_model`、`parser_model`、`output_model`。
>
> 参见 [模型继承示例（model inheritance example）](/examples/concepts/teams/basic/model_inheritance)。

详情请参阅 [TeamRunOutput 文档](/reference/teams/team-response)。

---

## 流式运行（Streaming）

设置 `stream=True` 可启用流式模式。此时，`run()` 将返回一个 `TeamRunOutputEvent` 对象的迭代器，而非单一响应。

```python
from typing import Iterator
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat

news_agent = Agent(name="新闻智能体", role="获取最新新闻")
weather_agent = Agent(name="天气智能体", role="获取未来7天的天气")

team = Team(
    name="新闻与天气团队",
    members=[news_agent, weather_agent],
    model=OpenAIChat(id="gpt-4o")
)

# 以流式方式运行团队
stream: Iterator[TeamRunOutputEvent] = team.run("东京的天气怎么样？", stream=True)
for chunk in stream:
    if chunk.event == "TeamRunContent":
        print(chunk.content)
```

> 💡 **提示：**
> 当使用 `arun()` 异步运行团队时，如果团队领导者将任务分派给多个成员，这些成员会**并发执行**。
> 这意味着事件会并行产生，事件顺序**不一定有序**。

---

### 流式所有事件（Streaming All Events）

默认情况下，流式输出仅包含 `RunContent` 类型事件。
若要流式传输团队内部所有事件，可设置 `stream_events=True`：

```python
# 启用全部事件流式输出
response_stream = team.run(
    "东京的天气怎么样？",
    stream=True,
    stream_events=True
)
```

这将实时输出团队的内部进程，如工具调用（tool call）或推理步骤（reasoning）。

---

### 处理事件（Handling Events）

你可以通过迭代响应流，逐个处理到达的事件：

```python
response_stream = team.run("你的提示词", stream=True, stream_events=True)

for event in response_stream:
    if event.event == "TeamRunContent":
        print(f"内容: {event.content}")
    elif event.event == "TeamToolCallStarted":
        print(f"开始调用工具: {event.tool}")
    elif event.event == "ToolCallStarted":
        print(f"成员开始调用工具: {event.tool}")
    elif event.event == "ToolCallCompleted":
        print(f"成员完成调用工具: {event.tool}")
    elif event.event == "TeamReasoningStep":
        print(f"推理步骤: {event.content}")
    ...
```

> 📘 **注意：**
> 团队成员事件会在团队执行期间产生。
> 若不希望接收这些事件，可设置 `stream_member_events=False`。

---

### 存储事件（Storing Events）

你可以在 `RunOutput` 对象中保存运行期间产生的所有事件。

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

team = Team(
    name="故事团队",
    members=[],
    model=OpenAIChat(id="gpt-4o"),
    store_events=True
)

response = team.run("讲一个5秒钟的关于狮子的短故事", stream=True, stream_events=True)
pprint_run_response(response)

for event in response.events:
    print(event.event)
```

默认情况下，`TeamRunContentEvent` 和 `RunContentEvent` 不会被存储。
你可以通过设置 `events_to_skip` 参数修改跳过的事件类型。例如：

```python
team = Team(
    name="故事团队",
    members=[],
    model=OpenAIChat(id="gpt-4o"),
    store_events=True,
    events_to_skip=["TeamRunStarted"]
)
```

---

### 事件类型（Event Types）

以下是 `Team.run()` 与 `Team.arun()` 根据配置可能产生的事件类型：

#### 核心事件（Core Events）

| 事件类型                         | 描述                              |
| ---------------------------- | ------------------------------- |
| `TeamRunStarted`             | 表示运行开始                          |
| `TeamRunContent`             | 包含模型响应的文本块                      |
| `TeamRunContentCompleted`    | 表示内容流式传输结束                      |
| `TeamRunIntermediateContent` | 包含模型的中间响应（当启用 `output_model` 时） |
| `TeamRunCompleted`           | 表示运行成功完成                        |
| `TeamRunError`               | 表示运行过程中发生错误                     |
| `TeamRunCancelled`           | 表示运行被取消                         |

#### 工具事件（Tool Events）

| 事件类型                    | 描述             |
| ----------------------- | -------------- |
| `TeamToolCallStarted`   | 团队工具调用开始       |
| `TeamToolCallCompleted` | 团队工具调用完成（包含结果） |

#### 推理事件（Reasoning Events）

| 事件类型                     | 描述     |
| ------------------------ | ------ |
| `TeamReasoningStarted`   | 推理开始   |
| `TeamReasoningStep`      | 单个推理步骤 |
| `TeamReasoningCompleted` | 推理完成   |

#### 记忆事件（Memory Events）

| 事件类型                        | 描述       |
| --------------------------- | -------- |
| `TeamMemoryUpdateStarted`   | 团队记忆更新开始 |
| `TeamMemoryUpdateCompleted` | 团队记忆更新完成 |

#### 会话摘要事件（Session Summary Events）

| 事件类型                          | 描述       |
| ----------------------------- | -------- |
| `TeamSessionSummaryStarted`   | 会话摘要生成开始 |
| `TeamSessionSummaryCompleted` | 会话摘要生成完成 |

#### 前置钩子事件（Pre-Hook Events）

| 事件类型                   | 描述       |
| ---------------------- | -------- |
| `TeamPreHookStarted`   | 前置钩子开始执行 |
| `TeamPreHookCompleted` | 前置钩子执行完成 |

#### 后置钩子事件（Post-Hook Events）

| 事件类型                    | 描述       |
| ----------------------- | -------- |
| `TeamPostHookStarted`   | 后置钩子开始执行 |
| `TeamPostHookCompleted` | 后置钩子执行完成 |

#### 解析模型事件（Parser Model Events）

| 事件类型                               | 描述       |
| ---------------------------------- | -------- |
| `TeamParserModelResponseStarted`   | 解析模型响应开始 |
| `TeamParserModelResponseCompleted` | 解析模型响应完成 |

#### 输出模型事件（Output Model Events）

| 事件类型                               | 描述       |
| ---------------------------------- | -------- |
| `TeamOutputModelResponseStarted`   | 输出模型响应开始 |
| `TeamOutputModelResponseCompleted` | 输出模型响应完成 |

详情请参阅 [TeamRunOutput 文档](/reference/teams/team-response)。

---

### 自定义事件（Custom Events）

如果你编写了自定义工具（custom tools），你可以定义并发送自定义事件。
这些事件会与 Agno 内置事件一同被处理。

可以通过继承内置的 `CustomEvent` 类来自定义事件类型，例如：

```python
from dataclasses import dataclass
from agno.run.team import CustomEvent

@dataclass
class CustomerProfileEvent(CustomEvent):
    """客户资料的自定义事件"""

    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
```

然后可以在自定义工具中产出该事件：

```python
from agno.tools import tool

@tool()
async def get_customer_profile():
    """示例工具，生成自定义事件"""

    yield CustomerProfileEvent(
        customer_name="John Doe",
        customer_email="john.doe@example.com",
        customer_phone="1234567890",
    )
```

详情请参阅 [完整示例](/examples/concepts/teams/events/custom_events)。

---

## 指定运行用户与会话（Specify Run User and Session）

你可以通过 `user_id` 和 `session_id` 参数指定运行所属的用户和会话：

```python
team.run("生成我的月度报告", user_id="john@example.com", session_id="session_123")
```

详情请参阅 [团队会话（Team Sessions）](/concepts/teams/sessions) 文档。

---

## 传入图片 / 音频 / 视频 / 文件（Passing Images / Audio / Video / Files）

你可以通过 `images`、`audio`、`video` 或 `files` 参数向团队传入多模态内容。例如：

```python
team.run("根据这张图片讲一个5秒钟的短故事", images=[Image(url="https://example.com/image.jpg")])
```

详情请参阅 [多模态（Multimodal）](/concepts/multimodal) 文档。

---

## 取消运行（Cancelling a Run）

可以通过调用 `Team.cancel_run()` 方法取消正在执行的运行。
详情请参阅 [取消运行（Cancelling a Run）](/concepts/teams/run-cancel) 文档。


