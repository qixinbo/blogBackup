---
title: 多智能体开发框架Agno教程2——Workflows
tags: [LLM]
categories: coding 
date: 2025-6-11
---

了解了Agno的基本运行原理，再来深入了解一下进阶的概念。
本文针对于Workflows概念进行深入研究。

# 🧩 什么是 Workflows（工作流）？

Agno 的 **工作流（Workflows）** 让你可以通过一系列**定义好的步骤（steps）** 来编排智能体（Agents）、团队（Teams）以及函数（Functions），从而构建出 **确定性（deterministic）**、**可控（controlled）** 的智能系统。

与自由形式（free-form）的智能体交互不同，**工作流提供结构化的自动化控制**，保证每次执行的逻辑一致、结果可预测，因此非常适合需要可靠性与可重复性的生产环境。

---

## 🚀 为什么要使用 Workflows？

工作流让你能够对智能系统实现 **可预测的控制（deterministic control）**，从而构建出可靠的自动化管线，每次执行都能得到一致结果。
在以下情况中，工作流尤为关键：

### ✅ 确定性执行（Deterministic Execution）

* 每个步骤都有明确的输入与输出；
* 每次运行都能得到一致结果；
* 具有清晰的日志与审计记录，适合生产环境。

### ⚙️ 复杂编排（Complex Orchestration）

* 多智能体之间的协作与任务交接；
* 支持并行处理与条件分支；
* 支持循环结构（loops）来执行迭代任务。

---

💡 **简而言之：**

* **Workflows（工作流）**：用于“确定性、可重复的自动化”；
* **Teams（团队）**：用于“动态、协作式的问题求解”。

| 场景          | 推荐方式           |
| ----------- | -------------- |
| 需要固定流程与可控输出 | ✅ 使用 Workflows |
| 需要灵活协作与智能推理 | ✅ 使用 Teams     |

---

## ⚖️ 工作流的确定性步骤执行（Deterministic Step Execution）

在工作流中，所有操作都按照严格定义的顺序执行，每个步骤都会生成确定性输出，作为下一个步骤的输入。
这让数据流变得可追踪、可预测，也避免了自由对话中可能出现的随机性。

### 🧱 Step 类型（Step Types）

| 类型               | 说明                          |
| ---------------- | --------------------------- |
| 🧠 **Agents**    | 具备特定能力和指令的单个智能体             |
| 👥 **Teams**     | 多个智能体协同工作的团队                |
| ⚙️ **Functions** | 自定义 Python 函数，用于执行特定逻辑或处理任务 |

---

### ✅ 确定性执行的优势（Deterministic Benefits）

通过工作流机制，智能体与团队仍然保留其独特的智能与能力，但在一个**受控的框架**中运行：

* **可预测执行**：步骤按照定义顺序运行；
* **可重复结果**：相同输入总能得到相同输出；
* **数据流清晰**：上一步输出即为下一步输入；
* **状态受控**：步骤之间可保持会话状态；
* **可靠容错**：内置重试与错误恢复机制。

> 💬 工作流 ≈ “智能体自动化的流水线版本”，在保持智能的同时，强调确定性与可控性。

---

## 💬 与用户的直接交互（Direct User Interaction）

如果用户希望**直接与工作流交互**（而不是通过程序调用），你可以添加一个 `WorkflowAgent`，让工作流具备自然语言对话的能力。

这样，工作流就能：

* 像聊天机器人一样进行对话；
* 判断是否能用已有结果回答；
* 或者根据用户的新问题自动重新执行工作流。

📚 详情请参考：[Conversational Workflows（会话型工作流）](/concepts/workflows/conversational-workflows)

---

## 🧠 总结对比

| 特性   | Workflows（工作流） | Teams（团队）      |
| ---- | -------------- | -------------- |
| 执行方式 | 确定性、线性步骤       | 动态协作、自由分工      |
| 控制   | 严格定义的输入输出      | 由团队领导动态调度      |
| 场景   | 自动化生产任务        | 复杂推理与多轮协作      |
| 典型用例 | 数据处理、报表生成、任务编排 | 问答系统、知识推理、内容生成 |


# 搭建工作流
## 🧩 一、Workflows 的作用

**Workflow** 是 Agno 的“编排层”，可以让你像搭积木一样组合多个智能体（Agent）、团队（Team）或函数（Function）来形成一个完整的处理流程。

比如你可以：

* 让一个 Agent 先抓取数据；
* 再让另一个函数或 Agent 清洗数据；
* 最后让一个 Team 生成报告或发布结果。

---

## ⚙️ 二、Workflows 的核心构件

| 组件          | 作用                             | 典型使用场景                             |
| ----------- | ------------------------------ | ---------------------------------- |
| `Workflow`  | 顶层 orchestrator（编排器），控制整个流程的执行 | 定义整体执行逻辑                           |
| `Step`      | 单个工作单元（核心执行节点）                 | 每个 Step 可以是 Agent、Team 或 Python 函数 |
| `Loop`      | 循环执行一个或多个 Step                 | 重复运行直到条件满足                         |
| `Parallel`  | 并行执行多个 Step                    | 同时调用多个 Agent/Team 并合并结果            |
| `Condition` | 条件分支执行                         | 根据条件决定是否执行某步                       |
| `Router`    | 动态路由执行                         | 根据内容决定下一步走向（if/else 多分支逻辑）         |

---

## 🔁 三、Step 的输入与输出

当 Step 是函数时，Agno 提供了标准化接口：

* `StepInput`：每步的输入结构体；
* `StepOutput`：输出结果，包含 `content` 字段（可包含 Agent 的返回内容）。

这样，不论 Step 是函数还是智能体，输入输出格式都统一了，方便后续编排和复用。

---

## 🧠 四、示例：混合执行工作流

```python
from agno.workflow import Step, Workflow, StepOutput

def data_preprocessor(step_input):
    # 自定义数据预处理逻辑
    return StepOutput(content=f"Processed: {step_input.input}")

workflow = Workflow(
    name="Mixed Execution Pipeline",
    steps=[
        research_team,      # 团队成员（Team）
        data_preprocessor,  # 自定义函数
        content_agent,      # Agent
    ]
)

workflow.print_response("Analyze the competitive landscape for fintech startups", markdown=True)
```

### 执行逻辑：

1. 输入“Analyze the competitive landscape for fintech startups”；
2. `research_team`（团队）先执行研究；
3. `data_preprocessor` 处理研究结果；
4. `content_agent` 生成最终输出；
5. 最终在终端打印格式化的结果。

---

## 💡 五、设计理念总结

Agno 的工作流设计遵循：

* **清晰（clarity）**：每个 Step 只负责一件事；
* **可组合（composability）**：Step 可以是 Agent、Team 或函数；
* **可扩展（extensibility）**：你能轻松添加循环、并行或条件分支；
* **数据流标准化（StepInput / StepOutput）**：简化了复杂流程中的数据传递。

# 运行工作流
## 🧩 一、Workflow 执行的核心接口

Agno 提供三种运行方式：

| 函数                          | 描述                    | 返回类型                       |
| --------------------------- | --------------------- | -------------------------- |
| `workflow.run()`            | 同步运行工作流               | `WorkflowRunOutput` 对象     |
| `workflow.arun()`           | 异步运行工作流               | `WorkflowRunOutput` 或异步迭代器 |
| `workflow.print_response()` | 封装版打印输出（内部调用 `run()`） | 直接打印 Markdown 输出           |

---

## ⚙️ 二、Workflow 示例结构（标准流程）

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Workflow
from agno.utils.pprint import pprint_run_response

# 1️⃣ 定义智能体
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIChat(id="gpt-5-mini"),
    tools=[HackerNewsTools()],
    role="Extract key insights from Hackernews posts",
)

web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-5-mini"),
    tools=[DuckDuckGoTools()],
    role="Search the web for the latest trends",
)

# 2️⃣ 定义团队
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)

# 3️⃣ 定义内容规划 Agent
content_planner = Agent(
    name="Content Planner",
    model=OpenAIChat(id="gpt-5-mini"),
    instructions=[
        "Plan a 4-week content schedule for the given topic",
        "Ensure 3 posts per week",
    ],
)

# 4️⃣ 定义工作流
content_creation_workflow = Workflow(
    name="Content Creation Workflow",
    description="Automated content creation from research to scheduling",
    db=SqliteDb(db_file="tmp/workflow.db"),
    steps=[research_team, content_planner],
)

# 5️⃣ 执行工作流
if __name__ == "__main__":
    response = content_creation_workflow.run(
        input="AI trends in 2024",
        markdown=True,
    )
    pprint_run_response(response, markdown=True)
```

✅ **执行逻辑：**

1. `research_team` 调用 HackerNews 和 DuckDuckGo 搜索；
2. 输出结果交给 `content_planner`；
3. 生成 4 周的内容计划。

---

## 🔁 三、异步执行（Async）

Agno 支持异步执行 `arun()`，可以与 FastAPI、AsyncIO 集成：

```python
response = await workflow.arun(input="Recent breakthroughs in quantum computing")
```

---

## 💧 四、流式输出（Streaming）

流式执行可以实时获取每个事件（例如步骤开始、结束、Agent 输出）：

```python
response = workflow.run(
    input="AI trends in 2024",
    stream=True,            # 打开流模式
    stream_events=True,     # 输出所有事件类型
)
```

可迭代输出：

```python
for event in response:
    print(event.event, event.data)
```

---

## 🧠 五、事件系统（Events）

Agno 的事件机制提供了完整的生命周期追踪。以下是关键事件类型表：

| 分类       | 事件类型                                                                                                         | 描述            |
| -------- | ------------------------------------------------------------------------------------------------------------ | ------------- |
| **核心事件** | `WorkflowStarted`, `WorkflowCompleted`, `WorkflowError`                                                      | 表示工作流开始/结束/错误 |
| **步骤事件** | `StepStarted`, `StepCompleted`, `StepError`                                                                  | 每个 Step 的执行状态 |
| **条件事件** | `ConditionExecutionStarted`, `ConditionExecutionCompleted`                                                   | 条件执行的开始和结束    |
| **并行事件** | `ParallelExecutionStarted`, `ParallelExecutionCompleted`                                                     | 并行执行的开始与结束    |
| **循环事件** | `LoopExecutionStarted`, `LoopIterationStartedEvent`, `LoopIterationCompletedEvent`, `LoopExecutionCompleted` | 循环过程中的生命周期    |
| **路由事件** | `RouterExecutionStarted`, `RouterExecutionCompleted`                                                         | 路由控制开始/结束     |

这些事件都封装在 `WorkflowRunOutputEvent` 对象中。

---

## 📦 六、事件存储与分析

工作流可以将所有执行事件存储到数据库，用于：

* 调试（Debugging）
* 审计（Audit Trails）
* 性能分析（Performance）
* 错误溯源（Error tracing）

```python
from agno.run.workflow import WorkflowRunEvent

workflow = Workflow(
    name="Debug Workflow",
    store_events=True,  # 启用事件存储
    events_to_skip=[
        WorkflowRunEvent.step_started,  # 可过滤无用事件
        WorkflowRunEvent.parallel_execution_started,
    ],
    steps=[...]
)
```

存储结果可以从：

* `workflow.run_response.events` 获取；
* 或直接在数据库中查询。

---

## 🚫 七、关闭遥测（Telemetry）

Agno 默认会记录模型使用统计，可关闭：

```bash
export AGNO_TELEMETRY=false
```

或在代码中：

```python
workflow = Workflow(..., telemetry=False)
```

---

## 🌐 八、适用场景总结

| 目标              | 建议用法                              |
| --------------- | --------------------------------- |
| 简单工作流快速测试       | `workflow.print_response()`       |
| 异步应用（如 FastAPI） | `await workflow.arun()`           |
| 实时输出进度          | `stream=True, stream_events=True` |
| 生产监控 / 调试       | `store_events=True`               |
| 性能优化            | 跳过不必要事件 `events_to_skip`          |

