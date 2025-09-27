---
title: Claude Code搭配Spec-Kit实现“规范驱动开发”
tags: [LLM]
categories: coding 
date: 2025-9-27
---
当前“规范驱动开发”（`Spec-Driven Development`）火热程度超过了“氛围编程”（`Vibe Coding`），本文尝试使用Claude Code搭配Spec-Kit来体验下。

# 安装必要软件
## 安装Claude Code
一条命令执行即可：
```sh
npm install -g @anthropic-ai/claude-code
```

## 设置模型
国内使用Claude系列模型的门槛较高，这里使用`Qwen3-Coder`来替代。可以参考[这里](https://help.aliyun.com/zh/model-studio/claude-code?spm=a2c4g.11186623.help-menu-2400256.d_0_9_2.3eab5835srkglG)。
因为`Qwen3-Coder`提供了对Anthropic格式的支持，所以直接设置环境变量即可。
对于其他模型，可以使用`claude-code-router`这个库来实现，具体可参考[这里](https://gameapp.club/post/2025-07-20-claude-code-with-litellm/)。

```
# 用您的百炼API KEY代替YOUR_DASHSCOPE_API_KEY
echo 'export ANTHROPIC_BASE_URL="https://dashscope.aliyuncs.com/api/v2/apps/claude-code-proxy"' >> ~/.zshrc
echo 'export ANTHROPIC_AUTH_TOKEN="YOUR_DASHSCOPE_API_KEY"' >> ~/.zshrc
```
设置完了别忘了`source`一下。

## 安装Specify
```sh
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
```

至此，需要安装的软件都已经安装完成。

# 规范驱动开发流程介绍
Spec-Kit所定义的规范驱动开发包含5个顺序步骤：
1. 建立项目原则
使用`/constitution`命令创建项目的治理原则和开发指导方针，比如：
```sh
/constitution Create principles focused on code quality, testing standards, user experience consistency, and performance requirements
```
这一步创建`memory/constitution.md`文件，包含项目的基础指导原则。

2. 定义功能
使用 /specify 命令描述要构建的内容，专注于`what`和`why`，而不是技术栈，比如：
```sh
/specify Build an application that can help me organize my photos in separate photo albums...
```
这个命令会自动创建新的功能分支并生成结构化的规格文档`spec.md`。

这一步完成后，最好马上查看下生成的该规格文档，检查其是否满足需求，这在“规范驱动开发”中非常重要，不要“默认正确”。
可以使用`/clarify`来进一步澄清模糊的需求。

3. 创建技术实现规划
使用`/plan`命令提供技术栈和架构选择，比如：
```sh
/plan The application uses Vite with minimal number of libraries. Use vanilla HTML, CSS, and JavaScript...
```
这会生成多个文件，包括`plan.md`、`research.md`、`data-model.md`、`contracts/` 和 `quickstart.md`。

4. 分解任务
使用`/tasks`命令从实现计划创建可执行的任务列表：
```sh
/tasks
```
这会分析计划和相关设计文档，生成`tasks.md`文件，包含有序的任务列表和依赖关系。

5. 实现
使用`/implement`命令执行所有任务并根据计划构建功能：
```sh
/implement
```

# 开始使用
如果是空白项目，则直接运行命令：
```sh
specify init <project-name>
```
如果是已有项目，则可以运行：
```sh
specify init --here
```
运行时会让选择是使用哪个编程智能体，这里选`Claude`，也可以在启动时直接指定：
```
specify init <project-name> --ai claude
```

启动Claude Code：
```sh
Claude
```
然后依次按5个步骤走即可。
