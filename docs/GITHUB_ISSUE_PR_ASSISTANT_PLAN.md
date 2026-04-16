# GitHub Issue / PR 助手项目方案

## 1. 项目定位

基于 `nanobot` 现有运行时，改造一个面向研发协作场景的 `GitHub Issue / PR 助手`。项目重点不是通用聊天，也不是通用 coding agent，而是围绕代码仓库协作流程提供以下能力：

- Issue 分类与优先级判断
- PR 结构化摘要与风险提示
- 基于代码、Issue、PR、文档的仓库问答
- 迭代日报 / 周报生成

项目目标是同时体现：

- 后端工程能力：消息、状态、工具、服务化、并发控制
- LLM 应用能力：RAG、LangChain、评估、微调
- 可量化结果：检索、生成、分类均可评估

推荐项目名：

- `RepoPilot`
- `PRSense`
- `IssueFlow`

优先推荐：`RepoPilot`

## 2. 为什么选这个方向

相比继续做一个通用 Agent 平台，这个方向更适合简历和面试：

- 场景边界清晰，面试官容易理解
- 能自然结合 RAG，而不是为了加 RAG 而加
- 检索评估、分类评估、摘要评估都可以做
- 可以局部引入 LangChain，而不需要把整个系统迁移到框架上
- 可以加入“小而明确”的微调模块，形成完整实验链路

## 3. 核心用户与使用场景

目标用户：

- 研发团队成员
- 开源项目维护者
- 实习/个人项目中的仓库协作参与者

典型使用场景：

- 输入一个 Issue，自动判断类型、优先级、相关模块
- 输入一个 PR，自动生成改动摘要、影响范围和潜在风险
- 询问“这个模块是做什么的”“这个 bug 可能和哪些文件有关”
- 汇总某段时间内仓库的 Issue / PR 活动，自动生成日报或周报

## 4. 功能范围

### 4.1 第一阶段必须完成

- 仓库问答
- PR 摘要
- Issue 分类
- 基础检索评估

### 4.2 第二阶段增强

- PR 风险点检测
- 历史 Issue / PR 关联推荐
- 日报 / 周报生成
- LangSmith tracing 与实验对比
- Ragas 生成评估

### 4.3 第三阶段增强

- Issue 分类微调
- PR 摘要风格化微调
- GitHub webhook / Slack 接入
- 自动评论或通知推送

## 5. 基于 nanobot 的改造策略

本项目不建议推倒重写，而是保留 `nanobot` 的通用运行时，只重构与场景相关的外层能力。

### 5.1 保留的模块

- `AgentLoop`
- `MessageBus`
- `SessionManager`
- `ToolRegistry`
- CLI / API 入口
- `ChannelManager`
- 工具执行与会话隔离机制

### 5.2 弱化或暂不使用的部分

- 与场景无关的聊天渠道
- 通用 web search 主流程
- 通用技能集合
- 非必要的 heartbeat / cron 能力

### 5.3 新增的场景工具

建议新增以下工具：

- `github_issue_fetch`
- `github_pr_fetch`
- `github_diff_fetch`
- `github_comments_fetch`
- `repo_file_search`
- `repo_file_read`
- `repo_chunk_retrieve`
- `issue_classify`
- `pr_summarize`
- `weekly_digest_generate`

如果后续接 GitHub App 或 webhook，还可以增加：

- `github_issue_comment`
- `github_pr_comment`
- `github_label_apply`

## 6. 系统架构

建议将系统拆成以下层次：

### 6.1 Runtime 层

继续使用 `nanobot` 的：

- 消息总线
- 会话管理
- 工具调用
- API / CLI
- 异步并发控制

### 6.2 Ingestion 层

负责抓取与清洗仓库数据：

- 源代码文件
- README / docs
- Issue
- PR
- Review comments
- Commit messages

### 6.3 Indexing 层

负责切分、向量化与索引存储：

- 代码 chunk
- 文档 chunk
- Issue / PR 元信息
- 检索 metadata

### 6.4 Retrieval 层

根据用户问题或任务类型，从多个索引中召回上下文：

- 代码检索
- 协作记录检索
- 文档检索
- rerank

### 6.5 Task 层

封装具体任务：

- 仓库问答
- Issue 分类
- PR 摘要
- 日报 / 周报

### 6.6 Evaluation 层

负责离线评估与实验对比：

- 检索评估
- 生成评估
- 分类评估
- 微调前后对比

## 7. RAG 设计

本项目不是单一“文档问答”RAG，而是混合检索 RAG。

### 7.1 三类索引

#### 代码索引

数据来源：

- `.py` / `.ts` / `.js` / `.java` / `.md` 等仓库文件

切分建议：

- 优先按函数 / 类 / 代码块切分
- 再补充固定 token chunk

元数据建议：

- 文件路径
- 模块名
- 函数 / 类名
- 仓库名
- commit hash（可选）

#### 协作索引

数据来源：

- Issue
- PR
- review comments
- commit message

元数据建议：

- issue / pr 编号
- 标题
- 标签
- 作者
- 时间
- 状态

#### 文档索引

数据来源：

- README
- docs
- ADR
- 设计文档

元数据建议：

- 文档路径
- 章节标题
- 文档类别

### 7.2 检索策略

建议采用混合检索：

- 向量检索：用于语义匹配
- 关键词检索：用于文件名、Issue 编号、错误关键词等精确召回
- rerank：用于最终排序

### 7.3 不同任务的检索重点

仓库问答：

- 代码索引 + 文档索引为主

Issue 分类：

- 协作索引 + 相关代码模块为主

PR 摘要：

- diff + 改动文件相关代码 + 历史 PR / issue

周报生成：

- 协作索引为主，按时间窗口聚合

## 8. LangChain 使用策略

不建议把整个项目迁移到 LangChain。更合理的做法是：

- 外层继续保留 `nanobot` 作为 runtime
- 在检索链路中局部接入 LangChain

适合使用 LangChain 的部分：

- document loader
- text splitter
- embedding pipeline
- vector store 封装
- retriever 组合
- retrieval chain

这样可以同时体现：

- 你具备系统设计能力
- 你也会使用主流 LLM 框架

## 9. 评估设计

本项目必须有评估，不然很容易变成“看起来能用”的 demo。

### 9.1 检索评估

目标：验证相关上下文是否被召回。

评估集构造：

- 人工选取一批代表性问题
- 为每个问题标注标准相关上下文
- 相关上下文可以是文件、函数、Issue、PR、文档片段

建议指标：

- Recall@k
- Hit@k
- MRR

示例问题：

- “登录超时问题可能在哪些模块？”
- “这个 PR 影响了哪些功能？”
- “与缓存失效相关的历史 issue 有哪些？”

### 9.2 生成评估

目标：验证回答和摘要是否正确、是否忠于上下文。

评估任务：

- 仓库问答
- PR 摘要
- 周报生成

建议指标：

- faithfulness
- answer correctness
- context precision
- context recall

工具建议：

- LangSmith：用于 tracing、实验记录和数据集管理
- Ragas：用于 RAG answer-level 指标计算

### 9.3 分类评估

目标：验证 Issue 分类和优先级预测效果。

任务示例：

- 类型分类：bug / feature / docs / refactor / question
- 优先级分类：low / medium / high
- 组件分类：auth / api / ui / infra 等

建议指标：

- Accuracy
- Macro F1
- Confusion Matrix

## 10. 微调设计

微调不建议作为第一阶段主线，而应该作为在 baseline 稳定后的增强项。

### 10.1 推荐优先做的微调任务

优先推荐：

- Issue 分类微调

可选增强：

- PR 摘要风格化微调

### 10.2 为什么优先做 Issue 分类微调

- 数据更容易构造
- 标签空间清晰
- 指标容易量化
- 面试时更容易讲清楚实验设计

### 10.3 建议对比实验

至少做三组：

- Prompt-only
- RAG + Prompt
- Fine-tuned model

这样能清楚回答：

- RAG 是否有效
- 微调是否真正带来增益
- 哪类问题最受益

## 11. 开发阶段计划

### 第一阶段：MVP

目标：做出能跑通的核心链路。

- 接入 GitHub 数据源
- 建立代码、Issue、PR、文档索引
- 实现仓库问答
- 实现 PR 摘要
- 实现基础 Issue 分类
- 做最基础检索评估

### 第二阶段：评估与增强

目标：让系统具备“可证明效果”的实验能力。

- 引入 LangChain 检索链
- 建立标准评估集
- 接入 LangSmith tracing
- 接入 Ragas
- 增加结果对比和误差分析

### 第三阶段：微调与产品化

目标：形成完整的亮点闭环。

- 构造 Issue 分类微调数据集
- 训练并对比模型效果
- 加入 GitHub webhook 或 Slack 通知
- 增加日报 / 周报

## 12. 预期交付物

建议最终至少交付这些内容：

- 可运行项目代码
- 一份架构文档
- 一份评估文档
- 一套小规模评估数据集
- 一份实验对比结果
- 一段项目 demo

## 13. 简历写法

项目名：

`RepoPilot：面向 GitHub Issue / PR 的智能研发协作助手`

可直接用于简历的描述：

- 基于现有 Agent Runtime 改造研发协作助手，支持仓库问答、Issue 分类、PR 摘要与周报生成。
- 构建面向代码、Issue、PR 和文档的混合检索 RAG 系统，使用 LangChain 编排检索链路，并对检索结果进行结构化评估。
- 设计检索评估、生成评估与分类评估体系，结合 LangSmith tracing 与 Ragas 指标分析系统效果。
- 针对 Issue 分类任务构建微调数据集，对比 prompt-only、RAG 和微调模型在 Accuracy / Macro F1 上的差异。

## 14. 面试表达建议

介绍时不要说“我做了一个聊天机器人”，建议这样表述：

> 我基于现有的 Agent Runtime 做了一个面向 GitHub 协作场景的智能助手。这个项目的重点不是通用对话，而是围绕代码仓库做混合检索、Issue 分类、PR 摘要和可评估的 LLM 应用。外层我保留了消息、会话、API 和工具执行骨架，在检索链路上局部引入 LangChain，并通过 LangSmith 和 Ragas 做实验与效果评估，后续再对 Issue 分类任务做微调增强。

## 15. 最小可行目标

如果时间有限，建议先做到以下版本：

- 支持一个仓库的代码 + Issue + PR 建索引
- 支持仓库问答
- 支持 PR 摘要
- 支持 Issue 分类
- 能跑一份基础评估

这个版本已经足够写进实习简历。
