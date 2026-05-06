# LegalBot (Legal legalbot) 项目详细文档

**项目**：LegalBot（法智）- 基于 legalbot 框架的法律 AI 助手
**框架版本**：legalbot
**文档版本**：1.0
**日期**：2026-04-19

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [配置系统](#4-配置系统)
5. [Agent 系统](#5-agent-系统)
6. [RAG 系统](#6-rag-系统)
7. [法律工具集](#7-法律工具集)
8. [文档生成系统](#8-文档生成系统)
9. [审计日志系统](#9-审计日志系统)
10. [技能系统](#10-技能系统)
11. [多渠道支持](#11-多渠道支持)
12. [CLI 命令参考](#12-cli-命令参考)
13. [测试框架](#13-测试框架)
14. [API 参考](#14-api-参考)
15. [部署指南](#15-部署指南)

---

## 1. 项目概述

### 1.1 什么是 LegalBot

LegalBot（法智）是基于 legalbot 框架构建的法律 AI 助手，专为中文法律场景设计。它结合了多 Agent 编排、检索增强生成（RAG）、法律文档生成和法律辩论等多种能力，为用户提供全面的法律辅助服务。

### 1.2 核心能力

| 能力         | 说明                                  |
| ---------- | ----------------------------------- |
| **法律检索**   | 基于混合 RAG（向量 + BM25 + RRF）的法律条文和案例检索 |
| **合同审查**   | 识别合同条款中的法律风险并引用相关法条                 |
| **法律辩论**   | 模拟原告/被告/法官三方辩论，分析争议焦点               |
| **案例对比**   | 对比多个案例的异同，生成结构化对比表                  |
| **法律文书生成** | 自动生成起诉状、答辩状、代理词、上诉状、执行申请书           |
| **引用幻觉检测** | 验证法律引用的准确性                          |
| **审计日志**   | 完整的法律查询审计跟踪，支持 PII 脱敏               |

### 1.3 技术栈

- **语言**：Python 3.12+
- **异步框架**：asyncio
- **配置管理**：Pydantic v2
- **向量数据库**：Chroma（支持持久化）
- **LLM 提供商**：支持 16+ 提供商（OpenAI、DeepSeek、Claude、Gemini 等）
- **CLI 框架**：Typer
- **测试框架**：pytest + pytest-asyncio

### 1.4 版本信息

```python
# legalbot/__init__.py
__version__ = "0.5.0"
__frame_version__ = "≥ 0.5.0"
```

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Gateway / API                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     MessageBus（消息总线）                    │
│              legalbot/bus/queue.py                            │
│         异步消息路由，支持多渠道并发处理                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 AgentLoop（Agent 主循环）                     │
│              legalbot/agent/loop.py                          │
│  接收消息 → 构建上下文 → 调用 LLM → 执行工具 → 返回响应        │
└────────┬──────────────┬──────────────┬──────────────┬───────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Context    │ │ToolRegistry │ │LegalOrchestr│ │  RAG 系统   │
│  Builder    │ │  工具注册   │ │  编排器     │ │  Retriever  │
│  上下文构建  │ │             │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Skills    │ │  各类工具   │ │辩论/案例对比│ │ Vector+BM25│
│   技能系统  │ │             │ │ 文档生成    │ │   + RRF    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │Chroma向量库  │
                                      └─────────────┘
```

### 2.2 消息流转

```
用户消息（任意渠道）
    │
    ▼
MessageBus.dispatch(message)
    │
    ▼
AgentLoop._process_message()
    │
    ├─→ ContextBuilder.build()  ──→ Memory + Skills
    │
    ├─→ AgentRunner.run()  ──→ LLM Provider.chat()
    │                              │
    │                              ▼
    │                         Tool Calls
    │                              │
    │                              ▼
    │                    ToolRegistry.execute()
    │                         /    │    \
    │                        ▼     ▼     ▼
    │              RAGSearch  Debate  DocGenerate
    │
    ├─→ LegalOrchestrator（意图分类）
    │         │
    │         ▼
    │    路由到专业 Agent
    │
    ├─→ LegalDocumentGenerator（文书生成）
    │
    └─→ LegalAuditLogger.log()（审计日志）
              │
              ▼
         JSONL 日志文件

响应
    │
    ▼
ChannelManager ──→ 对应渠道（Feishu/DingTalk/QQ/WebSocket）
```

### 2.3 关键设计原则

1. **事件驱动**：所有组件通过 MessageBus 异步通信
2. **工具即服务**：所有能力通过统一 ToolRegistry 暴露
3. **配置驱动**：所有行为通过 Pydantic 配置控制
4. **关注点分离**：RAG、Agent、Document、Audit 各自独立模块
5. **审计优先**：所有法律操作均记录审计日志

---

## 3. 目录结构

```
D:/workspace/legalbot/
├── legalbot/                          # 主 Python 包
│   ├── __init__.py                   # 版本信息，导出 legalbot Facade
│   ├── __main__.py                   # CLI 入口点
│   ├── legalbot.py                    # 高层编程接口
│   │
│   ├── agent/                        # Agent 系统核心
│   │   ├── loop.py                   # AgentLoop：主处理引擎
│   │   ├── runner.py                 # AgentRunner：LLM 调用和工具执行
│   │   ├── orchestrator.py           # LegalOrchestrator：意图分类和多 Agent 路由
│   │   ├── subagent.py               # SubagentManager：后台子 Agent 管理
│   │   ├── context.py                # ContextBuilder：上下文构建
│   │   ├── memory.py                 # Consolidator：记忆整合
│   │   ├── hook.py                   # AgentHook：生命周期钩子
│   │   ├── reasoner.py               # MultiStepLegalReasoner：多步推理
│   │   ├── skills.py                 # SkillsLoader：技能加载
│   │   └── tools/                    # 工具实现
│   │       ├── base.py               # Tool 基类
│   │       ├── registry.py           # ToolRegistry：工具注册中心
│   │       ├── filesystem.py         # 文件系统工具
│   │       ├── search.py             # 搜索工具
│   │       ├── shell.py              # ExecTool（沙箱支持）
│   │       ├── web.py                # WebSearch/WebFetch
│   │       ├── message.py            # 消息发送工具
│   │       ├── spawn.py              # SpawnTool：生成子 Agent
│   │       ├── cron.py               # CronTool：定时任务
│   │       ├── mcp.py                # MCP 服务器连接
│   │       ├── rag.py                # RAGSearchTool
│   │       ├── reasoner.py          # 多步推理工具
│   │       ├── debate.py             # 辩论工具
│   │       ├── case_compare.py       # 案例对比工具
│   │       ├── document.py           # 法律文书生成工具
│   │       ├── orchestrate.py        # OrchestrateTool：编排工具
│   │       ├── feedback.py           # 反馈工具
│   │       └── schema.py             # 工具参数 schema
│   │
│   ├── audit/                        # 审计日志系统
│   │   ├── __init__.py
│   │   └── logger.py                 # LegalAuditLogger
│   │
│   ├── bus/                          # 消息总线
│   │   ├── __init__.py
│   │   ├── message.py                # 消息模型
│   │   └── queue.py                  # MessageBus 实现
│   │
│   ├── channels/                      # 多渠道支持
│   │   ├── base.py                   # BaseChannel 抽象类
│   │   ├── manager.py                # ChannelManager
│   │   ├── registry.py               # 渠道插件发现
│   │   ├── dingtalk.py               # 钉钉实现
│   │   ├── feishu.py                 # 飞书实现
│   │   ├── qq.py                     # QQ 实现
│   │   └── websocket.py              # WebSocket 实现
│   │
│   ├── cli/                          # CLI 命令
│   │   ├── commands.py               # Typer 命令定义
│   │   ├── stream.py                 # 流式输出渲染
│   │   └── models.py                 # CLI 数据模型
│   │
│   ├── config/                        # 配置系统
│   │   ├── schema.py                 # Pydantic 配置模型
│   │   ├── loader.py                 # 配置加载/保存
│   │   └── paths.py                  # 路径工具
│   │
│   ├── document/                      # 法律文书生成
│   │   ├── generator.py              # LegalDocumentGenerator
│   │   ├── variables.py              # CaseFactsExtractor
│   │   ├── config.py                 # DocumentDraftConfig
│   │   └── templates/                # 文书模板
│   │       ├── base.py               # LegalDocumentTemplate 基类
│   │       ├── complaint.py          # 起诉状
│   │       ├── defense.py            # 答辩状
│   │       ├── appeal.py             # 上诉状
│   │       ├── enforcement.py        # 执行申请书
│   │       └── agent_opinion.py      # 代理词
│   │
│   ├── feedback/                      # 用户反馈系统
│   │   ├── collector.py              # FeedbackCollector
│   │   ├── storage.py                # FeedbackStorage
│   │   └── analyzer.py               # FeedbackAnalyzer
│   │
│   ├── providers/                    # LLM 提供商
│   │   ├── base.py                   # Provider 基类
│   │   ├── anthropic.py             # Anthropic/Claude
│   │   ├── openai.py                # OpenAI
│   │   ├── deepseek.py              # DeepSeek
│   │   ├── azure.py                 # Azure OpenAI
│   │   └── ...（其他提供商）
│   │
│   ├── rag/                          # RAG 系统
│   │   ├── chunker.py                # LegalChunker：文档分块
│   │   ├── embedding.py              # EmbeddingClient：嵌入模型
│   │   ├── vectorstore.py            # ChromaVectorStore
│   │   ├── retriever.py              # LegalRetriever：混合检索
│   │   ├── reranker.py               # DashScopeReranker
│   │   ├── loader.py                 # LegalDocumentLoader
│   │   ├── indexer.py                # LegalIndexer：索引构建
│   │   ├── case_types.py             # 案件类型定义
│   │   └── case_analyzer.py          # 案例分析器
│   │
│   ├── session/                       # 会话管理
│   │   └── manager.py                 # SessionManager
│   │
│   ├── skills/                        # 技能定义
│   │   ├── README.md                  # 技能系统说明
│   │   ├── legal-research/            # 法律检索技能
│   │   ├── legal-citation/            # 法律引用技能
│   │   ├── legal-reasoning/           # 法律推理技能
│   │   ├── legal-debate/              # 法律辩论技能
│   │   ├── legal-case-compare/        # 案例对比技能
│   │   ├── legal-document-draft/      # 文书起草技能
│   │   ├── cron/                      # 定时任务技能
│   │   ├── memory/                    # 记忆技能
│   │   └── summarize/                  # 摘要技能
│   │
│   └── utils/                          # 工具函数
│       ├── __init__.py
│       ├── env.py                     # 环境变量工具
│       └── ...
│
├── tests/                             # 测试套件
│   ├── agent/
│   │   ├── test_loop.py
│   │   ├── test_context.py
│   │   ├── test_orchestrate_tool.py   # ← 新增 6 个意图路由测试
│   │   ├── test_reasoner.py
│   │   └── ...
│   ├── rag/
│   │   ├── test_retriever.py          # ← 新增 3 个 BM25 懒加载测试
│   │   ├── test_chunker.py
│   │   └── ...
│   ├── audit/
│   │   └── test_audit_logger.py       # ← 新增 17 个审计测试
│   ├── document/
│   │   ├── test_document_generation.py
│   │   └── test_document_generation_full.py  # ← 新增 15 个生成测试
│   ├── channels/
│   ├── providers/
│   └── ...（其他测试）
│
├── docs/                               # 文档
│   ├── PROJECT_EVALUATION.md          # 项目评估报告
│   ├── EXAMPLE_CONFIG.md              # 配置示例
│   ├── LEGAL_AUDIT_LOGGING.md        # 审计日志设计
│   ├── HUMAN_REVIEW_WORKFLOW.md       # 人工复核工作流设计
│   └── ...
│
├── legal_data/                         # 法律知识库数据
│   └── ...
│
├── data/                              # 运行时数据
│
└── pyproject.toml                      # 项目配置
```

---

## 4. 配置系统

### 4.1 配置概览

所有配置通过 `legalbot/config/schema.py` 中的 Pydantic 模型定义。配置支持：
- YAML/JSON 配置文件
- 环境变量覆盖（`legalbot_` 前缀）
- CamelCase/snake_case 兼容

### 4.2 完整配置结构

```yaml
# legalbot 配置（默认路径：~/.legalbot/config.json）

# Agent 默认配置
agents:
  defaults:
    workspace: "~/.legalbot/workspace"      # 工作区路径
    model: "deepseek-v3"                   # 默认模型
    provider: "deepseek"                   # 默认提供商
    max_tokens: 8192                       # 最大输出 tokens
    context_window_tokens: 128000          # 上下文窗口大小
    temperature: 0.7                        # 生成温度
    max_tool_iterations: 10                 # 最大工具调用次数
    timezone: "Asia/Shanghai"              # 时区
    unified_session: true                    # 统一会话模式

# 渠道配置
channels:
  send_progress: true                       # 发送进度消息
  send_tool_hints: true                     # 发送工具提示
  send_max_retries: 3                       # 最大重试次数

# 提供商配置
providers:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"         # 环境变量引用
    api_base: "https://api.deepseek.com"
  openai:
    api_key: "${OPENAI_API_KEY}"
  # ... 其他提供商

# API 服务配置
api:
  host: "0.0.0.0"
  port: 8080
  timeout: 120

# 网关配置
gateway:
  host: "0.0.0.0"
  port: 8081

# 工具配置
tools:
  web:
    enable: true
    proxy: null
    search:
      provider: "duckduckgo"
      max_results: 5
      timeout: 30

  exec:
    enable: true
    timeout: 30
    sandbox: true                          # 启用 bwrap 沙箱

  mcp_servers: []                          # MCP 服务器列表

  rag:
    enable: true
    embedding_provider: "dashscope"
    embedding_model: "text-embedding-v3"
    embedding_api_key: "${DASHSCOPE_API_KEY}"
    embedding_dim: 1536
    vector_store: "chroma"
    persist_dir: "~/.legalbot/chroma"
    reranker: "dashscope"
    top_k: 5
    chunk_max_tokens: 512
    chunk_overlap_tokens: 64
    bm25_enable: true

  orchestrate:
    enable: true
    intent_model: ""                       # 空=使用默认模型
    agents:
      legal_research:
        system_prompt: "你是法律检索专家..."
        tools: ["legal_rag_search", "web_search", "read_file"]
        model: ""
      contract_review:
        system_prompt: "你是合同审查专家..."
        tools: ["legal_rag_search", "read_file"]
    debate:
      enable: true
      rounds: 2
      timeout_per_agent: 60
      timeout_total: 180
      max_retries: 2
      judge_model: ""                      # 空=默认
      plaintiff_model: ""
      defendant_model: ""
    case_compare:
      enable: true
      comparison_model: ""
      max_cases: 10
      top_k_default: 5

  feedback:
    enable: true
    storage_dir: "~/.legalbot/feedback"
    retention_days: 90
    rate_limit_per_minute: 10

  document_draft:
    enable: true
    template_dir: null                     # null=使用内置模板
    enabled_types:                         # 启用的文书类型
      - complaint
      - defense
      - agent_opinion
      - appeal
      - enforcement
    max_laws_retrieved: 8
    default_model: ""

  audit:
    enable: true
    audit_dir: "~/.legalbot/audit"
    retention_days: 90
    pii_masking: true
```

### 4.3 配置类层次

```
Base（camelCase/snake_case 兼容）
├── AgentsConfig
│   └── AgentDefaults
├── ChannelsConfig
├── ProvidersConfig
│   └── ProviderConfig（各提供商配置）
├── ApiConfig
├── GatewayConfig
├── ToolsConfig
│   ├── WebToolsConfig
│   │   └── WebSearchConfig
│   ├── ExecToolConfig
│   ├── MCPServerConfig
│   ├── RAGConfig
│   ├── OrchestrateConfig
│   │   ├── AgentDefConfig（子 Agent 定义）
│   │   ├── DebateConfig（辩论配置）
│   │   └── CaseCompareConfig（案例对比配置）
│   ├── FeedbackConfig
│   ├── DocumentDraftConfig
│   └── AuditConfig
└── Config（根配置）
```

### 4.4 配置加载

```python
# legalbot/config/loader.py
from legalbot.config import load_config, save_config

# 从默认路径加载配置
config = load_config()

# 从指定路径加载
config = load_config("~/custom/config.json")

# 保存配置
save_config(config, "~/custom/config.json")
```

### 4.5 环境变量覆盖

配置值可以使用 `${ENV_VAR}` 语法引用环境变量：

```yaml
providers:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
```

`loader.py` 中的 `resolve_config_env_vars()` 负责替换。

---

## 5. Agent 系统

### 5.1 AgentLoop（主循环）

`legalbot/agent/loop.py` 是系统的核心处理引擎。

#### 5.1.1 初始化

```python
class AgentLoop:
    def __init__(
        self,
        provider: Any,
        tools: ToolRegistry,
        config: Config,
        workspace: Path,
        bus: MessageBus,
        feedback_config: FeedbackConfig | None = None,
        audit_config: AuditConfig | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.config = config
        self.workspace = workspace
        self.bus = bus
        self.feedback_config = feedback_config
        self.audit_config = audit_config
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_init_lock = asyncio.Lock()  # 保护会话锁初始化
```

#### 5.1.2 会话锁（Double-Check 模式）

避免会话锁竞态条件：

```python
async def _get_session_lock(self, session_key: str) -> asyncio.Lock:
    lock = self._session_locks.get(session_key)
    if lock is not None:
        return lock
    async with self._session_init_lock:
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]
```

#### 5.1.3 消息处理流程

```
_process_message(msg: Message) → None
    │
    ├─→ _get_session_lock(msg.session_key)    # 获取会话锁
    │
    ├─→ _ensure_audit_logger()                 # 确保审计日志器初始化
    │
    ├─→ _build_context(msg)                   # 构建运行时上下文
    │       │
    │       ├─→ Memory.get_history()           # 获取历史消息
    │       ├─→ SkillsLoader.get_active()       # 获取活跃技能
    │       └─→ skills.inject_to_context()      # 注入技能到上下文
    │
    ├─→ AgentRunner.run(spec, context)         # 执行 Agent
    │       │
    │       ├─→ provider.chat(messages)        # 调用 LLM
    │       │
    │       └─→ ToolRegistry.execute()          # 执行工具调用
    │
    ├─→ _log_audit_event()                     # 记录审计日志
    │
    └─→ bus.publish(response)                  # 发布响应消息
```

#### 5.1.4 审计日志集成

每次处理消息后自动记录：

```python
async def _log_audit_event(self, msg, response_text, tools_used):
    if not self._audit_logger:
        return
    event_type = self._classify_legal_event(msg.content)
    citations = self._extract_citations(response_text)
    await self._audit_logger.log(
        event_type=event_type,
        session_id=msg.session_key,
        channel=msg.channel,
        query={"original_text": msg.content},
        response={
            "final_content": response_text,
            "tools_called": tools_used,
            "citations": citations,
            "disclaimer_shown": True,
        },
        metadata={"model": self.config.agents.defaults.model},
    )
```

### 5.2 AgentRunner（执行器）

`legalbot/agent/runner.py` 负责单次 Agent 执行。

#### 5.2.1 执行流程

```python
async def run(self, spec: AgentRunSpec, context: AgentContext) -> AgentRunResult:
    messages = context.to_llm_messages()

    while self.iteration < spec.max_tool_iterations:
        # 1. 调用 LLM
        response = await self.provider.chat_with_retry(messages)

        # 2. 处理停止原因
        if response.finish_reason == "stop":
            return AgentRunResult(
                final_content=response.content,
                stop_reason="stop",
                iterations=self.iteration,
            )

        # 3. 处理工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = await self.registry.execute(tool_call.name, tool_call.params)
                messages.append(response)          # LLM 响应
                messages.append(result.to_message()) # 工具结果

        self.iteration += 1

    return AgentRunResult(...)
```

### 5.3 LegalOrchestrator（法律编排器）

`legalbot/agent/orchestrator.py` 负责意图分类和多 Agent 路由。

#### 5.3.1 意图类型

```python
INTENT_LEGAL_QUERY = "legal_query"
INTENT_CONTRACT_REVIEW = "contract_review"
INTENT_CASE_SEARCH = "case_search"
INTENT_COMPLEX_LEGAL_QUERY = "complex_legal_query"
INTENT_DEBATE = "debate"
INTENT_CASE_COMPARE = "case_compare"
INTENT_DOCUMENT_DRAFT = "document_draft"
INTENT_GENERAL = "general"
```

#### 5.3.2 意图分类流程

```
用户查询
    │
    ▼
classify_intent(query)  ──→ LLM 调用
    │
    ├─→ LEGAL_QUERY ──→ _classify_complexity()
    │                       │
    │                       ├─→ "simple" → 简单法律检索（RAG 直接返回）
    │                       └─→ "complex" → 复杂法律检索（MultiStepReasoner）
    │
    ├─→ CONTRACT_REVIEW ──→ _contract_review_flow_sync()
    │
    ├─→ CASE_SEARCH ──→ 路由到 legal_research Agent
    │
    ├─→ DEBATE ──→ run_debate_sync()
    │                  │
    │                  ├─→ 原告 Agent
    │                  ├─→ 被告 Agent
    │                  └─→ 法官 Agent
    │
    ├─→ CASE_COMPARE ──→ CaseCompareTool
    │
    ├─→ DOCUMENT_DRAFT ──→ LegalDocumentGenerateTool
    │
    └─→ GENERAL ──→ general Agent 或直接 RAG
```

#### 5.3.3 辩论流程

```python
async def run_debate_sync(
    self,
    case_description: str,
    plaintiff_claims: str | None = None,
    defendant_response: str | None = None,
    debate_rounds: int = 1,
) -> str:
    # 创建三个 Agent：原告、被告、法官
    plaintiff = Subagent(agent_def, role="原告")
    defendant = Subagent(agent_def, role="被告")
    judge = Subagent(judge_agent_def, role="法官")

    for round_i in range(debate_rounds):
        # 原告论证
        p_result = await plaintiff.run(f"请就以下案件陈述原告方的诉讼请求和理由：{case_description}")
        # 被告答辩
        d_result = await defendant.run(f"请就以下案件和原告主张进行答辩：{case_description}\n\n原告主张：{p_result}")
        # 法官总结
        j_result = await judge.run(f"基于以下辩论，总结争议焦点：\n\n原告：{p_result}\n被告：{d_result}")

    return generate_debate_report(p_result, d_result, j_result)
```

### 5.4 SubagentManager（子 Agent 管理）

管理后台运行的子 Agent，支持自定义系统提示词和工具白名单。

```python
async def spawn_with_config(
    self,
    task: str,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
    label: str = "subagent",
) -> str:
    spec = AgentRunSpec(
        system_prompt=system_prompt or default_prompt,
        tools=self._build_available_tools(allowed_tools),
        model=model,
        max_iterations=10,
    )
    task_id = self._tasks.launch(runner.run, spec, task)
    return f"Subagent [{label}] started (id: {task_id})."
```

---

## 6. RAG 系统

### 6.1 系统概览

```
┌─────────────────────────────────────────────────────┐
│                    LegalIndexer                      │
│              legalbot/rag/indexer.py                 │
│  加载法律文档 → 分块 → 嵌入 → 向量存储 + BM25       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                  LegalRetriever                      │
│              legalbot/rag/retriever.py               │
│  1. Vector Search (top_k * 3)                       │
│  2. BM25 Keyword Search                             │
│  3. Reciprocal Rank Fusion (RRF) Merge              │
│  4. Optional Reranking (DashScope)                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              RetrievalResult[]
```

### 6.2 LegalChunker（文档分块器）

`legalbot/rag/chunker.py` 负责将法律文档分割成适合检索的块。

#### 6.2.1 分块策略

1. **按条文结构分割**：识别 `第X条` 模式作为主要分割点
2. **长条文二次分割**：超过 `chunk_max_tokens` 的条文按句子分割
3. **重叠机制**：相邻块之间保留 `chunk_overlap_tokens` 个 token 的重叠

#### 6.2.2 分块元数据

```python
@dataclass
class ChunkMeta:
    source: str                  # 文档来源
    law_name: str                # 法律名称，如"民法典"
    law_area: str                # 法律领域，如"民法"
    doc_type: str                # 文档类型
    article_no: str              # 条文编号，如"第五百七十五条"
    chunk_index: int             # 块在原文档中的索引
```

#### 6.2.3 分块示例

```python
# 输入法律文本
text = """
第五百七十五条 当事人一方不履行合同义务或者履行合同义务不符合约定的，
应当承担违约责任。
第五百七十六条 当事人一方不履行合同义务或者履行合同义务不符合约定的，
对方可以在催告后要求其承担违约责任。
"""

# 分块结果
chunks = [
    Chunk(id="c1", text="第五百七十五条 当事人一方不履行合同义务...",
          metadata=ChunkMeta(law_name="民法典", article_no="第五百七十五条")),
    Chunk(id="c2", text="第五百七十六条 当事人一方不履行合同义务...",
          metadata=ChunkMeta(law_name="民法典", article_no="第五百七十六条")),
]
```

### 6.3 EmbeddingClient（嵌入客户端）

`legalbot/rag/embedding.py` 支持多种嵌入模型。

#### 6.3.1 支持的提供商

| 提供商 | 模型 | 维度 |
|--------|------|------|
| DashScope | text-embedding-v3 | 1536 |
| OpenAI | text-embedding-3-small / v2 | 1536/3072 |
| Zhipu | embedding-2 | 2048 |
| BGE | bge-large-zh | 1024 |

#### 6.3.2 使用方式

```python
client = EmbeddingClient(
    provider="dashscope",
    api_key="${DASHSCOPE_API_KEY}",
    model="text-embedding-v3",
    dim=1536,
)

vectors = await client.embed(["法律文本1", "法律文本2"])
```

### 6.4 ChromaVectorStore（向量存储）

`legalbot/rag/vectorstore.py` 基于 Chroma 的向量存储实现。

#### 6.4.1 核心操作

```python
class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | None = None,    # 持久化目录
        collection_name: str = "legal",
    ):
        self._client = chromadb.PersistentClient(persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        texts: list[str],
    ) -> None:
        # 批量添加到 Chroma
        self._collection.add(ids, vectors, metadatas, texts)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter,
        )
        return [SearchResult(id=r['id'], text=r['text'], metadata=r['metadata'], score=r['distance'])
                for r in results]
```

### 6.5 BM25Store（关键词检索）

`legalbot/rag/retriever.py` 中的 `BM25Store` 类实现基于 BM25 的关键词检索。

#### 6.5.1 懒加载初始化

BM25 索引采用懒加载策略，避免 O(n) 重建：

```python
class BM25Store:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Any = None
        self._dirty: bool = False  # 标记是否需要重建索引

    def add(self, chunks: list[Chunk]) -> None:
        """添加块，标记 dirty，但不立即重建索引"""
        import jieba
        self._chunks.extend(chunks)
        new_tokenized = [list(jieba.cut(c.text)) for c in chunks]
        self._tokenized_corpus.extend(new_tokenized)
        self._dirty = True

    def _ensure_index(self) -> None:
        """仅在 dirty 时才重建索引"""
        if not self._dirty or not self._tokenized_corpus:
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._dirty = False

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        self._ensure_index()  # 首次 search 时才构建索引
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        # 返回 top_k 结果
```

### 6.6 LegalRetriever（混合检索器）

`legalbot/rag/retriever.py` 整合向量搜索、BM25 和 RRF。

#### 6.6.1 检索流程

```python
async def retrieve(
    self,
    query: str,
    law_area: str | None = None,
    doc_type: str | None = None,
    top_k: int = 5,
) -> RetrievalPipelineResult:
    # 1. 向量搜索（扩展 3 倍以补偿融合损失）
    vector_results = await self._vector_search(query, top_k * 3, law_area, doc_type)

    # 2. BM25 搜索
    bm25_results = self._bm25_search(query, top_k * 3)

    # 3. RRF 融合
    rrf_candidates = self._rrf_merge(vector_results, bm25_results)

    # 4. 可选重排序
    if self._reranker:
        reranked = await self._reranker.rerank(query, rrf_candidates, top_k=top_k)
        return RetrievalPipelineResult(rrf_candidates=rrf_candidates, top_k=reranked)

    return RetrievalPipelineResult(rrf_candidates=rrf_candidates, top_k=rrf_candidates[:top_k])
```

#### 6.6.2 Reciprocal Rank Fusion（互惠排名融合）

```python
@staticmethod
def _rrf_merge(
    vector_results: list[SearchResult],
    bm25_results: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[RetrievalResult]:
    """RRF 融合两个排名列表

    RRF_score(d) = Σ 1/(k + rank_i(d))

    Args:
        k: 融合参数，越大各排名系统权重越均衡
    """
    scores: dict[str, float] = {}

    for rank, result in enumerate(vector_results):
        rrf = 1 / (k + rank + 1)
        scores[result.id] = scores.get(result.id, 0) + rrf * 0.6  # 向量权重 0.6

    for rank, (chunk, bm25_score) in enumerate(bm25_results):
        rrf = 1 / (k + rank + 1)
        scores[chunk.id] = scores.get(chunk.id, 0) + rrf * 0.4  # BM25 权重 0.4

    # 按 RRF 分数排序
    sorted_ids = sorted(scores.keys(), key=lambda id: scores[id], reverse=True)
    # 构建结果（需要通过 ID 找回 Chunk 对象）
    ...
```

### 6.7 LegalIndexer（索引构建器）

`legalbot/rag/indexer.py` 负责从法律文档构建 RAG 索引。

#### 6.7.1 索引构建流程

```python
async def index(
    self,
    documents: list[Path] | None = None,
    clear_existing: bool = False,
) -> IndexResult:
    if clear_existing:
        await self.vector_store.clear()

    docs = documents or self._discover_documents()
    all_chunks = []

    for doc_path in docs:
        # 1. 加载文档
        loader = LegalDocumentLoader(doc_path)
        raw_text = loader.load()

        # 2. 分块
        chunks = self.chunker.chunk(raw_text)

        # 3. 嵌入
        texts = [c.text for c in chunks]
        embeddings = await self.embedding_client.embed(texts)

        # 4. 存储
        await self.vector_store.add(
            ids=[c.id for c in chunks],
            vectors=embeddings,
            metadatas=[dict(c.metadata) for c in chunks],
            texts=texts,
        )

        # 5. BM25 索引
        self.bm25_store.add(chunks)

        all_chunks.extend(chunks)

    return IndexResult(total_chunks=len(all_chunks), documents=len(docs))
```

---

## 7. 法律工具集

### 7.1 工具注册中心

`legalbot/agent/tools/registry.py` 实现了统一的工具注册和执行框架。

#### 7.1.1 核心 API

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]

    def tools_snapshot(self) -> dict[str, Tool]:
        """返回工具字典的快照（公共 API）"""
        return dict(self._tools)

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """执行工具"""
        tool, params, error = self.prepare_call(name, params)
        if error:
            raise ToolError(error, tool_name=name)
        try:
            result = await tool.execute(**params)
            return result
        except Exception as e:
            raise ToolError(f"Error executing {name}: {str(e)}", tool_name=name) from e
```

#### 7.1.2 Tool 基类

```python
class Tool(metaclass=ToolMeta):
    name: str
    description: str
    parameters: dict  # JSON Schema

    @property
    def exclusive(self) -> bool:
        """工具是否独占执行（编排类工具为 True）"""
        return False

    @property
    def read_only(self) -> bool:
        """工具是否只读"""
        return False

    async def execute(self, **kwargs) -> Any:
        """执行工具逻辑"""
        raise NotImplementedError
```

### 7.2 工具清单

| 工具 | 类名 | 功能 |
|------|------|------|
| 编排 | `OrchestrateTool` | 多 Agent 意图路由和调度 |
| 法律检索 | `RAGSearchTool` | 混合 RAG 检索法律条文 |
| 辩论 | `DebateTool` | 启动法律辩论分析 |
| 案例对比 | `CaseCompareTool` | 对比多个案例异同 |
| 文书生成 | `LegalDocumentGenerateTool` | 生成法律文书 |
| 多步推理 | `MultiStepReasoningTool` | 链式法律推理 |
| 文件读 | `ReadFileTool` | 读取文件内容 |
| 文件写 | `WriteFileTool` | 写入文件内容 |
| 文件编辑 | `EditFileTool` | 编辑文件内容 |
| 目录列表 | `ListDirTool` | 列出目录内容 |
| 文件搜索 | `GlobTool` | glob 模式搜索文件 |
| 内容搜索 | `GrepTool` | 正则搜索文件内容 |
| 命令执行 | `ExecTool` | 执行 Shell 命令（沙箱） |
| Web 搜索 | `WebSearchTool` | 搜索互联网 |
| Web 抓取 | `WebFetchTool` | 抓取网页内容 |
| 消息发送 | `MessageTool` | 向渠道发送消息 |
| 子 Agent | `SpawnTool` | 启动后台子 Agent |
| 定时任务 | `CronTool` | 创建定时任务 |
| 反馈 | `FeedbackTool` | 收集用户反馈 |
| MCP | `MCPTool` | MCP 服务器工具 |

### 7.3 OrchestrateTool（编排工具）

`legalbot/agent/tools/orchestrate.py` 是法律 Agent 的核心入口工具。

#### 7.3.1 关键字路由

编排工具在调用 LLM 之前先做关键字预检：

```python
# 辩论关键字
debate_keywords = (
    "辩论", "debate", "争议焦点", "原告", "被告",
    "诉讼请求", "答辩", "抗辩", "法律辩论",
)

# 案例对比关键字
case_compare_keywords = (
    "案例对比", "对比案例", "类似案例", "相似案例",
    "case compare", "类案", "案例检索",
)

# 文书起草关键字
doc_draft_keywords = (
    "起诉状", "答辩状", "代理词", "上诉状", "执行申请书",
    "写一份起诉书", "起草", "帮我写诉状", "写诉状",
    "complaint", "defense", "appeal", "enforcement",
)
```

#### 7.3.2 执行流程

```python
async def execute(self, query: str, intent: str | None = None, **kwargs):
    # 1. 关键字预检 - 辩论
    if any(kw in query for kw in debate_keywords):
        return await self._orchestrator.run_debate_sync(...)

    # 2. 关键字预检 - 案例对比
    if any(kw in query for kw in case_compare_keywords):
        tool = self._main_tools.get("legal_case_compare")
        return await tool.execute(dispute_facts=query)

    # 3. 关键字预检 - 文书起草
    if any(kw in query for kw in doc_draft_keywords):
        tool = self._main_tools.get("legal_document_generate")
        return await tool.execute(case_facts=query, doc_type="complaint")

    # 4. 显式 intent 路由
    if intent in ("legal_query", "case_search"):
        if self._retriever:
            results = await self._retriever.retrieve(query=query, top_k=5)
            return RAGSearchTool._format_results(query, results.top_k)
        return await self._orchestrator._run_agent_sync("legal_research", query)
    elif intent == "contract_review":
        return await self._orchestrator._contract_review_flow_sync(query)

    # 5. 自动意图分类
    return await self._orchestrator.dispatch_sync(query)
```

### 7.4 RAGSearchTool（检索工具）

`legalbot/agent/tools/rag.py` 提供法律检索能力。

```python
class RAGSearchTool(Tool):
    name = "legal_rag_search"
    description = "检索法律知识库中的相关法规、司法解释和指导性案例..."

    async def execute(
        self,
        query: str,
        law_area: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> str:
        results = await self.retriever.retrieve(
            query=query,
            law_area=law_area,
            doc_type=doc_type,
            top_k=top_k,
        )

        if not results.top_k:
            return "未找到相关法律条文，请尝试调整检索词。"

        return self._format_results(query, results.top_k)
```

### 7.5 DebateTool（辩论工具）

`legalbot/agent/tools/debate.py` 实现法律辩论模式。

### 7.6 CaseCompareTool（案例对比工具）

`legalbot/agent/tools/case_compare.py` 实现案例对比分析。

### 7.7 LegalDocumentGenerateTool（文书生成工具）

`legalbot/agent/tools/document.py` 封装文书生成能力。

---

## 8. 文档生成系统

### 8.1 系统概览

```
用户请求（"帮我写一份起诉状"）
    │
    ▼
LegalDocumentGenerator.generate(doc_type, case_facts)
    │
    ├─→ CaseFactsExtractor.extract()     # LLM 提取结构化事实
    │
    ├─→ LegalRetriever.retrieve()       # RAG 检索相关法律
    │
    ├─→ Template.build_prompt()          # 构建 LLM prompt
    │
    ├─→ Provider.chat()                  # 调用 LLM 生成文书
    │
    └─→ 添加免责声明
            │
            ▼
        最终文书文本
```

### 8.2 文书模板

`legalbot/document/templates/` 下有 5 种文书模板：

#### 8.2.1 模板基类

```python
class LegalDocumentTemplate(ABC):
    @property
    @abstractmethod
    def doc_type(self) -> str:
        """文书类型标识"""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """显示名称"""

    @property
    def law_keywords(self) -> list[str]:
        """用于 RAG 检索的法律关键词"""

    @property
    def required_variables(self) -> list[Variable]:
        """必需变量定义"""

    @abstractmethod
    def build_prompt(
        self,
        case_facts: str,
        relevant_laws: list[str],
        variable_set: dict,
    ) -> str:
        """构建 LLM 生成 prompt"""
```

#### 8.2.2 起诉状模板

```python
class ComplaintTemplate(LegalDocumentTemplate):
    doc_type = "complaint"
    display_name = "起诉状"

    law_keywords = [
        "中华人民共和国民事诉讼法",
        "民间借贷",
        "借款合同",
        "起诉条件",
        "管辖法院",
    ]

    required_variables = [
        Variable("plaintiff_name", "原告姓名", required=True),
        Variable("defendant_name", "被告姓名", required=True),
        Variable("litigation_requests", "诉讼请求", required=True),
        Variable("facts_and_reasons", "事实与理由", required=True),
        Variable("evidence", "证据", required=False),
        Variable("court", "管辖法院", required=False),
    ]

    def build_prompt(self, case_facts, relevant_laws, variable_set) -> str:
        return f"""你是一名专业律师，请根据以下案件事实和法律依据，
起草一份《民事起诉状》。

案件事实：
{self._format_case_facts(case_facts)}

相关法律条文：
{chr(10).join(relevant_laws)}

原告信息：{variable_set.get('原告姓名', '')}
被告信息：{variable_set.get('被告姓名', '')}
诉讼请求：{variable_set.get('诉讼请求', '')}
事实与理由：{variable_set.get('事实与理由', '')}

请按照民事起诉状的标准格式输出，包含：标题、当事人信息、诉讼请求、事实与理由、证据和证人、结尾（此致 XXX 人民法院）、署名日期。"""  # noqa: E501
```

#### 8.2.3 支持的文书类型

| 类型 | doc_type | 说明 |
|------|----------|------|
| 起诉状 | complaint | 民事起诉状 |
| 答辩状 | defense | 民事答辩状 |
| 代理词 | agent_opinion | 代理词 |
| 上诉状 | appeal | 民事上诉状 |
| 执行申请书 | enforcement | 执行申请书 |

### 8.3 CaseFactsExtractor（事实提取器）

使用 LLM 从自然语言案件描述中提取结构化信息：

```python
class CaseFactsExtractor:
    def __init__(self, provider: Any):
        self.provider = provider

    async def extract(self, case_facts: str, doc_type: str) -> CaseFacts:
        prompt = PROMPT_TEMPLATE.format(
            doc_type=doc_type,
            case_facts=case_facts,
        )
        response = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        # 解析 JSON 响应
        return CaseFacts.from_llm_response(response.content)
```

### 8.4 CaseFacts 数据类

```python
@dataclass
class CaseFacts:
    raw_text: str                    # 原始文本
    doc_type: str                   # 文书类型
    parties: dict                   # 当事人信息
    monetary_amount: float | None   # 争议金额
    case_type: str | None           # 案件类型
    dates: dict                    # 相关日期
    evidence: list                 # 证据清单

    def to_dict(self) -> dict:
        """转换为模板变量字典"""
        return {
            "原告姓名": self.parties.get("plaintiff", {}).get("name", ""),
            "被告姓名": self.parties.get("defendant", {}).get("name", ""),
            "争议金额": str(self.monetary_amount) if self.monetary_amount else "",
            "案由": self.case_type or "",
            # ... 更多字段
        }
```

### 8.5 生成的文书示例

```python
# 输入
case_facts = "张三借款给李四10万元，约定2024年1月1日还款，但李四至今未还"

# 输出（部分）
generated_document = """
民事起诉状

原告：张三，男，...（基本信息）
被告：李四，男，...（基本信息）

诉讼请求：
1. 判令被告返还借款本金人民币10万元；
2. 判令被告支付逾期利息...

事实与理由：
2023年X月X日，被告因资金周转需要向原告借款人民币10万元，
双方签订借款合同，约定于2024年1月1日归还...

相关法律依据：
《民法典》第六百七十六条：借款人未按照约定的期限返还借款的，
应当按照约定或者国家有关规定支付逾期利息。

此致
XXX 人民法院

原告（签名）：___
日期：___

---
【免责声明】
本文书由 AI 自动生成，仅供参考，不构成正式法律意见。
诉讼材料的正式提交应由执业律师审核确认。
如需正式法律意见，请咨询执业律师。
"""
```

---

## 9. 审计日志系统

### 9.1 系统架构

`legalbot/audit/logger.py` 实现了完整的法律操作审计跟踪。

```
┌─────────────────────────────────────────────────────┐
│              LegalAuditLogger                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  async log(event_type, session_id,           │   │
│  │         query, response, metadata)           │   │
│  └──────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌──────────────────────────────────────────────┐   │
│  │  PII 脱敏处理                                 │   │
│  │  - 身份证号：11010119900101123X               │   │
│  │         → 110101****123X                      │   │
│  │  - 手机号：13812345678                        │   │
│  │         → 138****5678                         │   │
│  │  - 邮箱：test@example.com                    │   │
│  │         → t***@example.com                    │   │
│  └──────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌──────────────────────────────────────────────┐   │
│  │  SHA256 Hash 计算                             │   │
│  │  record["_hash"] = SHA256(record)[:16]       │   │
│  └──────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│              ~/.legalbot/audit/YYYY-MM-DD.jsonl      │
│                                                      │
│  查询接口：async query()                             │
│  清理接口：async cleanup_old_logs()                  │
│  验证接口：async verify_integrity()                  │
└─────────────────────────────────────────────────────┘
```

### 9.2 事件类型

```python
class LegalEventType(str, Enum):
    LEGAL_QUERY = "legal_query"           # 法律咨询
    DOCUMENT_DRAFT = "document_draft"      # 文书起草
    CASE_COMPARE = "case_compare"          # 案例对比
    DEBATE = "debate"                     # 法律辩论
    CONTRACT_REVIEW = "contract_review"    # 合同审查
    SYSTEM_MESSAGE = "system_message"      # 系统消息
```

### 9.3 日志记录格式

```json
{
  "event_id": "f10e7c84-79f6-4725-bbb3-056d96de8023",
  "timestamp": "2026-04-19T08:24:36.998486+00:00",
  "event_type": "legal_query",
  "session_id": "cli:test",
  "channel": "cli",
  "user_id": "anonymous",
  "query": {
    "original_text": "张三借款给李四10万元"
  },
  "response": {
    "final_content": "根据《民法典》第675条...",
    "tools_called": ["legal_rag_search"],
    "citations": ["《民法典》第六百七十五条"],
    "disclaimer_shown": true
  },
  "metadata": {
    "model": "deepseek-v3"
  },
  "_hash": "828086ae17cb7c4e"
}
```

### 9.4 PII 脱敏

```python
_PII_PATTERNS = [
    ("id_card",   r"\b\d{17}[\dXx]\b"),           # 18位身份证
    ("phone_cn",  r"\b1[3-9]\d{9}\b"),            # 中国手机号
    ("email",     r"\b[\w.-]+@[\w.-]+\.\w+\b"),  # 邮箱
]

def _mask_pii(text: str) -> str:
    masked = text
    for label, pattern in _PII_PATTERNS:
        if label == "id_card":
            masked = pattern.sub(lambda m: f"{m.group()[:6]}****{m.group()[-4:]}", masked)
        elif label == "phone_cn":
            masked = pattern.sub(lambda m: f"{m.group()[:3]}****{m.group()[-4:]}", masked)
        elif label == "email":
            masked = pattern.sub(lambda m: m.group()[0] + "***@" + m.group().split("@")[1], masked)
    return masked
```

### 9.5 查询接口

```python
async def query(
    self,
    start_date: str | None = None,   # "2026-04-01"
    end_date: str | None = None,      # "2026-04-19"
    event_type: str | None = None,    # "legal_query"
    session_id: str | None = None,    # "cli:test"
    limit: int = 100,
) -> list[dict]:
    """查询审计日志，支持多条件过滤"""
    ...
```

### 9.6 完整性验证

```python
async def verify_integrity(self, event_id: str | None = None) -> dict:
    """验证记录未被篡改

    对于指定 event_id 或全部记录：
    1. 读取原始 JSON
    2. 提取 hash 字段
    3. 从记录中移除 hash 字段
    4. 重新计算 SHA256
    5. 比对 hash 值
    """
    result = {"valid": 0, "corrupted": [], "total": 0}
    for record in records:
        stored_hash = record.pop("_hash")
        computed = _compute_hash(record)
        if stored_hash == computed:
            result["valid"] += 1
        else:
            result["corrupted"].append(record["event_id"])
        record["_hash"] = stored_hash  # 恢复
    return result
```

### 9.7 CLI 命令

```bash
# 查询审计日志
legalbot audit query --event-type legal_query --limit 10
legalbot audit query --session-id "cli:test" --start-date 2026-04-01

# 清理旧日志
legalbot audit cleanup

# 验证完整性
legalbot audit verify
legalbot audit verify --event-id f10e7c84-79f6-4725-bbb3-056d96de8023
```

---

## 10. 技能系统

### 10.1 技能定义

技能是包含 `SKILL.md` 的目录，提供特殊能力的定义和使用说明。

```
legalbot/skills/
├── legal-research/
│   ├── SKILL.md           # 技能定义
│   └── skill.yaml         # 元数据（可选）
├── legal-debate/
│   └── SKILL.md
├── legal-document-draft/
│   └── SKILL.md           # always: true（强制加载）
└── ...
```

### 10.2 SKILL.md 格式

```markdown
---
name: legal-document-draft
description: 生成法律文书（起诉状、答辩状等）
always: true        # 强制加载，不在工具列表中显示但始终注入上下文
version: 1.0.0
---

# 法律文书起草技能

## 触发条件

当用户请求以下操作时激活：
- "写一份起诉状"
- "帮我起草答辩状"
- "生成代理词"

## 使用方法

调用 `legal_document_generate` 工具，参数：
- `case_facts`: 案件事实描述
- `doc_type`: 文书类型（complaint/defense/agent_opinion/appeal/enforcement）

## 约束

1. 所有生成的文书必须附加免责声明
2. 不生成虚假法律引用
3. 重大案件建议咨询执业律师
```

### 10.3 技能加载

```python
class SkillsLoader:
    def load(self, workspace: Path | None = None) -> list[Skill]:
        skills = []
        # 加载内置技能
        skills.extend(self._load_from_dir(Path(__file__).parent / "skills"))
        # 加载工作区技能
        if workspace:
            skills.extend(self._load_from_dir(workspace / "skills"))
        return skills

    def get_active(self, context: dict) -> list[Skill]:
        """根据上下文返回应激活的技能"""
        return [s for s in self.skills if s.always or self._matches_context(s, context)]
```

---

## 11. 多渠道支持

### 11.1 渠道架构

```
ChannelManager
├── Feishu (飞书)
├── DingTalk (钉钉)
├── QQ
└── WebSocket
```

### 11.2 渠道基类

```python
class BaseChannel(ABC):
    @abstractmethod
    async def start(self) -> None:
        """启动渠道"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止渠道"""
        pass

    @abstractmethod
    async def send(self, message: str, target: str | None = None) -> None:
        """发送消息"""
        pass

    @abstractmethod
    async def login(self) -> None:
        """登录/认证"""
        pass

    @abstractmethod
    async def on_message(self, message: Message) -> None:
        """接收消息的处理"""
        pass
```

### 11.3 消息模型

```python
@dataclass
class Message:
    id: str                       # 消息 ID
    channel: str                   # 渠道名称
    session_key: str               # 会话键
    role: str                      # "user" | "assistant" | "system"
    content: str                   # 消息内容
    metadata: dict                 # 渠道特定元数据
    timestamp: datetime
```

---

## 12. CLI 命令参考

### 12.1 命令清单

| 命令 | 说明 |
|------|------|
| `legalbot onboard` | 初始化配置 |
| `legalbot agent` | 交互式 CLI 聊天 |
| `legalbot serve` | 启动 OpenAI 兼容 API 服务器 |
| `legalbot gateway` | 启动完整网关（含渠道） |
| `legalbot status` | 显示系统状态 |
| `legalbot legal index` | 构建法律知识库索引 |
| `legalbot legal index-status` | 显示索引状态 |
| `legalbot legal search QUERY` | 从 CLI 检索法律 |
| `legalbot channels status` | 显示渠道状态 |
| `legalbot channels login` | 渠道登录认证 |
| `legalbot plugins list` | 列出可用插件 |
| `legalbot feedback list` | 列出反馈记录 |
| `legalbot feedback analyze` | 分析反馈数据 |
| `legalbot audit query` | 查询审计日志 |
| `legalbot audit cleanup` | 清理旧审计日志 |
| `legalbot audit verify` | 验证审计完整性 |

### 12.2 常用命令示例

```bash
# 初始化
legalbot onboard

# 交互式聊天
legalbot agent

# 构建法律索引
legalbot legal index --docs-dir ./legal_data

# 检索法律
legalbot legal search "民间借贷 利息"

# 查询审计日志
legalbot audit query --event-type legal_query --limit 20

# 启动 API 服务
legalbot serve --port 8080

# 启动网关
legalbot gateway
```

---

## 13. 测试框架

### 13.1 测试结构

```
tests/
├── agent/
│   ├── test_loop.py              # AgentLoop 测试
│   ├── test_context.py           # 上下文构建测试
│   ├── test_orchestrate_tool.py  # 编排工具测试（含新增意图路由测试）
│   ├── test_reasoner.py          # 多步推理测试
│   └── ...
├── rag/
│   ├── test_retriever.py         # 检索器测试（含新增 BM25 懒加载测试）
│   ├── test_chunker.py           # 分块器测试
│   ├── test_vectorstore.py       # 向量存储测试
│   ├── test_retrieval_metrics.py # 检索指标测试
│   └── ...
├── audit/
│   └── test_audit_logger.py      # 审计日志测试（17 个测试）
├── document/
│   ├── test_document_generation.py          # 文书模板测试
│   └── test_document_generation_full.py     # 文书生成全流程测试（15 个）
├── channels/
├── providers/
├── config/
├── cli/
├── feedback/
└── ...
```

### 13.2 测试工具

- **pytest**：主测试框架
- **pytest-asyncio**：`@pytest.mark.asyncio` 支持异步测试
- **pytest-cov**：代码覆盖率
- **pytest-anyio**：`anyio` 后端支持

### 13.3 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定目录
uv run pytest tests/rag/ -v

# 运行特定测试文件
uv run pytest tests/audit/test_audit_logger.py -v

# 运行带覆盖率
uv run pytest --cov=legalbot --cov-report=html

# 运行特定关键字测试
uv run pytest -k "lazy_init"

# 运行新增的测试
uv run pytest tests/rag/test_retriever.py tests/agent/test_orchestrate_tool.py tests/document/test_document_generation_full.py tests/audit/test_audit_logger.py -v
```

### 13.4 新增测试清单（2026-04-19）

| 文件 | 新增测试数 | 测试内容 |
|------|-----------|----------|
| `tests/audit/test_audit_logger.py` | 17 | PII 脱敏（身份证/手机/邮箱）、Hash 完整性、cleanup_old_logs、verify_integrity、篡改检测 |
| `tests/document/test_document_generation_full.py` | 15 | LegalDocumentGenerator 全流程、免责声明、各模板类型、错误处理、retriever 集成 |
| `tests/rag/test_retriever.py` | +3 | BM25Store 懒加载初始化（3 个测试） |
| `tests/agent/test_orchestrate_tool.py` | +6 | OrchestrateTool 意图关键字路由（辩论/案例对比/文书起草） |

---

## 14. API 参考

### 14.1 legalbot Facade

```python
from legalbot import legalbot

# 创建实例
legalbot = legalbot(config_path="~/.legalbot/config.json")

# 聊天
response = await legalbot.chat("民间借贷纠纷如何起诉？")

# 流式聊天
async for chunk in legalbot.stream_chat("写一份起诉状"):
    print(chunk, end="")
```

### 14.2 配置加载

```python
from legalbot.config import load_config

config = load_config()
print(config.tools.orchestrate.enable)
print(config.tools.rag.vector_store)
```

### 14.3 创建 AgentLoop

```python
from legalbot import AgentLoop
from legalbot.agent.tools.registry import ToolRegistry
from legalbot.bus.queue import MessageBus
from legalbot.providers import DeepSeekProvider

provider = DeepSeekProvider(api_key="sk-...")
registry = ToolRegistry()
# 注册工具...
bus = MessageBus()

agent = AgentLoop(
    provider=provider,
    tools=registry,
    config=config,
    workspace=Path("~/.legalbot/workspace"),
    bus=bus,
)
```

### 14.4 创建 LegalRetriever

```python
from legalbot.rag import create_retriever

retriever = await create_retriever(
    provider_name="dashscope",
    api_key="${DASHSCOPE_API_KEY}",
    vector_store="chroma",
    persist_dir="~/.legalbot/chroma",
)
results = await retriever.retrieve(
    query="民间借贷 利息",
    law_area="民法",
    top_k=5,
)
```

### 14.5 创建 LegalOrchestrator

```python
from legalbot.agent.orchestrator import LegalOrchestrator

orchestrator = LegalOrchestrator(
    provider=provider,
    retriever=retriever,
    config=config.tools.orchestrate,
    main_tools=registry.tools_snapshot(),
)
intent = await orchestrator.classify_intent("张三和李四的借款纠纷")
```

### 14.6 创建 LegalDocumentGenerator

```python
from legalbot.document.generator import LegalDocumentGenerator

generator = LegalDocumentGenerator(
    retriever=retriever,
    provider=provider,
    enabled_types=["complaint", "defense", "appeal"],
)

document = await generator.generate(
    doc_type="complaint",
    case_facts="张三借款给李四10万元，约定2024年1月1日还款，但李四至今未还",
    law_areas=["民法"],
)
```

### 14.7 创建 LegalAuditLogger

```python
from legalbot.audit import LegalAuditLogger

logger = LegalAuditLogger(
    audit_dir="~/.legalbot/audit",
    retention_days=90,
    pii_masking=True,
)

event_id = await logger.log(
    event_type="legal_query",
    session_id="cli:test",
    channel="cli",
    query={"original_text": "民间借贷如何起诉？"},
    response={"final_content": "根据《民法典》第675条..."},
)

results = await logger.query(event_type="legal_query", limit=10)
```

---

## 15. 部署指南

### 15.1 环境要求

- Python 3.12+
- uv（包管理器）
- Chroma 持久化存储需要磁盘空间
- LLM API 密钥（DeepSeek/OpenAI 等）

### 15.2 安装

```bash
# 克隆项目
git clone https://github.com/your-org/legalbot.git
cd legalbot

# 安装依赖
uv sync

# 初始化配置
legalbot onboard
```

### 15.3 配置环境变量

```bash
# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# DashScope（嵌入）
export DASHSCOPE_API_KEY="sk-..."

# 飞书（可选）
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
```

### 15.4 构建法律索引

```bash
# 将法律文档放入 legal_data/ 目录
cp your_laws/*.txt legal_data/

# 构建索引
legalbot legal index --docs-dir ./legal_data
```

### 15.5 启动方式

```bash
# 仅 API 服务（无渠道）
legalbot serve --port 8080

# 完整网关（含渠道）
legalbot gateway --port 8081

# 交互式 CLI
legalbot agent
```

### 15.6 Docker 部署（示例）

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["legalbot", "serve", "--port", "8080"]
```

### 15.7 生产环境检查清单

- [ ] 配置 HTTPS/TLS
- [ ] 配置防火墙
- [ ] 配置日志轮转
- [ ] 配置监控告警
- [ ] 配置定期索引更新
- [ ] 配置审计日志备份
- [ ] 配置会话历史备份
- [ ] 测试故障恢复

---

## 附录 A：完整配置字段参考

见第四章"配置系统"。

## 附录 B：工具参数 Schema

```python
# OrchestrateTool
{"type": "object", "properties": {
    "query": {"type": "string", "description": "用户的法律问题或请求"},
    "intent": {"type": "string", "description": "意图类别覆盖", "nullable": True}
}, "required": ["query"]}

# RAGSearchTool
{"type": "object", "properties": {
    "query": {"type": "string"},
    "law_area": {"type": "string", "nullable": True},
    "doc_type": {"type": "string", "nullable": True},
    "top_k": {"type": "integer", "default": 5}
}, "required": ["query"]}

# LegalDocumentGenerateTool
{"type": "object", "properties": {
    "case_facts": {"type": "string"},
    "doc_type": {"type": "string"}
}, "required": ["case_facts", "doc_type"]}

# DebateTool
{"type": "object", "properties": {
    "case_description": {"type": "string"},
    "plaintiff_claims": {"type": "string", "nullable": True},
    "defendant_response": {"type": "string", "nullable": True},
    "debate_rounds": {"type": "integer", "default": 1}
}, "required": ["case_description"]}

# CaseCompareTool
{"type": "object", "properties": {
    "dispute_facts": {"type": "string"}
}, "required": ["dispute_facts"]}
```

## 附录 C：文件编码说明

本项目源代码使用 UTF-8 编码。测试输出在 Windows GBK 终端下可能显示中文乱码，这是终端编码问题，不影响测试逻辑正确性。pytest 内部使用 UTF-8 处理字符串，断言基于正确的字符串值。

---

*文档结束*
