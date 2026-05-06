# legalbot 核心架构文档

**版本**：0.5.0 | **日期**：2026-05-05

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [Agent 主循环](#2-agent-主循环)
3. [上下文构建器](#3-上下文构建器)
4. [工具注册表](#4-工具注册表)
5. [会话管理器](#5-会话管理器)
6. [Agent 执行器](#6-agent-执行器)
7. [Hook 生命周期系统](#7-hook-生命周期系统)
8. [LLM Provider 抽象层](#8-llm-provider-抽象层)
9. [记忆系统与上下文压缩](#9-记忆系统与上下文压缩)
10. [子 Agent 管理器](#10-子-agent-管理器)
11. [消息总线与多渠道接入](#11-消息总线与多渠道接入)
12. [多 Agent 编排器](#12-多-agent-编排器)

---

## 1. 系统架构总览

### 1.1 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLI / API / Gateway                        │
│                   (Typer CLI | aiohttp API server)                │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     MessageBus（消息总线）                         │
│                    legalbot/bus/queue.py                           │
│          asyncio.Queue — 解耦渠道层与 Agent 核心层                  │
│          inbound: asyncio.Queue[InboundMessage]                   │
│          outbound: asyncio.Queue[OutboundMessage]                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                 AgentLoop（Agent 主循环）                          │
│                    legalbot/agent/loop.py                          │
│                                                                   │
│   ┌──────────────┐                                                │
│   │  run() 主循环  │  消费 inbound 消息 → 按 session 分发 →        │
│   │  while self.   │  并发处理（每 session 串行，多 session 并发）  │
│   │  _running:     │                                              │
│   │    msg = await │                                              │
│   │    bus.consume │                                              │
│   └──────┬────────┘                                              │
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────┐│
│   │  _dispatch(msg)                                              ││
│   │  1. 获取 per-session asyncio.Lock（同 session 串行）          ││
│   │  2. 获取 concurrency_gate Semaphore（全局并发控制）           ││
│   │  3. 调用 _process_message(msg)                               ││
│   └─────────────────────────────────────────────────────────────┘│
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────┐│
│   │  _process_message(msg)                                       ││
│   │  1. 从 SessionManager 获取/创建 session                       ││
│   │  2. 恢复未完成轮次的 checkpoint（如有）                       ││
│   │  3. 处理 slash commands（如 /help, /stop）                   ││
│   │  4. Consolidator.maybe_consolidate_by_tokens() 压缩旧消息    ││
│   │  5. ContextBuilder.build_messages() 构建消息列表             ││
│   │  6. _run_agent_loop() 执行 LLM 调用 + 工具执行循环           ││
│   │  7. _save_turn() 保存本轮消息到 session                       ││
│   │  8. 审计日志记录（legal 场景）                                ││
│   │  9. 返回 OutboundMessage                                     ││
│   └─────────────────────────────────────────────────────────────┘│
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────┐│
│   │  _run_agent_loop() → AgentRunner.run(AgentRunSpec)           ││
│   │                                                              ││
│   │  for iteration in range(max_iterations):                     ││
│   │    1. Context governance（截断/压缩历史消息）                 ││
│   │    2. before_iteration() hook                                ││
│   │    3. 调用 LLM（支持 streaming / non-streaming）             ││
│   │    4. 解析 response                                          ││
│   │    5. 如果有 tool_calls → before_execute_tools() hook        ││
│   │       → 执行工具（支持并发/串行）→ 回填 tool results          ││
│   │       → after_iteration() hook → continue                    ││
│   │    6. 如果无 tool_calls → 清洗内容 → 返回最终结果             ││
│   └─────────────────────────────────────────────────────────────┘│
└───────────────┬─────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬──────────────┬──────────────┐
    ▼           ▼           ▼              ▼              ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Context │ │ToolReg │ │ Session  │ │ Memory   │ │ Subagent │
│Builder │ │istry   │ │ Manager  │ │ Store    │ │ Manager  │
└────────┘ └────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 1.2 核心设计原则

| 原则                 | 实现方式                                                              |
| ------------------ | ----------------------------------------------------------------- |
| **无外部 Agent 框架依赖** | AgentLoop、AgentRunner 均为自研，不依赖 LangChain/LangGraph                |
| **异步全链路**          | 基于 asyncio，所有 I/O 操作均为非阻塞                                         |
| **Provider 无关**    | 通过 `LLMProvider` 抽象基类统一 Anthropic/OpenAI/Azure/DeepSeek 等 16+ 提供商 |
| **Session 隔离**     | 每个 `channel:chat_id` 独立 session，同 session 串行，多 session 并发         |
| **可控上下文窗口**        | Token 预算管理 + Consolidator 压缩 + 历史消息截断                             |
| **工具安全**           | 工作区限制（restrict_to_workspace）+ 沙盒执行 + 并发控制                         |
| **容错与重试**          | 瞬时错误自动重试（标准/持久模式），非瞬时错误快速失败                                       |

---

## 2. Agent 主循环

### 2.1 文件位置

`legalbot/agent/loop.py` — `AgentLoop` 类，约 950 行

### 2.2 关键数据结构

```python
class AgentLoop:
    """
    职责：
    1. 消费消息总线的 inbound 消息
    2. 按 session 分发处理（同 session 串行，不同 session 并发）
    3. 协调 ContextBuilder、ToolRegistry、AgentRunner 的协作
    4. 管理 MCP 连接、审计日志、slash commands
    """
```

### 2.3 核心属性

| 属性             | 类型                | 说明                                         |
| -------------- | ----------------- | ------------------------------------------ |
| `bus`          | `MessageBus`      | 异步消息总线，解耦渠道层和核心层                           |
| `provider`     | `LLMProvider`     | LLM 提供商实例（Anthropic/OpenAI/Azure 等）        |
| `context`      | `ContextBuilder`  | 系统提示词 + 消息列表构建器                            |
| `tools`        | `ToolRegistry`    | 工具注册表，管理所有可用工具                             |
| `runner`       | `AgentRunner`     | 无状态的执行器，负责 LLM 调用 + 工具执行循环                 |
| `sessions`     | `SessionManager`  | 会话管理器，持久化对话历史                              |
| `subagents`    | `SubagentManager` | 后台子 Agent 管理器（用于 spawn 工具）                 |
| `consolidator` | `Consolidator`    | Token 预算驱动的上下文压缩器                          |
| `commands`     | `CommandRouter`   | Slash command 路由器（/help, /stop, /memory 等） |

### 2.4 run() — 主事件循环

```python
async def run(self) -> None:
    self._running = True
    await self._connect_mcp()

    while self._running:
        msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)

        # 优先级命令（如 /stop）在主循环中直接处理
        if self.commands.is_priority(raw):
            result = await self.commands.dispatch_priority(ctx)
            await self.bus.publish_outbound(result)
            continue

        # 普通消息：创建后台 Task，不阻塞消息消费
        task = asyncio.create_task(self._dispatch(msg))
        self._active_tasks[session_key].append(task)
```

**设计要点**：

- 使用 `asyncio.wait_for(timeout=1.0)` 而非无限阻塞，使 `stop()` 能够在 1 秒内生效
- 每条消息通过 `asyncio.create_task()` 在后台处理，主循环立即回到 `consume_inbound()` 等待下一条
- 通过 `self._active_tasks` 追踪所有进行中的任务，用于 `/stop` 取消和优雅关闭

### 2.5 _dispatch() — 并发控制

```python
async def _dispatch(self, msg: InboundMessage) -> None:
    lock = await self._get_session_lock(msg.session_key)  # 每 session 一把锁
    gate = self._concurrency_gate  # 全局并发信号量（默认 3）
    async with lock, gate:
        response = await self._process_message(msg)
        await self.bus.publish_outbound(response)
```

**并发模型**：

```
Session A ──→ [Lock A] ──→ [Gate (3)] ──→ 处理
Session A ──→ [Lock A] ──→ 等待 Lock ... (串行)
Session B ──→ [Lock B] ──→ [Gate (3)] ──→ 处理  (并发)
Session C ──→ [Lock C] ──→ [Gate (3)] ──→ 处理  (并发)
Session D ──→ [Lock D] ──→ 等待 Gate ... (第4个并发请求)
```

- **同 Session 串行**：确保对话历史的一致性，不会出现"B 的消息在 A 的回复前插入"的问题
- **跨 Session 并发**：不同用户/渠道的消息可以并行处理
- **全局并发上限**：`legalbot_MAX_CONCURRENT_REQUESTS` 环境变量（默认 3），防止 LLM API 速率限制

### 2.6 _process_message() — 单条消息处理

```python
async def _process_message(self, msg):
    # 1. 获取或创建 Session
    session = self.sessions.get_or_create(session_key)

    # 2. 恢复未完成的轮次（power-loss recovery）
    self._restore_runtime_checkpoint(session)

    # 3. 处理 slash commands
    if result := await self.commands.dispatch(ctx):
        return result

    # 4. Token 预算检查，必要时压缩旧消息
    await self.consolidator.maybe_consolidate_by_tokens(session)

    # 5. 构建消息列表
    history = session.get_history(max_messages=0)
    messages = self.context.build_messages(
        history=history,
        current_message=msg.content,
        channel=msg.channel, chat_id=msg.chat_id,
    )

    # 6. 执行 Agent 循环
    final_content, tools_used, all_msgs, stop_reason = \
        await self._run_agent_loop(messages, session=session, ...)

    # 7. 保存本轮消息到 session
    self._save_turn(session, all_msgs, skip=1 + len(history))

    # 8. 审计日志
    await audit_logger.log(event_type=..., query=..., response=...)

    # 9. 触发后台 Consolidation
    self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

    return OutboundMessage(channel=..., chat_id=..., content=final_content)
```

### 2.7 Runtime Checkpoint — 异常中断恢复

当 Agent 正在执行工具调用时（如 LLM 返回了 tool_calls 但工具尚未执行完毕），系统会通过 session metadata 保存一个 checkpoint：

```python
checkpoint = {
    "phase": "awaiting_tools",       # 或 "tools_completed" / "final_response"
    "iteration": N,
    "assistant_message": {...},      # LLM 返回的 assistant 消息
    "completed_tool_results": [...], # 已执行完成的工具结果
    "pending_tool_calls": [...],     # 尚未执行的工具调用
}
```

下次同一 session 收到消息时，`_restore_runtime_checkpoint()` 会：
1. 将已完成的消息（assistant_message + completed_tool_results）追加到 session 历史
2. 为未执行的 tool_calls 插入合成的错误标记（"Task interrupted"）
3. 新请求可以从前一个中断点正常继续对话

---

## 3. 上下文构建器

### 3.1 文件位置

`legalbot/agent/context.py` — `ContextBuilder` 类

### 3.2 职责

将分散的上下文信息（系统提示词、记忆、技能、对话历史、用户消息）组装为 LLM 期望的消息列表格式。

### 3.3 build_system_prompt() — 系统提示词构建

系统提示词由以下部分组成（用 `\n\n---\n\n` 分隔）：

```
┌─────────────────────────────────────────────┐
│  1. Identity（身份声明）                      │
│     - workspace 路径                         │
│     - 运行时环境（OS, Python 版本）            │
│     - 平台策略（路径分隔符、Shell 语法）        │
│     - 渠道标识                                │
├─────────────────────────────────────────────┤
│  2. Bootstrap Files（工作区引导文件）          │
│     - AGENTS.md                              │
│     - SOUL.md                                │
│     - USER.md                                │
│     - TOOLS.md                               │
│     (如果文件存在于 workspace 目录)            │
├─────────────────────────────────────────────┤
│  3. Memory Context（长期记忆）                │
│     来自 memory/MEMORY.md                     │
├─────────────────────────────────────────────┤
│  4. Always-On Skills（常驻技能）              │
│     标记为 "always" 的技能定义               │
├─────────────────────────────────────────────┤
│  5. Skills Summary（可用技能列表）            │
│     所有已安装技能的摘要和触发条件            │
├─────────────────────────────────────────────┤
│  6. Recent History（近期历史摘要）            │
│     来自 memory/history.jsonl               │
│     取最近 50 条未处理的历史摘要              │
└─────────────────────────────────────────────┘
```

**示例系统提示词结构**：

```markdown
You are legalbot, a knowledgeable AI assistant.
Workspace: /home/user/projects/mybot
Runtime: Linux x86_64, Python 3.12.0
Platform: Linux (path separators: /)

---

## AGENTS.md
[项目配置和说明]

---

# Memory

## Long-term Memory
[长期记忆内容]

---

# Active Skills
[已激活的技能内容]

---

# Available Skills
- legal-debate: 法律辩论分析
- legal-document-draft: 法律文书起草
...

---

# Recent History
- [2026-05-05 10:00] User asked about contract review...
- [2026-05-05 09:30] Agent provided case analysis...
```

### 3.4 build_messages() — 消息列表构建

```python
def build_messages(
    self,
    history: list[dict],         # 从 session 获取的历史消息
    current_message: str,         # 当前用户输入
    skill_names: list[str] | None,
    media: list[str] | None,      # 图片路径列表
    channel: str | None,
    chat_id: str | None,
    current_role: str = "user",
) -> list[dict]:
```

**组装逻辑**：

1. **构建 Runtime Context** — 注入当前时间、渠道、聊天 ID 等运行时元数据
2. **构建 User Content** — 如果是纯文本直接使用；如果有图片则构建 `[{type: "image_url", ...}, {type: "text", ...}]` 格式
3. **合并 Runtime Context 和 User Content** — 避免连续两条同 role 的消息（部分 Provider 拒绝）
4. **解决连续同 role 消息** — 如果历史最后一条也是 `user`，将新消息内容 merge 进去

关键设计：**图片处理**
- 从文件读取原始字节，检测 MIME type（通过 magic bytes，fallback 到扩展名）
- Base64 编码后构造 `data:image/png;base64,...` 格式的 `image_url` block
- 在 block 中附加 `_meta.path` 元数据（持久化时会被剥离）

### 3.5 轻量级模板系统

ContextBuilder 不依赖 Jinja2 做系统提示词渲染，而是使用自定义的 `render_template()`：

```python
# 模板：legalbot/templates/agent/identity.md
# 变量：workspace_path, runtime, platform_policy, channel

from legalbot.utils.prompt_templates import render_template
return render_template("agent/identity.md",
    workspace_path=workspace_path,
    runtime=runtime,
    platform_policy=...,
    channel=channel,
)
```

模板文件位于 `legalbot/templates/`，使用 `{variable_name}` 语法，编译为 `.pyc` 时通过 hatch 打包进 wheel。

---

## 4. 工具注册表

### 4.1 文件位置

- `legalbot/agent/tools/registry.py` — `ToolRegistry` 类
- `legalbot/agent/tools/base.py` — `Tool` 抽象基类 + `Schema` 验证基类

### 4.2 Tool 抽象基类

```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...  # JSON Schema

    @property
    def read_only(self) -> bool:        # 无副作用，可并发
        return False

    @property
    def concurrency_safe(self) -> bool: # 可与其他安全工具并发
        return self.read_only and not self.exclusive

    @property
    def exclusive(self) -> bool:       # 必须独占执行
        return False

    @abstractmethod
    async def execute(self, **kwargs) -> Any: ...

    def cast_params(self, params: dict) -> dict:
        """Schema 驱动的参数类型转换（LLM 可能返回字符串而非整数等）"""

    def validate_params(self, params: dict) -> list[str]:
        """JSON Schema 验证，返回错误列表（空 = 通过）"""

    def to_schema(self) -> dict:
        """生成 OpenAI function-calling 格式的 tool definition"""
```

### 4.3 已有工具一览

| 工具名 | 文件 | read_only | 功能 |
|--------|------|-----------|------|
| `read_file` | `filesystem.py` | yes | 读取文件内容 |
| `write_file` | `filesystem.py` | no | 写入文件 |
| `edit_file` | `filesystem.py` | no | 精确字符串替换编辑 |
| `list_dir` | `filesystem.py` | yes | 列出目录内容 |
| `glob` | `search.py` | yes | 文件名模式匹配 |
| `grep` | `search.py` | yes | 正则表达式内容搜索 |
| `exec` | `shell.py` | no | 执行 Shell 命令 |
| `web_search` | `web.py` | yes | 网络搜索（DuckDuckGo） |
| `web_fetch` | `web.py` | yes | 获取网页内容 |
| `message` | `message.py` | no | 向渠道发送消息 |
| `spawn` | `spawn.py` | no | 创建后台子 Agent |
| `cron` | `cron.py` | no | 管理定时任务 |
| `mcp_*` | `mcp.py` | 依协议 | MCP 协议工具（动态注册） |
| `legal_rag_search` | `rag.py` | yes | 法律 RAG 检索 |
| `multi_step_reason` | `reasoner.py` | yes | 多步法律推理 |
| `legal_document_generate` | `document.py` | no | 法律文书生成 |
| `orchestrate` | `orchestrate.py` | yes | 多 Agent 编排 |
| `legal_debate` | `debate.py` | yes | 法律辩论分析 |
| `case_compare` | `case_compare.py` | yes | 案例对比分析 |
| `feedback_collect` | `feedback.py` | no | 收集用户反馈 |

### 4.4 ToolRegistry 核心方法

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict]:
        """生成 tool definitions，保证稳定排序以利用 prompt cache。
        内置工具按名称排序在前，MCP 工具按名称排序在后。
        这样即使 MCP 工具变化，内置工具的 schema 位置不变，
        Anthropic prompt cache 的 cache_breakpoint 仍然有效。"""

    def prepare_call(self, name, params) -> tuple[Tool, cast_params, error]:
        """三步：查找工具 → 类型转换 → 参数验证。返回 (tool, params, error)"""

    async def execute(self, name, params) -> Any:
        """执行工具调用。如果工具抛出异常，统一包装为 ToolError"""
```

**缓存友好的 tool definitions 排序**：

```
[read_file, edit_file, exec, glob, grep, ..., mcp_server1_toolA, mcp_server2_toolB]
└─────────── 内置工具（稳定排序）──────────┘  └──── MCP 工具（稳定排序）────┘
                   ↑                                    ↑
            cache_breakpoint                       cache_breakpoint
```

---

## 5. 会话管理器

### 5.1 文件位置

`legalbot/session/manager.py` — `Session` 和 `SessionManager` 类

### 5.2 Session 数据结构

```python
@dataclass
class Session:
    key: str                          # "channel:chat_id" 如 "dingtalk:user123"
    messages: list[dict]              # 完整对话历史
    created_at: datetime
    updated_at: datetime
    metadata: dict                    # 扩展元数据（含 runtime checkpoint）
    last_consolidated: int            # 已压缩的消息数量
```

### 5.3 存储格式：JSONL

```
{会话目录}/dingtalk_user123.jsonl
────────────────────────────────
{"_type": "metadata", "key": "dingtalk:user123", "created_at": "...", "updated_at": "...", "metadata": {...}, "last_consolidated": 42}
{"role": "system", "content": "..."}
{"role": "user", "content": "你好"}
{"role": "assistant", "content": "你好！有什么可以帮助你的？"}
{"role": "user", "content": "帮我查一下合同法"}
{"role": "assistant", "content": null, "tool_calls": [{"id": "...", "function": {"name": "legal_rag_search", "arguments": "..."}}]}
{"role": "tool", "tool_call_id": "...", "name": "legal_rag_search", "content": "[检索结果...]"}
{"role": "assistant", "content": "根据合同法的规定..."}
```

**为什么选 JSONL 而非 SQLite**：
- 人类可读，方便调试
- append-only 写入，无并发写入冲突
- 按行读取，支持流式处理
- 不需要额外依赖

### 5.4 get_history() — 历史消息提取

```python
def get_history(self, max_messages: int = 500) -> list[dict]:
    # 1. 只取 last_consolidated 之后的消息（已压缩的不再重复送入 LLM）
    unconsolidated = self.messages[self.last_consolidated:]

    # 2. 截取最近 max_messages 条
    sliced = unconsolidated[-max_messages:]

    # 3. 对齐到 user 消息边界（不以 mid-turn 开始）
    for i, msg in enumerate(sliced):
        if msg.get("role") == "user":
            sliced = sliced[i:]
            break

    # 4. 丢弃开头的孤儿 tool 结果（没有对应 assistant tool_calls 的 tool role 消息）
    start = find_legal_message_start(sliced)
    if start:
        sliced = sliced[start:]

    # 5. 只返回 role + content + tool_call_id + name + reasoning_content
    return [{"role": m["role"], "content": m.get("content", ""), ...} for m in sliced]
```

### 5.5 SessionManager 缓存策略

```python
class SessionManager:
    def __init__(self, workspace: Path):
        self._cache: dict[str, Session] = {}  # 内存缓存

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]           # 命中缓存
        session = self._load(key)              # 从磁盘加载 JSONL
        if session is None:
            session = Session(key=key)         # 新建 session
        self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        # 写入磁盘（全量覆盖 JSONL）+ 更新内存缓存
```

注意：`Session` 对象是可变引用。同一个 key 的后续请求拿到的是同一个 `Session` 实例，修改直接反映到内存中，调用 `save()` 时全量写回磁盘。

### 5.6 兼容迁移

`_load()` 会自动检测并迁移旧版全局路径 `~/.legalbot/sessions/` 到当前 workspace 的 `sessions/` 目录。

---

## 6. Agent 执行器

### 6.1 文件位置

`legalbot/agent/runner.py` — `AgentRunner` 类，约 720 行

### 6.2 设计哲学

`AgentRunner` 是一个**无状态的纯执行器**，不持有任何产品层配置（渠道、消息总线、审计等）。它只关心一件事：**给定消息和工具，循环调用 LLM 直到得到最终答案**。

`AgentLoop` 负责"产品层"（消息路由、session 管理、审计、streaming）
`AgentRunner` 负责"执行层"（LLM 调用、工具执行、重试、上下文治理）

### 6.3 AgentRunSpec — 执行配置

```python
@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict]       # 初始消息（含 system prompt + history + user message）
    tools: ToolRegistry                # 可用工具注册表
    model: str                         # 模型标识符
    max_iterations: int                # 最大 tool calling 轮次（默认 25）
    max_tool_result_chars: int         # 单次工具结果最大字符数（超出截断）
    temperature: float | None
    max_tokens: int | None
    reasoning_effort: str | None       # 推理深度（用于支持 reasoning 的模型）
    hook: AgentHook | None             # 生命周期回调
    concurrent_tools: bool             # 是否并发执行工具
    session_key: str | None
    context_window_tokens: int | None  # 上下文窗口 token 预算
    provider_retry_mode: str           # "standard" | "persistent"
    checkpoint_callback: Callable | None
```

### 6.4 run() — 核心执行循环

```
                     ┌──────────────────┐
                     │  iteration = 0   │
                     └────────┬─────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Context Governance            │
              │  1. backfill 缺失的 tool results│
              │  2. microcompact 旧的工具结果   │
              │  3. apply tool result budget   │
              │  4. snip_history (token 预算)  │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  before_iteration() hook       │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  调用 LLM                      │
              │  - 有 streaming hook → stream  │
              │  - 无 streaming hook → 普通调用 │
              └───────────────┬───────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
          has_tool_calls?          无 tool_calls
                    │                    │
                    ▼                    ▼
    ┌──────────────────────┐   ┌──────────────────┐
    │ before_execute_tools │   │ 内容清洗          │
    │ _execute_tools()      │   │ - 空白内容重试    │
    │ - 并发/串行执行        │   │ - length 恢复     │
    │ - 错误隔离            │   │ - 错误处理        │
    │ 检查点发射             │   └────────┬─────────┘
    │ after_iteration()     │            │
    └──────────┬───────────┘            ▼
               │               ┌──────────────────┐
               ▼               │ 返回 AgentRunResult│
          iteration += 1       │  - final_content   │
          continue             │  - messages        │
                               │  - tools_used      │
                               │  - usage           │
                               │  - stop_reason     │
                               └──────────────────┘
```

### 6.5 Context Governance — 上下文治理

每轮迭代开始前，AgentRunner 执行 4 步上下文治理：

**第一步：backfill 缺失的 tool results**

如果因为某种原因（会话恢复、API 重试、中间件截断等），assistant 消息中有 `tool_calls` 但后面缺少对应的 `tool` role 消息，自动插入合成错误结果：

```python
{"role": "tool", "tool_call_id": "...", "name": "...",
 "content": "[Tool result unavailable — call was interrupted or lost]"}
```

**第二步：microcompact — 旧工具结果压缩**

针对"可压缩"类型的工具（`read_file`, `exec`, `grep`, `glob`, `web_search`, `web_fetch`, `list_dir`），保留最近 10 条完整结果，更早的替换为一行摘要：

```python
# 原文可能是 5000 字符的文件内容
"{'name': 'read_file', 'content': '[read_file result omitted from context]'}"
```

**第三步：tool result budget — 单条结果截断**

对超过 `max_tool_result_chars`（默认 20000 字符）的 tool result 进行截断，保留头尾：

```python
truncate_text(content, max_chars)  # "前10000字符...\n\n[...truncated...]\n\n...后10000字符"
```

**第四步：snip_history — Token 预算截断**

计算当前消息列表的 token 估计值，如果超出 `context_window_tokens - max_output_tokens - 1024` 安全预算，则从历史头部开始丢弃消息：

1. System messages 始终保留
2. 从最近的消息向回累积，直到接近预算上限
3. 确保截断后第一条是 `user` role（不从中途 tool call 开始）
4. 处理孤儿 tool results
5. 极端情况下保留最后 4 条消息作为兜底

### 6.6 内容清洗与恢复

```
LLM 返回 content
        │
        ▼
  is_blank_text(clean)?
   ├── 是 → 空白重试（最多 2 次）
   │         └── 仍然空白 → finalization retry
   │              （不带 tools 的最后一次请求："请用纯文本回答以下问题"）
   │
   ├── finish_reason == "length"?
   │   └── 是 → length 恢复（最多 3 次）
   │          追加 "你被中断了，请继续" 消息，继续循环
   │
   ├── finish_reason == "error"?
   │   └── 是 → 返回错误消息
   │
   └── 正常 → 返回最终内容
```

### 6.7 工具并发执行

当 LLM 一次返回多个 tool calls 时：

```python
def _partition_tool_batches(spec, tool_calls):
    """
    将 tool_calls 分组：
    - concurrency_safe 的工具可以合批并发
    - 非安全工具单独串行
    """
```

示例：LLM 返回 `[read_file(A), read_file(B), exec(cmd), grep(pattern)]`
→ 分为 `[[read_file(A), read_file(B)], [exec(cmd)], [grep(pattern)]]`
→ 第一个 batch 并发执行两个 read_file，然后串行 exec，最后 grep

### 6.8 错误隔离

每个工具的执行错误都被捕获并转化为用户友好的错误消息返回给 LLM（而非让整个 agent 崩溃）：

```python
try:
    result = await tool.execute(**params)
except Exception as e:
    return f"Error: {type(e).__name__}: {e}"
```

---

## 7. Hook 生命周期系统

### 7.1 文件位置

`legalbot/agent/hook.py` — `AgentHook`、`AgentHookContext`、`CompositeHook`

### 7.2 设计模式：Template Method Pattern

`AgentRunner.run()` 在自己的执行循环中依次调用 hook 方法，业务逻辑通过子类化 `AgentHook` 注入：

```
迭代开始
    │
    ▼
before_iteration(context)    ← 可用于注入额外的 system message
    │
    ▼
LLM 调用
    │
    ├── streaming: on_stream(context, delta)  ← 每个文本增量
    │
    ▼
┌─ 有 tool_calls ──────────────────────────────────┐
│  on_stream_end(context, resuming=True)            │
│  before_execute_tools(context)   ← 记录工具调用    │
│  执行工具...                                       │
│  after_iteration(context)        ← 记录 token 用量 │
│  continue                                         │
└──────────────────────────────────────────────────┘
│
┌─ 无 tool_calls ──────────────────────────────────┐
│  on_stream_end(context, resuming=False)           │
│  finalize_content(context, content) ← 清洗内容     │
│  after_iteration(context)                         │
│  break                                            │
└──────────────────────────────────────────────────┘
```

### 7.3 AgentHookContext — 每轮状态

```python
@dataclass(slots=True)
class AgentHookContext:
    iteration: int                    # 当前迭代次数
    messages: list[dict]              # 当前消息列表
    response: LLMResponse | None      # LLM 响应
    usage: dict[str, int]             # token 用量
    tool_calls: list[ToolCallRequest]  # 待执行的工具调用
    tool_results: list[Any]           # 工具执行结果
    tool_events: list[dict]           # 工具执行事件（name, status, detail）
    final_content: str | None         # 最终回复
    stop_reason: str | None           # 停止原因
    error: str | None                 # 错误信息
```

### 7.4 CompositeHook — 多 Hook 组合

```python
class CompositeHook(AgentHook):
    """
    按注册顺序依次调用多个 hook。
    - async 方法（before_iteration, before_execute_tools 等）：
      错误隔离 —— 一个 hook 抛异常不影响其他 hook
      但标记了 reraise=True 的 hook 会向上传播错误
    - finalize_content：管道模式 —— 前一个的输出是后一个的输入
    """
```

### 7.5 _LoopHook — 主循环内置 Hook

`_LoopHook` 是 AgentLoop 内部使用的核心 hook，负责：
1. **流式输出**：将 LLM 文本增量通过 `on_stream` 回调传递
2. **进度通知**：将思考内容（strip `<think>` 后）和工具调用提示发送给用户
3. **Token 日志**：每轮结束后记录 prompt/completion/cached token 用量
4. **内容清洗**：通过 `finalize_content` 移除 `<think>...</think>` 块

---

## 8. LLM Provider 抽象层

### 8.1 文件位置

- `legalbot/providers/base.py` — `LLMProvider` 抽象基类
- `legalbot/providers/anthropic_provider.py` — Anthropic Claude
- `legalbot/providers/openai_compat_provider.py` — OpenAI 及兼容 API（DeepSeek, 通义千问, GLM 等）
- `legalbot/providers/azure_openai_provider.py` — Azure OpenAI
- `legalbot/providers/registry.py` — 16+ Provider 的元数据注册表

### 8.2 LLMProvider 抽象接口

```python
class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages, tools, model, ...) -> LLMResponse: ...

    async def chat_stream(self, messages, tools, model, ...,
                          on_content_delta: Callable[[str], Awaitable]) -> LLMResponse:
        """默认实现：fallback 到非 streaming chat，
        将完整内容作为单次 delta 发送。支持原生流式的 Provider 应 override"""

    async def chat_with_retry(self, ...) -> LLMResponse:
        """带重试的非流式调用"""

    async def chat_stream_with_retry(self, ...) -> LLMResponse:
        """带重试的流式调用"""
```

### 8.3 统一数据模型

所有 Provider 的实现都将各自的原生响应转换为统一的 `LLMResponse`：

```python
@dataclass
class LLMResponse:
    content: str | None                     # 文本内容
    tool_calls: list[ToolCallRequest]       # 工具调用
    finish_reason: str                      # "stop" | "length" | "tool_calls" | "error"
    usage: dict[str, int]                   # {prompt_tokens, completion_tokens, cached_tokens}
    reasoning_content: str | None           # Kimi/DeepSeek-R1 等推理内容
    thinking_blocks: list[dict] | None      # Anthropic extended thinking
    # 结构化错误元数据
    error_status_code: int | None
    error_kind: str | None
    error_type: str | None
    error_code: str | None
    error_retry_after_s: float | None
    error_should_retry: bool | None

@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict:
        """序列化为 OpenAI 兼容格式"""
```

### 8.4 重试策略

```python
# 重试延迟（指数退避）：1s → 2s → 4s
_CHAT_RETRY_DELAYS = (1, 2, 4)

# 瞬时错误判断：基于 metadata 优先，fallback 文本匹配
_TRANSIENT_ERROR_MARKERS = (
    "429", "rate limit", "500", "502", "503", "504",
    "overloaded", "timeout", "connection", "server error",
)

# 不可重试的 429 错误（欠费/配额耗尽）
_NON_RETRYABLE_429_ERROR_TOKENS = {
    "insufficient_quota", "billing_hard_limit_reached",
    "insufficient_balance", "payment_required",
}
```

两种重试模式：

| 模式 | 行为 |
|------|------|
| `standard` | 最多重试 3 次（1s → 2s → 4s），用完后返回最后一次错误响应 |
| `persistent` | 无限重试（最长延迟 60s），直到相同错误出现 10 次才放弃 |

重试期间支持 **heartbeat 进度回调**：每 30 秒通过 `on_retry_wait` 通知用户当前重试状态。

### 8.5 消息兼容层

Provider 基类提供了多个静态方法，供子类在发送请求前调用，确保不同 Provider 的消息格式兼容：

| 方法 | 功能 |
|------|------|
| `_sanitize_empty_content()` | 修复空 content（部分 Provider 拒绝空字符串）、剥离 `_meta` 字段 |
| `_enforce_role_alternation()` | 合并连续同 role 消息、移除末尾 assistant 消息（部分 Provider 不支持 prefilling） |
| `_strip_image_content()` | 将 image_url block 替换为文本占位符（不支持多模态的 Provider） |
| `_sanitize_request_messages()` | 仅保留 Provider 认可的 message key（如 Anthropic 不需要 `name` 字段） |

### 8.6 已支持的 Provider（16+）

通过 `legalbot/providers/registry.py` 注册，按 `backend` 分类：

| backend | 提供商示例 |
|---------|-----------|
| `anthropic` | Claude (claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5) |
| `openai_compat` | OpenAI (gpt-5.2, gpt-5.1), DeepSeek (deepseek-chat, deepseek-reasoner), 通义千问 (qwen-max), GLM (glm-4.7), Kimi (kimi-k2.5), MiniMax (minimax-m2.5), StepFun (step-3), 豆包 (doubao-1.5), Zhipu (GLM) |
| `azure_openai` | Azure OpenAI Service |

---

## 9. 记忆系统与上下文压缩

### 9.1 文件位置

- `legalbot/agent/memory.py` — `MemoryStore` 和 `Consolidator`

### 9.2 MemoryStore — 纯文件 I/O 层

```
workspace/
├── memory/
│   ├── MEMORY.md          ← 长期记忆（Agent 可读写）
│   ├── history.jsonl      ← 对话历史摘要（append-only JSONL）
│   └── .cursor            ← 历史游标（记录最后一条的 cursor ID）
├── SOUL.md                ← Agent 人格定义（模板文件）
└── USER.md                ← 用户信息（模板文件）
```

**MEMORY.md**：Agent 通过 tool calling 可以写入的长期记忆。包含 frontmatter 格式的结构化条目，由 `MEMORY.md` 索引文件和各 `.md` 记忆文件组成。

**history.jsonl**：压缩后的对话历史摘要。每行一条 JSON：

```json
{"cursor": 1, "timestamp": "2026-05-05 10:00", "content": "用户询问了关于劳动合同解除的法律问题..."}
{"cursor": 2, "timestamp": "2026-05-05 10:15", "content": "Agent 检索了《劳动合同法》第39-42条..."}
```

**兼容迁移**：`MemoryStore` 在初始化时会检测旧版 `HISTORY.md` 格式并自动迁移到 `history.jsonl`。

### 9.3 Consolidator — Token 预算驱动的上下文压缩

**问题**：长对话的 session history 会超过 LLM 的 context window。

**方案**：当检测到 prompt token 估算超过安全预算（`context_window - max_completion_tokens - 1024`），自动触发压缩：

```
┌──────────────────────────────────────────────────────────────┐
│  Session.messages                                             │
│  ┌──────────────────────┬──────────────────────────────────┐ │
│  │ 已压缩部分             │ 未压缩部分（Unconsolidated）        │ │
│  │ (last_consolidated=42) │ 42..150 = 108 条消息              │ │
│  │ 这些消息已经从 LLM      │ 这些消息还在 LLM 的 context 中     │ │
│  │ context 中移除          │                                  │ │
│  └──────────────────────┴──────────────────────────────────┘ │
│                            ↑                                  │
│            Consolidator 在这里找 user-turn 边界               │
│            如果 prompt tokens > budget:                        │
│              1. 从 start 向 end 扫描，找 user-turn 边界       │
│              2. 将 [last_consolidated : boundary] 发送给 LLM  │
│              3. LLM 总结为一句摘要                             │
│              4. 摘要写入 history.jsonl                        │
│              5. last_consolidated 推进到 boundary              │
│              6. 重复直到 prompt tokens < target (budget/2)    │
└──────────────────────────────────────────────────────────────┘
```

**边界选择**：只在 user 消息处截断，保证不会剪断 assistant + tool_calls + tool_results 的关联链。

**归档 LLM 调用**：使用独立的轻量级 LLM 调用（consolidator_archive system prompt），不占用 agent 的 context window。

**防并发**：每个 session 使用 `weakref.WeakValueDictionary` 存储 `asyncio.Lock`，确保同一 session 的 consolidate 和 process_message 不交错。

---

## 10. 子 Agent 管理器

### 10.1 文件位置

`legalbot/agent/subagent.py` — `SubagentManager` 类

### 10.2 用途

主 Agent 在对话中可以通过 `spawn` 工具创建后台子 Agent，执行长时间任务（如代码生成、大量文件搜索）而无需阻塞主对话。子 Agent 完成后通过系统消息通知主 Agent。

### 10.3 架构

```
用户消息 "帮我重构 utils/ 目录下的所有文件"
    │
    ▼
主 Agent (AgentLoop)
    │
    ├── LLM 决定: spawn(label="重构utils", task="读取并重构 utils/*.py")
    │
    ▼
SubagentManager.spawn()
    │
    ├── asyncio.create_task(_run_subagent(...))
    │
    ▼
子 Agent (独立 AgentRunner)
    ├── 自己的 ToolRegistry (read_file, write_file, exec, grep, glob, web_search...)
    ├── 自己的系统提示词 (subagent_system.md)
    ├── 独立的 15 轮 max_iterations
    └── 完成后 → MessageBus.publish_inbound(系统消息)
            │
            ▼
        主 Agent 收到系统消息 "Subagent [重构utils] completed successfully: [结果...]"
```

### 10.4 关键设计

- **独立执行**：子 Agent 拥有自己的 `AgentRunner` 和 `ToolRegistry`，与主 Agent 完全隔离
- **白名单工具**：`spawn_with_config()` 支持指定 `allowed_tools`，限制子 Agent 的可用工具
- **自定义提示词**：可通过 `system_prompt` 参数覆盖默认子 Agent 提示词
- **结果通知**：子 Agent 完成后通过 `MessageBus.publish_inbound()` 发送系统消息，主 Agent 将其作为对话上下文的一部分处理
- **Session 清理**：`cancel_by_session()` 支持按 session 批量取消子 Agent
- **部分失败展示**：即使子 Agent 中途失败（tool_error），也会总结已完成步骤和失败原因并通知主 Agent

---

## 11. 消息总线与多渠道接入

### 11.1 文件位置

- `legalbot/bus/queue.py` — `MessageBus`（异步消息队列）
- `legalbot/bus/events.py` — `InboundMessage` 和 `OutboundMessage` 数据结构
- `legalbot/channels/` — 渠道实现（CLI, 钉钉, 飞书, QQ, WebSocket）

### 11.2 MessageBus — 解耦层

```python
class MessageBus:
    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
```

极简的发布-订阅模型：两个无界 `asyncio.Queue`。渠道层向 `inbound` 发布用户消息，Agent 核心层消费并处理后向 `outbound` 发布响应，各渠道 worker 消费出站消息并投递到对应的聊天平台。

### 11.3 InboundMessage / OutboundMessage

```python
@dataclass
class InboundMessage:
    channel: str              # "cli" | "dingtalk" | "feishu" | "qq" | "websocket" | "system"
    sender_id: str            # 发送者标识
    chat_id: str              # 会话标识
    content: str              # 消息文本
    session_key: str          # 默认为 "{channel}:{chat_id}"
    media: list[str] | None   # 媒体文件路径
    metadata: dict | None     # 扩展元数据（流式标记等）
```

`OutboundMessage` 结构类似，含 `content` + `metadata`（用于流式控制标记 `_stream_delta`, `_stream_end`, `_progress` 等）。

### 11.4 渠道流式输出的实现

AgentLoop 在处理消息时，根据 `msg.metadata["_wants_stream"]` 决定是否启用流式输出：

```python
async def on_stream(delta: str) -> None:
    await self.bus.publish_outbound(OutboundMessage(
        content=delta,
        metadata={"_stream_delta": True, "_stream_id": stream_id},
    ))

async def on_stream_end(*, resuming: bool = False) -> None:
    await self.bus.publish_outbound(OutboundMessage(
        content="",
        metadata={"_stream_end": True, "_resuming": resuming, "_stream_id": stream_id},
    ))
```

渠道层根据 `_stream_delta` / `_stream_end` 标记决定是累积显示还是分段发送。

### 11.5 已有渠道

| 渠道 | 文件 | 协议 |
|------|------|------|
| CLI | `legalbot/cli/` | Typer + prompt_toolkit |
| 钉钉 | `channels/dingtalk.py` | dingtalk-stream SDK |
| 飞书 | `channels/feishu.py` | lark-oapi SDK |
| QQ | `channels/qq.py` | qq-botpy SDK |
| WebSocket | `channels/websocket.py` | websockets |
| API | `legalbot/api/server.py` | aiohttp HTTP server |

---

## 12. 多 Agent 编排器

### 12.1 文件位置

- `legalbot/agent/orchestrator.py` — `LegalOrchestrator`
- `legalbot/agent/tools/orchestrate.py` — `OrchestrateTool`

### 12.2 编排流程

```
用户法律问题
    │
    ▼
OrchestrateTool.execute()
    │
    ▼
LegalOrchestrator.classify_intent(query)
    ├── 两阶段分类:
    │   1. 如果是法律问题 → 判断复杂度: simple / complex
    │   2. 识别意图: legal_query / contract_review / case_search / debate
    │      / case_compare / document_draft / general
    │
    ▼
根据意图路由:
┌────────────────────┬──────────────────────────────────┐
│ 意图               │ 路由目标                          │
├────────────────────┼──────────────────────────────────┤
│ simple legal_query │ 直接 RAG 检索 + 一次 LLM 回答    │
│ complex legal_query│ MultiStepLegalReasoner (多步推理) │
│ contract_review    │ 合同审查 Agent                    │
│ case_search        │ 案例检索 Agent                    │
│ debate             │ LegalDebateTool (原告/被告/法官)   │
│ case_compare       │ CaseCompareTool (结构化对比表)    │
│ document_draft     │ LegalDocumentGenerator            │
│ general            │ 回退到主 Agent                    │
└────────────────────┴──────────────────────────────────┘
```

### 12.3 意图分类

两阶段 LLM 调用（轻量级、低延迟）：

**阶段 1**（仅对法律类）：
```
分析以下法律问题的复杂程度。
- simple: 仅需单条法律条文即可回答
- complex: 需要引用多条法律条文、逻辑推导
返回: simple 或 complex
```

**阶段 2**：
```
分析用户输入意图:
- legal_query / contract_review / case_search / debate
- case_compare / document_draft / general
返回: 意图类别名
```

### 12.4 与主 Agent 的集成

当 orchestrator 启用时，`AgentLoop._register_default_tools()` 会：
1. 注册 `OrchestrateTool` 作为顶层工具
2. 从主 Agent 移除 `legal_rag_search` 工具（防止直接调用绕过编排器）
3. 注册关联工具（`DebateTool`, `CaseCompareTool`）

这确保了所有法律类查询都必须通过编排器路由，不会被主 Agent 直接处理。

---

## 附录 A：完整生命周期示意图

```
用户发送消息
     │
     ▼
Channel Layer 构造 InboundMessage → bus.publish_inbound()
     │
     ▼
AgentLoop.run()
  while loop:
    msg = bus.consume_inbound()
    asyncio.create_task(_dispatch(msg))
     │
     ▼
_dispatch(msg)
  获取 session lock + concurrency gate
     │
     ▼
_process_message(msg)
  ├── session = sessions.get_or_create()
  ├── 恢复 checkpoint（如有）
  ├── slash command 处理
  ├── Consolidator.maybe_consolidate_by_tokens()
  ├── context.build_messages(history, current_message)
  ├── _run_agent_loop(messages)
  │     │
  │     ▼
  │   AgentRunner.run(AgentRunSpec)
  │     loop:
  │       ├── Context Governance (backfill → microcompact → trunc → snip)
  │       ├── before_iteration() hook
  │       ├── provider.chat_stream_with_retry()  ←─→  LLM API
  │       ├── 解析 LLMResponse
  │       ├── 有 tool_calls:
  │       │   ├── on_stream_end(resuming=True)
  │       │   ├── before_execute_tools() hook
  │       │   ├── _execute_tools() → 并发/串行 → tool.execute()
  │       │   ├── after_iteration() hook
  │       │   └── continue
  │       └── 无 tool_calls:
  │           ├── finalize_content() hook
  │           ├── 空白检测 → finalization retry
  │           ├── after_iteration() hook
  │           └── return AgentRunResult
  │
  ├── _save_turn(session, messages)
  ├── 审计日志记录
  ├── 后台 trigger Consolidator
  └── return OutboundMessage → bus.publish_outbound()
     │
     ▼
Channel Layer 投递响应到用户
```

## 附录 B：关键文件索引

| 文件 | 行数 | 核心类/函数 |
|------|------|------------|
| `legalbot/agent/loop.py` | ~950 | `AgentLoop`, `_LoopHook` |
| `legalbot/agent/runner.py` | ~720 | `AgentRunner`, `AgentRunSpec`, `AgentRunResult` |
| `legalbot/agent/context.py` | ~196 | `ContextBuilder` |
| `legalbot/agent/tools/registry.py` | ~128 | `ToolRegistry` |
| `legalbot/agent/tools/base.py` | ~280 | `Tool`, `Schema` |
| `legalbot/agent/hook.py` | ~104 | `AgentHook`, `AgentHookContext`, `CompositeHook` |
| `legalbot/agent/memory.py` | ~485 | `MemoryStore`, `Consolidator` |
| `legalbot/agent/subagent.py` | ~375 | `SubagentManager` |
| `legalbot/agent/orchestrator.py` | ~430 | `LegalOrchestrator` |
| `legalbot/session/manager.py` | ~237 | `Session`, `SessionManager` |
| `legalbot/bus/queue.py` | ~45 | `MessageBus` |
| `legalbot/providers/base.py` | ~704 | `LLMProvider`, `LLMResponse`, `ToolCallRequest` |
| `legalbot/legalbot.py` | ~175 | `legalbot` (SDK facade) |
