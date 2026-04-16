# 法智 (LegalBot) — 垂类法律 Agent MVP 设计文档

> 基于 nanobot 框架改造，以最小侵入方式引入 RAG + MultiAgent 能力

---

## 1. 产品定义

### 1.1 定位

法智是一个面向法律从业者和普通用户的垂类法律 AI 助手，核心能力：

- **法律知识检索**：精准检索法规、司法解释、指导性案例
- **法律问答**：基于检索结果回答法律问题，引用具体法条
- **合同审查**：识别合同条款中的法律风险
- **案例分析**：从判例中提取裁判要旨和法律适用逻辑

### 1.2 MVP 范围

| 能力 | MVP 包含 | 后续迭代 |
|------|---------|---------|
| 法规检索（RAG） | ✅ | - |
| 法律问答（引用法条） | ✅ | - |
| 合同风险点识别 | ✅ | 合同修订建议 |
| 案例检索 | ✅ | 案例对比分析 |
| MultiAgent 编排 | ✅ 意图路由 + 2个专业 Agent | 辩论模式、多轮协作 |
| 法律文书起草 | ❌ | ✅ |
| 多轮复杂推理 | ❌ | ✅ |

### 1.3 非目标

- 不替代律师出具正式法律意见
- 不处理实时法规更新（MVP 阶段知识库静态）
- 不做诉讼程序性指导

---

## 2. 架构总览

```
用户输入 (Channel / CLI / API)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  AgentLoop (现有，微调)                           │
│  ┌───────────────────────────────────────────┐  │
│  │  ContextBuilder                          │  │
│  │  - SOUL.md → 法律助手人设                  │  │
│  │  - AGENTS.md → 法律行为规范                │  │
│  │  - Skills → legal-research, legal-cite    │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  ToolRegistry (扩展)                      │  │
│  │  ├── 原有: read_file, web_search, exec... │  │
│  │  ├── NEW: legal_rag_search    ← RAG检索   │  │
│  │  └── NEW: legal_orchestrate   ← Agent调度  │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  SubagentManager (扩展)                   │  │
│  │  ├── 法律检索 Agent (legal_research)       │  │
│  │  └── 合同审查 Agent (contract_review)      │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
┌──────────────────┐  ┌──────────────────────┐
│  RAG 模块 (新增)  │  │  法律知识库 (新增)     │
│  - Embedding      │  │  - 法律法规           │
│  - VectorStore    │  │  - 司法解释           │
│  - Chunker        │  │  - 指导性案例         │
│  - Retriever      │  │  - 合同模板           │
│  - Reranker       │  │                      │
└──────────────────┘  └──────────────────────┘
```

---

## 3. 模块详细设计

### 3.1 RAG 模块

**新增目录**：`nanobot/rag/`

```
nanobot/rag/
├── __init__.py
├── embeddings.py       # Embedding 客户端
├── vectorstore.py      # 向量存储
├── chunker.py          # 法律文档分块器
├── loader.py           # 文档加载器
├── retriever.py        # 检索器（向量 + BM25 混合）
├── reranker.py         # 重排序
└── indexer.py          # 索引构建与管理
```

#### 3.1.1 Embedding 客户端 — `embeddings.py`

```python
class EmbeddingClient(ABC):
    """Embedding 抽象接口"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def dim(self) -> int:
        ...


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI / 兼容接口的 Embedding"""

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: str | None = None, api_base: str | None = None):
        # 支持任意 OpenAI 兼容端点（如 DashScope、SiliconFlow）
        ...


class LocalEmbeddingClient(EmbeddingClient):
    """本地 Embedding（sentence-transformers）"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        ...
```

**选型**：

| 方案     | 模型                            | 维度   | 适用场景        |
| ------ | ----------------------------- | ---- | ----------- |
| 远程 API | `text-embedding-3-small`      | 1536 | 快速验证、无需 GPU |
| 远程 API | DashScope `text-embedding-v3` | 1024 | 国内访问稳定      |
| 本地部署   | BGE-M3                        | 1024 | 数据隐私、离线使用   |
| 本地部署   | bce-embedding-base_v1         | 768  | 轻量本地部署      |

#### 3.1.2 向量存储 — `vectorstore.py`

```python
class VectorStore(ABC):
    """向量存储抽象接口"""

    @abstractmethod
    async def add(self, ids: list[str], vectors: list[list[float]],
                  metadatas: list[dict], documents: list[str]) -> None:
        ...

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 5,
                     filter: dict | None = None) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB 实现 — MVP 默认选择"""

    def __init__(self, persist_dir: Path, collection_name: str = "legal_kb"):
        import chromadb
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(collection_name)
        ...
```

**MVP 选型**：ChromaDB

- 纯 Python，无外部依赖服务
- 支持持久化存储（`PersistentClient`）
- 内置过滤（按 `law_area`、`doc_type` 等 metadata 过滤）
- 后续可平滑迁移到 Milvus / Qdrant

#### 3.1.3 法律文档分块器 — `chunker.py`

法律文档不能按固定 token 切分，需按条文结构切：

```python
class LegalChunker:
    """法律文档语义分块器 — 按条文/款/项结构切分"""

    # 法律文本结构模式
    ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+条")
    PARAGRAPH_PATTERN = re.compile(r"^[一二三四五六七八九十]+[、.]")  # 款
    ITEM_PATTERN = re.compile(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[)）]")  # 项

    def __init__(self, max_chunk_tokens: int = 800, overlap_tokens: int = 100):
        ...

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """分块策略：
        1. 优先按「第X条」切分
        2. 超长条文按「款」或 token 上限二次切分
        3. 保留上下文重叠（overlap）
        4. 每个 chunk 携带元数据：law_name, article_no, chapter, law_area
        """
        ...
```

**Chunk 元数据结构**：

```python
from typing import TypedDict

class ChunkMeta(TypedDict):
    law_name: str          # 法规名称，如"中华人民共和国民法典"
    article_no: str        # 条号，如"第五百八十三条"
    chapter: str           # 篇章，如"第三编 合同"
    section: str           # 节，如"第二节 合同的效力"
    law_area: str          # 法律领域，如"民法/合同法"
    doc_type: str          # 文档类型：law/judicial_interpretation/case/contract_template
    effective_date: str    # 生效日期
    source: str            # 来源

@dataclass
class Chunk:
    id: str
    text: str
    metadata: ChunkMeta
```

#### 3.1.4 检索器 — `retriever.py`

```python
class LegalRetriever:
    """混合检索器：向量检索 + BM25 + 重排序"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        bm25_store: BM25Store | None = None,
        reranker: Reranker | None = None,
    ):
        ...

    async def retrieve(
        self,
        query: str,
        law_area: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        检索流程：
        1. 向量检索 top_k * 3 候选
        2. BM25 检索 top_k * 3 候选
        3. 合并去重
        4. Reranker 精排
        5. 返回 top_k 结果
        """
        # Step 1: 向量检索
        query_vec = await self.embedding_client.embed([query])
        vector_results = await self.vector_store.search(
            query_vec[0], top_k=top_k * 3,
            filter=self._build_filter(law_area, doc_type),
        )
        # Step 2: BM25 检索
        bm25_results = self.bm25_store.search(query, top_k=top_k * 3) if self.bm25_store else []
        # Step 3: 合并去重
        candidates = self._merge_and_dedup(vector_results, bm25_results)
        # Step 4: 重排序
        if self.reranker:
            candidates = await self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]
```

**BM25 为什么要加**：法律术语精确匹配（如"善意第三人""不可抗力"）BM25 优于向量检索，混合检索可提升召回率 15-30%。

#### 3.1.5 重排序 — `reranker.py`

```python
class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, candidates: list, top_k: int) -> list:
        ...

class BGEReranker(Reranker):
    """BGE-Reranker-v2-m3 — 中文法律文本重排效果好"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        ...

class CohereReranker(Reranker):
    """Cohere Rerank API — 无需本地 GPU"""

    def __init__(self, api_key: str):
        ...
```

#### 3.1.6 文档加载器 — `loader.py`

```python
class LegalDocumentLoader:
    """法律文档加载器"""

    async def load_pdf(self, path: Path) -> list[RawDocument]:
        """解析法律 PDF（PyMuPDF + OCR fallback）"""
        ...

    async def load_html(self, url: str) -> list[RawDocument]:
        """从法律法规网站爬取（北大法宝、裁判文书网等）"""
        ...

    async def load_docx(self, path: Path) -> list[RawDocument]:
        """解析 Word 格式合同/文书"""
        ...
```

#### 3.1.7 索引管理 — `indexer.py`

```python
class LegalIndexer:
    """索引构建与管理"""

    def __init__(self, loader: LegalDocumentLoader, chunker: LegalChunker,
                 embedding_client: EmbeddingClient, vector_store: VectorStore):
        ...

    async def build_index(self, data_dir: Path) -> IndexStats:
        """全量构建索引"""
        ...

    async def incremental_update(self, data_dir: Path, since: datetime) -> IndexStats:
        """增量更新（按文件修改时间）"""
        ...
```

### 3.2 RAG Tool — 接入现有 Tool 体系

**新增文件**：`nanobot/agent/tools/rag_search.py`

```python
from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, IntegerSchema, tool_parameters_schema
from nanobot.rag.retriever import LegalRetriever


@tool_parameters(tool_parameters_schema(
    query=StringSchema("法律问题或关键词"),
    law_area=StringSchema("法律领域：民法/刑法/商法/劳动法/行政法等", required=False),
    doc_type=StringSchema("文档类型：law/judicial_interpretation/case", required=False),
    top_k=IntegerSchema(1, description="返回结果数", minimum=1, maximum=10, default=5),
    required=["query"],
))
class RAGSearchTool(Tool):
    """法律知识库检索工具 — 注册到 ToolRegistry"""

    name = "legal_rag_search"
    description = "搜索法律知识库，检索相关法规、司法解释和案例。输入法律问题或关键词，返回最相关的法律条文和案例。"

    @property
    def read_only(self) -> bool:
        return True

    def __init__(self, retriever: LegalRetriever):
        self.retriever = retriever

    async def execute(self, query: str, law_area: str | None = None,
                      doc_type: str | None = None, top_k: int = 5) -> str:
        results = await self.retriever.retrieve(
            query, law_area=law_area, doc_type=doc_type, top_k=top_k
        )
        return self._format_results(query, results)

    @staticmethod
    def _format_results(query: str, results: list) -> str:
        if not results:
            return f"未检索到与「{query}」相关的法律条文。"
        lines = [f"检索结果（{query}）：\n"]
        for i, r in enumerate(results, 1):
            meta = r.metadata
            lines.append(f"{i}. 【{meta.get('law_name', '未知')}】{meta.get('article_no', '')}")
            lines.append(f"   领域：{meta.get('law_area', '')} | 类型：{meta.get('doc_type', '')}")
            lines.append(f"   {r.text[:300]}")
            if len(r.text) > 300:
                lines.append("   ...")
            lines.append("")
        return "\n".join(lines)
```

**注册方式** — 修改 `nanobot/agent/loop.py:229`：

```python
# 在 AgentLoop._register_default_tools() 中新增：
if self.rag_config and self.rag_config.enable:
    from nanobot.agent.tools.rag_search import RAGSearchTool
    from nanobot.rag import create_retriever
    retriever = create_retriever(self.rag_config)
    self.tools.register(RAGSearchTool(retriever=retriever))
```

### 3.3 配置扩展

**修改**：`nanobot/config/schema.py`

```python
class RAGConfig(Base):
    """RAG 法律知识库配置"""

    enable: bool = False
    persist_dir: str = "~/.nanobot/legal_kb"  # 向量库持久化目录
    embedding_provider: str = "openai"  # openai / dashscope / local
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    vector_store: str = "chroma"  # chroma / milvus
    reranker: str = ""  # "" (关闭) / bge / cohere
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_api_key: str = ""
    bm25_enable: bool = True
    top_k: int = 5
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100


class OrchestrateConfig(Base):
    """MultiAgent 编排配置"""

    enable: bool = False
    intent_model: str = ""  # 意图分类模型，空则用默认模型
    agents: dict[str, AgentDefConfig] = {}  # 专业 Agent 定义


class AgentDefConfig(Base):
    """专业 Agent 定义"""

    system_prompt: str = ""
    tools: list[str] = []  # 允许使用的工具名列表
    model: str = ""  # 空则继承默认


# 修改 ToolsConfig
class ToolsConfig(Base):
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    ssrf_whitelist: list[str] = Field(default_factory=list)
    rag: RAGConfig = Field(default_factory=RAGConfig)               # 新增
    orchestrate: OrchestrateConfig = Field(default_factory=OrchestrateConfig)  # 新增
```

**配置文件示例** (`~/.nanobot/config.json`)：

```json
{
  "agents": {
    "defaults": {
      "model": "deepseek/deepseek-chat",
      "workspace": "~/.nanobot/workspace"
    }
  },
  "tools": {
    "rag": {
      "enable": true,
      "persist_dir": "~/.nanobot/legal_kb",
      "embedding_provider": "dashscope",
      "embedding_model": "text-embedding-v3",
      "embedding_api_key": "sk-xxx",
      "reranker": "bge",
      "bm25_enable": true,
      "top_k": 5
    },
    "orchestrate": {
      "enable": true,
      "agents": {
        "legal_research": {
          "system_prompt": "你是法律检索专家...",
          "tools": ["legal_rag_search", "web_search", "read_file"]
        },
        "contract_review": {
          "system_prompt": "你是合同审查专家...",
          "tools": ["legal_rag_search", "read_file", "edit_file"]
        }
      }
    }
  }
}
```

### 3.4 MultiAgent 编排

#### 3.4.1 意图分类

**新增文件**：`nanobot/agent/orchestrator.py`

```python
class LegalOrchestrator:
    """法律 MultiAgent 编排器"""

    INTENT_PROMPT = """分析以下用户输入的意图，返回最匹配的类别：

类别：
- legal_query: 法律问题咨询（需检索法条后回答）
- contract_review: 合同/协议审查
- case_search: 案例检索
- general: 通用对话（不需要专业 Agent）

用户输入：{query}

只返回类别名，不要解释。"""

    def __init__(self, provider: LLMProvider, subagent_mgr: SubagentManager,
                 config: OrchestrateConfig):
        self.provider = provider
        self.subagents = subagent_mgr
        self.config = config

    async def classify_intent(self, query: str) -> str:
        """用 LLM 分类用户意图"""
        prompt = self.INTENT_PROMPT.format(query=query)
        # 单次 LLM 调用，不进 Agent Loop
        ...

    async def dispatch(self, query: str, context: dict) -> str | None:
        """根据意图路由到合适的 Agent 或编排多 Agent"""
        if not self.config.enable:
            return None

        intent = await self.classify_intent(query)

        if intent == "general":
            return None  # 交给主 Agent 处理

        if intent == "legal_query":
            return await self._single_agent("legal_research", query, context)

        if intent == "contract_review":
            return await self._contract_review_flow(query, context)

        if intent == "case_search":
            return await self._single_agent("legal_research", query, context)

        return None

    async def _single_agent(self, agent_name: str, task: str, context: dict) -> str:
        """调度单个专业 Agent"""
        agent_def = self.config.agents[agent_name]
        # 扩展 SubagentManager，支持传入自定义 system_prompt 和 tool 白名单
        return await self.subagents.spawn_with_config(
            task=task,
            system_prompt=agent_def.system_prompt,
            allowed_tools=agent_def.tools,
            model=agent_def.model or None,
        )

    async def _contract_review_flow(self, query: str, context: dict) -> str:
        """合同审查流程：检索 → 审查"""
        # Step 1: 检索相关法规
        research = await self._single_agent(
            "legal_research",
            f"检索与以下合同相关的法律法规：\n{query}",
            context,
        )
        # Step 2: 基于检索结果审查合同
        review = await self._single_agent(
            "contract_review",
            f"基于以下法律检索结果：\n{research}\n\n审查合同内容：\n{query}",
            context,
        )
        return review
```

#### 3.4.2 SubagentManager 扩展

修改 `nanobot/agent/subagent.py`，新增 `spawn_with_config` 方法：

```python
class SubagentManager:
    # ... 现有代码 ...

    async def spawn_with_config(
        self,
        task: str,
        system_prompt: str | None = None,
        allowed_tools: list[str] | None = None,
        model: str | None = None,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """生成具有自定义 prompt 和工具集的子 Agent"""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent_with_config(
                task_id, task, display_label, origin,
                system_prompt=system_prompt,
                allowed_tools=allowed_tools,
                model=model,
            )
        )
        self._running_tasks[task_id] = bg_task
        # ... cleanup 逻辑同 spawn() ...
        return f"Subagent [{display_label}] started (id: {task_id})."

    async def _run_subagent_with_config(
        self, task_id, task, label, origin,
        system_prompt=None, allowed_tools=None, model=None,
    ):
        """执行自定义配置的子 Agent"""
        tools = ToolRegistry()

        # 按 allowed_tools 白名单注册工具
        available_tools = self._build_available_tools()
        if allowed_tools:
            for name in allowed_tools:
                if tool := available_tools.get(name):
                    tools.register(tool)
        else:
            for tool in available_tools.values():
                tools.register(tool)

        # 使用自定义 system_prompt
        prompt = system_prompt or self._build_subagent_prompt()

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task},
        ]

        result = await self.runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=tools,
            model=model or self.model,
            max_iterations=15,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=_SubagentHook(task_id),
            max_iterations_message="Task completed but no final response was generated.",
            error_message=None,
            fail_on_tool_error=True,
        ))
        # ... 结果处理同 _run_subagent() ...
```

#### 3.4.3 编排 Tool

**新增文件**：`nanobot/agent/tools/orchestrate.py`

```python
@tool_parameters(tool_parameters_schema(
    query=StringSchema("用户原始问题"),
    intent=StringSchema("意图类别：legal_query/contract_review/case_search", required=False),
    required=["query"],
))
class OrchestrateTool(Tool):
    """MultiAgent 编排工具 — 允许主 Agent 主动调度专业 Agent"""

    name = "legal_orchestrate"
    description = "调度法律专业 Agent 团队处理复杂法律任务。可根据任务类型自动路由到法律检索Agent或合同审查Agent。"

    def __init__(self, orchestrator: LegalOrchestrator):
        self.orchestrator = orchestrator

    @property
    def exclusive(self) -> bool:
        return True  # 编排任务独占执行

    async def execute(self, query: str, intent: str | None = None) -> str:
        context = {}
        result = await self.orchestrator.dispatch(query, context)
        return result or "未能调度专业 Agent，请直接使用 legal_rag_search 工具检索。"
```

### 3.5 领域定制

#### 3.5.1 SOUL.md — 法律助手人设

**修改**：`nanobot/templates/SOUL.md`

```markdown
# Soul

我是「法智」，一个专业的法律智能助手。

我的核心原则：
1. **引用先行** — 回答法律问题必须引用具体法条，注明法律全称和条文号
2. **区分边界** — 明确区分"法律规定"与"学理解释"，不作超越法律的推论
3. **时效意识** — 涉及时效性问题时，注明法规的生效/修订日期
4. **风险提示** — 始终提醒：AI 法律建议仅供参考，不替代正式律师意见
5. **精准用词** — 使用规范法律术语，避免口语化表述造成歧义

我擅长：
- 检索法律法规、司法解释和指导性案例
- 识别合同条款的法律风险
- 分析法律适用逻辑和裁判规则

我不擅长的边界：
- 不提供诉讼策略建议
- 不替代律师出具法律意见书
- 不处理实时法规变化
```

#### 3.5.2 法律领域 Skills

**新增目录**：`nanobot/skills/legal-research/`

```markdown
# nanobot/skills/legal-research/SKILL.md
---
name: legal-research
description: 法律知识检索与法条引用技能
always: false
---

## 检索策略

当用户提出法律问题时，按以下步骤检索：

1. **识别法律领域** — 判断问题属于民法/刑法/商法/劳动法/行政法等
2. **构造检索 query** — 提取核心法律概念，去除口语化表述
   - 差："公司不给加班费怎么办"
   - 优："加班费 劳动报酬 用人单位 支付义务"
3. **使用 legal_rag_search 检索** — 传入 law_area 缩小范围
4. **验证法条有效性** — 检查是否已被修订或废止
5. **交叉引用** — 相关法条之间建立关联

## 引用规范

引用法条格式：`《法律全称》第X条第X款`

示例：
- 《中华人民共和国民法典》第五百八十三条
- 《中华人民共和国劳动合同法》第三十一条第一款
- 《最高人民法院关于适用〈中华人民共和国民法典〉合同编通则若干问题的解释》第一条
```

**新增目录**：`nanobot/skills/legal-citation/`

```markdown
# nanobot/skills/legal-citation/SKILL.md
---
name: legal-citation
description: 法律引用规范技能
always: true
---

## 法条引用规则

1. 首次引用法律须使用全称，后续可用简称
2. 引用条文顺序：条 → 款 → 项
3. 引用司法解释须注明文号
4. 引用案例须注明案号和审理法院

## 免责声明

每次回答法律问题的末尾必须附加：
> 以上分析基于当前知识库中的法律法规，仅供参考，不构成正式法律意见。如需正式法律意见，请咨询执业律师。
```

---

## 4. 集成改动清单

以下列出所有需要修改的现有文件：

| 文件 | 改动 | 说明 |
|------|------|------|
| `nanobot/config/schema.py` | 新增 `RAGConfig`、`OrchestrateConfig`、`AgentDefConfig`；修改 `ToolsConfig` | 配置扩展 |
| `nanobot/agent/loop.py` | `_register_default_tools()` 中注册 `RAGSearchTool` 和 `OrchestrateTool` | 接入新 Tool |
| `nanobot/agent/loop.py` | `__init__` 中初始化 `LegalOrchestrator` | 编排器 |
| `nanobot/agent/subagent.py` | 新增 `spawn_with_config()` 和 `_run_subagent_with_config()` | 支持 Agent 定制 |
| `nanobot/templates/SOUL.md` | 替换为法律助手人设 | 领域定制 |
| `pyproject.toml` | 新增 `legal` optional dependency 组 | 依赖管理 |

**新增文件**：

| 文件 | 说明 |
|------|------|
| `nanobot/rag/__init__.py` | RAG 包入口 + `create_retriever()` 工厂 |
| `nanobot/rag/embeddings.py` | Embedding 客户端 |
| `nanobot/rag/vectorstore.py` | 向量存储 |
| `nanobot/rag/chunker.py` | 法律文档分块器 |
| `nanobot/rag/loader.py` | 文档加载器 |
| `nanobot/rag/retriever.py` | 混合检索器 |
| `nanobot/rag/reranker.py` | 重排序 |
| `nanobot/rag/indexer.py` | 索引管理 |
| `nanobot/agent/tools/rag_search.py` | RAG 检索 Tool |
| `nanobot/agent/tools/orchestrate.py` | 编排 Tool |
| `nanobot/agent/orchestrator.py` | MultiAgent 编排器 |
| `nanobot/skills/legal-research/SKILL.md` | 法律检索技能 |
| `nanobot/skills/legal-citation/SKILL.md` | 法条引用技能 |
| `tests/rag/` | RAG 模块测试 |
| `tests/agent/test_orchestrator.py` | 编排器测试 |

---

## 5. 依赖管理

**新增 optional dependency 组**（`pyproject.toml`）：

```toml
[project.optional-dependencies]
legal = [
    "chromadb>=0.5.0,<1.0.0",
    "rank-bm25>=0.2.2,<1.0.0",
    "jieba>=0.42.1,<1.0.0",
    "PyMuPDF>=1.25.0,<2.0.0",
    "sentence-transformers>=3.0.0,<4.0.0",  # 本地 Embedding（可选）
]
```

安装方式：`pip install nanobot-ai[legal]`

核心依赖（chromadb、rank-bm25、jieba、PyMuPDF）放在 `legal` optional 组，不污染 nanobot 核心。

---

## 6. 数据源与知识库构建

### 6.1 MVP 数据源

| 数据类型 | 来源 | 数量（MVP） | 格式 |
|---------|------|------------|------|
| 法律法规 | 国家法律法规数据库 (flk.npc.gov.cn) | 50 部核心法律 | PDF/HTML |
| 司法解释 | 最高人民法院官网 | 30 件常用解释 | PDF/HTML |
| 指导性案例 | 最高人民法院案例库 | 100 个指导案例 | HTML |
| 合同模板 | 自建 | 20 份典型合同 | DOCX |

### 6.2 知识库构建命令

新增 CLI 命令：

```bash
# 构建索引
nanobot legal index --data-dir ./legal_data --rebuild

# 增量更新
nanobot legal index --data-dir ./legal_data

# 查看索引状态
nanobot legal index-status
```

实现方式：在 `nanobot/cli/commands.py` 中注册 `legal` 子命令组。

---

## 7. 测试计划

### 7.1 单元测试

| 模块                          | 测试内容                   |
| --------------------------- | ---------------------- |
| `rag/chunker.py`            | 法条切分准确性、超长条文二次切分、元数据提取 |
| `rag/retriever.py`          | 向量检索、BM25 检索、混合检索、过滤   |
| `rag/reranker.py`           | 重排序逻辑                  |
| `agent/tools/rag_search.py` | Tool 参数验证、结果格式化        |
| `agent/orchestrator.py`     | 意图分类、Agent 路由          |

### 7.2 集成测试

| 场景 | 输入 | 预期 |
|------|------|------|
| 法律问答 | "用人单位不签劳动合同怎么办" | 引用《劳动合同法》第82条 |
| 法条检索 | "搜索合同违约金的相关规定" | 返回《民法典》第585条等 |
| 合同审查 | 提供一份租赁合同 | 识别风险条款并引用相关法条 |
| 意图路由 | "帮我看看这份合同" | 路由到 contract_review Agent |
| 通用对话 | "今天天气怎样" | 不触发 RAG，主 Agent 正常回答 |

### 7.3 评测指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 检索 Recall@5 | ≥ 80% | 前5条结果包含正确法条 |
| 检索 MRR | ≥ 0.7 | 正确法条的平均排名 |
| 法条引用准确率 | ≥ 90% | 引用的法条确实存在且相关 |
| 意图分类 F1 | ≥ 0.85 | 正确路由到专业 Agent |

---

## 8. 实施步骤

```
Phase 1: RAG 基础层
├── 1.1 创建 nanobot/rag/ 包结构
├── 1.2 实现 EmbeddingClient（OpenAI 兼容接口）
├── 1.3 实现 ChromaVectorStore
├── 1.4 实现 LegalChunker（法条结构切分）
├── 1.5 实现 LegalRetriever（向量 + BM25）
├── 1.6 实现 RAGSearchTool + 注册
└── 1.7 集成测试：法条检索可用

Phase 2: 知识库构建
├── 2.1 实现文档加载器（PDF/HTML）
├── 2.2 采集 MVP 数据源（50部法律 + 30件解释 + 100案例）
├── 2.3 实现索引构建 CLI 命令
├── 2.4 构建索引 + 质量验证
└── 2.5 端到端测试：法律问答可用

Phase 3: MultiAgent 编排
├── 3.1 扩展 SubagentManager（spawn_with_config）
├── 3.2 实现 LegalOrchestrator（意图分类 + 路由）
├── 3.3 定义 legal_research / contract_review Agent
├── 3.4 实现 OrchestrateTool + 注册
└── 3.5 集成测试：合同审查流程可用

Phase 4: 领域定制
├── 4.1 替换 SOUL.md 为法律助手人设
├── 4.2 编写 legal-research / legal-citation Skills
├── 4.3 新增 RAGConfig / OrchestrateConfig 到配置
└── 4.4 完整回归测试
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 法律文本分块不准 | 检索结果断章取义 | 按条文结构切分 + overlap 保留上下文 |
| BM25 中文分词差 | 法律术语召回不足 | 使用 jieba + 自定义法律词典 |
| 向量模型中文效果差 | 语义检索不准 | 选用 BGE-M3/bce-embedding（中文优化） |
| LLM 幻觉法条 | 引用不存在的法条 | RAG 结果硬约束 + Skill 中要求验证 |
| 意图分类错误 | Agent 路由不准 | 分类结果低置信度时 fallback 到主 Agent |

---

## 10. 后续迭代方向

- **法律文书起草**：起诉状、答辩状、代理词模板 + 填充
- **案例对比分析**：多案例裁判规则对比
- **法规更新追踪**：定时爬取 + 增量索引
- **Agent 辩论模式**：原告 Agent vs 被告 Agent，锻炼论证能力
- **多轮复杂推理**：Chain-of-Thought + 法条链式推理
- **用户反馈闭环**：标注检索不准 / 回答有误 → 优化 chunker 和 reranker
