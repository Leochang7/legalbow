# 用户反馈闭环 (User Feedback Loop)

> LegalBot MVP 扩展功能实施计划

---

## 一、功能概述与目标

### 1.1 功能描述

用户反馈闭环是一个数据/运营导向的功能，用于收集用户对 RAG 检索结果的评价，分析低质量检索的原因，并指导 chunker 和 reranker 的优化方向。

### 1.2 目标

- **即时**：允许用户标记检索结果 helpful/unhelpful，并提供文字修正
- **短期**：定期分析反馈数据，识别系统性问题
- **长期**：将反馈数据用于 chunker/reranker 调优

### 1.3 非目标

- 不做实时在线学习/模型更新
- 不做复杂的 NLP 反馈解析
- 不替代人工评估

---

## 二、架构设计

### 2.1 高层数据流

```
用户交互 (CLI/API)
         │
         ▼
┌─────────────────────────────────┐
│  反馈收集层                      │
│  - CLI 显式反馈命令              │
│  - 工具反馈                     │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  反馈存储层                      │
│  ~/.legalbot/feedback/           │
│  - 每日 JSONL 文件              │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  分析流水线（离线）              │
│  - 识别错误模式                 │
└─────────────────────────────────┘
```

### 2.2 模块结构

```
legalbot/
├── feedback/
│   ├── __init__.py
│   ├── models.py              # FeedbackRecord dataclass
│   ├── storage.py             # JSONL 文件存储
│   ├── collector.py           # FeedbackCollector
│   ├── analyzer.py            # FeedbackAnalyzer
│   └── cli.py                 # CLI 命令
└── docs/
    └── POST_MVP_FEEDBACK_LOOP.md
```

---

## 三、反馈数据模型

### 3.1 反馈记录 JSON Schema

```json
{
  "id": "fb-2026-04-17-abc123",
  "timestamp": "2026-04-17T10:30:00+08:00",
  "type": "helpful",
  "query": {
    "text": "劳动合同到期不续约有没有赔偿",
    "law_area": "劳动法",
    "doc_type": "law"
  },
  "results": [
    {
      "rank": 1,
      "chunk_id": "a1b2c3d4e5f6",
      "law_name": "中华人民共和国劳动合同法",
      "article_no": "第四十六条",
      "score": 0.95,
      "helpful": true
    }
  ],
  "correction": null,
  "session_id": "cli:user123",
  "channel": "cli",
  "model_version": "text-embedding-3-small",
  "reranker_enabled": true,
  "latency_ms": 234
}
```

**反馈类型**：`helpful`、`unhelpful`、`correction`

### 3.2 存储格式

```
~/.legalbot/feedback/
├── metadata.json
├── 2026/
│   └── 04/
│       ├── feedback-2026-04-01.jsonl
│       ├── feedback-2026-04-02.jsonl
│       └── ...
└── reports/
    └── 2026-04-week12-report.md
```

JSONL 文件中每一行是一个有效的 JSON 对象。

---

## 四、API / 接口设计

### 4.1 CLI 命令

| 命令 | 描述 |
|---------|-------------|
| `legalbot feedback rate --result-id <id> --helpful` | 评分 |
| `legalbot feedback correct --result-id <id> --text "..."` | 纠正 |
| `legalbot feedback list --limit 10` | 列出最近反馈 |
| `legalbot feedback export --from 2026-04-01 --to 2026-04-17 --output feedback.jsonl` | 导出 |
| `legalbot feedback analyze --period week` | 分析并生成报告 |

### 4.2 工具反馈

```python
class FeedbackTool(Tool):
    """提交检索结果反馈"""
    name = "legal_feedback"

    async def execute(
        self,
        result_id: str,
        helpful: bool | None = None,
        correction: str | None = None,
        reason: str | None = None,
    ) -> str:
        ...
```

### 4.3 内联反馈（RAG 检索工具）

在 RAG 检索结果后追加反馈引导：

```
检索结果：
1. 【劳动合同法】第四十六条...
2. 【劳动合同法】第四十四条...

以上结果对您有帮助吗？
- 有帮助 (+)  /  不太准确 (-)  /  纠正 [输入纠正内容]
```

---

## 五、核心类

### 5.1 models.py

**文件**：`legalbot/feedback/models.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class QueryInfo:
    text: str
    law_area: str | None = None
    doc_type: str | None = None


@dataclass
class ChunkResult:
    rank: int
    chunk_id: str
    law_name: str
    article_no: str
    score: float
    helpful: bool | None = None
    comment: str | None = None


@dataclass
class CorrectionInfo:
    chunk_id: str
    corrected_text: str
    reason: str | None = None


@dataclass
class FeedbackRecord:
    id: str
    timestamp: datetime
    type: Literal["helpful", "unhelpful", "correction"]
    query: QueryInfo
    results: list[ChunkResult] = field(default_factory=list)
    correction: CorrectionInfo | None = None
    session_id: str | None = None
    channel: str = "cli"
    model_version: str = ""
    reranker_enabled: bool = False
    latency_ms: int | None = None
```

### 5.2 storage.py

**文件**：`legalbot/feedback/storage.py`

```python
from pathlib import Path
from datetime import datetime
import json

from legalbot.feedback.models import FeedbackRecord


class FeedbackStorage:
    """基于 JSONL 的反馈存储。"""

    def __init__(self, feedback_dir: Path | None = None):
        self._feedback_dir = feedback_dir or self._default_dir()
        self._ensure_dir()

    def _default_dir(self) -> Path:
        return Path.home() / ".legalbot" / "feedback"

    def _ensure_dir(self) -> None:
        self._feedback_dir.mkdir(parents=True, exist_ok=True)

    def _daily_file(self, date: datetime) -> Path:
        return self._feedback_dir / date.strftime("%Y") / date.strftime("%m") / f"feedback-{date.strftime('%Y-%m-%d')}.jsonl"

    async def append(self, record: FeedbackRecord) -> None:
        file_path = self._daily_file(record.timestamp)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self._record_to_dict(record), ensure_ascii=False)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def query(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        feedback_type: str | None = None,
    ) -> list[FeedbackRecord]:
        records = []
        for file_path in self._feedback_dir.rglob("feedback-*.jsonl"):
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = self._dict_to_record(json.loads(line))
                        if since and record.timestamp < since:
                            continue
                        if until and record.timestamp > until:
                            continue
                        if feedback_type and record.type != feedback_type:
                            continue
                        records.append(record)
        return records

    def _record_to_dict(self, record: FeedbackRecord) -> dict:
        ...

    def _dict_to_record(self, data: dict) -> FeedbackRecord:
        ...
```

### 5.3 collector.py

**文件**：`legalbot/feedback/collector.py`

```python
from legalbot.feedback.models import FeedbackRecord, QueryInfo, ChunkResult, CorrectionInfo
from legalbot.feedback.storage import FeedbackStorage


class FeedbackCollector:
    """收集和处理用户反馈。"""

    def __init__(self, storage: FeedbackStorage):
        self._storage = storage

    async def submit(
        self,
        feedback_id: str,
        feedback_type: str,
        result_id: str | None = None,
        helpful: bool | None = None,
        correction: str | None = None,
        reason: str | None = None,
        query_info: dict | None = None,
    ) -> bool:
        record = FeedbackRecord(
            id=feedback_id,
            timestamp=datetime.now(),
            type=feedback_type,
            query=QueryInfo(**query_info) if query_info else QueryInfo(text=""),
            results=[ChunkResult(rank=0, chunk_id=result_id or "", law_name="",
                                  article_no="", score=0.0, helpful=helpful)] if result_id else [],
            correction=CorrectionInfo(chunk_id=result_id or "",
                                       corrected_text=correction or "",
                                       reason=reason) if correction else None,
        )
        await self._storage.append(record)
        return True
```

### 5.4 analyzer.py

**文件**：`legalbot/feedback/analyzer.py`

```python
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

from legalbot.feedback.storage import FeedbackStorage


@dataclass
class QueryStats:
    query_text: str
    total_reports: int
    helpful_count: int
    unhelpful_count: int
    law_area: str | None
    sample_unhelpful: list[str]


@dataclass
class ChunkStats:
    chunk_id: str
    law_name: str
    article_no: str
    total_reports: int
    correction_count: int
    outdated_count: int
    avg_helpful_score: float


@dataclass
class AnalysisReport:
    period_start: datetime
    period_end: datetime
    total_feedback: int
    helpful_rate: float
    top_problem_queries: list[QueryStats]
    low_score_chunks: list[ChunkStats]
    outdated_chunks: list[ChunkStats]


class FeedbackAnalyzer:
    """分析反馈数据并生成改进建议。"""

    def __init__(self, storage: FeedbackStorage):
        self._storage = storage

    async def analyze_period(self, since: datetime, until: datetime) -> AnalysisReport:
        records = self._storage.query(since=since, until=until)
        # 聚合统计
        total = len(records)
        helpful = sum(1 for r in records if r.type == "helpful")
        # 构建问题查询列表、chunk 统计等
        return AnalysisReport(
            period_start=since,
            period_end=until,
            total_feedback=total,
            helpful_rate=helpful / total if total > 0 else 0,
            top_problem_queries=[],
            low_score_chunks=[],
            outdated_chunks=[],
        )

    async def identify_low_score_chunks(self, threshold: float = 0.5, min_reports: int = 3) -> list[ChunkStats]:
        ...

    async def identify_outdated_chunks(self, min_corrections: int = 2) -> list[ChunkStats]:
        ...
```

---

## 六、集成点

### 6.1 工具注册

**文件**：`legalbot/agent/loop.py`

```python
# 在 _register_default_tools() 中 — RAG 工具注册之后：
if self.rag_config and self.rag_config.enable:
    from legalbot.agent.tools.feedback import FeedbackTool
    from legalbot.feedback import FeedbackCollector, FeedbackStorage

    storage = FeedbackStorage()
    collector = FeedbackCollector(storage)
    self.tools.register(FeedbackTool(collector=collector))
```

### 6.2 RAGSearchTool 增强

**文件**：`legalbot/agent/tools/rag.py`

修改 `_format_results()` 以包含反馈引导语，并在输出中嵌入 `data-feedback-id` 用于追踪。

---

## 七、CLI 命令

### 7.1 commands.py 扩展

**文件**：`legalbot/cli/commands.py`

```python
feedback_app = typer.Typer(help="管理法律 RAG 反馈")
app.add_typer(feedback_app, name="feedback")

@feedback_app.command("rate")
def feedback_rate(result_id: str, helpful: bool):
    """对检索结果进行评分。"""
    ...

@feedback_app.command("correct")
def feedback_correct(result_id: str, text: str, reason: str = "outdated"):
    """为检索结果提供纠正。"""
    ...

@feedback_app.command("list")
def feedback_list(limit: int = 20, feedback_type: str = None):
    """列出最近的反馈记录。"""
    ...

@feedback_app.command("export")
def feedback_export(from_date: str, to_date: str, output: str):
    """导出反馈数据。"""
    ...

@feedback_app.command("analyze")
def feedback_analyze(period: str = "day", output: str = None):
    """分析反馈并生成报告。"""
    ...
```

---

## 八、配置

### 8.1 FeedbackConfig

**文件**：`legalbot/config/schema.py`

```python
class FeedbackConfig(Base):
    """用户反馈收集配置"""
    enable: bool = True
    storage_dir: str = "~/.legalbot/feedback"
    retention_days: int = 90
    rate_limit_per_minute: int = 10
    require_confirmation_for_correction: bool = True


# 在 ToolsConfig 中：
class ToolsConfig(Base):
    # ... 现有字段 ...
    rag: RAGConfig = Field(default_factory=RAGConfig)
    orchestrate: OrchestrateConfig = Field(default_factory=OrchestrateConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
```

---

## 九、实施步骤

### 第一阶段：核心基础设施（第 1 周）
- 1.1 创建 `legalbot/feedback/` 包结构
- 1.2 实现 `models.py`、`storage.py`
- 1.3 实现 `collector.py`

### 第二阶段：CLI 集成（第 1-2 周）
- 2.1 添加 CLI 命令到 `commands.py`
- 2.2 在 AgentLoop 中连接 `FeedbackTool`

### 第三阶段：分析流水线（第 2-3 周）
- 3.1 实现 `analyzer.py`
- 3.2 报告生成
- 3.3 分析 CLI

### 第四阶段：集成优化（第 3-4 周）
- 4.1 RAGSearchTool 中的内联反馈
- 4.2 错误处理和边缘情况

### 第五阶段：测试（第 4 周）
- 所有模块的单元测试
- 集成测试

---

## 十、文件变更摘要

### 新增文件

| 文件 | 用途 |
|------|---------|
| `legalbot/feedback/__init__.py` | 包初始化 |
| `legalbot/feedback/models.py` | FeedbackRecord dataclass |
| `legalbot/feedback/storage.py` | JSONL 存储 |
| `legalbot/feedback/collector.py` | 反馈收集 |
| `legalbot/feedback/analyzer.py` | 分析引擎 |
| `legalbot/agent/tools/feedback.py` | FeedbackTool |
| `docs/POST_MVP_FEEDBACK_LOOP.md` | 本文档 |

### 修改文件

| 文件 | 变更 |
|------|---------|
| `legalbot/agent/loop.py` | 注册 FeedbackTool |
| `legalbot/agent/tools/rag.py` | 添加反馈引导 |
| `legalbot/cli/commands.py` | 添加反馈子命令 |
| `legalbot/config/schema.py` | 添加 `FeedbackConfig` |

---

## 十一、风险与缓解

| 风险 | 缓解措施 |
|------|------------|
| 用户采用率低 | 无摩擦的 +/- 内联按钮 |
| 存储增长 | 每月压缩（tar.gz） |
| 并发写入冲突 | 文件锁或原子重命名 |
| 反馈垃圾 | 速率限制（最多 10/分钟） |
| 错误反馈影响 | 纠正需要确认 |

---

## 十二、测试计划

| 测试 | 预期结果 |
|------|----------------|
| `test_append_new_file` | 记录写入新的 JSONL |
| `test_query_by_date_range` | 返回正确记录 |
| `test_identify_low_score_chunks` | 识别低于阈值的 chunks |
| `test_identify_outdated_chunks` | 识别有纠正的 chunks |
| `test_rate_limit` | 第 11 次提交在 1 分钟内被拒绝 |