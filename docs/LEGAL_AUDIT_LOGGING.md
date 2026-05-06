# 法律审计日志设计

## 背景

法律 AI 系统需要完整的审计跟踪，以满足合规要求和事后追溯需求。本文档描述审计日志的设计方案。

## 设计目标

1. **可追溯** - 记录每次法律咨询的时间、内容、结果
2. **不可篡改** - 日志写入后不应被修改或删除
3. **可查询** - 支持按时间、用户、会话、法律领域等维度查询
4. **最小化性能影响** - 异步写入，不阻塞主流程

## 审计日志内容

每次法律查询需记录以下字段：

```json
{
  "event_id": "uuid",
  "timestamp": "2026-04-19T15:00:00+08:00",
  "session_id": "cli:direct",
  "channel": "cli",
  "user_id": "anonymous",
  "event_type": "legal_query | document_draft | case_compare | debate | contract_review",
  "query": {
    "original_text": "民间借贷纠纷如何起诉？",
    "law_areas": ["民法", "民事诉讼法"],
    "intent": "legal_query"
  },
  "response": {
    "final_content": "根据《民法典》...",
    "tools_called": ["legal_rag_search"],
    "citations": [
      {"law": "《中华人民共和国民法典》", "article": "第六百七十五条", "valid": true}
    ],
    "disclaimer_shown": true
  },
  "metadata": {
    "model": "claude-opus-4-5",
    "tokens_used": 2048,
    "duration_ms": 3200,
    "success": true
  },
  "compliance": {
    "data_residency": "CN",
    "retention_until": "2027-04-19",
    "pii_detected": false
  }
}
```

## 事件类型

| 事件类型 | 说明 | 敏感字段 |
|----------|------|----------|
| `legal_query` | 法律咨询 | 查询内容 |
| `document_draft` | 文书生成 | 案件事实 |
| `case_compare` | 案例对比 | 对比内容 |
| `debate` | 法律辩论 | 辩论输入 |
| `contract_review` | 合同审查 | 合同内容 |

## 实现方案

### 1. 存储格式

使用 **JSONL**（JSON Lines）格式，每日一个文件：

```
~/.legalbot/audit/
  ├── 2026-04-19.jsonl
  ├── 2026-04-20.jsonl
  └── 2026-04-21.jsonl
```

优点：
- 追加写入，无需锁
- 按日期分离，便于清理
- 可被 `jq`、`pandas` 等工具直接处理

### 2. 核心实现

```python
# legalbot/audit/logger.py
import asyncio
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

class LegalAuditLogger:
    """Async audit logger for legal AI events."""

    def __init__(self, audit_dir: str = "~/.legalbot/audit"):
        self._audit_dir = Path(audit_dir).expanduser()
        self._today_file: Path | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def _get_today_file(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._audit_dir / f"{today}.jsonl"

    async def log(
        self,
        event_type: str,
        session_id: str,
        channel: str,
        query: dict,
        response: dict,
        metadata: dict | None = None,
        user_id: str = "anonymous",
    ) -> str:
        """Log a legal AI event. Returns event_id."""
        self._ensure_init()
        event_id = str(uuid4())
        record = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "channel": channel,
            "user_id": user_id,
            "event_type": event_type,
            "query": query,
            "response": response,
            "metadata": metadata or {},
        }
        # Compute integrity hash for tamper detection
        content_hash = hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        record["_hash"] = content_hash

        file_path = self._get_today_file()
        async with self._lock:
            async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return event_id

    async def query(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        event_type: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit logs with filters."""
        # Implementation: list matching .jsonl files, filter records
        ...
```

### 3. 与 Agent Loop 集成

在 `AgentLoop._process_message` 和 `_save_turn` 中添加审计钩子：

```python
# loop.py additions
from legalbot.audit.logger import LegalAuditLogger

class AgentLoop:
    def __init__(self, ...):
        self._audit_logger = LegalAuditLogger()

    async def _process_message(self, msg: InboundMessage, ...):
        # ... existing code ...

        # After generating response, log if legal intent
        if self._is_legal_event(msg.content):
            await self._audit_logger.log(
                event_type=self._classify_legal_event(msg.content),
                session_id=key,
                channel=msg.channel,
                query={"original_text": msg.content, "law_areas": [], "intent": ""},
                response={"final_content": final_content, "tools_called": [], "citations": [], "disclaimer_shown": True},
                metadata={"model": self.model, "duration_ms": 0},
            )
```

### 4. 过期数据清理

```python
# 保留 90 天，超期文件删除
async def cleanup_old_logs(self, retention_days: int = 90):
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for file_path in self._audit_dir.glob("*.jsonl"):
        if file_path.stat().st_mtime < cutoff.timestamp():
            file_path.unlink()
```

## 隐私保护

| 处理措施 | 说明 |
|----------|------|
| PII 检测 | 身份证号、手机号等自动打码 |
| 数据隔离 | 按 session_id 分开存储 |
| 保留期限 | 默认 90 天，可配置 |
| 加密存储 | 可选 GPG 加密（TODO） |

## 待实现

- [x] 创建 `legalbot/audit/` 包和 `LegalAuditLogger` 类 ✅
- [x] 在 `AgentLoop` 中集成审计日志调用 ✅
- [x] 实现 PII 自动检测和脱敏 ✅
- [x] 添加 `audit query` CLI 命令 ✅
- [x] 实现日志轮转和过期清理 ✅
- [x] 添加日志完整性验证（hash check） ✅
- [ ] 加密存储（GPG）—— 可选增强
