# 人工复核工作流设计

## 背景

AI 生成的法律文书存在错误风险，必须经过执业律师审核才能正式使用。本文档描述人工复核工作流的实现方案。

## 设计目标

1. **强制复核** - 文书生成后必须经过人工确认才能交付
2. **可编辑** - 允许律师在确认前修改文书内容
3. **版本记录** - 记录每次修改，保留修改历史
4. **不阻塞** - 复核流程异步进行，不影响用户体验

## 工作流设计

```
用户请求生成文书
       │
       ▼
  AI 生成草稿
（标记为 draft）
       │
       ▼
  进入复核队列
（通知律师）
       │
       ▼
  ┌─────────────┐
  │  律师审核   │
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
  修改      确认通过
    │         │
    ▼         ▼
  更新草稿    生成正式版
  重新提交    交付用户
```

## 实现方案

### 1. 文书状态机

```python
# legalbot/document/review.py
from enum import Enum

class DocumentStatus(Enum):
    DRAFT = "draft"           # AI 生成，待复核
    UNDER_REVIEW = "review"  # 律师复核中
    REVISION_REQUESTED = "revision"  # 律师要求修改
    APPROVED = "approved"     # 律师确认通过
    DELIVERED = "delivered"   # 已交付用户
    EXPIRED = "expired"       # 超时未复核
```

### 2. 复核队列

```python
# legalbot/document/review_manager.py
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4
import json

class ReviewManager:
    """Manages the legal document review workflow."""

    def __init__(
        self,
        review_dir: str = "~/.legalbot/review_queue",
        notification_callback: callable = None,
        review_timeout_hours: int = 48,
    ):
        self._review_dir = Path(review_dir).expanduser()
        self._review_dir.mkdir(parents=True, exist_ok=True)
        self._notification_callback = notification_callback
        self._timeout = timedelta(hours=review_timeout_hours)
        self._lock = asyncio.Lock()

    async def submit_for_review(
        self,
        doc_type: str,
        case_facts: str,
        generated_content: str,
        doc_path: str,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> str:
        """Submit a document for lawyer review. Returns review_id."""
        review_id = str(uuid4())
        record = {
            "review_id": review_id,
            "status": DocumentStatus.DRAFT.value,
            "doc_type": doc_type,
            "case_facts": case_facts[:500],  # 截断存储
            "generated_content": generated_content,
            "doc_path": doc_path,
            "session_key": session_key,
            "channel": channel,
            "chat_id": chat_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reviewer_notes": "",
            "revisions": [],
        }
        await self._save_record(review_id, record)

        # 通知律师
        if self._notification_callback:
            await self._notification_callback(
                event="new_document_review",
                review_id=review_id,
                doc_type=doc_type,
            )

        return review_id

    async def get_pending_reviews(self) -> list[dict]:
        """返回所有待复核的文书。"""
        records = []
        async with self._lock:
            for file_path in self._review_dir.glob("*.json"):
                record = json.loads(file_path.read_text(encoding="utf-8"))
                if record["status"] in (
                    DocumentStatus.DRAFT.value,
                    DocumentStatus.UNDER_REVIEW.value,
                    DocumentStatus.REVISION_REQUESTED.value,
                ):
                    records.append(record)
        return records

    async def approve(self, review_id: str) -> str:
        """律师确认通过，返回正式文件路径。"""
        record = await self._load_record(review_id)
        record["status"] = DocumentStatus.APPROVED.value
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._save_record(review_id, record)

        # 生成正式版（带水印）
        official_path = await self._generate_official_version(record)
        record["status"] = DocumentStatus.DELIVERED.value
        record["official_path"] = official_path
        await self._save_record(review_id, record)

        # 通知用户
        await self._notify_user(record, "approved")

        return official_path

    async def request_revision(
        self,
        review_id: str,
        notes: str,
        modified_content: str | None = None,
    ) -> None:
        """律师要求修改。"""
        record = await self._load_record(review_id)
        record["status"] = DocumentStatus.REVISION_REQUESTED.value
        record["reviewer_notes"] = notes
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record["revisions"].append({
            "at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "modified": modified_content is not None,
        })
        await self._save_record(review_id, record)

        # 通知用户
        await self._notify_user(record, "revision_requested")

    async def _generate_official_version(self, record: dict) -> str:
        """生成带水印和签章的正式版文书。"""
        # TODO: 实现正式版生成逻辑
        # - 添加"已审核"水印
        # - 添加审核人信息
        # - 添加审核时间戳
        ...
```

### 3. 与 Agent 集成

修改 `LegalDocumentGenerateTool`，生成后进入复核流程而非直接返回：

```python
# legalbot/agent/tools/document.py
async def execute(self, doc_type: str, case_facts: str, ...) -> str:
    # ... 生成文书 ...

    # 保存草稿文件
    draft_path = self._save_draft(content, doc_type)

    # 进入复核队列
    review_id = await self._review_manager.submit_for_review(
        doc_type=doc_type,
        case_facts=case_facts,
        generated_content=content,
        doc_path=draft_path,
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
    )

    return (
        f"法律文书《{doc_type_names[doc_type]}》已生成草稿，"
        f"正在等待律师复核（复核ID：{review_id}）。\n"
        f"草稿文件：{draft_path}\n\n"
        f"⚠️ 重要提示：此为 AI 生成草稿，须经执业律师审核方可正式使用。"
    )
```

### 4. CLI 命令

```python
# legalbot/cli/commands.py
@legal_app.command("review-list")
def review_list(
    status: str = typer.Option(None, "--status", help="按状态过滤"),
) -> None:
    """列出待复核的文书。"""
    ...

@legal_app.command("review-approve")
def review_approve(review_id: str) -> None:
    """确认文书通过。"""
    ...

@legal_app.command("review-reject")
def review_reject(
    review_id: str,
    notes: str = typer.Option(..., "--notes", help="修改意见"),
) -> None:
    """要求修改文书。"""
    ...
```

## 用户体验流程

```
用户：帮我写一份起诉状
AI：  已提交复核（复核ID: abc123）
      草稿文件已保存
      ⚠️ AI 生成草稿，须律师审核

[系统通知律师有新文书待审]

律师（CLI）：legalbot legal review-list
律师（CLI）：legalbot legal review-approve abc123

用户（收到通知）：您的起诉状已通过审核，正式文件已生成
```

## 待实现

- [ ] 创建 `legalbot/document/review.py` 状态机和 `ReviewManager`
- [ ] 修改 `LegalDocumentGenerateTool` 集成复核流程
- [ ] 添加 `legal review-list/approve/reject` CLI 命令
- [ ] 实现通知机制（邮件/Telegram/等）
- [ ] 实现正式版生成（水印、签章）
- [ ] 添加复核超时自动提醒
- [ ] 实现复核历史记录和统计
