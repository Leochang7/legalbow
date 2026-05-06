# legalbot

面向法律领域的多智能体 AI 协作框架，提供法律检索、多步推理、合同审查、案例对比、辩论分析、文书生成等核心能力。

## 核心特性

- **Multi-Agent 编排** — 主 Agent + 多个子 Agent（法律检索、合同审查、辩论分析等）并发调度，意图识别自动路由
- **混合 RAG 检索** — 向量语义检索 + BM25 关键词检索 + RRF 融合 + LLM 重排，精准引用法规和判例
- **多步链式推理** — "问题 → 初始检索 → 分析评估 → 补充检索 → 综合结论"循环迭代推理流水线
- **多渠道接入** — 飞书、钉钉、QQ、WebSocket，统一消息总线解耦
- **Hook 生命周期** — 审计日志、反馈收集等插件化扩展
- **多 Provider 支持** — Anthropic、OpenAI 兼容、Azure OpenAI、DashScope

## 快速开始

### 安装

```bash
# 安装依赖
pip install -e .

# 或使用 uv
uv sync
```

### 配置

创建 `~/.legalbot/config.json`：

```json
{
  "providers": {
    "openai_compat": {
      "api_key": "your-api-key",
      "api_base": "https://api.openai.com/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "gpt-4o",
      "workspace": "."
    }
  }
}
```

### 使用

```bash
# CLI 交互模式
python -m legalbot

# SDK 编程调用
```

```python
from legalbot import legalbot

bot = legalbot.from_config()
result = await bot.run("根据《合同法》第52条，分析这份合同是否存在无效情形")
print(result.content)
```

## 项目结构

```
legalbot/
├── agent/          # Agent 主循环、上下文构建、Session 管理、Hook 系统
│   └── tools/      # 工具实现（fs, shell, web, rag, orchestrate, debate 等）
├── channels/       # 渠道接入（feishu, dingtalk, qq, websocket）
├── cli/            # CLI 命令与交互
├── config/         # 配置加载与 Schema
├── providers/      # LLM Provider 抽象层
├── rag/            # RAG 检索（embedding, vectorstore, chunker, retriever, reranker）
├── document/       # 法律文书生成
├── feedback/       # 用户反馈闭环
├── audit/          # 审计日志
├── cron/           # 定时任务
├── session/        # 会话持久化
├── skills/         # 内置技能（legal-research, legal-citation, legal-reasoning 等）
└── bus/            # 消息总线
```

## 法律技能

| 技能 | 功能 |
|------|------|
| `legal-research` | 混合 RAG 法律知识检索 |
| `legal-citation` | 法条精确引用与验证 |
| `legal-reasoning` | 多步链式法律推理 |
| `legal-case-compare` | 类案对比分析 |
| `legal-debate` | 法律辩论自动化分析 |
| `legal-document-draft` | 法律文书自动生成 |

## 工具一览

| 工具 | 功能 |
|------|------|
| `legal_rag_search` | 法律知识库混合检索 |
| `multi_step_reason` | 多步法律推理 |
| `orchestrate` | 多 Agent 编排调度 |
| `legal_document_generate` | 法律文书生成 |
| `legal_debate` | 法律辩论分析 |
| `case_compare` | 案例对比分析 |
| `feedback_collect` | 用户反馈收集 |
| `web_search` / `web_fetch` | 网络搜索与内容获取 |
| `exec` | Shell 命令执行 |
| `glob` / `grep` | 文件搜索 |
| `read_file` / `write_file` / `edit_file` | 文件操作 |

## License

MIT
