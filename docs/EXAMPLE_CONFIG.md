# 示例配置文件

本文件展示 LegalBot 的完整配置结构。将此文件复制到 `~/.legalbot/config.json` 并根据实际情况修改。

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.legalbot/workspace",
      "model": "anthropic/claude-opus-4-5",
      "provider": "auto",
      "max_tokens": 8192,
      "context_window_tokens": 65536,
      "context_block_limit": null,
      "temperature": 0.1,
      "max_tool_iterations": 200,
      "max_tool_result_chars": 16000,
      "provider_retry_mode": "standard",
      "reasoning_effort": null,
      "timezone": "Asia/Shanghai",
      "unified_session": false
    }
  },
  "channels": {
    "send_progress": true,
    "send_tool_hints": false,
    "send_max_retries": 3
  },
  "providers": {
    "custom": {
      "api_key": "",
      "api_base": null,
      "extra_headers": null
    },
    "anthropic": {
      "api_key": "sk-ant-...",
      "api_base": null,
      "extra_headers": null
    },
    "deepseek": {
      "api_key": "sk-...",
      "api_base": "https://api.deepseek.com",
      "extra_headers": null
    },
    "dashscope": {
      "api_key": "sk-...",
      "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "extra_headers": null
    }
  },
  "api": {
    "host": "127.0.0.1",
    "port": 8900,
    "timeout": 120.0
  },
  "gateway": {
    "host": "0.0.0.0",
    "port": 18790
  },
  "tools": {
    "web": {
      "enable": true,
      "proxy": null,
      "search": {
        "provider": "brave",
        "apiKey": "",
        "baseUrl": "",
        "maxResults": 5,
        "timeout": 30
      }
    },
    "exec": {
      "enable": false,
      "timeout": 60,
      "pathAppend": "",
      "sandbox": ""
    },
    "restrictToWorkspace": false,
    "mcpServers": {},
    "ssrfWhitelist": [],
    "rag": {
      "enable": true,
      "embeddingProvider": "dashscope",
      "embeddingModel": "text-embedding-v4",
      "embeddingApiKey": "sk-...",
      "embeddingApiBase": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "embeddingDim": 1536,
      "vectorStore": "chroma",
      "persistDir": "~/.legalbot/legal_kb",
      "reranker": "qwen3-rerank",
      "rerankerApiKey": "sk-...",
      "bm25Enable": true,
      "topK": 5,
      "chunkMaxTokens": 800,
      "chunkOverlapTokens": 100
    },
    "feedback": {
      "enable": true,
      "storageDir": "~/.legalbot/feedback",
      "retentionDays": 90,
      "rateLimitPerMinute": 10,
      "requireConfirmationForCorrection": true
    },
    "document_draft": {
      "enable": true,
      "templateDir": "",
      "enabledTypes": [
        "complaint",
        "defense",
        "agent_opinion",
        "appeal",
        "enforcement"
      ]
    },
    "orchestrate": {
      "enable": true,
      "intentModel": "",
      "agents": {
        "legal_research": {
          "systemPrompt": "",
          "tools": ["legal_rag_search", "web_search", "read_file"],
          "model": ""
        },
        "contract_review": {
          "systemPrompt": "",
          "tools": ["legal_rag_search", "read_file"],
          "model": ""
        }
      },
      "debate": {
        "enable": true,
        "rounds": 1,
        "timeoutPerAgent": 120,
        "timeoutTotal": 300,
        "maxRetries": 2,
        "judgeModel": "",
        "plaintiffModel": "",
        "defendantModel": "",
        "agents": {}
      },
      "case_compare": {
        "enable": true,
        "comparisonModel": "",
        "maxCases": 10,
        "topKDefault": 5
      }
    }
  }
}
```

---

## 配置说明

### agents.defaults（Agent 全局默认配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `workspace` | string | `~/.legalbot/workspace` | 工作区目录 |
| `model` | string | `anthropic/claude-opus-4-5` | 默认模型 |
| `provider` | string | `auto` | 提供商，`auto` 自动检测 |
| `max_tokens` | int | `8192` | 最大输出 token 数 |
| `context_window_tokens` | int | `65536` | 上下文窗口大小 |
| `temperature` | float | `0.1` | 生成温度 |
| `max_tool_iterations` | int | `200` | 最大工具调用次数 |
| `max_tool_result_chars` | int | `16000` | 工具结果最大字符数 |
| `timezone` | string | `UTC` | 时区，如 `Asia/Shanghai` |
| `unified_session` | bool | `false` | 多渠道共享同一会话 |

### tools.rag（RAG 知识库配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `false` | 是否启用 RAG |
| `embeddingProvider` | string | `openai` | Embedding 提供商 |
| `embeddingModel` | string | `text-embedding-3-small` | Embedding 模型 |
| `embeddingApiKey` | string | `""` | Embedding API 密钥 |
| `embeddingApiBase` | string | `""` | API 地址（兼容 OpenAI 格式） |
| `embeddingDim` | int | `1536` | 向量维度 |
| `vectorStore` | string | `chroma` | 向量数据库 |
| `persistDir` | string | `~/.legalbot/legal_kb` | 知识库持久化目录 |
| `reranker` | string | `""` | 重排模型（空=禁用） |
| `bm25Enable` | bool | `true` | 是否启用 BM25 混合检索 |
| `topK` | int | `5` | 召回数量 |
| `chunkMaxTokens` | int | `800` | 分块最大 token 数 |
| `chunkOverlapTokens` | int | `100` | 分块重叠 token 数 |

### tools.orchestrate（多 Agent 编排配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `false` | 是否启用编排 |
| `intentModel` | string | `""` | 意图分类模型，空=使用默认 |
| `debate.enable` | bool | `false` | 是否启用辩论模式 |
| `case_compare.enable` | bool | `true` | 是否启用案例对比 |

### tools.exec（Shell 执行工具配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `true` | **生产环境建议设为 false** |
| `timeout` | int | `60` | 执行超时（秒） |
| `sandbox` | string | `""` | 沙箱后端（空=无，`bwrap`= bubblewrap） |

### tools.feedback（用户反馈配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `true` | 是否启用反馈收集 |
| `storageDir` | string | `~/.legalbot/feedback` | 反馈数据存储目录 |
| `retentionDays` | int | `90` | 数据保留天数 |
| `rateLimitPerMinute` | int | `10` | 每分钟最大反馈数 |

### tools.document_draft（法律文书生成配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `false` | 是否启用文书生成 |
| `enabledTypes` | list | 全部5种 | 启用的文书类型 |

---

## 快速开始

1. 复制本文件为 `~/.legalbot/config.json`
2. 填写必要的 API 密钥
3. 根据需要启用功能模块
4. 运行 `legalbot` 启动

## 生产环境建议

- `tools.exec.enable` 设为 `false`
- `agents.defaults.unified_session` 按需调整
- `tools.rag.persistDir` 使用绝对路径
- 敏感 API 密钥通过环境变量传入
