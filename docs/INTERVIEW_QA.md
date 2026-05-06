# legalbot 项目面试 Q&A

---

## 架构设计

### Q1：为什么从零写 Agent 框架，而不是用 LangChain/LangGraph？

用过，后来放弃了。LangChain 抽象太多，出了 bug 要翻它的 callback handler、agent executor、tool 适配层，而不是看自己的逻辑。LangGraph 的 StateGraph 概念是好的，但我们编排器一共 400 行，用意图分类加路由就够了，用不着一套状态机框架。

核心诉求是**控制权**。比如 tool result 截断、microcompact、token 预算管理是直接写在 AgentRunner 循环里的，每一步行为都看得见，改了就能生效。用 LangChain 要么用它提供的机制（不一定满足需求），要么给它写 plugin——最后花的精力不比自研少，还多了一层黑盒依赖。
SUBQ1:什么是microcompact

### Q2：Agent 主循环是怎么设计的？

两个类分工：`AgentLoop` 管产品层——消息路由、session 管理、审计、streaming；`AgentRunner` 管执行层——纯粹的工具调用循环，无状态。

执行循环本身很简单：每轮先做 context governance（截断超长历史、压缩旧工具结果），调 LLM，有 tool_calls 就执行工具然后继续，没有就返回。加上"空白响应重试"和"length 截断恢复"两个兜底。

并发模型是关键——同 session 串行（asyncio.Lock），不同 session 并发，全局用 Semaphore 限制并发数防止 API 限流。这样既保证了对话历史的一致性，又不会让一个用户阻塞其他人。

### Q3：为什么 AgentRunner 要设计成无状态的？

同一份执行逻辑要服务两个场景：主 Agent 的对话循环，和 SubagentManager 的后台子 Agent。区别只是传入的 ToolRegistry 和 max_iterations 不同，执行逻辑完全复用。

另外测试也方便——给 AgentRunSpec 传 mock tools 和 mock provider，不需要启动整个 AgentLoop。

### Q4：上下文窗口怎么管理？

四层防线：

1. **Token 预算截断**：从后往前累积消息，估算 token 数，超出 `context_window - max_output - 1024` 安全线就从历史头部丢弃。始终保证截断点在 user 消息处，不会剪断 tool_calls 和 tool_results 的关联。

2. **工具结果截断**：单条结果超过 20000 字符就截头去尾。`read_file` 或 `exec` 常返回巨大输出，不截断会把 context 撑爆。

3. **Microcompact**：旧轮次的文件读取、搜索类工具结果，保留最近 10 条完整内容，更早的替换为 `[xxx result omitted from context]`。这些"一次性查阅"的信息后续轮次基本不会再被引用。

4. **Consolidator**：前三层都挡不住时，用 LLM 把旧消息总结成一句话存到 history.jsonl，从活跃消息中移除。只在 user-turn 边界裁切，不破坏 tool calling 链。

### Q5：ToolRegistry 的 get_definitions() 为什么要排序？

为了**最大化 prompt cache 命中率**。内置工具按名排序放前面，MCP 工具按名排序放后面，这样 MCP 工具变化时内置工具的位置不变。Anthropic 的 cache_control 在两个边界各打一个 breakpoint，前缀匹配就能命中缓存。

### Q6：你是怎么适配 16+ LLM 提供商的？

所有 Provider 继承 `LLMProvider` 基类，实现 `chat()` 方法，内部把各自的原生响应统一转成 `LLMResponse` 这个中间表示。不同的原生格式（Anthropic 的 content blocks、OpenAI 的 tool_calls JSON、DeepSeek 的 reasoning_content）都在 Provider 内部消化。

基类提供了几个共享的 sanitize 方法，是踩坑踩出来的：`_enforce_role_alternation` 合并连续同 role 消息（有些 Provider 直接拒绝这种格式）、`_strip_image_content` 把图片替换为文本占位符（给不支持多模态的）、`_sanitize_request_messages` 只保留 Provider 认可的 message key。

### Q7：重试策略怎么设计的？

两档：standard 模式 3 次（1s/2s/4s 指数退避），persistent 模式一直重试到相同错误出现 10 次才放弃。

关键不是重试本身，而是**判断什么不该重试**。429 错误里 `rate_limit_exceeded` 应该重试，但 `insufficient_quota` 或 `billing_hard_limit_reached` 绝不能重试——重试一万次也不会好。策略是优先用结构化 error metadata（error_type、error_code），fallback 到文本匹配，不重试的规则比可重试的更"窄"——宁可少重试一次，不能死循环。

---

## RAG 与检索

### Q8：混合检索怎么做的？

向量检索（Chroma）+ BM25 关键词检索，RRF（Reciprocal Rank Fusion）融合。

选混合检索的原因很具体：法律条文对精确关键词匹配要求极高。"劳动合同法第三十九条"和"第四十条"，向量语义几乎一样，但法律后果完全不同。纯向量检索会把两者混在一起，BM25 对"第X条"这个精确词能给出明确区分。RRF 公式 `1/(k+rank)`，k=60，不需要调权重。

### Q9：BM25 懒加载解决了什么问题？

BM25 索引构建是 O(n) 遍历全部语料。如果每次 add chunk 都重建，100 次 add 就是 O(100n)。用懒加载模式——add 只追加 tokenized corpus 然后标记 dirty=True，search 时才检查 dirty 标记并按需重建，复杂度从 O(n*m) 降到 O(n)。

### Q10：分块策略有什么讲究？

优先按"第X条"结构分割——法律条文有天然的语义边界，不需要滑动窗口。只有单条条文超过 max_tokens 时才做二次分割，按句号、分号等句子边界断开。相邻块之间保留 overlap，防止边界处信息丢失。

拆法律条文用滑动窗口是错的——会把"第X条"从中间切断，检索结果就不完整了。

### Q11：多步法律推理（MultiStepLegalReasoner）怎么做的？

不是 CoT（Chain of Thought）。CoT 只是让 LLM 多想几步再输出，没有引入外部信息。这里是**检索增强的多步推理**——每一步都可能带回新的法条和案例，后续步骤可以看到前面所有检索结果的累积。

流程是：收到问题后先生成子问题列表（"要回答这个问题需要查哪些子问题"），然后逐个检索、逐个推理，每一步的结果作为下一步的 context。全部子问题推理完做一次汇总——检查各子问题结论之间有没有矛盾，决定是否补充检索。

---

## Agent 编排

### Q12：多 Agent 编排怎么做的？

两阶段 LLM 分类：先判断是不是 legal 问题 → 判断 simple/complex，再分到 7 种意图（legal_query / contract_review / case_search / debate / case_compare / document_draft / general）。每种意图路由到不同的处理器。

和主 Agent 的关系：当 orchestrator 启用时，主 Agent 的 `legal_rag_search` 工具会被移除，只保留 `orchestrate` 工具。这确保法律类查询必须走编排器，不会被主 Agent 绕过直接 RAG。

### Q13：辩论模式的三方论证怎么隔离？

原告和被告在立论阶段**信息隔离**——各自只看到案件描述，看不到对方的论证。这模拟了真实诉讼中"起诉状送达前被告不知道原告策略"的情况。对抗阶段才互相可见，产生真正的辩论火花。

技术上是用同一个 SubagentManager 开三个子 Agent，各自带不同的 system prompt（原告/被告/法官角色），立论阶段 asyncio.gather 并发执行。

---

## 工程实践

### Q14：流式输出在消息总线架构下怎么实现？

消息总线本身是无状态的。流式的关键是 outbound 消息上的 metadata 标记：`_stream_delta` + `_stream_id` 表示这是一个流式片段，`_stream_end` + `_resuming` 表示流结束（resuming=true 说明后面还有 tool calls，UI 应该继续显示加载状态）。渠道层根据这些标记决定如何渲染——CLI 直接 print，WebSocket 按帧 push。

### Q15：Runtime Checkpoint 解决什么问题？

Agent 执行到一半进程崩溃了（OOM、kill -9），下一次消息来时 `_restore_runtime_checkpoint()` 会把"上次执行到哪了"恢复出来。每个 iteration 的关键节点往 session.metadata 写 checkpoint，下次启动读到就把已完成的消息追加到历史，未执行的 tool_calls 补上 "Task interrupted" 标记。用户看到的就是"抱歉被中断了，请重试"，对话历史不会丢。

### Q16：Tool 参数校验为什么要分 cast 和 validate 两步？

LLM 返回的参数类型不可靠——经常把数字用字符串传过来 `{"limit": "10"}`。如果直接 JSON Schema 校验，`"10"` 不是 number 就失败了。所以先 cast 做宽松类型转换（"10" → 10, "true" → True），再 validate。这样 LLM 的类型错误大部分能自动修正。

### Q17：为什么 session 用 JSONL 而不是 SQLite？

1. 人类可读，出了 bug 直接 `cat` 看
2. 不用额外依赖
3. append-only 无并发写入冲突
4. 单 session 几百到几千条消息的规模，JSONL 顺序读写不是瓶颈

### Q18：测试怎么分层？

- **Tool 层**：单元测试 cast_params / validate_params 的边界行为
- **AgentRunner 层**：mock provider，验证工具并发、错误恢复、max_iterations 终止
- **RAG 检索**：用已知文档集合测 recall@k 和 MRR
- **集成测试**：端到端跑 `消息 → AgentLoop → 响应`

Provider 层不测真实 LLM 调用——成本高、结果不稳定——直接 mock HTTP 响应。

### Q19：全局并发数为什么默认是 3？

大多数 LLM API 的并发限制在 3-5 个请求。3 是一个保守但安全的默认值，通过 `legalbot_MAX_CONCURRENT_REQUESTS` 环境变量可调。这个 Semaphore 放在 session lock 之后获取，所以不会因为全局排队导致某个 session 被饿死——session lock 保证了同 session 只有一个协程在等待，gate 前面不会排长队。

---

## 动机与取舍

### Q20：自己做这个项目最大的技术收获是什么？

不要过早引入框架。这个项目 200+ 个 .py 文件，但核心链路（AgentRunner + ToolRegistry + ContextBuilder + Provider）加起来也就 2000 行，完全可控。框架能省的时间，在调试和定制化阶段往往会加倍还回来。

另一个是 **token 预算管理是 Agent 系统里最被低估的问题**。工具调用会产生巨大文本，几轮下来 context 就爆了。截断在哪、会不会破坏 tool_calls 的关联，这些细节决定了 Agent 会不会突然"失忆"或"胡说"。

### Q21：如果重新做，有什么设计会改？

1. Provider 的消息转换应该用 pipeline 而非继承——有些转换是正交的（"去图片"和"去 reasoning_content"），职责链模式更灵活
2. Session save 应改成增量写，现在的全量覆盖在长对话时性能差
3. Tool concurrency 可以更精细——现在只分 safe/unsafe 两级，但文件写入之间有顺序依赖，当前模型不管这个

### Q22：这个项目和用 LangChain 搭的 Agent 有什么区别？

大多数 LangChain 项目是在别人的框架里写配置和 prompt。这个项目是从 agent loop 到 provider 适配到 RAG pipeline 全部自研。面试官如果懂技术，应该能看出**控制力**的差别——你能解释 agent 循环里每一行为什么这样写，而不是"这是 LangChain 的默认行为"。
