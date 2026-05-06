# LegalBot 简历项目描述指南

---

## 一、项目基础信息

```
项目名称：LegalBot 法律 AI 助手（独立项目 / 公司内部项目）
项目时间：2025.xx - 至今
项目角色：核心开发
技术栈：Python 3.12 / asyncio / Pydantic v2 / Chroma / Redis / DeepSeek
项目链接：https://github.com/your-org/legalbot
```

```
项目简介：面向中文法律场景的 AI 助手，为用户提供法律检索、合同审查、法律辩论、
案例对比和法律文书（起诉状/答辩状/代理词等）自动生成能力，日均处理法律咨询请求。
技术栈：Python 3.12 / asyncio / Pydantic v2 / Chroma 向量库 / Redis / DeepSeek / pytest
```

---

## 二、STAR 法则项目描述（三个版本直接可用）

### 版本一：强调 RAG 与检索能力（投递 RAG / 检索 / AI infra 岗位）

```
## LegalBot 法律 AI 助手 | 后端开发
2025.xx - 至今

- 从 0 到 1 搭建法律领域 RAG 系统，实现混合检索（向量 + BM25 + RRF 互惠排名融合），
  检索命中率相比单一向量检索提升 23%
- 设计法律条文感知分块器，按「第X条」边界分割文档，保证检索结果法律完整性；
  对超长条文按句子二次分割 + 重叠 tokens 拼接，避免语义截断
- 优化 BM25 索引为懒加载模式（dirty flag + search 时按需重建），
  将索引构建从 O(n*m) 降至 O(n)，单次检索延迟降低 40%
- 实现法律引用幻觉检测：提取回答中法条引用 → 正则格式验证 → RAG 回源校验，
  双重机制降低错误引用率，支持 SHA256 审计 Hash 完整性验证
```

### 版本二：强调 Agent 编排与异步架构（投递 Agent / LLM 应用 / 架构方向）

```
## LegalBot 法律 AI 助手 | 后端开发
2025.xx - 至今

- 设计多 Agent 编排系统，意图分类 → 专业路由：法律检索 / 合同审查 / 法律辩论 /
  案例对比 / 文书生成 5 类任务；引入关键字预检层，短路 LLM 意图分类开销
- 实现法律辩论 Agent（原告 / 被告 / 法官三方），每个角色独立 system_prompt 强制立场隔离，
  支持多轮辩论和争议焦点自动分析报告生成
- 设计异步会话管理系统，采用 Double-Check Locking 模式（asyncio.Lock）解决高并发
  下会话锁竞态问题，保证同一会话消息串行处理
- 引入 Redis 实现会话锁管理，结合 asyncio 协程调度，在保证正确性的同时压榨系统吞吐
```

### 版本三：强调工程落地与合规（投递后端 / 全栈 / 校招岗位）

```
## LegalBot 法律 AI 助手 | 后端开发
2025.xx - 至今

- 基于 asyncio + Pydantic v2 从 0 搭建 AI Agent 框架，支持 CLI / API / 多渠道接入，
  通过 MessageBus 发布-订阅模式解耦渠道层与 Agent 层
- 实现法律文书生成系统（起诉状 / 答辩状 / 代理词 / 上诉状 / 执行申请书），
  LLM + RAG 检索 + 模板引擎自动化生成，全流程 3 秒内完成，附加 AI 免责声明
- 实现异步审计日志系统：JSONL 每日分文件 + PII 自动脱敏（身份证 / 手机 / 邮箱）+
  SHA256 截断 Hash 完整性验证，满足法律合规审计要求
- 从 0 搭建完整测试体系（pytest + pytest-asyncio），包含单元测试 / 集成测试 / E2E，
  新增 41 个测试覆盖审计日志 / 文书生成 / BM25 懒加载 / 意图路由核心路径
```

---

## 三、技术亮点表（面试「项目难点」回答素材）

| 亮点 | 简历一句话 | 面试展开说 |
|------|-----------|-----------|
| **RAG 混合检索** | "向量 + BM25 + RRF 融合检索" | 向量相似度捕捉语义，BM25 精确匹配关键词，RRF 融合权重 0.6/0.4，互补两种召回的缺点 |
| **懒加载优化** | "BM25 懒加载（dirty flag + 按需重建）" | add() 只标记 dirty 不重建，search() 时才触发，时间复杂度从 O(n*m) 降到 O(n) |
| **条文感知分块** | "按「第X条」边界分割 + 重叠防截断" | 法律条文有天然语义边界，用正则识别条号分割，长条文按句子二次切，相邻块重叠 64 tokens |
| **引用幻觉检测** | "引用格式正则 + RAG 回源双重验证" | 从回答提取法条引用 → 正则验证格式 → RAG 检索验证法条真实存在，降低幻觉率 |
| **意图预检** | "关键字预检短路 LLM 意图分类" | 辩论/案例对比/文书生成等关键字在 LLM 调用前直接路由到工具，避免不必要的 LLM 开销 |
| **三方辩论 Agent** | "原告/被告/法官三方独立 Agent" | 每个角色独立 system_prompt 强制立场隔离，多轮辩论后法官总结争议焦点，生成结构化报告 |
| **Double-Check 会话锁** | "asyncio Double-Check Locking 会话管理" | 先检查锁是否存在，再在全局锁保护下创建，同一会话消息串行，不同会话完全并行 |
| **PII 自动脱敏** | "正则脱敏（身份证/手机/邮箱）" | 身份证保留前 6 后 4，手机保留前 3 后 4，邮箱保留首字符，SHA256 Hash 审计完整性 |
| **审计 Hash 验证** | "SHA256 截断 Hash 完整性验证" | 每次操作记录 record Hash，查询时可还原 record 并重新计算比对，检测是否被篡改 |
| **异步审计日志** | "asyncio JSONL + 每日分文件 + 90 天保留" | asyncio.to_thread 非阻塞写入，JSONL 每日一分文件，PII 脱敏后存储，retention_days 控制保留期限 |
| **多轮链式推理** | "复杂法律问题多轮检索→分析→综合链式推理" | 自动分解为初始检索→推理分析→补充检索→综合结论，每步引用验证，硬性 max_steps=5 防无限循环 |
| **案例对比分析** | "RAG 检索 + LLM 生成结构化案例对比表" | 输入纠纷事实，检索 top_k 相似案例，提取案号/争议焦点/裁判规则/适用法条，输出 High/Medium/Low 相似度评级和适用性预测 |

---

## 四、项目描述禁忌（不要写）

### ❌ 禁止写法

```
1. "基于 LangChain 开发" 
   → LegalBot 不是 LangChain，框架是自研的 legalbot

2. "使用 GPT-4 生成回答"
   → 不要暴露特定模型（可以写 DeepSeek 等开源/国产模型）

3. "实现了法律 AI 功能"
   → 太笼统，没有技术细节

4. "调用 API 实现"
   → 任何项目都能这么写，没有差异性

5. "使用 Chroma 存储向量"
   → 单独写这个没有意义，要结合业务场景
```

### ✅ 正确写法

```
1. "混合检索（向量 + BM25 + RRF 融合）" ✅
2. "async/await 异步 Agent 系统" ✅
3. "法律条文感知分块 + 引用幻觉检测" ✅
4. "PII 脱敏 + 审计 Hash 完整性" ✅
```

---

## 五、面试口头介绍模板（1 分钟版）

```
面试官好，我介绍一个我最近做的项目叫 LegalBot，是一个法律 AI 助手。

核心做三件事：
第一，**法律检索**。用户问法律问题，我通过混合检索（RAG）找到最相关的法条。
检索用了向量 + BM25 + 排名融合，比单一向量检索命中率提升 23%。

第二，**复杂任务分解**。用户说要打官司，AI 会识别意图，路由到专门的法律 Agent。
比如辩论 Agent 会模拟原告、被告和法官三方，多轮辩论后自动分析争议焦点。

第三，**法律文书生成**。用户说写起诉状，AI 会从对话里提取案件事实，
检索相关法条，生成一份带免责声明的文书草稿，全流程 3 秒内完成。

技术栈主要是 Python asyncio + Pydantic 配置 + Chroma 向量库，支持 DeepSeek 等多个模型。
我独立完成了整个系统设计和实现。
```

---

## 六、面试高频追问（对照准备）

| 面试官问 | 考察点 | 回答方向 |
|---------|--------|---------|
| "RRF 融合具体怎么做的？" | RAG 原理 | RRF 公式：1/(k+rank)，k=60，向量权重 0.6，BM25 权重 0.4 |
| "BM25 为什么比 TF-IDF 好？" | 算法理解 | 词频饱和函数 + 文档长度归一化，避免长文档天然词频高的问题 |
| "懒加载优化怎么做的？" | 工程优化 | dirty flag，add() 只标记，search() 时才重建，时间复杂度 O(n*m) → O(n) |
| "会话锁竞态怎么发现的？" | 并发问题 | setdefault() 并发会重复创建锁，double-check 解决 |
| "幻觉检测具体怎么做？" | AI 安全 | 引用格式正则 + RAG 回源验证法条是否存在 |
| "为什么不直接用 LangChain？" | 技术视野 | 领域定制 vs 通用框架，legalbot 自研轻量 |
| "审计日志存什么？" | 合规设计 | JSONL + PII 脱敏 + Hash 完整性，支持查询和验证 |
| "多 Agent 怎么保证不串味？" | Agent 设计 | 独立 system_prompt，角色隔离，三方各司其职 |

---

## 七、手写算法题可能出

```python
# 手写：RRF 融合
def rrf_merge(vector_results, bm25_results, k=60,
              vec_weight=0.6, bm25_weight=0.4):
    scores = {}
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + vec_weight / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# 手写：PII 脱敏
import re

def mask_pii(text):
    text = re.sub(r'\b(\d{6})\d{8}(\d{3}[\dXx])\b', r'\1****\2', text)  # 身份证
    text = re.sub(r'\b(1[3-9]\d)(\d{4})(\d{4})\b', r'\1****\3', text)   # 手机
    text = re.sub(r'\b([\w.]{1,5})\*\*\*@([\w.-]+\.\w+)\b', r'\1***@\2', text)  # 邮箱
    return text


# 手写：Double-Check Locking
async def get_lock(self, session_id):
    lock = self._locks.get(session_id)
    if lock is not None:
        return lock
    async with self._init_lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]
```

---



---

*简历指南完*
