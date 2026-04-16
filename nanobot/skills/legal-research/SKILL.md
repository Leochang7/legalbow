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

## 检索工具使用

```
legal_rag_search(
    query="加班费 劳动报酬",
    law_area="劳动法",
    top_k=5
)
```

参数说明：
- `query`: 提取的核心法律概念，用空格分隔关键词
- `law_area`: 可选，法律领域过滤（民法/刑法/商法/劳动法/行政法）
- `doc_type`: 可选，文档类型过滤（law/judicial_interpretation/case/contract_template）
- `top_k`: 返回结果数，默认5

## 复杂问题处理

对于复杂法律问题，使用 legal_orchestrate 工具调度专业 Agent：
```
legal_orchestrate(
    query="用人单位拖欠工资且不签劳动合同如何维权",
    intent="legal_query"  # 可选，不传则自动分类
)
```

## 检索结果解读

每个检索结果包含：
- 法规名称和条文号
- 法律领域和文档类型
- 条文内容摘要

优先使用排名靠前的结果，但需验证法条是否现行有效。
