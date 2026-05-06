"""Case comparison data models."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CaseCoreData:
    """Extracted structured data from a single case."""
    case_no: str | None = None          # 案号
    case_name: str | None = None        # 案件名称
    court: str | None = None             # 审理法院
    judge_date: str | None = None        # 裁判日期
    dispute_type: str | None = None     # 纠纷类型
    dispute_focus: str | None = None    # 争议焦点
    ruling_rule: str | None = None       # 裁判规则
    applicable_laws: list[str] = field(default_factory=list)  # 适用法条
    source_chunk_id: str | None = None  # 来源 chunk ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaseCoreData":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApplicabilityPrediction:
    """Prediction of how applicable a case is to user's dispute."""
    most_similar_case: str           # Case number
    similarity_score: str             # "High" | "Medium" | "Low"
    key_similarities: list[str]      # Shared fact patterns
    suggested_strategies: list[str]   # Recommended strategies
    risk_warnings: list[str]         # Key risks
    applicable_laws: list[str]        # Most relevant laws


@dataclass
class CaseCompareConfig:
    """案例对比分析配置."""
    enable: bool = True
    comparison_model: str = ""  # empty = use default
    max_cases: int = 10
    top_k_default: int = 5
