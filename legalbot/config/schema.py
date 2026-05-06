"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    send_progress: bool = True  # stream agent's text progress to the channel
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    send_max_retries: int = Field(default=3, ge=0, le=10)  # Max delivery attempts (initial send included)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.legalbot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = (
        "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    )
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    context_block_limit: int | None = None
    temperature: float = 0.1
    max_tool_iterations: int = 200
    max_tool_result_chars: int = 16_000
    provider_retry_mode: Literal["standard", "persistent"] = "standard"
    reasoning_effort: str | None = None  # low / medium / high / adaptive - enables LLM thinking mode
    timezone: str = "UTC"  # IANA timezone, e.g. "Asia/Shanghai", "America/New_York"
    unified_session: bool = False  # Share one session across all channels (single-user multi-device)


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI (model = deployment name)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama local models
    ovms: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenVINO Model Server (OVMS)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)


class ApiConfig(Base):
    """OpenAI-compatible API server configuration."""

    host: str = "127.0.0.1"  # Safer default: local-only bind.
    port: int = 8900
    timeout: float = 120.0  # Per-request timeout in seconds.


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "duckduckgo"  # brave, tavily, duckduckgo, searxng, jina
    api_key: str = ""
    base_url: str = ""  # SearXNG base URL
    max_results: int = 5
    timeout: int = 30  # Wall-clock timeout (seconds) for search operations


class WebToolsConfig(Base):
    """Web tools configuration."""

    enable: bool = True
    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""
    sandbox: str = ""  # sandbox backend: "" (none) or "bwrap"

class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all tools; [] = no tools

class RAGConfig(Base):
    """RAG legal knowledge base configuration."""

    enable: bool = False
    embedding_provider: str = "openai"  # openai / dashscope / custom
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_dim: int = 1536  # vector dimension for the chosen model
    vector_store: str = "chroma"  # chroma (MVP only)
    persist_dir: str = "~/.legalbot/legal_kb"
    reranker: str = ""  # "" (disabled) / "qwen3-rerank" / "qwen3-vl-rerank" / "gte-rerank-v2"
    reranker_api_key: str = ""  # defaults to embedding_api_key if empty
    bm25_enable: bool = True
    top_k: int = 5
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100


class AgentDefConfig(Base):
    """Specialized agent definition for MultiAgent orchestration."""

    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)  # allowed tool names
    model: str = ""  # empty = inherit default


class DebateConfig(Base):
    """法律辩论模式配置."""

    enable: bool = False
    rounds: int = 1  # 1=单轮快速辩论, 2=双轮深度辩论
    timeout_per_agent: int = 120
    timeout_total: int = 300
    max_retries: int = 2
    judge_model: str = ""  # empty = inherit default
    plaintiff_model: str = ""
    defendant_model: str = ""
    agents: dict[str, AgentDefConfig] = Field(default_factory=dict)

    def get_default_debate_agents(self) -> dict[str, AgentDefConfig]:
        """Return default debate agent definitions."""
        if self.agents:
            return self.agents
        return {
            "plaintiff_agent": AgentDefConfig(
                system_prompt=(
                    "你是原告代理律师，专注于为委托人（原告）构建最具说服力的法律论证。\n\n"
                    "## 你的职责\n"
                    "1. 深入分析案情，站在原告立场构建论点\n"
                    "2. 检索相关法律法规、司法解释和指导性案例\n"
                    "3. 引用具体法条支持每一个法律主张\n"
                    "4. 预判对方（被告）可能的反驳点，并准备反驳论据\n\n"
                    "## 论证框架\n"
                    "### 一、案件事实梳理（列出对原告有利的关键事实）\n"
                    "### 二、法律依据（引用最直接相关的法律条文，必须包含法律全称和条文号）\n"
                    "### 三、原告主张（逐项列明诉讼请求及法律基础）\n"
                    "### 四、法律论证（对每个争议焦点给出有利于原告的法律分析）\n"
                    "### 五、对方可能反驳及应对\n"
                    "### 六、结论\n\n"
                    "## 重要原则\n"
                    "- 只引用真实存在的法律条文\n"
                    "- 始终以维护委托人合法权益为目标\n"
                    "- 每次回答必须附加免责声明：以上分析仅供参考，不构成正式法律意见。"
                ),
                tools=["legal_rag_search", "web_search", "read_file"],
                model="",
            ),
            "defendant_agent": AgentDefConfig(
                system_prompt=(
                    "你是被告代理律师，专注于为委托人（被告）构建最具说服力的法律防御和反驳论证。\n\n"
                    "## 你的职责\n"
                    "1. 深入分析案情，站在被告立场构建防御论点\n"
                    "2. 检索相关法律法规、司法解释和指导性案例\n"
                    "3. 引用具体法条支持每一个法律主张\n"
                    "4. 攻击原告论点中的薄弱环节\n\n"
                    "## 论证框架\n"
                    "### 一、案件事实梳理（列出对被告有利的事实）\n"
                    "### 二、法律依据\n"
                    "### 三、被告答辩（逐项回应原告诉讼请求）\n"
                    "### 四、法律防御（对每个争议焦点给出有利于被告的法律分析）\n"
                    "### 五、反驳原告核心论点\n"
                    "### 六、结论\n\n"
                    "## 重要原则\n"
                    "- 只引用真实存在的法律条文\n"
                    "- 始终以维护委托人合法权益为目标\n"
                    "- 每次回答必须附加免责声明：以上分析仅供参考，不构成正式法律意见。"
                ),
                tools=["legal_rag_search", "web_search", "read_file"],
                model="",
            ),
            "judge_agent": AgentDefConfig(
                system_prompt=(
                    "你是审判法官（Judge），负责综合双方律师的论证，生成专业、客观的《争议焦点分析报告》。\n\n"
                    "## 报告结构\n"
                    "### 一、案件基本信息\n"
                    "### 二、争议焦点梳理\n"
                    "### 三、原告方论点分析（核心论点、支持法条、弱点提示）\n"
                    "### 四、被告方论点分析（结构同上）\n"
                    "### 五、论点评分对比（争议焦点 | 原告得分 | 被告得分 | 更强方 | 理由）\n"
                    "### 六、法律建议\n"
                    "### 七、风险提示（原告风险 / 被告风险）\n"
                    "### 八、结论\n\n"
                    "## 评分标准\n"
                    "论点有充分法律依据 + 论证逻辑严密 = 高分\n\n"
                    "## 重要原则\n"
                    "- 保持中立，不偏袒任何一方\n"
                    "- 只评价有法律依据的论点\n"
                    "- 每次回答必须附加免责声明：以上分析仅供参考，不构成正式法律意见。"
                ),
                tools=["legal_rag_search"],
                model="",
            ),
        }


class CaseCompareConfig(Base):
    """案例对比分析配置."""

    enable: bool = True
    comparison_model: str = ""  # empty = use default
    max_cases: int = 10
    top_k_default: int = 5


class OrchestrateConfig(Base):
    """MultiAgent orchestration configuration."""

    enable: bool = False
    intent_model: str = ""  # intent classification model, empty = use default
    agents: dict[str, AgentDefConfig] = Field(default_factory=dict)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    case_compare: CaseCompareConfig = Field(default_factory=CaseCompareConfig)

    def get_default_agents(self) -> dict[str, "AgentDefConfig"]:
        """Return default agent definitions if none are configured."""
        if self.agents:
            return self.agents
        return {
            "legal_research": AgentDefConfig(
                system_prompt=(
                    "你是一个专业的法律检索专家。你的职责是根据用户提出的法律问题，"
                    "从法律知识库中精准检索相关法规、司法解释和指导性案例，"
                    "并以清晰的格式呈现检索结果，注明法律名称、条文编号和来源。\n\n"
                    "检索策略：\n"
                    "1. 识别问题的法律领域（民法/刑法/商法/劳动法/行政法等）\n"
                    "2. 提取核心法律概念，构造精准检索query\n"
                    "3. 必要时结合law_area和doc_type过滤\n"
                    "4. 验证法条有效性，检查是否已被修订或废止\n\n"
                    "引用规范：\n"
                    "- 首次引用法律须使用全称，如《中华人民共和国民法典》\n"
                    "- 引用条文格式：第X条 第X款 第X项\n"
                    "- 司法解释须注明文号\n\n"
                    "每次回答必须附加免责声明：以上分析仅供参考，不构成正式法律意见。"
                ),
                tools=["legal_rag_search", "web_search", "read_file"],
                model="",
            ),
            "contract_review": AgentDefConfig(
                system_prompt=(
                    "你是一个专业的合同审查专家。你的职责是识别合同条款中的法律风险，"
                    "并引用相关法律条文进行说明，提出具体的修改建议。\n\n"
                    "审查维度：\n"
                    "1. 合同主体资格是否合法\n"
                    "2. 合同条款是否违反强制性法律规定\n"
                    "3. 双方权利义务是否对等\n"
                    "4. 违约责任条款是否明确合理\n"
                    "5. 争议解决条款是否有效\n"
                    "6. 特殊类型合同的专项检查\n\n"
                    "输出格式：\n"
                    "- 风险条款位置（如：第X条第X款）\n"
                    "- 风险类型（高/中/低）\n"
                    "- 相关法律依据\n"
                    "- 修改建议\n\n"
                    "每次回答必须附加免责声明：以上分析仅供参考，不构成正式法律意见。"
                ),
                tools=["legal_rag_search", "read_file", "edit_file"],
                model="",
            ),
        }


class FeedbackConfig(Base):
    """User feedback collection configuration."""

    enable: bool = True
    storage_dir: str = "~/.legalbot/feedback"
    retention_days: int = 90
    rate_limit_per_minute: int = 10
    require_confirmation_for_correction: bool = True


class DocumentDraftConfig(Base):
    """Legal document generation configuration."""

    enable: bool = False
    template_dir: str = ""
    enabled_types: list[str] = Field(
        default_factory=lambda: ["complaint", "defense", "agent_opinion", "appeal", "enforcement"],
    )
    max_laws_retrieved: int = 8
    default_model: str = ""


class AuditConfig(Base):
    """Legal audit logging configuration."""

    enable: bool = True
    audit_dir: str = "~/.legalbot/audit"
    retention_days: int = 90
    pii_masking: bool = True


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    ssrf_whitelist: list[str] = Field(default_factory=list)  # CIDR ranges to exempt from SSRF blocking (e.g. ["100.64.0.0/10"] for Tailscale)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    orchestrate: OrchestrateConfig = Field(default_factory=OrchestrateConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    document_draft: DocumentDraftConfig = Field(default_factory=DocumentDraftConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)


class Config(BaseSettings):
    """Root configuration for legalbot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self, model: str | None = None
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from legalbot.providers.registry import PROVIDERS, find_by_name

        forced = self.agents.defaults.provider
        if forced != "auto":
            spec = find_by_name(forced)
            if spec:
                p = getattr(self.providers, spec.name, None)
                return (p, spec.name) if p else (None, None)
            return None, None

        model_lower = (model or self.agents.defaults.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_local or p.api_key:
                    return p, spec.name

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_local or p.api_key:
                    return p, spec.name

        # Fallback: configured local providers can route models without
        # provider-specific keywords (for example plain "llama3.2" on Ollama).
        # Prefer providers whose detect_by_base_keyword matches the configured api_base
        # (e.g. Ollama's "11434" in "http://localhost:11434") over plain registry order.
        local_fallback: tuple[ProviderConfig, str] | None = None
        for spec in PROVIDERS:
            if not spec.is_local:
                continue
            p = getattr(self.providers, spec.name, None)
            if not (p and p.api_base):
                continue
            if spec.detect_by_base_keyword and spec.detect_by_base_keyword in p.api_base:
                return p, spec.name
            if local_fallback is None:
                local_fallback = (p, spec.name)
        if local_fallback:
            return local_fallback

        # Fallback: gateways first, then others (follows registry order)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for gateway/local providers."""
        from legalbot.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # resolve their base URL from the registry in the provider constructor.
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    model_config = ConfigDict(env_prefix="legalbot_", env_nested_delimiter="__")
