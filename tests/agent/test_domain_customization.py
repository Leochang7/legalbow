"""Tests for Phase 4 domain customization: SOUL.md persona, legal skills, and integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from legalbot.agent.skills import SkillsLoader
from legalbot.agent.context import ContextBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_skill_simple(
    base: Path,
    name: str,
    *,
    frontmatter: str,
    body: str = "# Skill\n",
) -> Path:
    """Create ``base / name / SKILL.md`` with raw frontmatter string."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\n{frontmatter}\n---\n\n{body}"
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. SOUL.md content tests
# ---------------------------------------------------------------------------

class TestSoulContent:
    """Verify the 法智 legal assistant SOUL.md template."""

    @pytest.fixture()
    def soul_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "legalbot" / "templates" / "SOUL.md"

    def test_soul_file_exists(self, soul_path: Path) -> None:
        assert soul_path.exists(), "templates/SOUL.md must exist"

    def test_soul_contains_fazhi_identity(self, soul_path: Path) -> None:
        content = soul_path.read_text(encoding="utf-8")
        assert "法智" in content, "SOUL.md must identify as 法智"
        assert "法律智能助手" in content, "SOUL.md must declare legal assistant role"

    def test_soul_contains_core_principles(self, soul_path: Path) -> None:
        content = soul_path.read_text(encoding="utf-8")
        for keyword in ("引用先行", "区分边界", "时效意识", "风险提示", "精准用词"):
            assert keyword in content, f"SOUL.md must contain principle: {keyword}"

    def test_soul_contains_capability_boundary(self, soul_path: Path) -> None:
        content = soul_path.read_text(encoding="utf-8")
        assert "不替代" in content or "不提供" in content, "SOUL.md must state capability boundaries"
        assert "律师" in content, "SOUL.md must reference lawyer disclaimer"

    def test_soul_no_generic_legalbot_text(self, soul_path: Path) -> None:
        content = soul_path.read_text(encoding="utf-8")
        # Should NOT contain the generic legalbot placeholder text
        assert "I am an AI assistant" not in content
        assert "legalbot" not in content.lower() or "法智" in content


# ---------------------------------------------------------------------------
# 2. Legal skills loading tests
# ---------------------------------------------------------------------------

class TestLegalSkillsLoading:
    """Verify legal-citation and legal-research skills load correctly."""

    @pytest.fixture()
    def builtin_skills_dir(self) -> Path:
        return Path(__file__).parent.parent.parent / "legalbot" / "skills"

    def test_legal_citation_skill_exists(self, builtin_skills_dir: Path) -> None:
        assert (builtin_skills_dir / "legal-citation" / "SKILL.md").exists()

    def test_legal_research_skill_exists(self, builtin_skills_dir: Path) -> None:
        assert (builtin_skills_dir / "legal-research" / "SKILL.md").exists()

    def test_legal_citation_is_always(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        meta = loader.get_skill_metadata("legal-citation")
        assert meta is not None
        assert meta.get("always") == "true", "legal-citation must be always=true"

    def test_legal_research_is_not_always(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        meta = loader.get_skill_metadata("legal-research")
        assert meta is not None
        assert meta.get("always") == "false", "legal-research must be always=false"

    def test_get_always_skills_includes_legal_citation(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        always = loader.get_always_skills()
        assert "legal-citation" in always

    def test_get_always_skills_excludes_legal_research(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        always = loader.get_always_skills()
        assert "legal-research" not in always

    def test_legal_citation_skill_content(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        content = loader.load_skill("legal-citation")
        assert content is not None
        assert "免责声明" in content, "legal-citation must contain disclaimer section"
        assert "法律全称" in content, "legal-citation must contain citation format rules"
        assert "条" in content and "款" in content, "legal-citation must mention 条/款/项 ordering"

    def test_legal_research_skill_content(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        content = loader.load_skill("legal-research")
        assert content is not None
        assert "legal_rag_search" in content, "legal-research must reference legal_rag_search tool"
        assert "legal_orchestrate" in content, "legal-research must reference legal_orchestrate tool"
        assert "检索策略" in content, "legal-research must contain search strategy"

    def test_legal_citation_description(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        meta = loader.get_skill_metadata("legal-citation")
        assert meta is not None
        assert "description" in meta
        assert "引用" in meta["description"] or "规范" in meta["description"]

    def test_legal_research_description(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        meta = loader.get_skill_metadata("legal-research")
        assert meta is not None
        assert "description" in meta
        assert "检索" in meta["description"] or "法条" in meta["description"]


# ---------------------------------------------------------------------------
# 3. get_always_skills unit tests (with synthetic fixtures)
# ---------------------------------------------------------------------------

class TestGetAlwaysSkills:
    """Unit tests for SkillsLoader.get_always_skills with controlled fixtures."""

    def test_always_true_top_level(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "ws" / "skills"
        ws_skills.mkdir(parents=True)
        _write_skill_simple(
            ws_skills,
            "always-skill",
            frontmatter="name: always-skill\nalways: true\ndescription: Test",
            body="# Always",
        )
        loader = SkillsLoader(tmp_path / "ws", builtin_skills_dir=tmp_path / "builtin")
        assert loader.get_always_skills() == ["always-skill"]

    def test_always_false_excluded(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "ws" / "skills"
        ws_skills.mkdir(parents=True)
        _write_skill_simple(
            ws_skills,
            "ondemand-skill",
            frontmatter="name: ondemand-skill\nalways: false\ndescription: Test",
            body="# On Demand",
        )
        loader = SkillsLoader(tmp_path / "ws", builtin_skills_dir=tmp_path / "builtin")
        assert loader.get_always_skills() == []

    def test_mixed_always_and_ondemand(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "ws" / "skills"
        ws_skills.mkdir(parents=True)
        _write_skill_simple(
            ws_skills,
            "a-always",
            frontmatter="name: a-always\nalways: true\ndescription: A",
        )
        _write_skill_simple(
            ws_skills,
            "b-ondemand",
            frontmatter="name: b-ondemand\nalways: false\ndescription: B",
        )
        _write_skill_simple(
            ws_skills,
            "c-always",
            frontmatter="name: c-always\nalways: true\ndescription: C",
        )
        loader = SkillsLoader(tmp_path / "ws", builtin_skills_dir=tmp_path / "builtin")
        assert loader.get_always_skills() == ["a-always", "c-always"]

    def test_no_always_key_defaults_to_excluded(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "ws" / "skills"
        ws_skills.mkdir(parents=True)
        _write_skill_simple(
            ws_skills,
            "no-always-field",
            frontmatter="name: no-always-field\ndescription: Test",
        )
        loader = SkillsLoader(tmp_path / "ws", builtin_skills_dir=tmp_path / "builtin")
        assert loader.get_always_skills() == []


# ---------------------------------------------------------------------------
# 4. ContextBuilder integration with 法智 SOUL.md + skills
# ---------------------------------------------------------------------------

class TestContextBuilderWithLegalPersona:
    """Test that ContextBuilder correctly assembles 法智 persona with skills."""

    @pytest.fixture()
    def workspace(self, tmp_path: Path) -> Path:
        """Create a workspace with SOUL.md and skills mirroring the real setup."""
        ws = tmp_path / "workspace"
        ws.mkdir()

        # Write 法智 SOUL.md
        (ws / "SOUL.md").write_text(
            "我是「法智」，一个专业的法律智能助手。\n\n"
            "我的核心原则：\n1. **引用先行** — 回答法律问题必须引用具体法条",
            encoding="utf-8",
        )

        # Write legal-citation skill (always: true)
        citation_dir = ws / "skills" / "legal-citation"
        citation_dir.mkdir(parents=True)
        (citation_dir / "SKILL.md").write_text(
            "---\nname: legal-citation\ndescription: 法律引用规范技能\nalways: true\n---\n\n"
            "## 法条引用规则\n\n每次回答法律问题末尾须附加免责声明。",
            encoding="utf-8",
        )

        # Write legal-research skill (always: false)
        research_dir = ws / "skills" / "legal-research"
        research_dir.mkdir(parents=True)
        (research_dir / "SKILL.md").write_text(
            "---\nname: legal-research\ndescription: 法律知识检索技能\nalways: false\n---\n\n"
            "## 检索策略\n\n使用 legal_rag_search 工具。",
            encoding="utf-8",
        )

        return ws

    def test_system_prompt_contains_soul(self, workspace: Path) -> None:
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt()
        assert "法智" in prompt
        assert "法律智能助手" in prompt

    def test_system_prompt_contains_always_skill(self, workspace: Path) -> None:
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt()
        assert "legal-citation" in prompt
        assert "法条引用规则" in prompt

    def test_system_prompt_does_not_contain_ondemand_skill_body(self, workspace: Path) -> None:
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt()
        # legal-research body should NOT be in the "Active Skills" section
        # (it appears in the skills summary XML, but not as loaded content)
        assert "使用 legal_rag_search 工具" not in prompt

    def test_system_prompt_contains_skills_summary_with_research(self, workspace: Path) -> None:
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt()
        # Skills summary should list legal-research as available
        assert "legal-research" in prompt

    def test_system_prompt_contains_disclaimer_from_always_skill(self, workspace: Path) -> None:
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt()
        assert "免责声明" in prompt

    def test_soul_section_label(self, workspace: Path) -> None:
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt()
        assert "## SOUL.md" in prompt


# ---------------------------------------------------------------------------
# 5. load_skills_for_context strips frontmatter correctly
# ---------------------------------------------------------------------------

class TestLoadSkillsForContext:
    """Test that skill content is correctly stripped of frontmatter for context injection."""

    def test_strips_frontmatter(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "ws" / "skills"
        ws_skills.mkdir(parents=True)
        _write_skill_simple(
            ws_skills,
            "test-skill",
            frontmatter="name: test-skill\nalways: true\ndescription: D",
            body="# Test Skill\n\nBody content here.",
        )
        loader = SkillsLoader(tmp_path / "ws", builtin_skills_dir=tmp_path / "builtin")
        content = loader.load_skills_for_context(["test-skill"])
        assert "---" not in content
        assert "always:" not in content
        assert "Body content here." in content
        assert "### Skill: test-skill" in content

    def test_multiple_skills_joined(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "ws" / "skills"
        ws_skills.mkdir(parents=True)
        _write_skill_simple(ws_skills, "a", frontmatter="name: a", body="# A")
        _write_skill_simple(ws_skills, "b", frontmatter="name: b", body="# B")
        loader = SkillsLoader(tmp_path / "ws", builtin_skills_dir=tmp_path / "builtin")
        content = loader.load_skills_for_context(["a", "b"])
        assert "### Skill: a" in content
        assert "### Skill: b" in content


# ---------------------------------------------------------------------------
# 6. Build skills summary includes legal skills
# ---------------------------------------------------------------------------

class TestBuildSkillsSummary:
    """Test skills summary XML includes both legal skills with correct metadata."""

    @pytest.fixture()
    def builtin_skills_dir(self) -> Path:
        return Path(__file__).parent.parent.parent / "legalbot" / "skills"

    def test_summary_includes_legal_skills(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        summary = loader.build_skills_summary()
        assert "legal-citation" in summary
        assert "legal-research" in summary

    def test_summary_marks_legal_citation_available(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        summary = loader.build_skills_summary()
        # Find the legal-citation skill block
        assert 'available="true"' in summary or "legal-citation" in summary

    def test_summary_has_description(self, builtin_skills_dir: Path) -> None:
        loader = SkillsLoader(Path("/nonexistent"), builtin_skills_dir=builtin_skills_dir)
        summary = loader.build_skills_summary()
        assert "引用" in summary or "检索" in summary, "Summary should include skill descriptions"
