"""Agent core module."""

from legalbot.agent.context import ContextBuilder
from legalbot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from legalbot.agent.loop import AgentLoop
from legalbot.agent.memory import MemoryStore
from legalbot.agent.skills import SkillsLoader
from legalbot.agent.subagent import SubagentManager

__all__ = [
    "AgentHook",
    "AgentHookContext",
    "AgentLoop",
    "CompositeHook",
    "ContextBuilder",
    "MemoryStore",
    "SkillsLoader",
    "SubagentManager",
]
