"""Message bus module for decoupled channel-agent communication."""

from legalbot.bus.events import InboundMessage, OutboundMessage
from legalbot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
