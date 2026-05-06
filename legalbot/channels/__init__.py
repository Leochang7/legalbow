"""Chat channels module with plugin architecture."""

from legalbot.channels.base import BaseChannel
from legalbot.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
