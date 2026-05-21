"""Conversation memory for the SCF agent."""

from __future__ import annotations

from langchain_core.messages import BaseMessage


class ConversationMemory:
    """Sliding-window message history for LangGraph ReAct agent (last k turns)."""

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self._messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._messages.extend(messages)
        # keep last k human+ai pairs (2*k messages)
        self._messages = self._messages[-(self.k * 2):]

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []


def get_memory() -> ConversationMemory:
    return ConversationMemory(k=10)
