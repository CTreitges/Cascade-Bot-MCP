"""Harness-Abstraktion: jede Rolle (Plan/Implement/Review/Sub-Agent) läuft durch
eine Harness, die Tools/MCPs/Sandbox bereitstellt — Modell-Wahl ist orthogonal.

Aktuell unterstützt: ClaudeCodeHarness (claude-agent-sdk + claude-code-router).
Codex-Adapter ist als Stub vorbereitet, aber bewusst nicht implementiert.
"""
from cascade.harness.base import (
    Harness,
    HarnessRequest,
    HarnessResult,
    HarnessEvent,
    ToolUseEvent,
    ToolResultEvent,
    AssistantTextEvent,
    DoneEvent,
    ToolCall,
    TokenUsage,
)
from cascade.harness.router import get_harness, list_harnesses

__all__ = [
    "Harness",
    "HarnessRequest",
    "HarnessResult",
    "HarnessEvent",
    "ToolUseEvent",
    "ToolResultEvent",
    "AssistantTextEvent",
    "DoneEvent",
    "ToolCall",
    "TokenUsage",
    "get_harness",
    "list_harnesses",
]
