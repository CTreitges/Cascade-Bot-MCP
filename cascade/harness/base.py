"""Common types + Protocol für alle Harness-Implementierungen.

Eine Harness führt einen Agent-Run durch, der ein Tool-Set in einem Workspace
benutzt. Sie streamt strukturierte Events während des Runs und liefert am
Ende ein HarnessResult mit Tool-Calls, Files-Diff, Token-Verbrauch.

Provider/Modell-Wahl ist Teil des HarnessRequest — die Harness selbst macht
ggf. Routing (z.B. ClaudeCodeHarness → claude-code-router für non-Anthropic-
Modelle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)


HarnessName = Literal["claude-code", "codex"]
ProviderName = Literal["anthropic", "openai", "ollama"]
RoleName = Literal["planner", "implementer", "reviewer", "subagent", "triage", "quick-review"]


# ──────────────────────────────────────────────────────────────────────────────
#  Datenklassen
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


@dataclass
class ToolCall:
    """Ein einzelner Tool-Aufruf während des Agent-Runs (für Logging + Telemetry)."""
    name: str
    args: Dict[str, Any]
    is_error: bool = False
    result_preview: str = ""
    duration_ms: int = 0
    started_at: float = 0.0


@dataclass
class HarnessRequest:
    """Eingabe für jede Harness-Run.

    permission_mode wird absichtlich NICHT konfigurierbar gehalten — alle
    Harnesses laufen mit bypassPermissions (User-Vorgabe 2026-05-03). Edits
    werden ohne Rückfrage akzeptiert; Workspaces sollen pro Sub-Task isoliert
    sein, dann ist das sicher.
    """
    role: RoleName
    harness: HarnessName = "claude-code"
    provider: ProviderName = "anthropic"
    model: str = "claude-sonnet-4-6"

    prompt: str = ""
    system: Optional[str] = None
    allowed_tools: List[str] = field(
        default_factory=lambda: ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
    )
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    cwd: Path = field(default_factory=lambda: Path("."))
    max_turns: int = 20
    timeout_s: int = 600
    max_cost_usd: Optional[float] = None
    enable_subagents: bool = False  # gibt das Task-Tool frei

    # Routing-Hint: wenn provider != "anthropic" und harness == "claude-code",
    # wird claude-code-router (Port 3456 default) für die SDK-Calls genutzt.
    router_url: str = "http://127.0.0.1:3456"


@dataclass
class HarnessResult:
    """Ergebnis eines Agent-Runs."""
    final_text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0  # 0.0 wenn Provider kein Cost-Tracking liefert (z.B. Ollama)
    wall_clock_s: float = 0.0
    num_turns: int = 0

    # Optional: was hat die Harness im Workspace verändert?
    # Wird vom Caller via worktree-diff nachgeholt, wenn die Harness es nicht selbst tracken kann.
    files_changed: List[str] = field(default_factory=list)

    # True wenn Run sauber durch den DoneEvent abgeschlossen — False bei Timeout / Crash.
    success: bool = True
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
#  Stream-Events (alle Harnesses emittieren in derselben Form)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class HarnessEvent:
    """Basis für alle Stream-Events während eines Runs."""
    timestamp: float = 0.0


@dataclass
class ToolUseEvent(HarnessEvent):
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""


@dataclass
class ToolResultEvent(HarnessEvent):
    tool_id: str = ""
    is_error: bool = False
    content_preview: str = ""


@dataclass
class AssistantTextEvent(HarnessEvent):
    text: str = ""


@dataclass
class DoneEvent(HarnessEvent):
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    num_turns: int = 0
    success: bool = True
    error: Optional[str] = None


# Callback-Signatur für Live-Stream (Telegram-Bot-Integration etc.)
EventCallback = Callable[[HarnessEvent], Awaitable[None]]


# ──────────────────────────────────────────────────────────────────────────────
#  Protocol
# ──────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class Harness(Protocol):
    """Ein Provider+Modell-agnostischer Agent-Runner."""

    name: HarnessName

    async def run(
        self,
        request: HarnessRequest,
        on_event: Optional[EventCallback] = None,
    ) -> HarnessResult:
        """Führt einen Run aus und gibt das aggregierte Ergebnis zurück.

        Wenn `on_event` angegeben ist, werden Stream-Events live ausgegeben —
        nützlich für Telegram-Bot-Heartbeats oder Logging.
        """
        ...
