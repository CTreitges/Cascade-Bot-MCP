"""Plan v5 R5 — Observability: structured logging + JSONL-Export + Trace-IDs.

Inspiration: Ruflo's observability-Plugin. Cascade hat schon error_log.py
+ telegram_audit, aber für Production-Debugging fehlt:
  - Korrelierte Events: alle Events eines Runs/Sub-Tasks haben gleiche trace_id
  - Strukturiertes JSONL für externes Tooling (Grafana/Datadog/etc.)
  - Per-Run-Summary: total-cost, total-tokens, tool-call-Histogram, success-rate

Design:
  - TraceContext (contextvar) → trace_id wird automatisch in alle log-Records gemerged
  - JSONLEmitter: append-only, eine Zeile pro Event, rotiert bei N MB
  - RunSummary builder: aggregiert Events eines Runs am Ende

Speicherort: <CASCADE_HOME>/store/metrics.jsonl
Format: {ts, trace_id, role, event, payload}

Event-Beispiele:
  {"ts": …, "trace_id": "abc", "role": "implementer", "event": "llm_call",
   "payload": {"model": "claude-sonnet-4-6", "input_tokens": 1500,
               "output_tokens": 320, "cost_usd": 0.012, "latency_ms": 4500,
               "provider": "anthropic"}}
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("cascade.observability")


# ──────────────────────────────────────────────────────────────────────
#  Trace-Context (contextvar)
# ──────────────────────────────────────────────────────────────────────
_TRACE_ID: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_TASK_ID: ContextVar[Optional[str]] = ContextVar("task_id", default=None)
_SUBTASK: ContextVar[Optional[str]] = ContextVar("subtask", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_trace_context(
    *,
    trace_id: Optional[str] = None,
    task_id: Optional[str] = None,
    subtask: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Setzt Trace-Context für nachfolgende Events. Returns alte Werte
    fürs zurücksetzen (try/finally pattern)."""
    old = {
        "trace_id": _TRACE_ID.get(),
        "task_id": _TASK_ID.get(),
        "subtask": _SUBTASK.get(),
    }
    if trace_id is not None:
        _TRACE_ID.set(trace_id)
    if task_id is not None:
        _TASK_ID.set(task_id)
    if subtask is not None:
        _SUBTASK.set(subtask)
    return old


def restore_trace_context(prev: Dict[str, Optional[str]]) -> None:
    _TRACE_ID.set(prev.get("trace_id"))
    _TASK_ID.set(prev.get("task_id"))
    _SUBTASK.set(prev.get("subtask"))


def current_trace() -> Dict[str, Optional[str]]:
    return {
        "trace_id": _TRACE_ID.get(),
        "task_id": _TASK_ID.get(),
        "subtask": _SUBTASK.get(),
    }


# ──────────────────────────────────────────────────────────────────────
#  JSONL-Emitter
# ──────────────────────────────────────────────────────────────────────
@dataclass
class JSONLEmitter:
    """Append-only JSONL-Logger mit size-based-Rotation.

    Jede Zeile = 1 JSON-Object: {ts, trace_id, task_id, subtask, event, payload}
    Bei file > rotate_at_mb wird zu file.1, .2, … rotiert (max keep_files).
    """
    path: Path
    rotate_at_mb: float = 50.0
    keep_files: int = 5
    enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.time(),
            "event": event,
            **{k: v for k, v in current_trace().items() if v is not None},
            "payload": payload or {},
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"observability: failed to serialize {event}: {e}")
            return
        try:
            self._maybe_rotate()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.warning(f"observability: failed to write {event}: {e}")

    def _maybe_rotate(self) -> None:
        try:
            if not self.path.exists():
                return
            size_mb = self.path.stat().st_size / (1024 * 1024)
            if size_mb < self.rotate_at_mb:
                return
            # rotate: file → file.1 → file.2 → … → drop oldest
            for i in range(self.keep_files - 1, 0, -1):
                older = self.path.with_suffix(self.path.suffix + f".{i}")
                newer_idx = i + 1
                target = self.path.with_suffix(self.path.suffix + f".{newer_idx}")
                if older.exists():
                    if newer_idx > self.keep_files:
                        older.unlink(missing_ok=True)
                    else:
                        older.rename(target)
            first = self.path.with_suffix(self.path.suffix + ".1")
            self.path.rename(first)
        except Exception as e:
            logger.debug(f"observability: rotation failed: {e}")


# ──────────────────────────────────────────────────────────────────────
#  Per-Run-Summary
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RunSummary:
    """Aggregiert Events eines Runs für Final-Card / RLM-Insight."""
    trace_id: str
    task_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_tokens: int = 0
    by_role_cost: Dict[str, float] = field(default_factory=dict)
    by_model_cost: Dict[str, float] = field(default_factory=dict)
    by_provider_cost: Dict[str, float] = field(default_factory=dict)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    failover_attempts: int = 0
    errors: List[str] = field(default_factory=list)
    success: bool = False

    def add_llm_call(
        self,
        *,
        role: str,
        model: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.total_cost_usd += cost_usd
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_tokens += cache_tokens
        self.by_role_cost[role] = self.by_role_cost.get(role, 0.0) + cost_usd
        self.by_model_cost[model] = self.by_model_cost.get(model, 0.0) + cost_usd
        self.by_provider_cost[provider] = self.by_provider_cost.get(provider, 0.0) + cost_usd

    def add_tool_call(self, tool_name: str) -> None:
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

    def add_failover_attempt(self) -> None:
        self.failover_attempts += 1

    def add_error(self, msg: str) -> None:
        self.errors.append(msg[:300])

    def finalize(self, success: bool) -> Dict[str, Any]:
        self.ended_at = time.time()
        self.success = success
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        wall = (self.ended_at or time.time()) - self.started_at
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_clock_s": round(wall, 1),
            "success": self.success,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "cache": self.total_cache_tokens,
            },
            "by_role_cost": {k: round(v, 6) for k, v in self.by_role_cost.items()},
            "by_model_cost": {k: round(v, 6) for k, v in self.by_model_cost.items()},
            "by_provider_cost": {k: round(v, 6) for k, v in self.by_provider_cost.items()},
            "tool_call_counts": dict(self.tool_call_counts),
            "failover_attempts": self.failover_attempts,
            "errors": self.errors[:10],
        }

    def render_telegram(self, lang: str = "de") -> str:
        """Kompakter Final-Card-Text fürs Telegram-UI."""
        wall = (self.ended_at or time.time()) - self.started_at
        m, s = divmod(int(wall), 60)
        ok = "✅" if self.success else "❌"
        lines = [
            f"{ok} Run-Summary  • {m}:{s:02d}",
            f"💰 Total: ${self.total_cost_usd:.4f}",
        ]
        if self.by_role_cost:
            lines.append("By role:")
            for role, cost in sorted(self.by_role_cost.items(), key=lambda x: -x[1]):
                lines.append(f"  {role}: ${cost:.4f}")
        if self.tool_call_counts:
            top_tools = sorted(self.tool_call_counts.items(), key=lambda x: -x[1])[:5]
            lines.append("Top tools: " + ", ".join(f"{t}({n})" for t, n in top_tools))
        if self.failover_attempts:
            lines.append(f"⚠️ Failover: {self.failover_attempts} attempts")
        if self.errors:
            lines.append(f"❌ Errors ({len(self.errors)}):")
            for e in self.errors[:3]:
                lines.append(f"  • {e[:100]}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  Singleton-Konvenience
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_EMITTER: Optional[JSONLEmitter] = None


def configure_emitter(
    *,
    path: Path | str,
    rotate_at_mb: float = 50.0,
    keep_files: int = 5,
    enabled: bool = True,
) -> JSONLEmitter:
    """Setup the process-global emitter (typisch beim Bot-Start)."""
    global _DEFAULT_EMITTER
    _DEFAULT_EMITTER = JSONLEmitter(
        path=Path(path),
        rotate_at_mb=rotate_at_mb,
        keep_files=keep_files,
        enabled=enabled,
    )
    return _DEFAULT_EMITTER


def emit(event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Convenience: emit via global emitter wenn konfiguriert."""
    if _DEFAULT_EMITTER is None:
        return
    _DEFAULT_EMITTER.emit(event, payload)
