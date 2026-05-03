"""Harness-Auswahl per Name."""
from __future__ import annotations

from typing import Dict, List

from cascade.harness.base import Harness, HarnessName
from cascade.harness.claude_code import ClaudeCodeHarness
from cascade.harness.codex import CodexHarness


_HARNESSES: Dict[str, Harness] = {
    "claude-code": ClaudeCodeHarness(),
    "codex": CodexHarness(),
}


def get_harness(name: HarnessName) -> Harness:
    """Liefert Singleton der angeforderten Harness."""
    if name not in _HARNESSES:
        raise ValueError(
            f"Unbekannte Harness '{name}'. Bekannt: {sorted(_HARNESSES.keys())}"
        )
    return _HARNESSES[name]


def list_harnesses() -> List[str]:
    return sorted(_HARNESSES.keys())
