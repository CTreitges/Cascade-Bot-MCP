"""CodexHarness — STUB.

Bewusst nicht implementiert (User-Vorgabe 2026-05-03: „Codex erstmal außen
vor lassen"). Skeleton-Klasse damit `cascade.harness.router.get_harness('codex')`
einen klaren Fehler wirft und die Architektur erkennbar bleibt für die spätere
Erweiterung.
"""
from __future__ import annotations

from typing import Optional

from cascade.harness.base import EventCallback, HarnessRequest, HarnessResult


class CodexHarness:
    name = "codex"

    async def run(
        self,
        request: HarnessRequest,
        on_event: Optional[EventCallback] = None,
    ) -> HarnessResult:
        raise NotImplementedError(
            "CodexHarness ist noch nicht implementiert. "
            "Aktuell wird ausschließlich ClaudeCodeHarness unterstützt. "
            "Erweiterung via @openai/codex CLI Subprocess + JSON-Stream "
            "ist als Phase A im Plan v4 vorgesehen, aber zurückgestellt."
        )
