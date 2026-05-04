"""Sub-Task-Runner für den Orchestrator.

Statt der existing JSON-only call_implementer-Logik in core.py rennt hier
ein einzelner Sub-Task als ECHTER Agent-Run mit Tool-Access via
ClaudeCodeHarness. Der Implementer kann Read/Edit/Write/Bash/Glob/Grep
nutzen — also exakt wie Claude Code direkt.

Per Sub-Task wird ein Prompt gebaut der dem Modell sagt:
  - Was die Aufgabe ist (sub-task summary + steps + acceptance_criteria)
  - Welche Files er anfassen soll (files_to_touch — als Hinweis, nicht
    als Constraint)
  - Welche Quality-Checks er bestehen muss (am Ende selbst laufen lassen
    können via Bash)
  - Wo er Code findet (cwd ist worktree-Pfad)

Der Run endet wenn das Modell sagt es ist fertig (max_turns oder freiwillig).
Final wird ein Diff aus dem Worktree extrahiert.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from cascade.harness import (
    AssistantTextEvent,
    DoneEvent,
    HarnessEvent,
    HarnessRequest,
    ToolResultEvent,
    ToolUseEvent,
    get_harness,
)
from cascade.orchestrator.result import SubTaskResult
from cascade.orchestrator.worktree import Worktree, WorktreeManager
from cascade.role_config import RoleConfig


logger = logging.getLogger("cascade.orchestrator.runner")


def _build_subtask_prompt(plan_summary: str, sub_task: Any, prior_results: Dict[str, SubTaskResult]) -> str:
    """Baut einen klaren Implementer-Prompt für genau diesen Sub-Task."""
    parts = [
        f"# OBERSTE AUFGABE\n{plan_summary}\n",
        f"# DEIN SUB-TASK: {sub_task.name}\n",
        f"## Zusammenfassung\n{sub_task.summary}",
    ]
    if sub_task.steps:
        parts.append("## Schritte\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(sub_task.steps, 1)))
    if sub_task.files_to_touch:
        parts.append(
            "## Voraussichtlich relevante Files (Hinweis, nicht zwingend)\n"
            + "\n".join(f"- `{f}`" for f in sub_task.files_to_touch)
        )
    if sub_task.acceptance_criteria:
        parts.append(
            "## Akzeptanzkriterien — diese MÜSSEN am Ende erfüllt sein\n"
            + "\n".join(f"- {c}" for c in sub_task.acceptance_criteria)
        )
    if sub_task.quality_checks:
        parts.append(
            "## Quality-Checks die nach deinem Run laufen werden\n"
            "Stelle sicher dass deine Implementierung diese besteht. Du kannst sie "
            "selbst via Bash laufen lassen wenn du willst:\n"
            + "\n".join(
                f"- `{getattr(c, 'name', '?')}`: `{getattr(c, 'command', '?')}`"
                for c in sub_task.quality_checks
            )
        )
    if prior_results:
        parts.append("## Was vorherige Sub-Tasks bereits geliefert haben")
        for dep_name, dep_res in prior_results.items():
            files = ", ".join(dep_res.files_changed[:6]) or "—"
            parts.append(f"- **{dep_name}** ({dep_res.status}): {files}")
    parts.append(
        "\n# WORKFLOW\n"
        "Du arbeitest in einem isolierten git worktree. Nutze Read/Glob/Grep um "
        "den Code zu erkunden, Edit/Write um Änderungen zu machen, Bash für Tests "
        "und Verifikation. Wenn du fertig bist, schreibe in 1-2 Sätzen WAS du "
        "geändert hast und WARUM."
    )
    return "\n\n".join(parts)


async def run_subtask_via_harness(
    *,
    plan_summary: str,
    sub_task: Any,
    role_config: RoleConfig,
    workspace_path: Path | str,
    prior_results: Optional[Dict[str, SubTaskResult]] = None,
    on_event: Optional[Callable[[str, HarnessEvent], Awaitable[None]]] = None,
    max_turns: Optional[int] = None,
) -> SubTaskResult:
    """Führt einen Sub-Task als Agent-Run via ClaudeCodeHarness aus.

    Args:
        plan_summary:    Top-Level-Plan-Summary (für Kontext)
        sub_task:        cascade.agents.planner.SubTask (oder kompatibel)
        role_config:     RoleConfig für Implementer-Rolle (model/provider/harness)
        workspace_path:  cwd für die Harness (= worktree-pfad bei Orchestrator-Run)
        prior_results:   Ergebnisse anderer Sub-Tasks die bereits liefen
                         (für Kontext welche Files schon angefasst wurden)
        on_event:        Callback (sub_task_name, event) — für Telegram-Live-
                         Stream (Phase H)
        max_turns:       Override; sonst nimmt role_config.max_turns

    Returns:
        SubTaskResult mit status="done" oder "failed", final_text,
        tool_calls, cost_usd, wall_clock_s.
    """
    prior_results = prior_results or {}
    name = sub_task.name
    t0 = time.monotonic()
    result = SubTaskResult(sub_task_name=name, status="running")

    prompt = _build_subtask_prompt(plan_summary, sub_task, prior_results)

    enable_subagents = sub_task.sub_agents_mode == "implementer-dispatched"

    req = HarnessRequest(
        role="implementer",
        harness=role_config.harness,
        provider=role_config.provider,
        model=role_config.model,
        prompt=prompt,
        allowed_tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash"],
        cwd=Path(workspace_path),
        max_turns=int(max_turns or role_config.max_turns or 20),
        enable_subagents=enable_subagents,
    )

    harness = get_harness(role_config.harness)

    # Event-Forwarding mit Sub-Task-Tag — Phase H konsumiert das
    async def _wrapped_event(ev: HarnessEvent) -> None:
        if on_event:
            try:
                await on_event(name, ev)
            except Exception:
                logger.exception("on_event callback raised — ignoring")

    try:
        hr = await harness.run(req, on_event=_wrapped_event if on_event else None)
        result.final_text = hr.final_text
        result.cost_usd = hr.cost_usd
        result.num_turns = hr.num_turns
        result.tool_calls = [
            {
                "name": tc.name,
                "args": tc.args,
                "is_error": tc.is_error,
                "duration_ms": tc.duration_ms,
            }
            for tc in hr.tool_calls
        ]
        if not hr.success:
            result.status = "failed"
            result.error = hr.error or "harness reported success=False"
        else:
            result.status = "done"
    except Exception as e:
        logger.exception("subtask %s crashed", name)
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"

    result.wall_clock_s = time.monotonic() - t0
    return result
