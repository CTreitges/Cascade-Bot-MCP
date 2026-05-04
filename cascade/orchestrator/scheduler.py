"""Orchestrator: DAG-basierter parallel Sub-Task-Scheduler.

Nimmt einen Plan, sortiert die Sub-Tasks via cascade.dag.topological_batches,
gibt jedem Sub-Task einen git worktree, führt sie via Harness parallel
(per Batch) aus, sammelt SubTaskResults.

Failure-Policy:
  - Ein Sub-Task der failed → alle transitiv von ihm abhängigen werden
    "blocked" markiert (nicht ausgeführt)
  - Andere parallele Sub-Tasks im gleichen Batch laufen weiter
  - Per-Sub-Task-Replan ist Phase I (separater Code, hier Hook bereit)

Concurrency:
  - asyncio.Semaphore mit Default max_concurrent=3
  - Per-Provider-Limit kann später hier eingehängt werden (Anthropic 3,
    OpenAI 5, Ollama 10)
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from cascade.dag import topological_batches, validate_dag
from cascade.harness import HarnessEvent
from cascade.orchestrator.result import (
    OrchestratorResult,
    SubTaskResult,
)
from cascade.orchestrator.runner import run_subtask_via_harness
from cascade.orchestrator.worktree import WorktreeManager
from cascade.role_config import RoleConfig


logger = logging.getLogger("cascade.orchestrator.scheduler")


class Orchestrator:
    """Führt einen Plan mit Sub-Tasks parallel aus.

    Args:
        plan:                   Plan mit subtasks-Liste (cascade.agents.planner.Plan)
        repo_root:              Root des Source-Repos (wo Worktrees erzeugt werden)
        implementer_role:       RoleConfig für Sub-Task-Implementer
        max_concurrent:         max parallel laufende Sub-Tasks (default 3)
        base_ref:               git ref von dem Worktrees abzweigen (default HEAD)
        on_event:               (sub_task_name, HarnessEvent) Callback für Live-
                                Stream (Phase H)
        on_subtask_status:      (sub_task_name, status_dict) Callback wenn ein
                                Sub-Task seinen Status ändert (started/done/failed)
    """

    def __init__(
        self,
        *,
        plan: Any,
        repo_root: Path | str,
        implementer_role: RoleConfig,
        max_concurrent: int = 3,
        base_ref: Optional[str] = None,
        on_event: Optional[Callable[[str, HarnessEvent], Awaitable[None]]] = None,
        on_subtask_status: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self.plan = plan
        self.repo_root = Path(repo_root).resolve()
        self.role = implementer_role
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))
        self.worktree_mgr = WorktreeManager(self.repo_root, base_ref=base_ref)
        self.on_event = on_event
        self.on_subtask_status = on_subtask_status
        self.results: Dict[str, SubTaskResult] = {}

    async def _emit_status(self, name: str, **kwargs: Any) -> None:
        if self.on_subtask_status:
            try:
                await self.on_subtask_status(name, kwargs)
            except Exception:
                logger.exception("on_subtask_status raised — ignoring")

    async def _run_one(self, sub_task: Any, prior: Dict[str, SubTaskResult]) -> SubTaskResult:
        """Führt einen einzelnen Sub-Task im Worktree aus."""
        async with self.semaphore:
            await self._emit_status(sub_task.name, status="starting")
            try:
                wt = await self.worktree_mgr.create(sub_task.name)
            except Exception as e:
                logger.exception("worktree create failed for %s", sub_task.name)
                res = SubTaskResult(
                    sub_task_name=sub_task.name,
                    status="failed",
                    error=f"worktree create: {type(e).__name__}: {e}",
                )
                await self._emit_status(sub_task.name, status="failed", error=res.error)
                return res

            await self._emit_status(sub_task.name, status="running", branch=wt.branch)
            res = await run_subtask_via_harness(
                plan_summary=getattr(self.plan, "summary", ""),
                sub_task=sub_task,
                role_config=self.role,
                workspace_path=wt.path,
                prior_results={k: v for k, v in prior.items() if v.status == "done"},
                on_event=self.on_event,
            )
            res.branch = wt.branch

            # ZUERST commiten (committed alle uncommitted changes), DANN diff —
            # sonst ist `git diff base..branch` leer und files_changed bleibt [].
            try:
                if res.status == "done":
                    committed = await self.worktree_mgr.commit_changes(
                        sub_task.name,
                        f"cascade-sub: {sub_task.name}\n\n{res.final_text[:500]}",
                    )
                    if committed:
                        files = await self.worktree_mgr.get_files_changed(sub_task.name)
                        res.files_changed = files
            except Exception:
                logger.exception("post-run worktree ops failed for %s", sub_task.name)

            await self._emit_status(
                sub_task.name,
                status=res.status,
                files=len(res.files_changed),
                cost=res.cost_usd,
                turns=res.num_turns,
            )
            return res

    def _transitive_dependents(self, failed_names: Iterable[str], all_subtasks: List[Any]) -> set[str]:
        """Sammelt alle Sub-Tasks die transitiv von einem aus failed_names abhängen."""
        failed_set = set(failed_names)
        affected = set(failed_set)
        # Iterativ wachsen lassen
        changed = True
        while changed:
            changed = False
            for st in all_subtasks:
                if st.name in affected:
                    continue
                if any(dep in affected for dep in st.depends_on):
                    affected.add(st.name)
                    changed = True
        # Die ursprünglich failed gehören NICHT als "blocked" — die sind ja "failed"
        return affected - failed_set

    async def run(self) -> OrchestratorResult:
        """Hauptmethode: validiert DAG, läuft Batch für Batch, sammelt Result."""
        t0 = time.monotonic()
        result = OrchestratorResult()

        subtasks = list(getattr(self.plan, "subtasks", []) or [])
        if not subtasks:
            result.success = True
            result.total_wall_clock_s = time.monotonic() - t0
            return result

        # DAG-Validation: sollte schon in core.py passiert sein (Phase D),
        # aber defensive nochmal prüfen.
        errors = validate_dag(subtasks)
        if errors:
            result.error = "DAG-Validation: " + "; ".join(errors[:3])
            result.success = False
            return result

        # Skipped-Tasks vom Caller mitgeben (resumed_completed_subtasks Logik
        # in core.py) wäre Phase J — hier setzen wir alle auf pending.
        for st in subtasks:
            self.results[st.name] = SubTaskResult(sub_task_name=st.name, status="pending")

        try:
            batches = topological_batches(subtasks)
        except ValueError as e:
            result.error = f"topological_batches: {e}"
            result.success = False
            return result

        result.batches_run = 0
        for batch_idx, batch in enumerate(batches):
            result.batches_run += 1

            # Filter: blockierte Sub-Tasks (durch failed-deps) skippen
            already_failed = {n for n, r in self.results.items() if r.status == "failed"}
            blocked = self._transitive_dependents(already_failed, subtasks)

            ready = [st for st in batch if st.name not in blocked]
            for st in batch:
                if st.name in blocked:
                    self.results[st.name] = SubTaskResult(
                        sub_task_name=st.name,
                        status="blocked",
                        error="upstream sub-task failed",
                    )
                    await self._emit_status(st.name, status="blocked")

            if not ready:
                continue

            # Parallel-Execution dieses Batches
            prior_for_batch = {k: v for k, v in self.results.items() if v.status == "done"}
            tasks = [
                asyncio.create_task(self._run_one(st, prior_for_batch))
                for st in ready
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for st, res in zip(ready, batch_results):
                if isinstance(res, BaseException):
                    self.results[st.name] = SubTaskResult(
                        sub_task_name=st.name,
                        status="failed",
                        error=f"{type(res).__name__}: {res}",
                    )
                else:
                    self.results[st.name] = res

        # Aggregation
        result.sub_task_results = self.results
        result.total_cost_usd = sum(r.cost_usd for r in self.results.values())
        result.total_wall_clock_s = time.monotonic() - t0
        result.success = (
            result.num_failed == 0 and result.num_skipped == 0
        )
        return result

    async def cleanup(self, keep_branches: bool = True) -> None:
        """Räumt Worktrees auf. Branches bleiben (für nachfolgenden Reviewer-
        Diff oder Merge), wenn keep_branches=True."""
        for name in list(self.worktree_mgr.active.keys()):
            await self.worktree_mgr.remove(name, keep_branch=keep_branches)
