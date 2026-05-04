"""Result-Datenklassen für den Orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


SubTaskStatus = Literal["pending", "running", "done", "failed", "skipped", "blocked"]


@dataclass
class SubTaskResult:
    """Ergebnis eines einzelnen Sub-Task-Runs im Orchestrator."""
    sub_task_name: str
    status: SubTaskStatus = "pending"
    final_text: str = ""
    files_changed: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    num_turns: int = 0
    error: Optional[str] = None
    # Branch-Name in der Hauptcheckout falls Worktree benutzt wurde
    branch: Optional[str] = None


@dataclass
class OrchestratorResult:
    """Gesamt-Ergebnis eines Orchestrator-Runs über alle Sub-Tasks."""
    sub_task_results: Dict[str, SubTaskResult] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_wall_clock_s: float = 0.0
    batches_run: int = 0
    integration_branch: Optional[str] = None
    # True wenn alle Sub-Tasks done sind. Wenn auch nur ein failed/blocked → False.
    success: bool = False
    error: Optional[str] = None

    @property
    def num_done(self) -> int:
        return sum(1 for r in self.sub_task_results.values() if r.status == "done")

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.sub_task_results.values() if r.status == "failed")

    @property
    def num_skipped(self) -> int:
        return sum(1 for r in self.sub_task_results.values() if r.status in ("skipped", "blocked"))

    def failed_subtask_names(self) -> List[str]:
        return [n for n, r in self.sub_task_results.items() if r.status == "failed"]
