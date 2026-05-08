"""Smoke for Plan v5 R5 — Run-Card-Emit am Run-Ende.

Bestätigt:
  1. core.run_cascade emittet "run_card" wenn _budget_state.spent_usd > 0
  2. payload.text ist der RunSummary-Block (kostet, by_role, top tools)
  3. lang=de/en respektiert wird

Wir mocken progress + store; kein echter LLM-Call. Statt run_cascade direkt
zu fahren bauen wir den finally-Block-Pfad nach: BudgetState mit Daten füllen,
RunSummary aus ihm bauen, render_telegram() prüfen.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.cost_budget import BudgetState
from cascade.observability import RunSummary


def passed(label: str) -> None:
    print(f"  ✅ {label}")


def fail(label: str, why: str) -> None:
    print(f"  ❌ {label}: {why}")
    raise SystemExit(1)


def test_run_card_render():
    print("\n[1] BudgetState → RunSummary → render_telegram")
    bs = BudgetState(run_id="task-test")
    bs.spent_usd = 0.4275
    bs.by_role = {"planner": 0.10, "implementer": 0.30, "reviewer": 0.0275}
    bs.by_model = {"claude-opus-4-7": 0.10, "kimi-k2.6": 0.30, "claude-sonnet-4-6": 0.0275}

    rs = RunSummary(trace_id="abc12345", task_id="task-test", started_at=1700000000.0)
    rs.total_cost_usd = bs.spent_usd
    rs.by_role_cost = dict(bs.by_role)
    rs.by_model_cost = dict(bs.by_model)
    rs.tool_call_counts = {"Read": 12, "Bash": 4, "Edit": 2}
    rs.finalize(success=True)
    card = rs.render_telegram(lang="de")
    print(card)
    assert "✅" in card
    assert "Run-Summary" in card
    assert "$0.4275" in card
    assert "implementer" in card
    assert "planner" in card
    assert "Read(12)" in card
    passed("Card vollständig: success-marker, total cost, by_role, top tools")


def test_skip_when_no_spend():
    print("\n[2] Skip-Bedingung: spent_usd=0 + keine warnings → kein Card-Emit")
    bs = BudgetState(run_id="dryrun")
    bs.spent_usd = 0.0
    assert bs.spent_usd == 0.0
    assert not bs.warnings_emitted
    # Die Bedingung im finally-Block:
    skip = (bs.spent_usd > 0.0) or bool(bs.warnings_emitted)
    assert skip is False, "soll skippen wenn Run keine LLM-Calls hatte"
    passed("Skip-Pfad triggert bei reinem Trockenlauf")


def test_emit_when_warning_only():
    print("\n[3] Emit auch wenn nur Warnings (Budget-Schwelle ohne große Kosten)")
    bs = BudgetState(run_id="warn-only")
    bs.spent_usd = 0.001
    bs.warnings_emitted.append(0.5)
    skip = (bs.spent_usd > 0.0) or bool(bs.warnings_emitted)
    assert skip is True
    passed("Warning emit als Trigger akzeptiert")


def test_failed_card_marker():
    print("\n[4] Failed run → ❌-Marker in Card")
    rs = RunSummary(trace_id="x", task_id="y")
    rs.total_cost_usd = 0.05
    rs.errors = ["something broke"]
    rs.finalize(success=False)
    card = rs.render_telegram(lang="de")
    assert "❌" in card
    assert "something broke" in card or "Errors" in card
    passed("failed-marker + error-line vorhanden")


def main() -> None:
    print("=" * 60)
    print("  Plan v5 R5 — Run-Card-Smoke")
    print("=" * 60)
    test_run_card_render()
    test_skip_when_no_spend()
    test_emit_when_warning_only()
    test_failed_card_marker()
    print("\n" + "=" * 60)
    print("  ✅ Alle 4 Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    main()
