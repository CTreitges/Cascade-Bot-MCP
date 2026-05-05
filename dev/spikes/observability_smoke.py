"""Plan v5 R5 — Observability Smoke."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.observability import (
    JSONLEmitter,
    RunSummary,
    configure_emitter,
    current_trace,
    emit,
    new_trace_id,
    restore_trace_context,
    set_trace_context,
)


def passed(label):
    print(f"  ✅ {label}")


def fail(label, why):
    print(f"  ❌ {label}: {why}")
    raise SystemExit(1)


def test_trace_context():
    print("\n[1] trace_context set/restore")
    assert current_trace() == {"trace_id": None, "task_id": None, "subtask": None}
    prev = set_trace_context(trace_id="t1", task_id="task-A")
    assert current_trace()["trace_id"] == "t1"
    assert current_trace()["task_id"] == "task-A"
    restore_trace_context(prev)
    assert current_trace()["trace_id"] is None
    passed("set + restore round-trip")


def test_jsonl_emitter():
    print("\n[2] JSONLEmitter writes line per emit")
    tmp = Path(tempfile.mkdtemp(prefix="cascade-obs-"))
    em = JSONLEmitter(path=tmp / "metrics.jsonl")
    set_trace_context(trace_id="t1", task_id="A")
    em.emit("llm_call", {"model": "claude-sonnet-4-6", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.0015})
    em.emit("tool_use", {"name": "Read", "args": {"file_path": "alpha.py"}})
    set_trace_context(trace_id="t2")
    em.emit("done", {})
    restore_trace_context({"trace_id": None, "task_id": None, "subtask": None})

    lines = (tmp / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3, f"erwartet 3 Zeilen, got {len(lines)}"
    rec1 = json.loads(lines[0])
    assert rec1["event"] == "llm_call"
    assert rec1["trace_id"] == "t1"
    assert rec1["task_id"] == "A"
    assert rec1["payload"]["model"] == "claude-sonnet-4-6"
    rec3 = json.loads(lines[2])
    assert rec3["trace_id"] == "t2"
    print(f"     3 lines written, trace + task switching klappt")
    shutil.rmtree(tmp, ignore_errors=True)
    passed("JSONL serialization + trace-merge")


def test_run_summary_aggregate():
    print("\n[3] RunSummary aggregiert correctly")
    s = RunSummary(trace_id="t-abc", task_id="task-1")
    s.add_llm_call(role="planner", model="claude-opus-4-7", provider="anthropic",
                   input_tokens=1000, output_tokens=200, cost_usd=0.030)
    s.add_llm_call(role="implementer", model="claude-sonnet-4-6", provider="anthropic",
                   input_tokens=2000, output_tokens=500, cost_usd=0.014)
    s.add_llm_call(role="implementer", model="claude-sonnet-4-6", provider="anthropic",
                   input_tokens=1500, output_tokens=300, cost_usd=0.009)
    s.add_tool_call("Read")
    s.add_tool_call("Read")
    s.add_tool_call("Edit")
    s.add_failover_attempt()
    final = s.finalize(success=True)
    assert abs(final["total_cost_usd"] - 0.053) < 1e-6
    # planner (0.030) > implementer (0.023): aggregation ist korrekt
    assert final["by_role_cost"]["planner"] > final["by_role_cost"]["implementer"]
    assert abs(final["by_role_cost"]["implementer"] - 0.023) < 1e-6
    assert final["tool_call_counts"]["Read"] == 2
    assert final["failover_attempts"] == 1
    assert final["success"] is True
    passed(f"total ${final['total_cost_usd']:.4f}, {sum(final['tool_call_counts'].values())} tools")


def test_run_summary_telegram_render():
    print("\n[4] RunSummary.render_telegram lesbar")
    s = RunSummary(trace_id="t-xyz")
    s.add_llm_call(role="planner", model="opus", provider="anthropic", cost_usd=0.05)
    s.add_llm_call(role="implementer", model="sonnet", provider="anthropic", cost_usd=0.02)
    s.add_tool_call("Read")
    s.add_tool_call("Read")
    s.add_tool_call("Bash")
    s.finalize(success=True)
    out = s.render_telegram(lang="de")
    assert "✅" in out
    assert "$0.0700" in out
    assert "implementer" in out and "planner" in out
    assert "Read(2)" in out or "Read (2)" in out
    print(out)
    passed("kompakter Telegram-Block korrekt")


def test_rotation():
    print("\n[5] rotation bei rotate_at_mb erreicht")
    tmp = Path(tempfile.mkdtemp(prefix="cascade-obs-rot-"))
    em = JSONLEmitter(path=tmp / "x.jsonl", rotate_at_mb=0.0001, keep_files=3)
    for i in range(20):
        em.emit("evt", {"i": i, "filler": "x" * 200})
    files = sorted(tmp.iterdir())
    print(f"     files after burst: {[f.name for f in files]}")
    assert any(".1" in f.name for f in files), f"expected rotation: {files}"
    shutil.rmtree(tmp, ignore_errors=True)
    passed("rotation triggert + .1 entstanden")


def test_global_emitter():
    print("\n[6] configure_emitter + global emit()")
    tmp = Path(tempfile.mkdtemp(prefix="cascade-obs-g-"))
    em = configure_emitter(path=tmp / "g.jsonl")
    set_trace_context(trace_id="globaltest")
    emit("my_event", {"k": "v"})
    restore_trace_context({"trace_id": None, "task_id": None, "subtask": None})
    lines = (tmp / "g.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "my_event"
    assert rec["trace_id"] == "globaltest"
    assert rec["payload"]["k"] == "v"
    shutil.rmtree(tmp, ignore_errors=True)
    passed("global emitter + emit() klappt")


def main():
    print("=" * 60)
    print("  Plan v5 R5 — Observability Smoke")
    print("=" * 60)
    test_trace_context()
    test_jsonl_emitter()
    test_run_summary_aggregate()
    test_run_summary_telegram_render()
    test_rotation()
    test_global_emitter()
    print("\n" + "=" * 60)
    print("  ✅ Alle 6 Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    main()
