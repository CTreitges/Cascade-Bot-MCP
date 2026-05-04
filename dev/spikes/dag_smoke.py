"""Smoke-Tests für cascade/dag.py — DAG-Validation + Topological-Sort."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.agents.planner import SubTask
from cascade.dag import (
    has_file_overlap,
    topological_batches,
    validate_dag,
)


def make(name, depends_on=None, files=None):
    return SubTask(
        name=name,
        summary=f"Test sub-task {name}",
        depends_on=depends_on or [],
        files_to_touch=files or [],
    )


def passed(label):
    print(f"  ✅ {label}")


def failed(label, why):
    print(f"  ❌ {label}: {why}")
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────────────
def test_empty():
    print("\n[1] empty plan")
    assert validate_dag([]) == []
    assert topological_batches([]) == []
    passed("empty leer-list-safe")


def test_linear_chain():
    print("\n[2] linear chain a→b→c")
    sts = [make("a"), make("b", ["a"]), make("c", ["b"])]
    assert validate_dag(sts) == []
    batches = topological_batches(sts)
    names = [[s.name for s in b] for b in batches]
    assert names == [["a"], ["b"], ["c"]], f"got {names}"
    passed(f"linear → {names}")


def test_diamond():
    print("\n[3] diamond a→{b,c}→d")
    sts = [
        make("a", files=["docs/intro.md"]),
        make("b", ["a"], files=["src/x.py"]),
        make("c", ["a"], files=["src/y.py"]),
        make("d", ["b", "c"], files=["src/main.py"]),
    ]
    assert validate_dag(sts) == []
    batches = topological_batches(sts)
    names = [sorted(s.name for s in b) for b in batches]
    assert names == [["a"], ["b", "c"], ["d"]], f"got {names}"
    passed(f"diamond → {names}")


def test_cycle_detection():
    print("\n[4] cycle a→b→a")
    sts = [make("a", ["b"]), make("b", ["a"])]
    errors = validate_dag(sts)
    assert any("Zykel" in e for e in errors), f"got {errors}"
    passed(f"cycle erkannt: {errors[0]}")


def test_self_loop():
    print("\n[5] self-loop a→a")
    sts = [make("a", ["a"])]
    errors = validate_dag(sts)
    assert any("sich selbst" in e for e in errors), f"got {errors}"
    passed(f"self-loop erkannt: {errors[0]}")


def test_dangling_ref():
    print("\n[6] dangling depends_on")
    sts = [make("a", ["nonexistent"])]
    errors = validate_dag(sts)
    assert any("unbekannt" in e for e in errors), f"got {errors}"
    passed(f"dangling ref erkannt: {errors[0]}")


def test_duplicate_name():
    print("\n[7] duplicate sub-task name")
    sts = [make("a"), make("a")]
    errors = validate_dag(sts)
    assert any("mehrfach" in e for e in errors), f"got {errors}"
    passed(f"duplicate erkannt: {errors[0]}")


def test_file_overlap_serializes():
    print("\n[8] file overlap zwischen parallel-fähigen → serialisiert")
    sts = [
        make("a"),
        make("b", ["a"], files=["src/x.py"]),
        make("c", ["a"], files=["src/x.py"]),  # gleicher File!
    ]
    assert validate_dag(sts) == []
    batches = topological_batches(sts)
    names = [sorted(s.name for s in b) for b in batches]
    # b und c müssen in unterschiedliche Batches
    assert len(batches) >= 2, f"got {names}"
    flat = [n for b in batches for n in b]
    assert "b" in [s.name for s in flat] and "c" in [s.name for s in flat]
    # Sie dürfen NICHT in derselben Batch sein
    for batch in batches:
        names_in = [s.name for s in batch]
        assert not ("b" in names_in and "c" in names_in), f"b+c parallel trotz overlap: {names}"
    passed(f"overlap erzwingt serialisiert: {names}")


def test_no_overlap_parallel():
    print("\n[9] disjunkte Files → echte Parallelisierung")
    sts = [
        make("a"),
        make("b", ["a"], files=["src/foo.py"]),
        make("c", ["a"], files=["src/bar.py"]),
        make("d", ["a"], files=["docs/readme.md"]),
    ]
    batches = topological_batches(sts)
    # Batch[1] muss b+c+d alle enthalten
    assert len(batches) == 2
    par = sorted(s.name for s in batches[1])
    assert par == ["b", "c", "d"], f"got {par}"
    passed(f"alle 3 parallel: {par}")


def test_glob_overlap():
    print("\n[10] Glob-Pattern-Overlap erkannt")
    sts = [
        make("tests-x", files=["tests/test_x.py"]),
        make("tests-all", files=["tests/test_*.py"]),
    ]
    confs = has_file_overlap(sts)
    assert confs, f"erwartet overlap, got {confs}"
    passed(f"glob overlap: {confs[0]}")


def test_batches_with_subagents_field():
    print("\n[11] sub_agents_mode field roundtrip")
    st = SubTask(
        name="x", summary="y",
        sub_agents_mode="dag",
    )
    assert st.sub_agents_mode == "dag"
    assert validate_dag([st]) == []
    passed(f"sub_agents_mode='dag' geschrieben+gelesen")


def main():
    print("=" * 60)
    print("  cascade/dag.py Smoke-Tests")
    print("=" * 60)
    test_empty()
    test_linear_chain()
    test_diamond()
    test_cycle_detection()
    test_self_loop()
    test_dangling_ref()
    test_duplicate_name()
    test_file_overlap_serializes()
    test_no_overlap_parallel()
    test_glob_overlap()
    test_batches_with_subagents_field()
    print("\n" + "=" * 60)
    print("  ✅ Alle 11 Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    main()
