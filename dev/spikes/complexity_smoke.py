"""Plan v5 R2 — Complexity-Tier-Routing Smoke."""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.complexity import (
    Tier,
    TierDecision,
    classify_via_heuristic,
    decide_tier,
    model_for_tier,
)


@dataclass
class FakePlan:
    direct_ops: list = field(default_factory=list)
    subtasks: list = field(default_factory=list)
    files_to_touch: list = field(default_factory=list)


def passed(label):
    print(f"  ✅ {label}")


def fail(label, why):
    print(f"  ❌ {label}: {why}")
    raise SystemExit(1)


def test_trivial_keyword():
    print("\n[1] Trivial via Keyword")
    d = classify_via_heuristic("Rename foo() to bar() in alpha.py")
    assert d.tier == Tier.TRIVIAL, d
    assert d.confidence >= 0.85
    passed(f"trivial: {d.reason}")


def test_trivial_direct_ops():
    print("\n[2] Trivial via direct_ops")
    plan = FakePlan(direct_ops=[1, 2, 3])
    d = classify_via_heuristic("eine kleine Änderung", plan=plan)
    assert d.tier == Tier.TRIVIAL, d
    assert d.confidence == 1.0
    passed(f"direct_ops: {d.reason}")


def test_complex_keyword():
    print("\n[3] Complex via Architektur-Keyword")
    d = classify_via_heuristic(
        "Bitte überprüfe die gesamte Architektur unseres Auth-Systems "
        "und schlage Verbesserungen vor — security review nötig."
    )
    assert d.tier == Tier.COMPLEX, d
    passed(f"complex: {d.reason}")


def test_complex_via_subtasks():
    print("\n[4] Complex via ≥3 sub_tasks")
    plan = FakePlan(subtasks=[1, 2, 3, 4])
    d = classify_via_heuristic("doesn't matter", plan=plan)
    assert d.tier == Tier.COMPLEX
    assert d.confidence == 1.0
    passed(f"sub_tasks: {d.reason}")


def test_complex_many_files():
    print("\n[5] Complex via ≥10 files_to_touch")
    plan = FakePlan(files_to_touch=[f"f{i}.py" for i in range(12)])
    d = classify_via_heuristic("standard task", plan=plan)
    assert d.tier == Tier.COMPLEX
    passed(f"many files: {d.reason}")


def test_standard_default():
    print("\n[6] Standard als default für 1-2 sub_tasks")
    plan = FakePlan(subtasks=[1, 2], files_to_touch=["a.py", "b.py"])
    d = classify_via_heuristic("normal task", plan=plan)
    assert d.tier == Tier.STANDARD
    passed(f"standard: {d.reason}")


def test_ambivalent_falls_to_standard():
    print("\n[7] Ambivalent → Standard mit niedriger Confidence")
    d = classify_via_heuristic("do something")
    assert d.tier == Tier.STANDARD
    assert d.confidence < 0.7
    passed(f"ambivalent: confidence={d.confidence}")


def test_short_task_no_match():
    print("\n[8] Kurzer Task ohne Trivial-Keyword → Standard")
    d = classify_via_heuristic("fix the bug")
    assert d.tier == Tier.STANDARD
    passed(f"short generic: {d.reason}")


def test_model_for_tier():
    print("\n[9] model_for_tier defaults + override")
    assert model_for_tier(Tier.TRIVIAL) == "claude-haiku-4-5"
    assert model_for_tier(Tier.STANDARD) == "claude-sonnet-4-6"
    assert model_for_tier(Tier.COMPLEX) == "claude-opus-4-7"
    assert model_for_tier(Tier.STANDARD, {Tier.STANDARD: "kimi-k2.6"}) == "kimi-k2.6"
    passed("defaults + override greifen")


async def test_decide_tier_skips_llm_when_confident():
    print("\n[10] decide_tier: skipt LLM bei hoher Confidence")
    plan = FakePlan(subtasks=[1, 2, 3])
    # confidence 1.0 → kein LLM-call (auch ohne settings)
    d = await decide_tier("anything", plan=plan, settings=None)
    assert d.tier == Tier.COMPLEX
    assert d.via_heuristic is True
    passed("high-confidence heuristic kein LLM")


async def main():
    print("=" * 60)
    print("  Plan v5 R2 — Complexity-Tier-Routing Smoke")
    print("=" * 60)
    test_trivial_keyword()
    test_trivial_direct_ops()
    test_complex_keyword()
    test_complex_via_subtasks()
    test_complex_many_files()
    test_standard_default()
    test_ambivalent_falls_to_standard()
    test_short_task_no_match()
    test_model_for_tier()
    await test_decide_tier_skips_llm_when_confident()
    print("\n" + "=" * 60)
    print("  ✅ Alle 10 Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
