"""Phase G Smoke: call_reviewer_via_harness mit echtem Workspace + Diff.

Aufgabe: ein synthetischer „Plan" + „Diff" wo wir wissen ob Reviewer
bestehen soll. Reviewer soll Read/Glob nutzen, dann JSON liefern.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.agents.planner import Plan
from cascade.agents.reviewer import call_reviewer_via_harness
from cascade.workspace import CheckResult, QualityCheck

WORKSPACE = Path("/tmp/cascade-spike-workspace")


def make_plan() -> Plan:
    return Plan(
        summary="Eine kurze hello()-Funktion in alpha.py die 'hello' returnt.",
        steps=["Schreibe def hello() in alpha.py die 'hello' returnt"],
        files_to_touch=["alpha.py"],
        acceptance_criteria=[
            "alpha.py existiert",
            "alpha.py enthält def hello() die 'hello' (string) zurückgibt",
        ],
    )


SAMPLE_DIFF = """\
diff --git a/alpha.py b/alpha.py
new file mode 100644
+def hello():
+    return 'hello'
"""


async def main():
    if not WORKSPACE.exists():
        WORKSPACE.mkdir(parents=True)
        (WORKSPACE / "alpha.py").write_text("def hello():\n    return 'hello'\n")

    plan = make_plan()

    print("=" * 60)
    print("  Phase G — call_reviewer_via_harness")
    print("=" * 60)
    print(f"\n📋 Plan: {plan.summary}")
    print(f"📁 Workspace: {WORKSPACE}")
    print(f"🤖 Reviewer-Modell: claude-sonnet-4-6 (default)")
    print()

    result = await call_reviewer_via_harness(
        plan=plan,
        diff=SAMPLE_DIFF,
        workspace_root=WORKSPACE,
        check_results=[
            CheckResult(name="py-compile", ok=True, exit_code=0, output="", duration_s=0.1),
        ],
        lang="de",
    )

    print(f"\n✅ Verdikt: pass={result.passed}")
    print(f"   severity: {result.severity}")
    print(f"   passing_criteria: {result.passing_criteria}")
    print(f"   failing_criteria: {result.failing_criteria}")
    print(f"   feedback: {result.feedback[:300]}")
    print()
    if result.passed:
        print("  🎉 Phase-G-Smoke grün — Reviewer-via-Harness funktioniert end-to-end.")
    else:
        print("  ⚠️  Reviewer hat pass=false geliefert — vermutlich strenger als erwartet.")
        print("       Phase-G-Smoke trotzdem ok wenn JSON valid + ReviewResult parsed.")


if __name__ == "__main__":
    asyncio.run(main())
