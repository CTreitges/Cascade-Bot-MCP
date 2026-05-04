"""Phase E E2E-Smoke: vollständiger Orchestrator-Run mit 3 Sub-Tasks.

DAG: explore → (fix-a ‖ fix-b)
Workspace: temp git-Repo wo wir 2 unabhängige Files modifizieren lassen.

Erfolg = alle 3 Sub-Tasks status="done", 2 Batches gelaufen, Worktrees
hinterher sauber aufgeräumt.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.agents.planner import Plan, SubTask
from cascade.orchestrator import Orchestrator
from cascade.role_config import RoleConfig


def shell(cmd, cwd):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def setup_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="cascade-orch-"))
    shell("git init -q -b main", repo)
    shell("git config user.email t@t", repo)
    shell("git config user.name t", repo)
    (repo / "alpha.py").write_text("def hello():\n    return 'hello'\n")
    (repo / "beta.py").write_text("def world():\n    return 'world'\n")
    (repo / "README.md").write_text("# Test Repo\n\nTwo files: alpha.py, beta.py.\n")
    shell("git add -A && git commit -q -m base", repo)
    return repo


def make_plan() -> Plan:
    return Plan(
        summary="Refactor: alle Funktionen sollen Docstrings bekommen.",
        subtasks=[
            SubTask(
                name="explore",
                summary="Erkunde welche Funktionen es gibt und in welchen Files.",
                steps=["Glob alle .py Files", "Grep nach 'def '"],
                files_to_touch=[],
                acceptance_criteria=["mindestens eine Erkenntnis im final_text"],
            ),
            SubTask(
                name="docs-alpha",
                summary="Füge eine Docstring zu hello() in alpha.py hinzu.",
                depends_on=["explore"],
                files_to_touch=["alpha.py"],
                acceptance_criteria=[
                    "alpha.py:hello() hat eine 1-zeilige Docstring",
                    "Funktion returnt weiterhin 'hello'",
                ],
            ),
            SubTask(
                name="docs-beta",
                summary="Füge eine Docstring zu world() in beta.py hinzu.",
                depends_on=["explore"],
                files_to_touch=["beta.py"],
                acceptance_criteria=[
                    "beta.py:world() hat eine 1-zeilige Docstring",
                    "Funktion returnt weiterhin 'world'",
                ],
            ),
        ],
    )


async def main():
    repo = setup_repo()
    print(f"📁 Test-Repo: {repo}")

    plan = make_plan()
    role = RoleConfig(
        role="implementer",
        harness="claude-code",
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_turns=10,
    )

    events_log = []

    async def on_event(name, ev):
        kind = type(ev).__name__
        events_log.append((name, kind))

    async def on_status(name, info):
        print(f"  [{name}] {info}")

    orch = Orchestrator(
        plan=plan,
        repo_root=repo,
        implementer_role=role,
        max_concurrent=3,
        on_event=on_event,
        on_subtask_status=on_status,
    )

    print("\n🚀 Orchestrator.run() …")
    result = await orch.run()

    print(f"\n📊 Result")
    print(f"  success: {result.success}")
    print(f"  batches: {result.batches_run}")
    print(f"  cost:    ${result.total_cost_usd:.4f}")
    print(f"  wall:    {result.total_wall_clock_s:.1f}s")
    print(f"  done={result.num_done} failed={result.num_failed} skipped={result.num_skipped}")
    print()
    for name, r in result.sub_task_results.items():
        print(f"  {name}: status={r.status}, files={r.files_changed}, "
              f"cost=${r.cost_usd:.4f}, turns={r.num_turns}")
        if r.error:
            print(f"     ERROR: {r.error[:200]}")

    # Verify: die 2 docs-Sub-Tasks haben tatsächlich Files geändert
    if "docs-alpha" in result.sub_task_results:
        files_a = result.sub_task_results["docs-alpha"].files_changed
        if "alpha.py" in files_a:
            print(f"\n  ✅ docs-alpha hat alpha.py modifiziert")
        else:
            print(f"\n  ⚠️  docs-alpha hat KEIN alpha.py modifiziert: {files_a}")

    # Cleanup
    print("\n🧹 cleanup …")
    await orch.cleanup(keep_branches=False)
    shutil.rmtree(repo, ignore_errors=True)

    print(f"\n📈 Total Events: {len(events_log)}")
    by_kind = {}
    for _, k in events_log:
        by_kind[k] = by_kind.get(k, 0) + 1
    for k, n in sorted(by_kind.items()):
        print(f"  {k}: {n}")

    if result.success and result.num_done == 3:
        print("\n  ✅ Phase E E2E grün — Orchestrator + Worktrees + Parallelism funktionieren.")
    else:
        print(f"\n  ⚠️  Nicht alle Sub-Tasks done — schau die Errors oben.")


if __name__ == "__main__":
    asyncio.run(main())
