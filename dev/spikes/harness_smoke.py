"""Verifiziert die Harness-Abstraktion gegen denselben Test wie sdk_smoke.py.

Ruft ClaudeCodeHarness sowohl direkt (Anthropic) als auch via Router (Ollama)
und vergleicht: gleiche Tool-Call-Anzahl, gleiche Final-Answer-Qualität.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Cascade-Root in Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.harness import (
    HarnessRequest,
    HarnessResult,
    ToolUseEvent,
    ToolResultEvent,
    DoneEvent,
    AssistantTextEvent,
    get_harness,
)


WORKSPACE = Path("/tmp/cascade-spike-workspace")


async def run_one(label: str, model: str, provider: str) -> HarnessResult:
    print(f"\n{'='*60}\n  {label}\n  model={model}  provider={provider}\n{'='*60}")

    harness = get_harness("claude-code")

    async def on_event(ev):
        if isinstance(ev, ToolUseEvent):
            args_preview = str(ev.args)[:80].replace("\n", " ")
            print(f"  → {ev.name}({args_preview})")
        elif isinstance(ev, ToolResultEvent):
            err = "ERR" if ev.is_error else "ok"
            print(f"  ← ({err}) {ev.content_preview[:120]}")
        elif isinstance(ev, DoneEvent):
            print(f"  ⏹  done — turns={ev.num_turns} cost=${ev.cost_usd:.4f} success={ev.success}")

    req = HarnessRequest(
        role="implementer",
        harness="claude-code",
        provider=provider,
        model=model,
        prompt=(
            "Im aktuellen Verzeichnis liegen Python-Dateien. "
            "Finde die längste Funktion (in Zeilen) und nenne ihren Namen + Zeilenzahl. "
            "Antworte am Ende in einer Zeile."
        ),
        allowed_tools=["Read", "Glob", "Grep", "Bash"],
        cwd=WORKSPACE,
        max_turns=10,
    )
    result = await harness.run(req, on_event=on_event)

    print(f"\n  📝 final ({len(result.final_text)} chars):")
    print(f"     {result.final_text[:300]}")
    print(f"  ⏱  {result.wall_clock_s:.1f}s | tools={len(result.tool_calls)} | turns={result.num_turns}")
    print(f"  💰 ${result.cost_usd:.4f} | tokens-total={result.usage.total}")
    return result


async def main():
    results = []
    results.append(await run_one(
        label="A) Claude direkt",
        model="claude-sonnet-4-6",
        provider="anthropic",
    ))
    results.append(await run_one(
        label="B) kimi-k2.6 via Router",
        model="kimi-k2.6",
        provider="ollama",
    ))

    print("\n\n" + "="*60 + "\n  ZUSAMMENFASSUNG\n" + "="*60)
    for r, label in zip(results, ["A) Claude direkt", "B) kimi-k2.6 via Router"]):
        ok = "✅" if r.success else "❌"
        print(f"{ok} {label}: {len(r.tool_calls)} tools, {r.wall_clock_s:.1f}s, ${r.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
