"""Phase-0-Spike: claude-agent-sdk Smoke-Test.

Test 1: Claude-Modell direkt (Anthropic API, kein Router) — soll Tools nutzen.
Test 2: Per Env-Var auf Router umleiten — gleiche Aufgabe mit Ollama-Modell.

Verifiziert:
  - SDK importierbar in cascade venv
  - claude-agent-sdk findet `claude` CLI in PATH
  - tool_use / tool_result Events streamen
  - Modell führt echte Read/Glob/Grep Calls aus
  - Auth funktioniert in subprocess-Kontext (für späteren systemd-Service)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Spike-Workspace: kleiner Test-Ordner, gleich existiert
WORKSPACE = Path("/tmp/cascade-spike-workspace")


def setup_workspace():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "alpha.py").write_text("""
def hello():
    return 'hello'

def world():
    print('world')
    return 42

def long_function_with_many_lines():
    x = 0
    for i in range(100):
        x += i * 2
        if x > 500:
            x -= 100
    return x
""".lstrip())
    (WORKSPACE / "beta.py").write_text("""
import alpha

def main():
    print(alpha.hello())
    print(alpha.world())
""".lstrip())
    (WORKSPACE / "README.md").write_text("# Spike Workspace\n\nTwo files: alpha.py + beta.py\n")


async def run_query(label: str, model: str, prompt: str, allowed_tools: list[str], use_router: bool):
    from claude_agent_sdk import query, ClaudeAgentOptions

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  model={model}  router={use_router}")
    print(f"{'='*60}")

    if use_router:
        os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:3456"
        os.environ["ANTHROPIC_API_KEY"] = "router-dummy"
    else:
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        # Lass die ANTHROPIC_API_KEY in Ruhe — Claude CLI nutzt eigene Auth

    options = ClaudeAgentOptions(
        model=model,
        allowed_tools=["Read", "Glob", "Grep"],
        max_turns=10,
        cwd=str(WORKSPACE),
        permission_mode="bypassPermissions",
    )

    t0 = time.monotonic()
    tool_uses = []
    final_text = ""
    last_msg_type = None
    error = None

    def _scan_blocks(blocks):
        """Inspect ContentBlocks within AssistantMessage / UserMessage (tool_result lebt
        meist in UserMessage)."""
        nonlocal final_text
        if not isinstance(blocks, list):
            return
        for b in blocks:
            btype = type(b).__name__.lower()
            if "tooluse" in btype:
                name = getattr(b, "name", "?")
                inp = getattr(b, "input", {})
                args_preview = str(inp)[:80].replace("\n", " ")
                tool_uses.append(name)
                print(f"  → tool_use: {name}({args_preview})")
            elif "toolresult" in btype:
                err = getattr(b, "is_error", False)
                content = getattr(b, "content", "")
                if isinstance(content, list):
                    content = " ".join(getattr(c, "text", str(c)) for c in content)
                preview = str(content)[:120].replace("\n", " ")
                print(f"  ← tool_result ({'ERR' if err else 'ok'}): {preview}")
            elif "text" in btype or hasattr(b, "text"):
                final_text += getattr(b, "text", "")

    try:
        async for msg in query(prompt=prompt, options=options):
            mtype = type(msg).__name__
            last_msg_type = mtype
            mtype_str = mtype.lower()
            if "assistantmessage" in mtype_str or "usermessage" in mtype_str:
                _scan_blocks(getattr(msg, "content", []))
            elif "result" in mtype_str:
                cost = getattr(msg, "total_cost_usd", None)
                if cost is not None:
                    print(f"  💰 cost: ${cost:.4f}")
                turns = getattr(msg, "num_turns", None)
                if turns is not None:
                    print(f"  🔁 turns: {turns}")
    except Exception as e:
        error = e
        print(f"  ❌ FEHLER: {type(e).__name__}: {e}")

    dur = time.monotonic() - t0
    print(f"\n  ⏱  {dur:.1f}s | tool_uses={len(tool_uses)} ({tool_uses}) | last={last_msg_type}")
    if final_text:
        print(f"  📝 final-text (first 300 chars):")
        print(f"     {final_text[:300]}")
    return {
        "label": label,
        "model": model,
        "ok": error is None,
        "duration_s": dur,
        "tool_uses": tool_uses,
        "final_len": len(final_text),
        "error": str(error) if error else None,
    }


async def main():
    setup_workspace()
    print(f"📁 Spike-Workspace: {WORKSPACE}")

    results = []

    # TEST 1: Claude-Modell, kein Router
    results.append(await run_query(
        label="TEST 1 — Claude (kein Router)",
        model="claude-sonnet-4-6",
        prompt=(
            "Im aktuellen Verzeichnis liegen Python-Dateien. "
            "Finde die längste Funktion (in Zeilen) und nenne ihren Namen + Zeilenzahl. "
            "Nutze Read und Glob/Grep wie nötig. Antworte am Ende in einer Zeile."
        ),
        allowed_tools=["Read", "Glob", "Grep"],
        use_router=False,
    ))

    # TEST 2: Ollama-Modell via Router (nur wenn Router läuft)
    router_running = os.system("curl -fsS -m 2 http://127.0.0.1:3456 >/dev/null 2>&1") == 0
    if router_running:
        results.append(await run_query(
            label="TEST 2 — kimi-k2.6 via claude-code-router",
            model="kimi-k2.6",
            prompt="Same as Test 1 — find the longest function in the current directory and name it with line count.",
            allowed_tools=["Read", "Glob", "Grep"],
            use_router=True,
        ))
    else:
        print("\n⚠️  Router (port 3456) nicht erreichbar — Test 2 übersprungen.")
        print("   Setup-Anleitung: npm i -g @musistudio/claude-code-router && ccr start")

    print(f"\n\n{'='*60}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    for r in results:
        ok = "✅" if r["ok"] else "❌"
        print(f"{ok} {r['label']} — {r['duration_s']:.1f}s, {len(r['tool_uses'])} tools")
        if r["error"]:
            print(f"     Fehler: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())
