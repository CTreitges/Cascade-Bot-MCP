"""Inspiziert die SDK-Message-Struktur — speziell ResultMessage.usage —
damit wir die Token-Felder im claude_code.py-Adapter richtig extrahieren.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_agent_sdk import ClaudeAgentOptions, query

WORKSPACE = Path("/tmp/cascade-spike-workspace")


def dump_obj(name, obj, indent=2):
    pad = "  " * indent
    print(f"{pad}{name}: type={type(obj).__name__}")
    if obj is None:
        return
    for attr in sorted(dir(obj)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
        except Exception as e:
            print(f"{pad}  .{attr}: <err: {e}>")
            continue
        if callable(val):
            continue
        # Recurse für strukturierte Objekte
        if hasattr(val, "__dict__") and not isinstance(val, (str, int, float, bool, list, dict, type(None))):
            print(f"{pad}  .{attr} =")
            dump_obj(attr, val, indent + 2)
        else:
            preview = repr(val)
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"{pad}  .{attr} = {preview}")


async def main():
    # Sicherstellen dass Anthropic-Auth genutzt wird, nicht Router
    os.environ.pop("ANTHROPIC_BASE_URL", None)

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        allowed_tools=["Read", "Glob"],
        max_turns=4,
        cwd=str(WORKSPACE),
        permission_mode="bypassPermissions",
    )

    print("=" * 60)
    print("Sammle alle Top-Level-Messages …")
    print("=" * 60)
    saw = []
    async for msg in query(prompt="Liste die python-Files im aktuellen Verzeichnis.", options=options):
        mtype = type(msg).__name__
        saw.append(mtype)
        print(f"\n>>> Msg #{len(saw)}: {mtype}")
        # ResultMessage volle Inspektion
        if "result" in mtype.lower():
            dump_obj(mtype, msg, indent=1)

    print("\n=== Gesamte Sequence ===")
    for i, t in enumerate(saw, 1):
        print(f"  {i}. {t}")


if __name__ == "__main__":
    asyncio.run(main())
