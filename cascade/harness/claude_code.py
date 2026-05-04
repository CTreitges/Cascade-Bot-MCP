"""ClaudeCodeHarness — claude-agent-sdk Wrapper mit Router-Support.

Nutzt die `claude` CLI im Hintergrund (claude-agent-sdk spawnt den Subprozess).
Wenn `provider` != "anthropic", wird ANTHROPIC_BASE_URL auf den lokalen
claude-code-router gesetzt — der routet dann zu OpenAI/Ollama/etc. weiter.

Permission-Mode ist hardcoded "bypassPermissions" (User-Vorgabe). Workspaces
sollen per Sub-Task isoliert sein (git worktree), dann ist das sicher.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from cascade.harness.base import (
    AssistantTextEvent,
    DoneEvent,
    EventCallback,
    Harness,
    HarnessRequest,
    HarnessResult,
    TokenUsage,
    ToolCall,
    ToolResultEvent,
    ToolUseEvent,
)
from cascade.pricing import compute_cost as _compute_cost
from cascade.pricing import extract_token_counts as _extract_tokens


# Hardcoded — User-Vorgabe 2026-05-03: alle Harness-Runs immer bypass.
_PERMISSION_MODE = "bypassPermissions"

# Default-Tools wenn der Caller nichts überschreibt.
_DEFAULT_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
_SUBAGENT_TOOL = "Task"


class ClaudeCodeHarness:
    name = "claude-code"

    async def run(
        self,
        request: HarnessRequest,
        on_event: Optional[EventCallback] = None,
    ) -> HarnessResult:
        # Lazy-Import: SDK soll keine Hard-Dep für andere cascade-Module sein.
        from claude_agent_sdk import ClaudeAgentOptions, query

        # Router-Routing: wenn non-Anthropic-Provider, BASE_URL auf Router umlenken.
        # Das geschieht über os.environ — die SDK reicht es an die `claude` CLI weiter.
        prev_env: Dict[str, Optional[str]] = {}
        if request.provider != "anthropic":
            prev_env["ANTHROPIC_BASE_URL"] = os.environ.get("ANTHROPIC_BASE_URL")
            prev_env["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
            os.environ["ANTHROPIC_BASE_URL"] = request.router_url
            os.environ["ANTHROPIC_API_KEY"] = "router-dummy"

        tools = list(request.allowed_tools or _DEFAULT_TOOLS)
        if request.enable_subagents and _SUBAGENT_TOOL not in tools:
            tools.append(_SUBAGENT_TOOL)

        opts_kwargs: Dict[str, Any] = {
            "model": request.model,
            "allowed_tools": tools,
            "max_turns": request.max_turns,
            "cwd": str(request.cwd),
            "permission_mode": _PERMISSION_MODE,
        }
        if request.system:
            opts_kwargs["system_prompt"] = request.system
        if request.mcp_servers:
            # Format: list of {name, url, type, ...} — direkt an SDK weitergeben.
            opts_kwargs["mcp_servers"] = request.mcp_servers
        options = ClaudeAgentOptions(**opts_kwargs)

        result = HarnessResult()
        t0 = time.monotonic()
        tool_started_at: Dict[str, float] = {}

        try:
            async for msg in query(prompt=request.prompt, options=options):
                await self._dispatch_message(msg, result, tool_started_at, on_event)
        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {e}"
            if on_event:
                await on_event(DoneEvent(
                    timestamp=time.time(),
                    usage=result.usage,
                    cost_usd=result.cost_usd,
                    num_turns=result.num_turns,
                    success=False,
                    error=result.error,
                ))
        finally:
            # Env wiederherstellen
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        result.wall_clock_s = time.monotonic() - t0

        if result.success and on_event:
            await on_event(DoneEvent(
                timestamp=time.time(),
                usage=result.usage,
                cost_usd=result.cost_usd,
                num_turns=result.num_turns,
                success=True,
            ))
        return result

    # ──────────────────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────────────────
    async def _dispatch_message(
        self,
        msg: Any,
        result: HarnessResult,
        tool_started_at: Dict[str, float],
        on_event: Optional[EventCallback],
    ) -> None:
        """Übersetzt SDK-Messages in unsere unified HarnessEvents + füllt result."""
        mtype = type(msg).__name__.lower()

        if "result" in mtype and "tool" not in mtype:
            # ResultMessage am Ende — Turns, Usage (DICT, nicht Object!), Cost
            turns = getattr(msg, "num_turns", None)
            if turns is not None:
                result.num_turns = int(turns)

            # SDK gibt usage als dict (input_tokens etc.) — nicht als Object.
            usage = getattr(msg, "usage", None)
            counts = _extract_tokens(usage)
            result.usage = TokenUsage(
                input_tokens=counts["input"],
                output_tokens=counts["output"],
                cache_read_input_tokens=counts["cache_read"],
                cache_creation_input_tokens=counts["cache_creation"],
            )

            # Cost: bevorzugt SDK-eigener Wert (für Anthropic akkurat).
            # Falls 0 oder fehlt (z.B. Ollama-via-Router), eigene Berechnung
            # via cascade.pricing als Fallback.
            sdk_cost = getattr(msg, "total_cost_usd", None)
            if sdk_cost is not None and float(sdk_cost) > 0:
                result.cost_usd = float(sdk_cost)
            else:
                # Fallback: pricing-Tabelle. Bei Ollama-Modellen liefert
                # die Tabelle 0.0 (Subscription-Pricing) — okay.
                result.cost_usd = _compute_cost(usage, getattr(msg, "model", None) or "")
                if result.cost_usd == 0.0:
                    # Letzter Fallback: probiere model aus model_usage-dict
                    mu = getattr(msg, "model_usage", None) or {}
                    if isinstance(mu, dict) and mu:
                        first_model = next(iter(mu.keys()))
                        result.cost_usd = _compute_cost(usage, first_model)
            return

        if "assistantmessage" in mtype or "usermessage" in mtype:
            blocks = getattr(msg, "content", [])
            if isinstance(blocks, list):
                for b in blocks:
                    await self._dispatch_block(b, result, tool_started_at, on_event)

    async def _dispatch_block(
        self,
        block: Any,
        result: HarnessResult,
        tool_started_at: Dict[str, float],
        on_event: Optional[EventCallback],
    ) -> None:
        btype = type(block).__name__.lower()

        if "tooluse" in btype:
            tool_id = getattr(block, "id", str(uuid.uuid4()))
            name = getattr(block, "name", "?")
            args = getattr(block, "input", {}) or {}
            tool_started_at[tool_id] = time.monotonic()
            tc = ToolCall(name=name, args=args, started_at=time.time())
            result.tool_calls.append(tc)
            if on_event:
                await on_event(ToolUseEvent(
                    timestamp=time.time(), name=name, args=args, tool_id=tool_id,
                ))
            return

        if "toolresult" in btype:
            tool_id = getattr(block, "tool_use_id", "")
            is_error = bool(getattr(block, "is_error", False))
            content = getattr(block, "content", "")
            if isinstance(content, list):
                content = " ".join(getattr(c, "text", str(c)) for c in content)
            preview = str(content)[:200].replace("\n", " ")
            # Letztes Tool im result.tool_calls passt vermutlich (in-order); robuster:
            # find by mapping
            for tc in reversed(result.tool_calls):
                if not tc.is_error and tc.result_preview == "":
                    started = tool_started_at.get(tool_id, tc.started_at or time.time())
                    tc.is_error = is_error
                    tc.result_preview = preview
                    tc.duration_ms = int((time.monotonic() - started) * 1000)
                    break
            if on_event:
                await on_event(ToolResultEvent(
                    timestamp=time.time(), tool_id=tool_id, is_error=is_error,
                    content_preview=preview,
                ))
            return

        if "text" in btype or hasattr(block, "text"):
            text = getattr(block, "text", "")
            if text:
                result.final_text += text
                if on_event:
                    await on_event(AssistantTextEvent(timestamp=time.time(), text=text))
