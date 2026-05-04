"""Smoke-Test für cascade.role_config — verifiziert die Resolution-Reihenfolge."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.role_config import (
    RoleConfig,
    detect_provider,
    encode_role_overrides,
    get_role_config,
    parse_role_overrides,
)


class FakeSettings:
    cascade_planner_model = "claude-opus-4-7"
    cascade_implementer_model = "kimi-k2.6"
    cascade_reviewer_model = "claude-sonnet-4-6"
    cascade_triage_model = "claude-haiku-4-5"
    cascade_planner_effort = "high"
    cascade_implementer_effort = ""
    cascade_reviewer_effort = ""
    cascade_triage_effort = ""


def test_detect_provider():
    assert detect_provider("claude-opus-4-7") == "anthropic"
    assert detect_provider("claude-sonnet-4-6") == "anthropic"
    assert detect_provider("gpt-5") == "openai"
    assert detect_provider("o3-mini") == "openai"
    assert detect_provider("o4-mini") == "openai"
    assert detect_provider("kimi-k2.6") == "ollama"
    assert detect_provider("qwen3-coder:480b") == "ollama"
    assert detect_provider("glm-5.1") == "ollama"
    assert detect_provider("") == "ollama"
    print("✅ detect_provider — 9 cases")


def test_defaults_only():
    rc = get_role_config("planner", FakeSettings())
    assert rc.model == "claude-opus-4-7"
    assert rc.provider == "anthropic"
    assert rc.harness == "claude-code"
    assert rc.effort == "high"
    assert rc.enable_subagents is False
    print(f"✅ defaults_only — {rc.role}: {rc.model} ({rc.provider}/{rc.harness}, effort={rc.effort})")


def test_legacy_chat_session():
    session = {"planner_model": "kimi-k2.6", "planner_effort": "low"}
    rc = get_role_config("planner", FakeSettings(), session)
    assert rc.model == "kimi-k2.6"
    assert rc.provider == "ollama", f"auto-detect failed: {rc.provider}"
    assert rc.effort == "low"
    print(f"✅ legacy_chat_session override — model→{rc.model}, provider→{rc.provider}")


def test_role_overrides_json():
    session = {
        "planner_model": "claude-opus-4-7",
        "role_overrides_json": '{"planner": {"harness": "codex", "provider": "openai", "model": "gpt-5"}}',
    }
    rc = get_role_config("planner", FakeSettings(), session)
    # JSON model überschreibt legacy
    assert rc.model == "gpt-5"
    assert rc.harness == "codex"
    assert rc.provider == "openai"
    print(f"✅ role_overrides_json — harness→{rc.harness}, model→{rc.model}")


def test_run_overrides():
    session = {"planner_model": "kimi-k2.6"}
    run_ovr = {"model": "claude-sonnet-4-6", "enable_subagents": True}
    rc = get_role_config("planner", FakeSettings(), session, run_overrides=run_ovr)
    assert rc.model == "claude-sonnet-4-6"
    assert rc.provider == "anthropic"
    assert rc.enable_subagents is True
    print(f"✅ run_overrides win — model→{rc.model}, subagents→{rc.enable_subagents}")


def test_subagent_role_inherits_implementer():
    rc = get_role_config("subagent", FakeSettings())
    assert rc.model == "kimi-k2.6"  # Default = implementer-Modell
    assert rc.provider == "ollama"
    print(f"✅ subagent inherits implementer-default — {rc.model}")


def test_invalid_json_safe():
    session = {"role_overrides_json": "{invalid json"}
    rc = get_role_config("planner", FakeSettings(), session)
    assert rc.model == "claude-opus-4-7"  # fällt auf defaults zurück
    print(f"✅ invalid_json_safe — fallback ok")


def test_encode_decode_roundtrip():
    src = {
        "planner": {"harness": "claude-code", "model": "claude-opus-4-7"},
        "implementer": {"model": "kimi-k2.6", "enable_subagents": True},
    }
    encoded = encode_role_overrides(src)
    decoded = parse_role_overrides(encoded)
    assert decoded["planner"]["model"] == "claude-opus-4-7"
    assert decoded["implementer"]["enable_subagents"] is True
    print(f"✅ encode/decode roundtrip ok ({len(encoded)} chars)")


def test_to_harness_request_kwargs():
    rc = get_role_config("implementer", FakeSettings())
    kw = rc.to_harness_request_kwargs()
    assert kw["role"] == "implementer"
    assert kw["model"] == "kimi-k2.6"
    assert kw["provider"] == "ollama"
    assert kw["harness"] == "claude-code"
    assert kw["enable_subagents"] is False
    assert kw["max_turns"] == 20
    print(f"✅ to_harness_request_kwargs — {kw}")


def main():
    print("=" * 60)
    print("  role_config Smoke-Tests")
    print("=" * 60)
    test_detect_provider()
    test_defaults_only()
    test_legacy_chat_session()
    test_role_overrides_json()
    test_run_overrides()
    test_subagent_role_inherits_implementer()
    test_invalid_json_safe()
    test_encode_decode_roundtrip()
    test_to_harness_request_kwargs()
    print("=" * 60)
    print("  ✅ Alle Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    main()
