"""Tests für cascade/pricing.py — Cost-Berechnung pro Modell."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.pricing import (
    PRICES,
    ModelPrice,
    compute_cost,
    extract_token_counts,
    get_model_price,
)


def passed(label):
    print(f"  ✅ {label}")


def failed(label, why):
    print(f"  ❌ {label}: {why}")
    raise SystemExit(1)


def test_extract_anthropic_dict():
    print("\n[1] usage als Anthropic-snake_case dict")
    usage = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 1000,
    }
    counts = extract_token_counts(usage)
    assert counts == {"input": 100, "output": 200, "cache_read": 1000, "cache_creation": 50}, counts
    passed(f"snake_case → {counts}")


def test_extract_camelcase_dict():
    print("\n[2] usage als camelCase dict (model_usage Format)")
    usage = {
        "inputTokens": 4,
        "outputTokens": 119,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 27000,
    }
    counts = extract_token_counts(usage)
    assert counts == {"input": 4, "output": 119, "cache_read": 0, "cache_creation": 27000}, counts
    passed(f"camelCase → {counts}")


def test_extract_mixed_or_partial():
    print("\n[3] partielle / leere usage")
    assert extract_token_counts(None) == {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    assert extract_token_counts({}) == {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    only_in = extract_token_counts({"input_tokens": 5})
    assert only_in == {"input": 5, "output": 0, "cache_read": 0, "cache_creation": 0}
    passed("None/{}/partial alle 0-safe")


def test_get_model_price_exact():
    print("\n[4] exakter Modell-Lookup")
    p = get_model_price("claude-sonnet-4-6")
    assert p.input_per_mtok == 3.00
    assert p.output_per_mtok == 15.00
    p_opus = get_model_price("claude-opus-4-7")
    assert p_opus.input_per_mtok == 15.00
    passed(f"sonnet=$3/$15, opus=$15/$75")


def test_get_model_price_fallback():
    print("\n[5] Prefix-Fallback für unbekannte Tags")
    p = get_model_price("claude-haiku-4-5-20251101")  # nicht in Tabelle
    assert p.input_per_mtok == 1.00
    p2 = get_model_price("gpt-5-turbo")
    assert p2.input_per_mtok == 5.00
    passed(f"prefix fallback: claude-haiku-* → $1, gpt-* → $5")


def test_get_model_price_ollama():
    print("\n[6] Ollama-Modelle: 0.0 (Subscription)")
    for m in ("kimi-k2.6", "qwen3-coder:480b", "glm-5.1", "deepseek-v4-pro:cloud"):
        p = get_model_price(m)
        assert p.input_per_mtok == 0.0 and p.output_per_mtok == 0.0
    passed("alle Ollama-Modelle bei $0/$0")


def test_compute_cost_sonnet():
    print("\n[7] compute_cost — Sonnet-4-6, realistic Run")
    usage = {
        "input_tokens": 4,
        "output_tokens": 119,
        "cache_creation_input_tokens": 27047,
        "cache_read_input_tokens": 0,
    }
    cost = compute_cost(usage, "claude-sonnet-4-6")
    # 4 * 3.00/1M + 119 * 15.00/1M + 27047 * 3.75/1M = ~0.10
    expected = 4*3/1e6 + 119*15/1e6 + 27047*3.75/1e6
    assert abs(cost - expected) < 1e-6, f"{cost} vs {expected}"
    print(f"     cost = ${cost:.6f}")
    passed(f"sonnet 27k cache+119 out → ${cost:.4f}")


def test_compute_cost_opus():
    print("\n[8] compute_cost — Opus-4-7")
    usage = {"input_tokens": 1000, "output_tokens": 500}
    cost = compute_cost(usage, "claude-opus-4-7")
    # 1000*15/1M + 500*75/1M = 0.015 + 0.0375 = 0.0525
    assert abs(cost - 0.0525) < 1e-6
    passed(f"opus 1k in + 500 out → ${cost:.4f}")


def test_compute_cost_ollama_zero():
    print("\n[9] compute_cost — Ollama liefert 0")
    cost = compute_cost({"input_tokens": 100000, "output_tokens": 50000}, "kimi-k2.6")
    assert cost == 0.0
    passed("kimi 100k in + 50k out → $0")


def test_unknown_model_default():
    print("\n[10] unbekanntes Modell-Tag → Ollama-Fallback")
    p = get_model_price("ll4ma-9b-experimental")
    assert p.input_per_mtok == 0.0
    cost = compute_cost({"input_tokens": 1e6}, "ll4ma-9b-experimental")
    assert cost == 0.0
    passed("unbekannt → Fallback $0")


def main():
    print("=" * 60)
    print("  cascade/pricing.py Smoke-Tests")
    print("=" * 60)
    test_extract_anthropic_dict()
    test_extract_camelcase_dict()
    test_extract_mixed_or_partial()
    test_get_model_price_exact()
    test_get_model_price_fallback()
    test_get_model_price_ollama()
    test_compute_cost_sonnet()
    test_compute_cost_opus()
    test_compute_cost_ollama_zero()
    test_unknown_model_default()
    print("\n" + "=" * 60)
    print("  ✅ Alle 10 Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    main()
