"""Modell-Preis-Tabelle für Cost-Berechnung.

Preise in USD pro 1M Tokens (input / output / cache_read / cache_creation).
Stand 2026-05-04 — bei Provider-Preisänderungen hier nachpflegen.

`compute_cost(usage, model)` liefert die USD-Summe für ein konkretes Run-Result.
Nutzbar für claude-agent-sdk usage-dicts UND für selbst-getrackte Token-Counts
(z.B. tiktoken-Estimate auf Ollama-Pfaden).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPrice:
    """Preise in USD pro 1 Million Tokens."""
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0          # 0 = wie input
    cache_creation_per_mtok: float = 0.0      # 0 = wie input

    def cost_for(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        cr = self.cache_read_per_mtok or self.input_per_mtok
        cc = self.cache_creation_per_mtok or self.input_per_mtok
        return (
            input_tokens         * self.input_per_mtok    / 1_000_000
            + output_tokens      * self.output_per_mtok   / 1_000_000
            + cache_read_tokens  * cr                     / 1_000_000
            + cache_creation_tokens * cc                  / 1_000_000
        )


# ──────────────────────────────────────────────────────────────────────────
#  Preis-Tabelle (Stand 2026-05-04)
# ──────────────────────────────────────────────────────────────────────────
# Anthropic — claude.com/pricing
# Cache-Read = 1/10 Input, Cache-Creation = 1.25× Input (Standard-Anthropic-Schema)
PRICES: Dict[str, ModelPrice] = {
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-opus-4-7":      ModelPrice(15.00, 75.00, 1.50, 18.75),
    "claude-opus-4-6":      ModelPrice(15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-4-6":    ModelPrice( 3.00, 15.00, 0.30,  3.75),
    "claude-sonnet-4-5":    ModelPrice( 3.00, 15.00, 0.30,  3.75),
    "claude-haiku-4-5":     ModelPrice( 1.00,  5.00, 0.10,  1.25),
    "claude-haiku-4-5-20251001": ModelPrice(1.00, 5.00, 0.10, 1.25),

    # ── OpenAI ───────────────────────────────────────────────────────────
    # Stand 2026-04, kann driften
    "gpt-5":         ModelPrice(  5.00,  15.00),
    "gpt-5-mini":    ModelPrice(  0.50,   2.00),
    "o3":            ModelPrice( 60.00, 240.00),
    "o3-mini":       ModelPrice(  3.00,  12.00),
    "o4-mini":       ModelPrice(  3.00,  12.00),

    # ── Ollama Cloud ─────────────────────────────────────────────────────
    # Subscription-Preis ($20/Monat) → wir tracken Tokens für Budget-
    # Hinweise, aber USD wird auf 0 gehalten (kein per-call Pricing).
    "kimi-k2.6":          ModelPrice(0.0, 0.0),
    "qwen3-coder:480b":   ModelPrice(0.0, 0.0),
    "qwen3-coder":        ModelPrice(0.0, 0.0),
    "glm-5.1":            ModelPrice(0.0, 0.0),
    "deepseek-v4-pro:cloud": ModelPrice(0.0, 0.0),
    "deepseek-v4-pro":    ModelPrice(0.0, 0.0),
    "deepseek-v4-flash":  ModelPrice(0.0, 0.0),
    "minimax-m2.7":       ModelPrice(0.0, 0.0),
}


# Fallback-Preise wenn ein Modell nicht in der Tabelle steht
_DEFAULTS_BY_PREFIX: Tuple[Tuple[str, ModelPrice], ...] = (
    ("claude-opus",   ModelPrice(15.00, 75.00, 1.50, 18.75)),
    ("claude-sonnet", ModelPrice( 3.00, 15.00, 0.30,  3.75)),
    ("claude-haiku",  ModelPrice( 1.00,  5.00, 0.10,  1.25)),
    ("gpt-",          ModelPrice( 5.00, 15.00)),
    ("o3",            ModelPrice(60.00,240.00)),
    ("o4",            ModelPrice( 3.00, 12.00)),
)
_OLLAMA_FALLBACK = ModelPrice(0.0, 0.0)


def get_model_price(model: str) -> ModelPrice:
    """Liefert ModelPrice für `model`, mit Prefix-Fallback. Unbekannte
    Non-Anthropic-/Non-OpenAI-Modelle landen im Ollama-Fallback (0.0)."""
    if model in PRICES:
        return PRICES[model]
    m = model.lower()
    for prefix, price in _DEFAULTS_BY_PREFIX:
        if m.startswith(prefix):
            return price
    return _OLLAMA_FALLBACK


# ──────────────────────────────────────────────────────────────────────────
#  Token-Extraktion aus claude-agent-sdk usage-dict
# ──────────────────────────────────────────────────────────────────────────
def extract_token_counts(usage: Any) -> Dict[str, int]:
    """Extrahiert Token-Counts aus dem SDK ResultMessage.usage dict.

    Akzeptiert dict (snake_case wie Anthropic-API) oder model_usage-camelCase.
    Liefert immer {input, output, cache_read, cache_creation} mit ints.
    """
    if not usage:
        return {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}

    if isinstance(usage, dict):
        return {
            "input": int(
                usage.get("input_tokens")
                or usage.get("inputTokens")
                or 0
            ),
            "output": int(
                usage.get("output_tokens")
                or usage.get("outputTokens")
                or 0
            ),
            "cache_read": int(
                usage.get("cache_read_input_tokens")
                or usage.get("cacheReadInputTokens")
                or 0
            ),
            "cache_creation": int(
                usage.get("cache_creation_input_tokens")
                or usage.get("cacheCreationInputTokens")
                or 0
            ),
        }

    # Object mit Attributen (defensive Variante)
    return {
        "input": int(getattr(usage, "input_tokens", 0) or 0),
        "output": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cache_creation": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    }


def compute_cost(usage: Any, model: str) -> float:
    """USD-Kosten aus usage-dict + Modell-Tag. 0.0 wenn unbekanntes Modell
    oder Ollama (kein per-call Pricing)."""
    price = get_model_price(model)
    counts = extract_token_counts(usage)
    return price.cost_for(
        input_tokens=counts["input"],
        output_tokens=counts["output"],
        cache_read_tokens=counts["cache_read"],
        cache_creation_tokens=counts["cache_creation"],
    )
