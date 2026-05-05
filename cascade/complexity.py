"""Plan v5 R2 — 3-Tier Komplexitäts-basiertes Routing.

Inspiration: Ruflo's intelligence-Plugin. Statt jeden Task ans gleiche
(meist teure) Modell zu schicken, klassifiziert Cascade vor dem Run die
Komplexität und wählt:

  Tier 1 — TRIVIAL    : kein LLM-Call, direkt apply (Renames, Imports,
                        Bool-Flips). Plan hat direct_ops mit ≤3 ops.
                        $0 Kosten.
  Tier 2 — STANDARD   : Sonnet-Niveau (claude-sonnet-4-6 / kimi-k2.6).
                        Bug-Fixes, Feature-Adds, Refactoring im
                        kleinen Maßstab. Standard-Wahl.
  Tier 3 — COMPLEX    : Opus-Niveau (claude-opus-4-7). Architektur-
                        Decisions, ≥3 Sub-Tasks, Cross-File-Refactors,
                        Security-Reviews, Distributed-Systems-Design.

Heuristik first (kein LLM-Call), bei Ambivalenz fallback zu Haiku-
Klassifikation (~$0.001).

Spar-Effekt erwartet: 30-50% Cost (Trivial geht $0, Standard statt Opus).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


logger = logging.getLogger("cascade.complexity")


class Tier(Enum):
    TRIVIAL = "trivial"     # Tier 1
    STANDARD = "standard"   # Tier 2
    COMPLEX = "complex"     # Tier 3


@dataclass
class TierDecision:
    tier: Tier
    confidence: float       # 0.0–1.0
    reason: str             # menschenlesbar
    via_heuristic: bool = True  # False wenn LLM-classifier benutzt wurde


# ──────────────────────────────────────────────────────────────────────
#  Defaults pro Tier
# ──────────────────────────────────────────────────────────────────────
TIER_DEFAULT_MODELS = {
    Tier.TRIVIAL:  "claude-haiku-4-5",      # Fallback wenn doch LLM gebraucht
    Tier.STANDARD: "claude-sonnet-4-6",
    Tier.COMPLEX:  "claude-opus-4-7",
}


# Keywords im Task-Text die auf hohe Komplexität hinweisen.
_COMPLEX_KEYWORDS = [
    "architecture", "architektur",
    "refactor", "refaktor",
    "redesign", "umbau",
    "migration",
    "distributed", "verteilte",
    "security review", "sicherheitsanalyse",
    "performance optimization", "performance-optimierung",
    "race condition", "concurrency",
    "schema migration", "datenbank-migration",
    "authentication", "authentifizierung",
    "authorization", "autorisierung",
]

# Keywords die eindeutig auf Trivial hinweisen.
_TRIVIAL_KEYWORDS = [
    "rename ", "umbenennen ",
    "typo", "tippfehler",
    "comment", "kommentar",
    "format", "formatierung",
    "lint",
    "import sort", "imports sortieren",
    "remove unused", "ungenutzten code entfernen",
    "constant ", "konstante ",
]


# ──────────────────────────────────────────────────────────────────────
#  Heuristic-Klassifikation
# ──────────────────────────────────────────────────────────────────────
def classify_via_heuristic(
    task_text: str,
    *,
    plan: Optional[Any] = None,
) -> TierDecision:
    """Klassifiziert anhand Task-Text + optional Plan-Struktur.

    Rückgabe Confidence:
      1.0  — ganz klares Signal (z.B. >5 sub-tasks ODER klare keywords)
      0.7  — solides Signal aber nicht überwältigend
      0.4  — ambivalent (Caller könnte LLM-Klassifikation triggern)
    """
    txt = (task_text or "").lower()

    # ── Tier 1 (Trivial) ────────────────────────────────────────────
    # Sehr kurzer Task + Trivial-Keyword → Tier 1
    if len(txt) < 200:
        for kw in _TRIVIAL_KEYWORDS:
            if kw in txt:
                return TierDecision(
                    tier=Tier.TRIVIAL,
                    confidence=0.85,
                    reason=f"trivial-keyword '{kw.strip()}' in <200 chars",
                )

    # Plan hat direct_ops mit ≤3 ops und KEINE sub_tasks → Tier 1
    if plan is not None:
        direct_ops = getattr(plan, "direct_ops", None) or []
        sub_tasks = getattr(plan, "subtasks", None) or []
        if direct_ops and len(direct_ops) <= 3 and not sub_tasks:
            return TierDecision(
                tier=Tier.TRIVIAL,
                confidence=1.0,
                reason=f"direct_ops with {len(direct_ops)} ops, no sub_tasks",
            )

    # ── Tier 3 (Complex) ────────────────────────────────────────────
    # Complex-Keyword + längerer Task → Tier 3
    matching_complex = [kw for kw in _COMPLEX_KEYWORDS if kw in txt]
    if matching_complex and len(txt) > 100:
        return TierDecision(
            tier=Tier.COMPLEX,
            confidence=0.85,
            reason=f"complex-keywords: {matching_complex[:3]}",
        )

    # Plan hat ≥3 sub_tasks ODER ≥10 files_to_touch → Tier 3
    if plan is not None:
        sub_tasks = getattr(plan, "subtasks", None) or []
        files_to_touch = getattr(plan, "files_to_touch", None) or []
        if len(sub_tasks) >= 3:
            return TierDecision(
                tier=Tier.COMPLEX,
                confidence=1.0,
                reason=f"{len(sub_tasks)} sub-tasks (decomposed plan)",
            )
        if len(files_to_touch) >= 10:
            return TierDecision(
                tier=Tier.COMPLEX,
                confidence=0.9,
                reason=f"{len(files_to_touch)} files_to_touch",
            )

    # ── Tier 2 (Standard) — der häufigste Default ─────────────────
    # Plan mit 1-2 sub-tasks oder 1-9 files
    if plan is not None:
        sub_tasks = getattr(plan, "subtasks", None) or []
        files_to_touch = getattr(plan, "files_to_touch", None) or []
        if 1 <= len(sub_tasks) <= 2:
            return TierDecision(
                tier=Tier.STANDARD,
                confidence=0.85,
                reason=f"{len(sub_tasks)} sub-task(s), files={len(files_to_touch)}",
            )
        if 1 <= len(files_to_touch) <= 9:
            return TierDecision(
                tier=Tier.STANDARD,
                confidence=0.7,
                reason=f"{len(files_to_touch)} files_to_touch (no sub-tasks)",
            )

    # ── Ambivalent ─────────────────────────────────────────────────
    # Kein klares Signal — defaultmäßig Standard mit niedriger Confidence
    return TierDecision(
        tier=Tier.STANDARD,
        confidence=0.4,
        reason="no clear signal, defaulting to Standard",
    )


# ──────────────────────────────────────────────────────────────────────
#  LLM-Klassifikation (optional, wenn Heuristik unsicher)
# ──────────────────────────────────────────────────────────────────────
async def classify_via_llm(
    task_text: str,
    *,
    settings: Any,
    timeout_s: float = 30.0,
) -> TierDecision:
    """Ruft Haiku-classifier (~$0.001) wenn Heuristik confidence < 0.7.

    Erwartet vom Modell ein einzelnes Wort: TRIVIAL / STANDARD / COMPLEX.
    Bei ungültiger Antwort → Fallback auf Heuristik.
    """
    from cascade.llm_client import agent_chat

    prompt = (
        "Classify the complexity of this software task into ONE of three tiers. "
        "Reply with ONLY the tier name in caps, no other text.\n\n"
        "TIER_1_TRIVIAL: typo fixes, renames, format/lint, single one-line edits, "
        "imports cleanup. ≤3 file operations. <30s of work.\n"
        "TIER_2_STANDARD: bug fixes, small feature additions, refactoring of "
        "1-3 files. Most common. 1-15 minutes of work.\n"
        "TIER_3_COMPLEX: architecture decisions, multi-file refactors, security "
        "design, distributed systems, schema migrations. ≥30 minutes of work.\n\n"
        f"TASK:\n{task_text[:1500]}\n\n"
        "Answer (one word: TRIVIAL, STANDARD, or COMPLEX):"
    )
    try:
        raw = await agent_chat(
            prompt=prompt,
            model="claude-haiku-4-5",
            system_prompt="You are a complexity classifier. Reply with ONE word only.",
            output_json=False,
            temperature=0.0,
            s=settings,
            timeout_s=timeout_s,
            retry_max_total_wait_s=60.0,  # cap kurz halten
            retry_min_backoff_s=5.0,
        )
    except Exception as e:
        logger.warning(f"complexity LLM-classifier failed, fallback heuristic: {e}")
        return classify_via_heuristic(task_text)

    answer = (raw or "").strip().upper()
    if "TRIVIAL" in answer or "TIER_1" in answer:
        return TierDecision(Tier.TRIVIAL, 0.9, "LLM:trivial", via_heuristic=False)
    if "COMPLEX" in answer or "TIER_3" in answer:
        return TierDecision(Tier.COMPLEX, 0.9, "LLM:complex", via_heuristic=False)
    if "STANDARD" in answer or "TIER_2" in answer:
        return TierDecision(Tier.STANDARD, 0.9, "LLM:standard", via_heuristic=False)
    logger.warning(f"complexity LLM returned unparseable: {answer[:80]}")
    return classify_via_heuristic(task_text)


# ──────────────────────────────────────────────────────────────────────
#  Combined Entry-Point
# ──────────────────────────────────────────────────────────────────────
async def decide_tier(
    task_text: str,
    *,
    plan: Optional[Any] = None,
    settings: Optional[Any] = None,
    use_llm_for_ambivalent: bool = True,
    confidence_threshold: float = 0.6,
) -> TierDecision:
    """Hauptentry: Heuristik first, bei Confidence < threshold optional
    LLM-Classifier (wenn settings gegeben + use_llm_for_ambivalent=True).
    """
    h = classify_via_heuristic(task_text, plan=plan)
    if h.confidence >= confidence_threshold:
        return h
    if not use_llm_for_ambivalent or settings is None:
        return h
    return await classify_via_llm(task_text, settings=settings)


def model_for_tier(tier: Tier, override_models: Optional[dict] = None) -> str:
    """Modell-String für einen Tier — mit optionalem Override-Map."""
    if override_models and tier in override_models:
        return override_models[tier]
    return TIER_DEFAULT_MODELS[tier]
