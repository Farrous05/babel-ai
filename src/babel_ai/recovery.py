"""Recovery evaluation after a post-collapse injection (spec step 4).

Given a run that collapsed, was injected once, and continued, decide whether
the model *recovered* — i.e. moved away from the attractor and stayed diverse,
without just parroting the injected text or descending into gibberish.

Recovery is declared when, for ``hold_k`` consecutive post-injection rounds,
ALL of the spec's criteria hold:

1. **Diversity rose above the collapsed level** — operationalized as the
   spec's version-(b) signal: cosine distance of the turn from the *collapsed
   window's* mean embedding grows past a cutoff (the turn moved away from the
   attractor; equivalently cosine similarity to the attractor dropped).
2. **It holds for K rounds** — not a one-round blip (the ``hold_k`` streak).
3. **Perplexity stays sane** — GPT-2 perplexity within a band (not gibberish).
4. **Not just repeating the injection** — Jaccard distance between the turn and
   the injected text stays well above ~0.

Thresholds are provisional (like the collapse cutoffs) and meant to be
calibrated against hand-labeled recoveries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class RecoveryConfig:
    """Provisional recovery thresholds (calibrate against labeled runs)."""

    hold_k: int = 5  # consecutive qualifying rounds to declare recovery
    distance_cutoff: float = (
        0.30  # cosine dist from collapsed window to qualify
    )
    max_perplexity: float = 150.0  # GPT-2 perplexity gibberish guard
    min_jaccard_to_injection: float = (
        0.5  # turn must differ from the injection
    )


@dataclass
class RecoveryResult:
    """Outcome of recovery evaluation for one run."""

    recovered: bool
    recovery_round: Optional[int]  # absolute self-loop round (start of streak)
    hold_length: int  # length of the qualifying streak
    # cosine distance of each post-injection turn from the collapsed window
    post_injection_distances: List[float] = field(default_factory=list)


def _word_set(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard_distance(a: str, b: str) -> float:
    """1 - |A∩B| / |A∪B| over word sets. 1.0 if either side is empty."""
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - (inter / union if union else 0.0)


def _embed(texts: Sequence[str]):
    """Embed texts with the analyzer's Sentence-BERT model (lazy import)."""
    from babel_ai.analyzer import SimilarityAnalyzer

    return SimilarityAnalyzer.semantic_model.encode(
        list(texts), convert_to_tensor=True
    )


def version_b_distances(
    contents: Sequence[str], collapsed_window: Sequence[str]
) -> List[float]:
    """Spec version-(b) distance of each content from the collapsed window.

    Embeds the (non-empty) collapsed-window turns, takes their centroid, and
    returns ``1 - cosine_similarity`` of every entry in ``contents`` to that
    centroid (clamped to a valid cosine range). This is the single definition
    of the "distance from the collapsed window" signal, shared by the recovery
    evaluator (post-injection turns) and the per-round CSV logger (all turns).

    ``collapsed_window`` must contain at least one non-empty string; callers
    guard this.
    """

    from sentence_transformers.util import cos_sim

    window = [c for c in collapsed_window if c.strip()]
    centroid = _embed(window).mean(dim=0)
    emb = _embed(list(contents))
    distances: List[float] = []
    for i in range(len(contents)):
        sim = float(cos_sim(emb[i], centroid).item())
        distances.append(1.0 - max(-1.0, min(1.0, sim)))
    return distances


def evaluate_recovery(
    agent_contents: Sequence[str],
    analyses: Sequence[object],
    injection_round: int,
    injection_text: str,
    window: int,
    config: Optional[RecoveryConfig] = None,
) -> RecoveryResult:
    """Evaluate whether the run recovered after the injection.

    Args:
        agent_contents: per-round self-loop turn texts (index = round). These
            are the *original* model outputs (the injected span is excluded).
        analyses: per-round ``AnalysisResult`` aligned with ``agent_contents``
            (used for perplexity).
        injection_round: round at which injection was applied.
        injection_text: the injected text (criterion 4).
        window: collapsed-window size (the turns up to & incl. injection_round).
        config: thresholds; defaults to ``RecoveryConfig()``.
    """

    config = config or RecoveryConfig()
    n = len(agent_contents)

    post_start = injection_round + 1
    if post_start >= n:
        return RecoveryResult(False, None, 0, [])

    # Collapsed window = the `window` turns up to and including injection_round.
    win_lo = max(0, injection_round - window + 1)
    collapsed_window = [
        c for c in agent_contents[win_lo : injection_round + 1] if c.strip()
    ]
    post_contents = list(agent_contents[post_start:])
    if not collapsed_window or not any(c.strip() for c in post_contents):
        return RecoveryResult(False, None, 0, [])

    # Version-(b) distances: each post-injection turn vs collapsed centroid.
    distances = version_b_distances(post_contents, collapsed_window)

    # Per-round qualification against all four criteria.
    streak = 0
    best_start: Optional[int] = None
    best_len = 0
    cur_start = 0
    for i, content in enumerate(post_contents):
        round_idx = post_start + i
        analysis = analyses[round_idx] if round_idx < len(analyses) else None
        perplexity = getattr(analysis, "token_perplexity", None)

        moved_away = distances[i] >= config.distance_cutoff
        perplexity_ok = (
            perplexity is not None and perplexity <= config.max_perplexity
        )
        not_parroting = (
            _jaccard_distance(content, injection_text)
            >= config.min_jaccard_to_injection
        )

        if moved_away and perplexity_ok and not_parroting:
            if streak == 0:
                cur_start = round_idx
            streak += 1
            if streak > best_len:
                best_len = streak
                best_start = cur_start
        else:
            streak = 0

    recovered = best_len >= config.hold_k
    if recovered:
        logger.info(
            "Recovery declared: round %d, held %d rounds", best_start, best_len
        )
    else:
        logger.info("No recovery (longest qualifying streak %d)", best_len)

    return RecoveryResult(
        recovered=recovered,
        recovery_round=best_start if recovered else None,
        hold_length=best_len,
        post_injection_distances=distances,
    )
