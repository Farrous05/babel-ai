"""Injection module: knock a collapsed self-loop out of repetition.

Implements spec step 3. After the collapse detector fires, we take the model's
last output, optionally append an ``<end>`` marker, then append an injection
text, and feed the result back as the next input. The injection's exact
character span is tagged so it can be excluded from scoring, and its cosine
distance from the collapsed window is measured and logged.

Parameters (per intervention):
  - **size**:  1 word / 1 sentence / 1 paragraph  (``InjectionSize``)
  - **source**: real (off-topic dataset snippet) / noise (shuffled tokens)
  - **marker**: whether to prepend ``<end>`` (kept toggleable for the later
    marker-on/off control); on by default.

This module has no heavy import-time dependencies: ``build_injection`` works on
a plain text corpus, and the embedding model is imported lazily only when a
distance is actually measured.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from babel_ai.enums import InjectionSize, InjectionSource

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class InjectionEvent:
    """Record of one injection, for logging and span exclusion."""

    round: int  # self-loop round at which injection was applied
    size: str
    source: str
    use_marker: bool
    marker: str
    text: str  # the injected text (excludes the marker)
    distance: Optional[float]  # cosine distance vs the collapsed window
    span_start: int  # char offset of the injected span in the new message
    span_end: int  # char offset (exclusive) of the end of the injected span


def load_corpus(data_path: str) -> List[str]:
    """Load a flat list of text snippets from a ShareGPT-format JSON file."""

    with open(data_path, "r") as f:
        data = json.load(f)

    texts: List[str] = []
    for conv in data:
        for item in conv.get("items", []):
            value = item.get("value")
            if isinstance(value, str) and value.strip():
                texts.append(value)
    logger.info("Loaded injection corpus of %d snippets", len(texts))
    return texts


def _sample_snippet(
    size: InjectionSize, corpus: Sequence[str], rng: random.Random
) -> str:
    """Sample a coherent off-topic snippet of the requested granularity."""

    if not corpus:
        raise ValueError("Injection corpus is empty")

    if size == InjectionSize.WORD:
        for _ in range(20):
            words = _WORD_RE.findall(rng.choice(corpus))
            if words:
                return rng.choice(words)
        raise ValueError("No usable word found in corpus")

    if size == InjectionSize.SENTENCE:
        for _ in range(20):
            sents = [
                s.strip()
                for s in _SENTENCE_SPLIT.split(rng.choice(corpus))
                if len(s.split()) >= 3
            ]
            if sents:
                return rng.choice(sents)
        return rng.choice(corpus).strip()

    # PARAGRAPH
    for _ in range(20):
        paras = [
            p.strip()
            for p in _PARAGRAPH_SPLIT.split(rng.choice(corpus))
            if len(p.split()) >= 15
        ]
        if paras:
            return rng.choice(paras)
    return rng.choice(corpus).strip()


def build_injection(
    size: InjectionSize,
    source: InjectionSource,
    corpus: Sequence[str],
    rng: Optional[random.Random] = None,
) -> str:
    """Build the injection text of the given size and source.

    REAL returns a coherent off-topic snippet. NOISE returns the meaning-
    destroyed control: the same snippet with its words shuffled (or, for a
    single word, its characters scrambled) -- matched length/vocabulary, no
    structure.
    """

    rng = rng or random.Random()
    snippet = _sample_snippet(size, corpus, rng)

    if source == InjectionSource.REAL:
        return snippet

    words = snippet.split()
    if len(words) <= 1:
        chars = list(snippet)
        rng.shuffle(chars)
        return "".join(chars)
    rng.shuffle(words)
    return " ".join(words)


def build_far_injection(
    size: InjectionSize,
    source: InjectionSource,
    corpus: Sequence[str],
    window_texts: Sequence[str],
    num_candidates: int,
    rng: Optional[random.Random] = None,
) -> str:
    """Build an injection biased toward being *far off-topic*.

    Samples ``num_candidates`` injection texts and returns the one with the
    greatest cosine distance from the collapsed window's mean embedding (the
    most semantically different from what the loop is stuck on). With
    ``num_candidates == 1`` this is just ``build_injection`` (plain random).
    """

    rng = rng or random.Random()
    if num_candidates <= 1:
        return build_injection(size, source, corpus, rng=rng)

    candidates = [
        build_injection(size, source, corpus, rng=rng)
        for _ in range(num_candidates)
    ]
    window = [t for t in window_texts if t and t.strip()]
    if not window:
        return candidates[0]

    best, best_dist = candidates[0], -1.0
    for cand in candidates:
        dist = measure_injection_distance(cand, window)
        if dist is not None and dist > best_dist:
            best, best_dist = cand, dist
    logger.info(
        "Picked farthest of %d candidates (distance %.3f): %r",
        num_candidates,
        best_dist,
        best[:80],
    )
    return best


def measure_injection_distance(
    injection_text: str, window_texts: Sequence[str]
) -> Optional[float]:
    """Cosine distance between the injection and the collapsed window's mean
    embedding. ``None`` if there is nothing to compare. Reuses the analyzer's
    embedding model (imported lazily to avoid a heavy import here)."""

    window = [t for t in window_texts if t and t.strip()]
    if not injection_text.strip() or not window:
        return None

    from sentence_transformers.util import cos_sim

    from babel_ai.analyzer import SimilarityAnalyzer

    model = SimilarityAnalyzer.semantic_model
    inj = model.encode(injection_text, convert_to_tensor=True)
    win = model.encode(list(window), convert_to_tensor=True)
    win_mean = win.mean(dim=0)
    sim = float(cos_sim(inj, win_mean).item())
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def apply_injection(
    last_content: str,
    size: InjectionSize,
    source: InjectionSource,
    corpus: Sequence[str],
    window_texts: Sequence[str],
    round_idx: int,
    use_marker: bool = True,
    marker: str = "<end>",
    rng: Optional[random.Random] = None,
    num_candidates: int = 1,
) -> "tuple[str, InjectionEvent]":
    """Append marker + injection to ``last_content`` and return the new message
    plus an :class:`InjectionEvent` tagging the injected span.

    When ``num_candidates > 1`` the injection text is the farthest-off-topic of
    that many sampled candidates (see :func:`build_far_injection`).
    """

    text = build_far_injection(
        size, source, corpus, window_texts, num_candidates, rng=rng
    )
    prefix = last_content
    if use_marker:
        prefix = f"{prefix} {marker} "
    else:
        prefix = f"{prefix} "
    new_content = prefix + text

    event = InjectionEvent(
        round=round_idx,
        size=size.value,
        source=source.value,
        use_marker=use_marker,
        marker=marker if use_marker else "",
        text=text,
        distance=measure_injection_distance(text, window_texts),
        span_start=len(prefix),
        span_end=len(new_content),
    )
    logger.info(
        "Injected %s/%s at round %d (distance %s): %r",
        size.value,
        source.value,
        round_idx,
        event.distance,
        text[:80],
    )
    return new_content, event
