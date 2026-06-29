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

    if size == InjectionSize.MULTI_PARAGRAPH:
        # A *bigger* dose: gather several paragraphs (≥15 words each) from
        # across the corpus and concatenate them, blank-line separated.
        n_target = 4
        paras: List[str] = []
        for _ in range(60):
            for p in _PARAGRAPH_SPLIT.split(rng.choice(corpus)):
                p = p.strip()
                if len(p.split()) >= 15:
                    paras.append(p)
            if len(paras) >= n_target:
                break
        if paras:
            return "\n\n".join(paras[:n_target])
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


def _ensure_terminal(text: str) -> str:
    """Guarantee the injection ends on sentence-terminal punctuation, so a
    self-loop model treats it as a finished utterance to respond to rather than
    a fragment to continue. Code-only corpus snippets (ending in ``}``, ``)``,
    ``;``) are the main offenders; we append a period when needed."""

    text = (text or "").rstrip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _skim_first_half(text: str) -> "tuple[str, int]":
    """Keep the first half of the output's *whole sentences*; return the kept
    text and the **word count of the dropped half**. Cutting on a sentence
    boundary avoids leaving a broken fragment. Falls back to a word-split if the
    text is a single sentence."""

    sents = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if len(sents) <= 1:
        words = text.split()
        half = len(words) // 2
        return " ".join(words[:half]), len(words) - half
    keep_n = max(1, len(sents) // 2)
    kept = " ".join(sents[:keep_n])
    dropped_words = len(" ".join(sents[keep_n:]).split())
    return kept, dropped_words


def build_sized_injection(
    target_words: int,
    source: InjectionSource,
    corpus: Sequence[str],
    window_texts: Sequence[str],
    num_candidates: int,
    rng: Optional[random.Random] = None,
) -> str:
    """Build an off-topic injection of approximately ``target_words`` words.

    Prefers a **single coherent corpus item** long enough to reach the target on
    its own, so the injection is *one topic* rather than a multi-snippet jumble
    (stitching several unrelated snippets confounds "big dose" with "many
    topics"). Among the long-enough candidates it picks the farthest-off-topic
    one and takes its leading **whole sentences** up to the target -- always
    ending on a sentence boundary, never a mid-sentence fragment (which a
    self-loop would just *complete* instead of responding to). Only if no single
    item is long enough does it fall back to stitching. Length therefore
    approximates ``target_words``. Used by OUTPUT_SIZED (target = the model's
    last output length) and SKIM_HALF (target = the dropped half).
    """

    rng = rng or random.Random()

    def _sentences(text: str) -> List[str]:
        # Keep only *complete* sentences -- those that end in terminal
        # punctuation. The ShareGPT corpus is full of code and clipped snippets
        # whose tail piece has no '.', '!' or '?'; including such a fragment as
        # the last line is exactly what makes the self-loop model *continue* it
        # instead of responding to the injection.
        return [
            s.strip()
            for s in _SENTENCE_SPLIT.split(text)
            if s.strip() and s.strip()[-1] in ".!?"
        ]

    def _lead(text: str) -> "tuple[str, int]":
        """Leading whole sentences of ``text`` up to ~target_words."""
        out: List[str] = []
        n = 0
        for s in _sentences(text):
            out.append(s)
            n += len(s.split())
            if n >= target_words:
                break
        return " ".join(out), n

    # 1) Prefer ONE coherent item long enough to reach the target alone; among
    #    such candidates, keep the farthest off-topic. Sample as REAL text;
    #    NOISE is applied once at the end.
    long_enough: List[str] = []
    tries = 0
    while len(long_enough) < max(num_candidates, 1) and tries < 80:
        tries += 1
        lead, n = _lead(rng.choice(corpus))
        if n >= target_words:
            long_enough.append(lead)

    if long_enough:
        text = (
            long_enough[0]
            if len(long_enough) == 1
            else max(
                long_enough,
                key=lambda c: measure_injection_distance(c, window_texts)
                or -1.0,
            )
        )
    else:
        # 2) Fallback: no single item is long enough -> stitch whole sentences
        #    from far-off-topic paragraphs until the target is reached.
        pool = _sentences(
            build_far_injection(
                InjectionSize.PARAGRAPH, InjectionSource.REAL, corpus,
                window_texts, num_candidates, rng=rng,
            )
        )
        out: List[str] = []
        count = 0
        guard = 0
        while count < target_words and guard < 200:
            if not pool:
                pool = _sentences(
                    build_injection(
                        InjectionSize.PARAGRAPH, InjectionSource.REAL,
                        corpus, rng=rng,
                    )
                )
                guard += 1
                if not pool:
                    break
            s = pool.pop(0)
            out.append(s)
            count += len(s.split())
        text = " ".join(out) if out else build_injection(
            InjectionSize.PARAGRAPH, InjectionSource.REAL, corpus, rng=rng
        )

    # NOISE control: same length / vocabulary, structure destroyed.
    if source == InjectionSource.NOISE:
        words = text.split()
        rng.shuffle(words)
        text = " ".join(words)
    return text


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

    from babel_ai.embeddings import reference_embedder

    # Reference-based distance (injection vs the collapsed window) -> use the
    # text-embedding-3-large reference embedder, not the SBERT detector model.
    model = reference_embedder()
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

    rng = rng or random.Random()
    # `base` is the model text we keep before the injection. For most sizes it
    # is the full last output (pure append). SKIM_HALF replaces the dropped half
    # of the output with off-topic text of equal size, so `base` is the kept
    # first half and the input length stays ≈ one output.
    base = last_content
    if size == InjectionSize.OUTPUT_SIZED:
        target = max(15, len(last_content.split()))
        text = build_sized_injection(
            target, source, corpus, window_texts, num_candidates, rng=rng
        )
    elif size == InjectionSize.SKIM_HALF:
        base, dropped_words = _skim_first_half(last_content)
        text = build_sized_injection(
            max(15, dropped_words), source, corpus, window_texts,
            num_candidates, rng=rng,
        )
    else:
        text = build_far_injection(
            size, source, corpus, window_texts, num_candidates, rng=rng
        )
    # Every injection except a bare WORD must read as a finished utterance, so
    # the loop responds to it instead of completing a dangling fragment.
    if size != InjectionSize.WORD:
        text = _ensure_terminal(text)
    prefix = base
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
