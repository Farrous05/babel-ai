"""Lightweight, dependency-free language/script detection.

The collapse study runs from an English seed, but some models (notably Qwen,
trained heavily on Chinese) *drift into another language* as the self-loop
deepens -- a form of collapse toward the model's dominant pretraining
distribution. That drift silently breaks two recovery guards:

* **Perplexity** is scored with GPT-2, which only knows English -- a fluent
  Chinese turn looks like "gibberish" (or, worse, passes by accident).
* **Not-parroting** is a word-overlap (Jaccard) check against the *English*
  injection -- a Chinese turn shares no words, so it trivially "passes".

So a language switch must be *detected*, not ignored. We don't need to tell
English from French (same script); the failure mode is a **script** switch
(Latin -> CJK / Cyrillic / Arabic / ...), which a Unicode-block tally catches
exactly, with no model and no extra dependency.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

# Unicode "name" substrings that identify the script of an alphabetic char.
# Order matters only for readability; each char maps to the first that hits.
_SCRIPT_KEYS = (
    "CJK",  # Chinese (and shared Han ideographs)
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",  # Korean
    "CYRILLIC",
    "ARABIC",
    "HEBREW",
    "DEVANAGARI",
    "GREEK",
    "LATIN",
)


def _script_of(ch: str) -> str | None:
    """Return the script key for a single alphabetic char, or None."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for key in _SCRIPT_KEYS:
        if key in name:
            return "JAPANESE" if key in ("HIRAGANA", "KATAKANA") else key
    return "OTHER"


def script_histogram(text: str) -> dict[str, float]:
    """Fraction of alphabetic characters in each script. Empty if no letters."""
    counts: Counter[str] = Counter()
    for ch in text:
        s = _script_of(ch)
        if s:
            counts[s] += 1
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items()}


def dominant_script(text: str) -> str:
    """The script of the majority of a turn's letters.

    Returns ``"NONE"`` when the text has no alphabetic characters (e.g. pure
    code, numbers, or punctuation) -- callers should treat that as "no
    language signal", not as a switch.
    """
    hist = script_histogram(text)
    if not hist:
        return "NONE"
    return max(hist, key=hist.get)


def switched_language(text: str, baseline_script: str) -> bool:
    """True if ``text``'s dominant script differs from ``baseline_script``.

    ``NONE`` on either side (no letters / unknown baseline) is *not* a switch:
    we only flag a confident change of script.
    """
    if baseline_script == "NONE":
        return False
    here = dominant_script(text)
    if here == "NONE":
        return False
    return here != baseline_script
