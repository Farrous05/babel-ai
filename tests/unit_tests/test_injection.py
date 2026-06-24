"""Tests for the injection module (no embedding model needed)."""

import random

import pytest
from pydantic import ValidationError

from babel_ai.enums import InjectionSize, InjectionSource, InjectionTrigger
from babel_ai.injection import (
    InjectionEvent,
    apply_injection,
    build_injection,
)
from models import InjectionConfig

CORPUS = [
    "The mitochondria is the powerhouse of the cell. It produces energy. "
    "This process is called respiration.",
    "Paris is the capital of France.\n\nIt sits on the river Seine and is "
    "famous for the Eiffel Tower, the Louvre, and many historic cafes that "
    "line its broad and busy boulevards every single day of the year.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
]


def _rng():
    return random.Random(42)


def test_build_real_word_is_single_word():
    out = build_injection(
        InjectionSize.WORD, InjectionSource.REAL, CORPUS, rng=_rng()
    )
    assert out and len(out.split()) == 1


def test_build_real_sentence_nonempty():
    out = build_injection(
        InjectionSize.SENTENCE, InjectionSource.REAL, CORPUS, rng=_rng()
    )
    assert out and len(out.split()) >= 3


def test_build_real_paragraph_is_longer():
    out = build_injection(
        InjectionSize.PARAGRAPH, InjectionSource.REAL, CORPUS, rng=_rng()
    )
    assert out and len(out.split()) >= 10


def test_noise_sentence_preserves_word_multiset():
    """Shuffled-token noise keeps the same bag of words, different order."""
    rng = random.Random(7)
    real = build_injection(
        InjectionSize.SENTENCE, InjectionSource.REAL, CORPUS, rng=rng
    )
    rng2 = random.Random(7)  # same draw -> same snippet, then shuffled
    noise = build_injection(
        InjectionSize.SENTENCE, InjectionSource.NOISE, CORPUS, rng=rng2
    )
    assert sorted(real.split()) == sorted(noise.split())


def test_noise_word_scrambles_characters():
    rng = random.Random(7)
    real = build_injection(
        InjectionSize.WORD, InjectionSource.REAL, CORPUS, rng=rng
    )
    rng2 = random.Random(7)
    noise = build_injection(
        InjectionSize.WORD, InjectionSource.NOISE, CORPUS, rng=rng2
    )
    assert sorted(real) == sorted(noise)  # same characters


def test_build_injection_is_deterministic_with_seed():
    a = build_injection(
        InjectionSize.SENTENCE, InjectionSource.REAL, CORPUS, rng=_rng()
    )
    b = build_injection(
        InjectionSize.SENTENCE, InjectionSource.REAL, CORPUS, rng=_rng()
    )
    assert a == b


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        build_injection(
            InjectionSize.WORD, InjectionSource.REAL, [], rng=_rng()
        )


def test_apply_injection_tags_span_with_marker():
    last = "I'm doing well, thanks!"
    new_content, event = apply_injection(
        last_content=last,
        size=InjectionSize.SENTENCE,
        source=InjectionSource.REAL,
        corpus=CORPUS,
        window_texts=[],  # no window -> distance None, no model loaded
        round_idx=12,
        use_marker=True,
        marker="<end>",
        rng=_rng(),
    )
    assert isinstance(event, InjectionEvent)
    # the original output is preserved verbatim at the start
    assert new_content.startswith(last)
    assert "<end>" in new_content
    # the tagged span is exactly the injected text
    assert new_content[event.span_start : event.span_end] == event.text
    assert event.round == 12
    assert event.source == "real"
    assert event.distance is None  # empty window -> not measured


def test_apply_injection_without_marker_omits_marker():
    last = "Sure thing."
    new_content, event = apply_injection(
        last_content=last,
        size=InjectionSize.WORD,
        source=InjectionSource.REAL,
        corpus=CORPUS,
        window_texts=[],
        round_idx=5,
        use_marker=False,
        rng=_rng(),
    )
    assert "<end>" not in new_content
    assert event.marker == ""
    assert new_content[event.span_start : event.span_end] == event.text


# --- InjectionConfig trigger/fixed_round validation (spec step 5) ---


def test_config_after_collapse_default_ok():
    cfg = InjectionConfig(
        size=InjectionSize.SENTENCE, source=InjectionSource.REAL
    )
    assert cfg.trigger == InjectionTrigger.AFTER_COLLAPSE
    assert cfg.fixed_round is None


def test_config_fixed_round_requires_round():
    with pytest.raises(ValidationError):
        InjectionConfig(
            size=InjectionSize.SENTENCE,
            source=InjectionSource.REAL,
            trigger=InjectionTrigger.FIXED_ROUND,
        )


def test_config_fixed_round_with_round_ok():
    cfg = InjectionConfig(
        size=InjectionSize.SENTENCE,
        source=InjectionSource.REAL,
        trigger=InjectionTrigger.FIXED_ROUND,
        fixed_round=5,
    )
    assert cfg.fixed_round == 5


def test_config_fixed_round_forbidden_for_after_collapse():
    with pytest.raises(ValidationError):
        InjectionConfig(
            size=InjectionSize.SENTENCE,
            source=InjectionSource.REAL,
            trigger=InjectionTrigger.AFTER_COLLAPSE,
            fixed_round=5,
        )
