"""Tests for the recovery evaluator.

The embedding model is monkeypatched: any text containing "far" embeds to a
vector orthogonal to the collapsed window, so cosine distance from the window
is 1.0 (moved away); otherwise distance is 0.0 (stayed near the attractor).
This lets us test the recovery *logic* without loading Sentence-BERT.
"""

import types

import pytest
import torch

from babel_ai.recovery import (
    RecoveryConfig,
    _jaccard_distance,
    evaluate_recovery,
)

WINDOW = 10
INJECTION_ROUND = 9
INJECTION_TEXT = "completely unrelated injected phrase about campaigns"


def _fake_embed(texts):
    rows = [[0.0, 1.0] if "far" in t.lower() else [1.0, 0.0] for t in texts]
    return torch.tensor(rows, dtype=torch.float)


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr("babel_ai.recovery._embed", _fake_embed)


def _analyses(n, perplexity=30.0):
    return [
        types.SimpleNamespace(token_perplexity=perplexity) for _ in range(n)
    ]


def _run(post_texts, perplexity=30.0, injection_text=INJECTION_TEXT):
    """Build a 10-round collapsed window + given post-injection texts."""
    collapsed = [f"near attractor thank you {i}" for i in range(10)]
    contents = collapsed + list(post_texts)
    return evaluate_recovery(
        agent_contents=contents,
        analyses=_analyses(len(contents), perplexity),
        injection_round=INJECTION_ROUND,
        injection_text=injection_text,
        window=WINDOW,
    )


def test_jaccard_distance_basics():
    assert _jaccard_distance("a b c", "a b c") == 0.0
    assert _jaccard_distance("a b c", "x y z") == 1.0
    assert _jaccard_distance("", "a") == 1.0


def test_recovers_when_moved_away_sane_not_parroting():
    post = [f"far diverse essay on biology number {i}" for i in range(10)]
    res = _run(post)
    assert res.recovered
    assert res.recovery_round == INJECTION_ROUND + 1  # first post round = 10
    assert res.hold_length == 10
    assert all(d == pytest.approx(1.0) for d in res.post_injection_distances)


def test_no_recovery_when_stays_near_attractor():
    post = [f"near attractor thank you again {i}" for i in range(10)]
    res = _run(post)
    assert not res.recovered
    assert res.recovery_round is None
    assert all(d == pytest.approx(0.0) for d in res.post_injection_distances)


def test_no_recovery_when_parroting_injection():
    # moved away (contains "far") but lexically identical to the injection.
    parrot = "far " + INJECTION_TEXT
    res = _run([parrot] * 10, injection_text=parrot)
    assert not res.recovered  # criterion 4 (not-parroting) fails


def test_no_recovery_when_gibberish_perplexity():
    post = [f"far diverse essay number {i}" for i in range(10)]
    res = _run(post, perplexity=999.0)  # above max_perplexity guard
    assert not res.recovered


def test_no_recovery_when_streak_too_short():
    # 3 far rounds (< hold_k=5), then back near the attractor.
    post = [f"far text {i}" for i in range(3)] + [
        f"near attractor {i}" for i in range(7)
    ]
    res = _run(post)
    assert not res.recovered
    assert res.hold_length == 3


def test_recovery_round_is_first_of_qualifying_streak():
    # 2 near, then 6 far -> streak starts at round 12.
    post = [f"near attractor {i}" for i in range(2)] + [
        f"far diverse {i}" for i in range(6)
    ]
    res = _run(post)
    assert res.recovered
    assert res.recovery_round == INJECTION_ROUND + 1 + 2  # round 12
    assert res.hold_length == 6


def test_custom_config_hold_k():
    post = [f"far diverse {i}" for i in range(3)]
    res = evaluate_recovery(
        agent_contents=[f"near {i}" for i in range(10)] + post,
        analyses=_analyses(13),
        injection_round=INJECTION_ROUND,
        injection_text=INJECTION_TEXT,
        window=WINDOW,
        config=RecoveryConfig(hold_k=3),
    )
    assert res.recovered and res.hold_length == 3
