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


def _analyses(n, perplexity=30.0, window_sim=0.1):
    # window_sim low => windowed turn-to-turn distance high => "diverse now"
    # (criterion 6 passes). Tests that need a collapsed turn-to-turn signal
    # pass a high window_sim explicitly.
    return [
        types.SimpleNamespace(
            token_perplexity=perplexity,
            semantic_similarity_window=window_sim,
        )
        for _ in range(n)
    ]


def _run(
    post_texts, perplexity=30.0, injection_text=INJECTION_TEXT, window_sim=0.1
):
    """Build a 10-round collapsed window + given post-injection texts."""
    collapsed = [f"near attractor thank you {i}" for i in range(10)]
    contents = collapsed + list(post_texts)
    return evaluate_recovery(
        agent_contents=contents,
        analyses=_analyses(len(contents), perplexity, window_sim),
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


def test_no_recovery_when_turn_to_turn_collapsed():
    """Criterion 6: moved far from the OLD attractor (version-b high) but the
    turn-to-turn windowed distance is low -- i.e. looping on a NEW topic. That
    is migration, not recovery, so it must not count."""
    post = [f"far diverse essay number {i}" for i in range(10)]
    res = _run(post, window_sim=0.9)  # windowed distance 0.1 << 0.40 cutoff
    assert not res.recovered
    assert res.hold_length == 0


def test_no_recovery_when_language_switched():
    """Criterion 7: a turn that moved away and varies turn-to-turn but switched
    script (Latin collapsed window -> CJK turns) is drift, not recovery -- and
    it invalidates the English perplexity/parroting guards."""
    # Each turn contains "far" (so it embeds as moved-away) but is dominantly
    # CJK, so its script differs from the Latin collapsed window.
    post = [f"far 确实市场投资策略非常重要分析 {i}" for i in range(10)]
    res = _run(post)
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


def _run_with_recollapse(post_texts, recollapse_rounds):
    collapsed = [f"near attractor thank you {i}" for i in range(10)]
    contents = collapsed + list(post_texts)
    return evaluate_recovery(
        agent_contents=contents,
        analyses=_analyses(len(contents)),
        injection_round=INJECTION_ROUND,
        injection_text=INJECTION_TEXT,
        window=WINDOW,
        recollapse_rounds=recollapse_rounds,
    )


def test_no_recovery_when_loop_recollapses():
    """Criterion 5: a turn that moved away but on which the detector re-declared
    collapse is a hop into a NEW loop, not recovery -- it can't count."""
    # post rounds 10-19 all 'far'; re-collapse at 13 and 17 splits them into
    # streaks of length 3, 3, 2 -- none reaches hold_k=5.
    post = [f"far diverse essay number {i}" for i in range(10)]
    res = _run_with_recollapse(post, recollapse_rounds=[13, 17])
    assert not res.recovered
    assert res.hold_length < 5


def test_recovery_survives_late_recollapse():
    """Criterion 5 is surgical: a re-collapse AFTER a full qualifying streak
    still leaves recovery intact (rounds 10-18 = a 9-round hold)."""
    post = [f"far diverse essay number {i}" for i in range(10)]
    res = _run_with_recollapse(post, recollapse_rounds=[19])
    assert res.recovered
    assert res.recovery_round == INJECTION_ROUND + 1  # round 10
    assert res.hold_length == 9
