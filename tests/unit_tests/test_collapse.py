"""Tests for the collapse detector."""

from babel_ai.collapse import CollapseConfig, CollapseDetector


def _feed(det, sem_window_series, sem_series=None, lex_window_series=None):
    """Feed a series of per-round windowed-similarity values to the detector."""
    for i, sw in enumerate(sem_window_series):
        det.update(
            round_idx=i,
            semantic_similarity_window=sw,
            semantic_similarity=(
                None if sem_series is None else sem_series[i]
            ),
            lexical_similarity_window=(
                None if lex_window_series is None else lex_window_series[i]
            ),
        )
    return det


# Streak-mechanics tests disable the warm-up guard (warmup=0) so they test the
# consecutive-low-distance logic in isolation. The warm-up itself is tested
# separately in test_warmup_*.
def _detector(warmup=0, **kwargs):
    return CollapseDetector(CollapseConfig(warmup=warmup, **kwargs))


def test_default_config_matches_locked_constants():
    cfg = CollapseConfig()
    assert cfg.window == 10
    assert cfg.persistence == 3  # K=3 (matches Maiti's window-size=3)
    assert cfg.cosine_cutoff == 0.40
    assert cfg.jaccard_cutoff == 0.40
    assert cfg.warmup == 10
    assert cfg.rearm_cutoff == 0.50  # hysteresis upper threshold (> cutoff)


def test_collapse_declared_after_persistence():
    """Diverse start, then sustained high similarity -> collapse at the round
    the K-th consecutive low-distance turn lands."""
    # cosine distance = 1 - sim. cutoff 0.40 -> sim >= 0.60 is "below cutoff".
    # rounds 0-3 diverse (sim 0.2), rounds 4+ collapsed (sim 0.9).
    det = _detector()
    series = [0.2, 0.2, 0.2, 0.2] + [0.9] * 6
    _feed(det, series)
    assert det.collapsed
    # K=3 consecutive: rounds 4,5,6 -> declared at round 6.
    assert det.onset_round == 6


def test_no_collapse_when_distance_stays_high():
    """A run that never sustains low distance must not fire."""
    det = _detector()
    _feed(det, [0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5])  # sim, dist 0.5-0.7
    assert not det.collapsed
    assert det.onset_round is None
    assert det.collapse_rate() is None


def test_streak_resets_on_diverse_blip():
    """A one-round blip above cutoff resets the streak (no premature fire)."""
    det = _detector()
    # 2 collapsed (< K=3), 1 diverse blip, then 3 collapsed -> 2nd streak fires.
    series = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9]
    _feed(det, series)
    assert det.collapsed
    assert det.onset_round == 5  # rounds 3,4,5 are the qualifying streak


def test_collapse_rate_is_positive_when_similarity_rises():
    """Rate = slope of turn-to-turn semantic similarity up to onset."""
    det = _detector()
    sem_window = [0.2, 0.3, 0.5, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    sem_tt = [0.1, 0.25, 0.45, 0.7, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9]
    _feed(det, sem_window, sem_series=sem_tt)
    assert det.collapsed
    rate = det.collapse_rate()
    assert rate is not None and rate > 0


def test_jaccard_tracked_but_does_not_trigger():
    """Semantic stays diverse while lexical collapses: no primary collapse,
    but the secondary Jaccard onset is recorded."""
    det = _detector()
    # semantic distance high (sim 0.2 -> dist 0.8, above 0.40 cutoff):
    sem_window = [0.2] * 8
    # lexical collapsed (sim 0.9 -> dist 0.1, below cutoff) throughout:
    lex_window = [0.9] * 8
    _feed(det, sem_window, lex_window_series=lex_window)
    assert not det.collapsed  # primary (semantic) never fires
    assert det.onset_round is None
    assert det.jaccard_onset_round == 2  # K=3 consecutive -> round 2


def test_custom_config_thresholds():
    """A stricter cutoff / persistence changes the onset."""
    det = _detector(persistence=3, cosine_cutoff=0.30)
    # sim 0.75 -> dist 0.25 <= 0.30 cutoff.
    _feed(det, [0.5, 0.75, 0.75, 0.75])
    assert det.collapsed
    assert det.onset_round == 3  # rounds 1,2,3


def test_rearm_requires_recovery_before_second_collapse():
    """Hysteresis: after rearm() the detector is DISARMED and will not declare a
    new collapse until the windowed cosine distance first rises above
    rearm_cutoff (the loop must visibly leave the attractor). Until then the
    still-collapsed (contaminated) window is ignored, not re-triggered."""
    det = _detector()  # rearm_cutoff default 0.50
    _feed(det, [0.9, 0.9, 0.9])  # K=3 -> collapse at round 2
    assert det.collapsed and det.onset_round == 2
    det.rearm(2)
    assert not det.collapsed and det.onset_round is None
    assert det.last_rearm_round == 2
    # the still-low post-injection window (sim 0.9 -> dist 0.1) must NOT fire a
    # spurious re-collapse while disarmed:
    for i in range(3, 7):
        det.update(round_idx=i, semantic_similarity_window=0.9)
    assert not det.collapsed
    # recovery: dist 0.8 (> 0.50 rearm_cutoff) re-arms the detector
    det.update(round_idx=7, semantic_similarity_window=0.2)
    # now a fresh K-run of low distance re-declares collapse
    for i in range(8, 11):
        det.update(round_idx=i, semantic_similarity_window=0.9)
    assert det.collapsed and det.onset_round == 10


def test_warmup_blocks_premature_collapse():
    """With the default warmup=10, a collapse in the unstable first rounds is
    not declared until the window has filled."""
    det = CollapseDetector()  # warmup = 10
    # collapsed from round 0, but only 8 rounds -> never reaches warmup.
    _feed(det, [0.9] * 8)
    assert not det.collapsed
    assert det.onset_round is None


def test_warmup_declares_at_boundary_when_collapse_persists():
    """If collapse persists through the warm-up, it is declared at the first
    qualifying round at/after the warmup boundary."""
    det = CollapseDetector()  # warmup = 10, K = 3
    _feed(det, [0.9] * 15)
    assert det.collapsed
    # streak >= 3 from round 2 on, but gated until round 10.
    assert det.onset_round == 10
