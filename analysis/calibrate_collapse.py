"""Calibrate the collapse detector from real self-loop runs.

Part A of Project_spec.md steps 1-2. Loads one or more
`drift_experiment_*.csv` runs produced by ``src/main.py``, derives the two
distance signals the detector will use::

    cosine_distance  = 1 - semantic_similarity
    jaccard_distance = 1 - lexical_similarity

then (a) plots the three headline metrics per round so collapse is visible,
and (b) sweeps the paper's "consecutive steps below a cutoff" rule to suggest
calibration constants (window ``W``, persistence ``K``, distance cutoffs) and
report the collapse-onset round (TTC) and collapse rate per run.

Nothing here detects collapse *online* -- this is an offline calibration aid.
The chosen constants get baked into ``babel_ai/collapse.py`` in Part B.

Usage::

    poetry run python analysis/calibrate_collapse.py results/drift_experiment_*.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Reuse the existing CSV loader / iteration-alignment helpers.
sys.path.insert(0, "src")
from graphical_analysis.data_utils import (  # noqa: E402
    compute_relative_iteration_from_first_llm,
    load_experiment_rows,
)

# Smoothing window and persistence to evaluate (spec defaults W=10, K=10;
# the referenced paper used a shorter persistence of 3 -- we report both).
WINDOWS: Sequence[int] = (1, 5, 10)
PERSISTENCES: Sequence[int] = (3, 5, 10)


@dataclass
class RunSeries:
    """Per-round metric series for one run, aligned to the first LLM turn."""

    name: str
    rel_iter: np.ndarray  # round index, 0 = first model self-loop turn
    cosine_sim: np.ndarray
    cosine_dist: np.ndarray
    jaccard_dist: np.ndarray
    coherence: np.ndarray
    perplexity: np.ndarray


def load_run(csv_path: str) -> RunSeries:
    """Load one CSV into a RunSeries, keeping only LLM self-loop rounds."""

    rows = load_experiment_rows(csv_path)
    df = compute_relative_iteration_from_first_llm(rows)
    df = df[df["rel_iter_from_llm"] >= 0].sort_values("rel_iter_from_llm")

    sem = df["semantic_similarity"].to_numpy(dtype=float)
    lex = df["lexical_similarity"].to_numpy(dtype=float)
    return RunSeries(
        name=csv_path.split("/")[-1].replace("drift_experiment_", "")[:-4],
        rel_iter=df["rel_iter_from_llm"].to_numpy(dtype=float),
        cosine_sim=sem,
        cosine_dist=1.0 - sem,
        jaccard_dist=1.0 - lex,
        coherence=df["coherence_score"].to_numpy(dtype=float),
        perplexity=df["token_perplexity"].to_numpy(dtype=float),
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean (min_periods=1), NaN-tolerant."""

    s = pd.Series(values)
    return s.rolling(window, min_periods=1).mean().to_numpy()


def first_onset(
    distance: np.ndarray, cutoff: float, persistence: int, window: int
) -> Optional[int]:
    """First round index where smoothed distance stays <= cutoff for
    ``persistence`` consecutive rounds. Returns the round the run is
    *declared* collapsed (the end of that streak), or None."""

    smoothed = _smooth(distance, window)
    run = 0
    for i, v in enumerate(smoothed):
        if not np.isnan(v) and v <= cutoff:
            run += 1
            if run >= persistence:
                return i
        else:
            run = 0
    return None


def collapse_rate(cosine_sim: np.ndarray, onset: int) -> float:
    """Slope of semantic similarity rising from round 0 to onset (TTC).

    Equivalent to the rate at which cosine distance falls toward the
    attractor. Linear-regression slope over [0, onset]."""

    if onset is None or onset < 1:
        return float("nan")
    x = np.arange(onset + 1, dtype=float)
    y = cosine_sim[: onset + 1]
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def summarize(run: RunSeries) -> None:
    """Print early-vs-late metric levels and the onset/rate sweep."""

    n = len(run.rel_iter)
    early = slice(0, min(10, n))
    late = slice(max(0, n - 10), n)

    def avg(a: np.ndarray, sl: slice) -> float:
        v = a[sl]
        v = v[~np.isnan(v)]
        return float(np.mean(v)) if len(v) else float("nan")

    print(f"\n=== run {run.name}  ({n} self-loop rounds) ===")
    print(
        f"  cosine_sim   early={avg(run.cosine_sim, early):.3f} "
        f"late={avg(run.cosine_sim, late):.3f}"
    )
    print(
        f"  cosine_dist  early={avg(run.cosine_dist, early):.3f} "
        f"late={avg(run.cosine_dist, late):.3f}"
    )
    print(
        f"  jaccard_dist early={avg(run.jaccard_dist, early):.3f} "
        f"late={avg(run.jaccard_dist, late):.3f}"
    )
    print(
        f"  coherence    early={avg(run.coherence, early):.3f} "
        f"late={avg(run.coherence, late):.3f}"
    )

    # Suggest a cosine cutoff: midpoint between early and late distance.
    mid = (avg(run.cosine_dist, early) + avg(run.cosine_dist, late)) / 2.0
    print(f"  suggested cosine cutoff (early/late midpoint) ~ {mid:.3f}")
    print("  onset (round collapse declared) by (cutoff, K, W):")
    for cutoff in (round(mid, 2), round(mid - 0.05, 2)):
        for k in PERSISTENCES:
            for w in WINDOWS:
                onset = first_onset(run.cosine_dist, cutoff, k, w)
                if onset is not None:
                    rate = collapse_rate(run.cosine_sim, onset)
                    print(
                        f"    cutoff={cutoff:.2f} K={k:<2} W={w:<2} "
                        f"-> onset={onset:<3} rate={rate:+.4f}"
                    )
                else:
                    print(
                        f"    cutoff={cutoff:.2f} K={k:<2} W={w:<2} "
                        f"-> no collapse"
                    )


def plot_runs(runs: List[RunSeries], out_path: str) -> None:
    """Plot cosine distance, jaccard distance, coherence, perplexity."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("cosine_dist", "cosine distance (1 - sem sim)"),
        ("jaccard_dist", "jaccard distance (1 - lex sim)"),
        ("coherence", "coherence (unique/total words)"),
        ("perplexity", "GPT-2 perplexity"),
    ]
    # Raw per-turn metrics are jumpy (collapse is an A,A,B,B 2-cycle). Overlay
    # the W=10 rolling mean as a bold line. Distance panels show the windowed
    # signal the detector uses; coherence is smoothed for readability;
    # perplexity is left raw so per-turn gibberish spikes stay visible.
    smooth_attrs = {"cosine_dist", "jaccard_dist", "coherence"}

    # Legend labels carry each run's measured collapse onset (not engineered).
    labels = {}
    for run in runs:
        onset = first_onset(run.cosine_dist, 0.40, 5, 10)
        labels[run.name] = (
            f"{run.name}  (collapse ~r{onset})"
            if onset is not None
            else f"{run.name}  (no collapse)"
        )

    for ax, (attr, title) in zip(axes.flat, panels):
        for i, run in enumerate(runs):
            color = f"C{i}"
            values = getattr(run, attr)
            faint = 0.25 if attr in smooth_attrs else 0.7
            ax.plot(
                run.rel_iter,
                values,
                marker=".",
                alpha=faint,
                color=color,
                label=None if attr in smooth_attrs else labels[run.name],
            )
            if attr in smooth_attrs:
                ax.plot(
                    run.rel_iter,
                    _smooth(values, 10),
                    color=color,
                    linewidth=2.2,
                    label=labels[run.name],
                )
        if attr == "cosine_dist":
            ax.axhline(
                0.40, color="k", linestyle="--", linewidth=1, alpha=0.6
            )
        ax.set_title(title)
        ax.set_xlabel("round (0 = first self-loop turn)")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=8, title="run (measured collapse round)")
    fig.suptitle("Collapse calibration — metrics per round", fontsize=14)
    caption = (
        "Three replicate single-agent self-loop runs (gpt-4o-mini, ShareGPT "
        "seed, identical config); they collapse at different rounds purely "
        "from seed/API randomness.\n"
        "Distance panels: faint = raw turn-to-turn (jumpy because collapse is "
        "an A,A,B,B 2-cycle); bold = windowed W=10, the signal the detector "
        "uses; dashed = cutoff 0.40.\n"
        "Collapse = windowed cosine distance falling and staying below the "
        "cutoff. Coherence stays high (not a collapse signal); perplexity "
        "stays low (fluent repetition, not gibberish)."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", help="drift_experiment_*.csv paths")
    parser.add_argument(
        "--out",
        default="results/collapse_calibration.png",
        help="output figure path",
    )
    args = parser.parse_args()

    runs = [load_run(p) for p in args.csvs]
    for run in runs:
        summarize(run)
    plot_runs(runs, args.out)


if __name__ == "__main__":
    main()
