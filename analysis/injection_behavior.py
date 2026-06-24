"""Characterise what the model *does* with an injection (read-only).

The recovery evaluator (spec step 4) collapses a run to a single
recovered/not flag. This analysis goes one level deeper -- the project's
core interest is the model's *behaviour* when off-topic text is injected
after collapse -- using only the three metrics the spec already defines
(NOT an LLM-judge):

  * **version-(b) cosine distance** of each post-injection turn from the
    collapsed window (already logged in ``_meta.json`` as
    ``recovery.post_injection_distances``): did the loop move away from the
    attractor, and did it stay away?
  * **Jaccard distance to the injected text** (recomputed per round with the
    exact function the recovery evaluator uses): is the model just parroting
    the injection, or generating genuinely new content?
  * **GPT-2 perplexity** (per-round ``token_perplexity`` from the run CSV):
    the gibberish guard.

The trajectory of those three across the post-injection turns *is* the
behaviour story. Each run is labelled with the decision tree below. The
spec's recovered flag (a sustained ``hold_k``-round streak meeting all
recovery criteria) is the authoritative "clean escape" signal, so it is
honoured first; the remaining labels describe what a *non*-recovered run did
instead, using the recovery evaluator's own thresholds (so the labels stay
consistent with the flag):

  escape    -- the run *recovered* (recovered flag True): it moved away from
               the attractor, stayed novel + sane, and held for ``hold_k``
               rounds. A clean recovery.
  parrot    -- did not recover; on average it echoed the injection (mean
               Jaccard distance to the injection < ``min_jaccard_to_injection``).
  gibberish -- did not recover; mean perplexity > ``max_perplexity``.
  blend     -- did not recover, but it *reacted*: it moved away from the
               attractor for most post rounds (>= half have version-(b)
               distance >= ``distance_cutoff``) without sustaining it.
  ignore    -- did not recover and barely left the attractor (most post
               rounds below ``distance_cutoff``): it kept collapsing.

This is strictly read-only: it reads each run's ``*.csv`` + ``*_meta.json``
under a results directory, writes tidy tables (and an optional figure) to an
output directory, and never re-runs an experiment.

Usage::

    poetry run python analysis/injection_behavior.py
    poetry run python analysis/injection_behavior.py \\
        --runs-dir results/grid --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "src")

from babel_ai.recovery import (  # noqa: E402
    RecoveryConfig,
    _jaccard_distance,
)

SIZE_ORDER = ["word", "sentence", "paragraph"]
SOURCE_ORDER = ["real", "noise"]
LABEL_ORDER = ["ignore", "parrot", "gibberish", "blend", "escape"]


def _parse_analysis(raw: str) -> dict:
    """Parse the stringified ``AnalysisResult`` dict from a CSV cell.

    ``ast.literal_eval`` is used first; it fails on bare ``nan``/``inf``
    (which a perplexity field can contain), so we fall back to a sandboxed
    ``eval`` that only exposes those float constants.
    """
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return eval(  # noqa: S307 - trusted, our own output
            raw,
            {"__builtins__": {}},
            {"nan": float("nan"), "inf": float("inf")},
        )


@dataclass
class RunBehavior:
    """Per-run behaviour summary across the post-injection window."""

    run: str
    temp: float
    size: str
    source: str
    collapse_onset: Optional[int]
    injection_round: int
    injection_distance: Optional[float]
    n_post: int
    mean_db: float  # mean version-(b) distance from collapsed window
    frac_moved: float  # fraction of post rounds with db >= cutoff
    db_slope: float  # trend of db over post rounds (away vs back)
    mean_jaccard_inj: float  # mean Jaccard distance to injection text
    mean_perplexity: float
    recovered: bool
    label: str


def _read_run(run_dir: str) -> Optional[Tuple[dict, List[dict]]]:
    """Return ``(meta, agent_turns)`` for a run dir, or ``None`` to skip.

    ``agent_turns`` is the ordered list of per-turn dicts (each with parsed
    ``analysis``) for the self-loop turns only; its index is the agent round.
    Runs without an applied injection are skipped (nothing to characterise).
    """
    metas = glob.glob(os.path.join(run_dir, "*_meta.json"))
    csvs = [c for c in glob.glob(os.path.join(run_dir, "*.csv"))]
    if not metas or not csvs:
        return None
    with open(metas[0]) as f:
        meta = json.load(f)
    if not meta.get("injection"):
        return None  # no injection (floor / never-collapsed) -> skip

    agent_turns: List[dict] = []
    with open(csvs[0], newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("agent_id"):
                continue  # fetcher seed message, not a self-loop turn
            analysis = row.get("analysis")
            row["_analysis"] = _parse_analysis(analysis) if analysis else {}
            agent_turns.append(row)
    return meta, agent_turns


def _agent_temp(meta: dict) -> float:
    cfg = meta.get("config", {})
    agents = cfg.get("agent_configs") or [{}]
    return float(agents[0].get("temperature", float("nan")))


def _slope(values: List[float]) -> float:
    """Least-squares slope of ``values`` vs their index (0 if < 2 points)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def _classify(
    frac_moved: float,
    mean_jaccard_inj: float,
    mean_perplexity: float,
    recovered: bool,
    cfg: RecoveryConfig,
) -> str:
    """Label a run from its post-injection metric summary (see module doc).

    The recovered flag is honoured first (authoritative sustained-escape
    signal); the rest describe what a non-recovered run did instead.
    """
    if recovered:
        return "escape"
    if mean_jaccard_inj < cfg.min_jaccard_to_injection:
        return "parrot"
    if mean_perplexity > cfg.max_perplexity:
        return "gibberish"
    if frac_moved >= 0.5:
        return "blend"
    return "ignore"


def analyze_run(
    run_dir: str, cfg: RecoveryConfig
) -> Optional[Tuple[RunBehavior, List[dict]]]:
    """Build a :class:`RunBehavior` plus a per-round record list for one run."""
    loaded = _read_run(run_dir)
    if loaded is None:
        return None
    meta, agent_turns = loaded

    inj = meta["injection"]
    inj_round = int(inj["round"])
    inj_text = inj.get("text", "")
    rec = meta.get("recovery") or {}

    post = agent_turns[inj_round + 1 :]
    # Prefer the first-class per-round CSV column (future_work §C) when every
    # post turn carries it; fall back to the recovery meta's list for older
    # runs that predate the column.
    csv_db = [t["_analysis"].get("version_b_distance") for t in post]
    if post and all(d is not None for d in csv_db):
        distances = [float(d) for d in csv_db]
    else:
        meta_db = rec.get("post_injection_distances") or []
        n = min(len(post), len(meta_db))
        post = post[:n]
        distances = [float(x) for x in meta_db[:n]]
    n = len(post)
    if n == 0:
        return None

    per_round: List[dict] = []
    dbs: List[float] = []
    jaccards: List[float] = []
    ppls: List[float] = []
    for i in range(n):
        turn = post[i]
        db = float(distances[i])
        content = turn.get("content", "") or ""
        jac = _jaccard_distance(content, inj_text) if inj_text else 1.0
        ppl = turn["_analysis"].get("token_perplexity")
        ppl = float(ppl) if ppl is not None else float("nan")
        dbs.append(db)
        jaccards.append(jac)
        if ppl == ppl:  # not NaN
            ppls.append(ppl)
        per_round.append(
            {
                "run": os.path.basename(run_dir),
                "temp": _agent_temp(meta),
                "size": inj["size"],
                "source": inj["source"],
                "post_round": i,  # rounds since injection (1 = first after)
                "agent_round": inj_round + 1 + i,
                "version_b_distance": db,
                "jaccard_to_injection": jac,
                "perplexity": ppl,
            }
        )

    frac_moved = sum(1 for d in dbs if d >= cfg.distance_cutoff) / len(dbs)
    mean_db = sum(dbs) / len(dbs)
    mean_jac = sum(jaccards) / len(jaccards)
    mean_ppl = sum(ppls) / len(ppls) if ppls else float("nan")
    recovered = bool(rec.get("recovered"))
    label = _classify(frac_moved, mean_jac, mean_ppl, recovered, cfg)

    behavior = RunBehavior(
        run=os.path.basename(run_dir),
        temp=_agent_temp(meta),
        size=inj["size"],
        source=inj["source"],
        collapse_onset=meta.get("collapse_onset_round"),
        injection_round=inj_round,
        injection_distance=inj.get("distance"),
        n_post=n,
        mean_db=mean_db,
        frac_moved=frac_moved,
        db_slope=_slope(dbs),
        mean_jaccard_inj=mean_jac,
        mean_perplexity=mean_ppl,
        recovered=recovered,
        label=label,
    )
    return behavior, per_round


def _fnum(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "  -  "
    return f"{x:.{nd}f}"


def print_summary(behaviors: List[RunBehavior]) -> None:
    """Print the per-run table and the behaviour-mix table by size x source."""
    temps = sorted({b.temp for b in behaviors})
    for temp in temps:
        runs = [b for b in behaviors if b.temp == temp]
        print(f"\n############### TEMPERATURE {temp} ###############")

        print("\n--- Per-run behaviour (post-injection window) ---")
        head = (
            f"  {'size':<9} {'source':<6} {'label':<9} "
            f"{'mean_db':>8} {'frac>cut':>8} {'db_slope':>9} "
            f"{'jac_inj':>8} {'ppl':>7} {'recov':>6}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        order = {s: i for i, s in enumerate(SIZE_ORDER)}
        for b in sorted(runs, key=lambda r: (order.get(r.size, 9), r.source)):
            print(
                f"  {b.size:<9} {b.source:<6} {b.label:<9} "
                f"{_fnum(b.mean_db):>8} {_fnum(b.frac_moved,2):>8} "
                f"{_fnum(b.db_slope,4):>9} {_fnum(b.mean_jaccard_inj):>8} "
                f"{_fnum(b.mean_perplexity,1):>7} "
                f"{('yes' if b.recovered else 'no'):>6}"
            )

        print("\n--- Behaviour mix by size x source (labels) ---")
        hdr = f"  {'size':<10} | " + " | ".join(
            f"{s:^22}" for s in SOURCE_ORDER
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for size in SIZE_ORDER:
            cells = []
            for source in SOURCE_ORDER:
                cell = [
                    b for b in runs if b.size == size and b.source == source
                ]
                if not cell:
                    cells.append("  -  ")
                    continue
                counts: Dict[str, int] = {}
                for b in cell:
                    counts[b.label] = counts.get(b.label, 0) + 1
                cells.append(
                    ",".join(
                        f"{lab}:{counts[lab]}"
                        for lab in LABEL_ORDER
                        if lab in counts
                    )
                )
            print(f"  {size:<10} | " + " | ".join(f"{c:^22}" for c in cells))


def write_tables(
    behaviors: List[RunBehavior],
    per_round: List[dict],
    out_dir: str,
) -> Tuple[str, str]:
    """Write the per-run summary and tidy per-round CSVs; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "injection_behavior_summary.csv")
    rounds_path = os.path.join(out_dir, "injection_behavior_rounds.csv")

    summary_fields = [
        "run",
        "temp",
        "size",
        "source",
        "collapse_onset",
        "injection_round",
        "injection_distance",
        "n_post",
        "mean_db",
        "frac_moved",
        "db_slope",
        "mean_jaccard_inj",
        "mean_perplexity",
        "recovered",
        "label",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for b in behaviors:
            w.writerow({k: getattr(b, k) for k in summary_fields})

    if per_round:
        with open(rounds_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_round[0].keys()))
            w.writeheader()
            w.writerows(per_round)
    return summary_path, rounds_path


def plot_trajectories(
    per_round: List[dict],
    behaviors: List[RunBehavior],
    out_dir: str,
    cfg: RecoveryConfig,
) -> List[str]:
    """One figure per temperature: a size x source grid of the three-metric
    trajectories (mean across seeds), version-(b) distance + Jaccard-to-
    injection on the left axis and perplexity on the right. Best-effort."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped (matplotlib unavailable): {e}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    temps = sorted({r["temp"] for r in per_round})
    paths: List[str] = []
    for temp in temps:
        fig, axes = plt.subplots(
            len(SIZE_ORDER),
            len(SOURCE_ORDER),
            figsize=(11, 9),
            sharex=True,
        )
        fig.suptitle(
            f"Post-injection behaviour (temp {temp}) -- "
            "version-(b) distance & Jaccard-to-injection (left), "
            "perplexity (right)"
        )
        for si, size in enumerate(SIZE_ORDER):
            for qi, source in enumerate(SOURCE_ORDER):
                ax = axes[si][qi]
                rows = [
                    r
                    for r in per_round
                    if r["temp"] == temp
                    and r["size"] == size
                    and r["source"] == source
                ]
                labs = sorted(
                    {
                        b.label
                        for b in behaviors
                        if b.temp == temp
                        and b.size == size
                        and b.source == source
                    }
                )
                title = f"{size} / {source}"
                if labs:
                    title += f"  [{','.join(labs)}]"
                ax.set_title(title, fontsize=9)
                if not rows:
                    ax.text(
                        0.5,
                        0.5,
                        "no run",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="grey",
                    )
                    continue
                by_round: Dict[int, List[dict]] = {}
                for r in rows:
                    by_round.setdefault(r["post_round"], []).append(r)
                xs = sorted(by_round)

                def _mean(key: str, rr: List[dict]) -> float:
                    vals = [v[key] for v in rr if v[key] == v[key]]  # drop NaN
                    return sum(vals) / len(vals) if vals else float("nan")

                db = [_mean("version_b_distance", by_round[x]) for x in xs]
                jac = [_mean("jaccard_to_injection", by_round[x]) for x in xs]
                ppl = [_mean("perplexity", by_round[x]) for x in xs]

                ax.plot(xs, db, "-o", ms=3, label="dist (b)", color="C0")
                ax.plot(xs, jac, "-s", ms=3, label="jac->inj", color="C2")
                ax.axhline(
                    cfg.distance_cutoff,
                    ls="--",
                    lw=0.8,
                    color="C0",
                    alpha=0.5,
                )
                ax.axhline(
                    cfg.min_jaccard_to_injection,
                    ls=":",
                    lw=0.8,
                    color="C2",
                    alpha=0.5,
                )
                ax.set_ylim(0, 1.05)
                axr = ax.twinx()
                axr.plot(xs, ppl, "-^", ms=3, label="ppl", color="C3")
                axr.axhline(
                    cfg.max_perplexity,
                    ls="--",
                    lw=0.8,
                    color="C3",
                    alpha=0.4,
                )
                axr.set_ylabel("ppl", fontsize=8, color="C3")
                if si == len(SIZE_ORDER) - 1:
                    ax.set_xlabel("rounds since injection")

        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], color="C0", marker="o", label="version-(b) dist"),
            Line2D([], [], color="C2", marker="s", label="Jaccard->inj"),
            Line2D([], [], color="C3", marker="^", label="perplexity (R)"),
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            ncol=3,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(out_dir, f"injection_behavior_t{temp}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-dir",
        default="results/grid",
        help="directory holding per-run folders (each with csv + meta.json)",
    )
    ap.add_argument(
        "--out-dir",
        default="results/analysis",
        help="where to write the summary/rounds CSVs and the figures",
    )
    ap.add_argument(
        "--include-temp0",
        action="store_true",
        help="include temp-0 runs (excluded by default: degenerate)",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = RecoveryConfig()
    run_dirs = sorted(
        d
        for d in glob.glob(os.path.join(args.runs_dir, "*"))
        if os.path.isdir(d)
    )

    behaviors: List[RunBehavior] = []
    per_round: List[dict] = []
    for run_dir in run_dirs:
        result = analyze_run(run_dir, cfg)
        if result is None:
            continue
        behavior, rounds = result
        if not args.include_temp0 and behavior.temp == 0.0:
            continue
        behaviors.append(behavior)
        per_round.extend(rounds)

    if not behaviors:
        print(f"No injection runs found under {args.runs_dir}")
        return

    print_summary(behaviors)
    summary_path, rounds_path = write_tables(
        behaviors, per_round, args.out_dir
    )
    print(f"\nWrote {summary_path}")
    print(f"Wrote {rounds_path}")
    if not args.no_plot:
        for p in plot_trajectories(per_round, behaviors, args.out_dir, cfg):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
