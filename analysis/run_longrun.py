"""One long, free-generation run with REPEATED injections (pilot).

Explores three things the v1 grid could not, on a single long conversation:

  * **Long horizon** — many more rounds than the 60-round grid, so collapse has
    room to form *and* we can watch the response to each injection play out.
  * **Free generation** — no brevity system prompt and a large token cap, so
    each turn is the model's natural-length utterance (closer to Maiti/Multi_LLM,
    which use no instructions).
  * **Repeated dosing** — two cadences, to compare:
      - ``recollapse``: re-inject every time the loop *re-collapses* (dose on
        demand) — ``after_collapse`` + ``repeat``;
      - ``interval``: inject every N rounds regardless (dose on a clock) —
        ``fixed_interval``.
    Both inject the **most off-topic** real snippet (farthest of N candidates).

NOTE on length: the self-loop feeds the model the *full* history each turn, so
cost/context grow with round count. 150 rounds is a safe ceiling on full
history; truly long runs (300–1000, like Multi_LLM) want a windowed input first
(a separate decision — see methodology_notes / future_work).

Usage::

    poetry run python analysis/run_longrun.py                 # both cadences
    poetry run python analysis/run_longrun.py --rounds 150 --temp 0.7
    poetry run python analysis/run_longrun.py --mode recollapse  # just one
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, "src")

from api.enums import OpenAIModels, Provider  # noqa: E402
from babel_ai.enums import (  # noqa: E402
    AgentSelectionMethod,
    AnalyzerType,
    FetcherType,
    InjectionSize,
    InjectionSource,
    InjectionTrigger,
)
from babel_ai.experiment import Experiment  # noqa: E402
from models import (  # noqa: E402
    AgentConfig,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
    InjectionConfig,
)

OUT_DIR = "results/longrun"


def _injection(mode: str, interval: int, seed: int) -> InjectionConfig:
    """Sentence-sized, far-off-topic real injection, dosed by `mode`."""
    common = dict(
        size=InjectionSize.SENTENCE,
        source=InjectionSource.REAL,
        num_candidates=8,  # pick the most off-topic of 8 samples
        rng_seed=seed,
    )
    if mode == "recollapse":
        return InjectionConfig(
            trigger=InjectionTrigger.AFTER_COLLAPSE, repeat=True, **common
        )
    if mode == "interval":
        return InjectionConfig(
            trigger=InjectionTrigger.FIXED_INTERVAL,
            interval=interval,
            **common,
        )
    raise ValueError(f"unknown mode: {mode}")


def _config(
    injection: InjectionConfig, rounds: int, temp: float, seed: int
) -> ExperimentConfig:
    return ExperimentConfig(
        fetcher_config=FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="data/sharegpt_real.json",
            min_messages=2,
            max_messages=6,
        ),
        analyzer_config=AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=10
        ),
        agent_configs=[
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4O_MINI,
                system_prompt=None,  # free generation (no brevity prompt)
                temperature=temp,
                # Large cap so turns finish on their own (free generation).
                # 512 was too low -- gpt-4o-mini's natural unprompted reply is
                # ~540+ tokens, so it cut nearly every turn mid-sentence and
                # (with last-message feeding) the next turn continued the
                # fragment. 2048 lets replies end naturally.
                max_tokens=2048,
                seed=seed,
            )
        ],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=rounds,
        max_total_characters=10**7,  # don't stop the long run early
        history_window=1,  # feed only the last message (Maiti-style self-loop)
        output_dir=OUT_DIR,
        injection_config=injection,
    )


def _run_one(
    mode: str, rounds: int, temp: float, seed: int, interval: int
) -> str:
    import random

    random.seed(seed)  # fix the fetched seed conversation
    inj = _injection(mode, interval=interval, seed=seed)
    exp = Experiment(_config(inj, rounds, temp, seed))
    exp.run()
    m = exp.metadata
    n_inj = len(m.injections)
    print(
        f"[longrun:{mode}] done — collapse_onset={m.collapse_onset_round}, "
        f"injections={n_inj} at rounds "
        f"{[e['round'] for e in m.injections]}"
    )
    # find the run dir we just wrote (most recent under OUT_DIR)
    dirs = sorted(glob.glob(os.path.join(OUT_DIR, "drift_experiment_*")))
    return dirs[-1] if dirs else ""


def _parse_analysis(raw: str) -> dict:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return eval(  # noqa: S307 - our own output
            raw,
            {"__builtins__": {}},
            {"nan": float("nan"), "inf": float("inf")},
        )


def _word_set(text: str) -> set:
    import re

    return set(re.findall(r"\w+", (text or "").lower()))


def _jaccard_distance(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 1.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def plot_run(run_dir: str, mode: str) -> Optional[str]:
    """Plot the whole-run trajectory: version-(b) distance + Jaccard-to-latest-
    injection (left), perplexity (right), with a vertical line at each
    injection. Reads the per-round ``version_b_distance`` CSV column directly.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}")
        return None

    csv = glob.glob(os.path.join(run_dir, "*.csv"))[0]
    meta = json.load(open(glob.glob(os.path.join(run_dir, "*_meta.json"))[0]))
    df = pd.read_csv(csv)
    agents = df[df["agent_id"].notna()].reset_index(drop=True)

    injections = meta.get("injections") or []
    inj_rounds = [e["round"] for e in injections]

    rounds, db, jac, ppl, trunc = [], [], [], [], 0
    for i, row in agents.iterrows():
        a = _parse_analysis(row["analysis"]) if row.get("analysis") else {}
        rounds.append(i)
        db.append(a.get("version_b_distance"))
        ppl.append(a.get("token_perplexity"))
        # Jaccard distance to the most recent injection text before this round
        prior = [e for e in injections if e["round"] < i]
        jac.append(
            _jaccard_distance(row.get("content", ""), prior[-1]["text"])
            if prior
            else None
        )
        content = str(row.get("content", "")).rstrip()
        if content and content[-1] not in ".!?\"')":
            trunc += 1

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(rounds, db, "-o", ms=2, color="C0", label="version-(b) distance")
    ax.plot(rounds, jac, "-s", ms=2, color="C2", label="Jaccard->injection")
    ax.axhline(0.40, ls="--", lw=0.8, color="C0", alpha=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("agent round")
    ax.set_ylabel("distance (0-1)")
    axr = ax.twinx()
    axr.plot(
        rounds, ppl, "-^", ms=2, color="C3", alpha=0.6, label="perplexity"
    )
    axr.set_ylabel("perplexity", color="C3")
    for r in inj_rounds:
        ax.axvline(r, ls=":", lw=1.0, color="grey")
    onset = meta.get("collapse_onset_round")
    if onset is not None:
        ax.axvline(onset, ls="-", lw=1.2, color="orange", alpha=0.7)
    ax.set_title(
        f"Long run [{mode}] — {len(inj_rounds)} injections (dotted), "
        f"collapse onset (orange), truncated turns: "
        f"{trunc}/{len(rounds)}"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"trajectory_{mode}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument(
        "--mode",
        choices=["recollapse", "interval", "both"],
        default="both",
    )
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    modes = ["recollapse", "interval"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_dir = _run_one(
            mode, args.rounds, args.temp, args.seed, args.interval
        )
        if run_dir:
            out = plot_run(run_dir, mode)
            if out:
                print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
