"""Run the v1 experiment grid (spec step 6).

Sweeps the two independent variables -- injection **size** (word/sentence/
paragraph) x **source** (real/noise) -- across several seeds, plus the
no-intervention floor (and optionally the fresh-run injection baseline) for
each seed. Each run's collapse/injection/recovery metadata is collected into
one tidy CSV (``results/grid/grid_results.csv``) that ``aggregate_grid.py``
turns into the headline "smallest size that recovers, per source" table.

Seed pairing: ``random.seed(s)`` is set before each run so every condition for
seed ``s`` starts from the *same* fetched conversation; the injection sampler
uses ``rng_seed=s`` so real vs noise at a given size share the same snippet.

Usage::

    poetry run python analysis/run_grid.py --seeds 3
    poetry run python analysis/run_grid.py --seeds 3 --fresh   # + baseline
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from typing import Dict, List, Optional

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

SIZES = [InjectionSize.WORD, InjectionSize.SENTENCE, InjectionSize.PARAGRAPH]
SOURCES = [InjectionSource.REAL, InjectionSource.NOISE]

FIELDS = [
    "seed",
    "temp",
    "condition",
    "size",
    "source",
    "collapse_onset",
    "injection_round",
    "injection_distance",
    "recovered",
    "recovery_round",
    "hold_length",
]


def _make_config(
    injection: Optional[InjectionConfig],
    max_iters: int,
    temp: float = 0.2,
    model_seed: Optional[int] = None,
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
                system_prompt=(
                    "You are a helpful assistant in a conversation. Reply in "
                    "at most 3 sentences, and always finish your final "
                    "sentence."
                ),
                temperature=temp,
                max_tokens=150,
                seed=model_seed,
            )
        ],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=max_iters,
        max_total_characters=10**6,
        output_dir="results/grid",
        injection_config=injection,
    )


def _run_one(
    seed: int,
    condition: str,
    size: Optional[InjectionSize],
    source: Optional[InjectionSource],
    injection: Optional[InjectionConfig],
    max_iters: int,
    temp: float,
) -> Dict[str, object]:
    random.seed(seed)  # fix the fetched seed conversation for pairing
    # also seed model sampling so real vs noise share the same trajectory
    exp = Experiment(_make_config(injection, max_iters, temp, model_seed=seed))
    exp.run()
    m = exp.metadata
    rec = m.recovery or {}
    inj = m.injection or {}
    return {
        "seed": seed,
        "temp": temp,
        "condition": condition,
        "size": size.value if size else "",
        "source": source.value if source else "",
        "collapse_onset": m.collapse_onset_round,
        "injection_round": inj.get("round"),
        "injection_distance": inj.get("distance"),
        "recovered": rec.get("recovered"),
        "recovery_round": rec.get("recovery_round"),
        "hold_length": rec.get("hold_length"),
    }


def _write(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument(
        "--temp",
        type=float,
        default=0.2,
        help="single sampling temperature (ignored if --temps is given)",
    )
    ap.add_argument(
        "--temps",
        type=str,
        default=None,
        help=(
            "comma-separated temperatures to sweep, e.g. 0.0,0.5,1.0. "
            "temp 0 (greedy) is the cleanest collapse; higher temps test "
            "whether stochasticity prevents it."
        ),
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="also run the fresh-run injection baseline per condition",
    )
    ap.add_argument("--out", default="results/grid/grid_results.csv")
    args = ap.parse_args()

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    temps = (
        [float(t) for t in args.temps.split(",")]
        if args.temps
        else [args.temp]
    )

    rows: List[Dict[str, object]] = []
    for temp in temps:
        for seed in range(args.seeds):
            # no-intervention floor
            rows.append(
                _run_one(
                    seed,
                    "no_intervention",
                    None,
                    None,
                    None,
                    args.max_iters,
                    temp,
                )
            )
            _write(rows, args.out)
            for size in SIZES:
                for source in SOURCES:
                    inj = InjectionConfig(
                        size=size,
                        source=source,
                        trigger=InjectionTrigger.AFTER_COLLAPSE,
                        rng_seed=seed,
                    )
                    rows.append(
                        _run_one(
                            seed,
                            "intervention",
                            size,
                            source,
                            inj,
                            args.max_iters,
                            temp,
                        )
                    )
                    _write(rows, args.out)
                    if args.fresh:
                        finj = InjectionConfig(
                            size=size,
                            source=source,
                            trigger=InjectionTrigger.FIXED_ROUND,
                            fixed_round=5,
                            rng_seed=seed,
                        )
                        rows.append(
                            _run_one(
                                seed,
                                "fresh",
                                size,
                                source,
                                finj,
                                args.max_iters,
                                temp,
                            )
                        )
                        _write(rows, args.out)
            print(
                f"[grid] temp {temp} seed {seed} done "
                f"({len(rows)} runs so far)"
            )

    print(f"[grid] wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
