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

from api.enums import (  # noqa: E402
    OllamaCompanyModels,
    OpenAIModels,
    Provider,
)
from babel_ai.enums import (  # noqa: E402
    AgentSelectionMethod,
    AnalyzerType,
    FetcherType,
    InjectionSize,
    InjectionSource,
    InjectionTrigger,
)
from babel_ai.experiment import Experiment  # noqa: E402
from llm_judge import judge_recovery  # noqa: E402
from models import (  # noqa: E402
    AgentConfig,
    AgentMetric,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
    InjectionConfig,
)

SIZES = [InjectionSize.WORD, InjectionSize.SENTENCE, InjectionSize.PARAGRAPH]
SOURCES = [InjectionSource.REAL, InjectionSource.NOISE]

# Selectable loop models (same aliases as run_longrun.py). Default keeps the
# original gpt-4o-mini behaviour so existing invocations are unchanged; the
# company-served open models each resolve their own <NAME>_BASE_URL/_API_KEY.
MODELS = {
    "gpt-4o-mini": (Provider.OPENAI, OpenAIModels.GPT4O_MINI),
    "qwen-2.5-72b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.QWEN_2_5_72B),
    "qwen-2.5-7b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.QWEN_2_5_7B),
    "llama-3.3-70b": (
        Provider.OLLAMA_COMPANY,
        OllamaCompanyModels.LLAMA_3_3_70B,
    ),
}

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
    "recovered_confirmed",  # recovered AND the judge agrees it's varied
    "judge_score",
    "recovery_round",
    "hold_length",
]


def _make_config(
    injection: Optional[InjectionConfig],
    max_iters: int,
    temp: float = 0.2,
    model_seed: Optional[int] = None,
    model_key: str = "gpt-4o-mini",
) -> ExperimentConfig:
    provider, model = MODELS[model_key]
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
                provider=provider,
                model=model,
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
    model_key: str = "gpt-4o-mini",
    stochastic: bool = False,
) -> Dict[str, object]:
    random.seed(seed)  # fix the fetched seed conversation across conditions
    # model_seed: None => stochastic sampling (required for served vLLM, which
    # is frozen/deterministic when seeded -- a seed makes it ignore injections
    # and emit byte-identical turns). Seeded only for exact reproducibility.
    model_seed = None if stochastic else seed
    exp = Experiment(
        _make_config(
            injection, max_iters, temp, model_seed=model_seed,
            model_key=model_key,
        )
    )
    exp.run()
    m = exp.metadata
    rec = m.recovery or {}
    inj = m.injection or {}
    # Confirm any flagged recovery with the discourse judge (one call per
    # candidate recovery; no-op when nothing recovered). This catches the
    # cheerleader/confused ruts the cheap criteria can't see.
    agent_contents = [
        mm.content
        for mm in exp.result_metrics
        if isinstance(mm, AgentMetric)
    ]
    judged = judge_recovery(agent_contents, rec)
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
        "recovered_confirmed": judged["recovered_confirmed"],
        "judge_score": judged["judge_score"],
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
        "--model",
        default="gpt-4o-mini",
        choices=sorted(MODELS),
        help="loop model alias (company-served models need their "
        "<NAME>_BASE_URL/_API_KEY in .env)",
    )
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
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="no sampling seed (varied output each run). Required for the "
        "company vLLM, which is frozen/deterministic when seeded.",
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
                    model_key=args.model,
                    stochastic=args.stochastic,
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
                            model_key=args.model,
                            stochastic=args.stochastic,
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
                                model_key=args.model,
                                stochastic=args.stochastic,
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
