"""Run the injection-study grid (free-generation self-loop).

Sweeps the axes that matter for "does injection ever help?":

  * **model**        -- Qwen-7B / Qwen-72B / Llama-3.3-70B / Llama-3.2-3B
  * **temperature**  -- e.g. 0.7, 1.0
  * **seed**         -- different starting conversations
  * **prompt-type**  -- none + 4 system prompts (see PROMPTS)
  * **injection size** -- paragraph / output-sized / skim-half (real snippets)
  * **timing**       -- after-collapse (rescue) / early@5 (prevent) /
                        every-5-rounds (steady dosing)
  * plus a no-injection **floor** per cell.

Injections are drawn from a deliberately **cross-domain** corpus
(``data/injection_diverse.json``) and the *farthest-of-N* candidate is chosen,
so an injection is always a genuinely different topic from the conversation
(the old grid injected tech-into-tech, which slid straight off).

Each run's collapse/injection/recovery metadata -- including the
**judge-confirmed** recovery flag -- lands in one CSV, and every run folder gets
its per-metric PNGs (same as the long-run maker).

Usage::

    poetry run python analysis/run_grid.py --models qwen-2.5-7b --stochastic \\
        --temps 0.7,1.0 --seeds 3 --max-iters 120 --out results/grid/stage1.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from babel_ai.recovery import RecoveryConfig  # noqa: E402
from llm_judge import judge_recovery  # noqa: E402
from run_longrun import plot_run  # noqa: E402  (reuse the graph maker)
from models import (  # noqa: E402
    AgentConfig,
    AgentMetric,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
    InjectionConfig,
)

# Injection *sizes* -- the same three "clean injection" conditions as the
# long-run study (real off-topic snippets only; no word/sentence, no noise).
SIZES = [
    InjectionSize.PARAGRAPH,
    InjectionSize.OUTPUT_SIZED,
    InjectionSize.SKIM_HALF,
]

# Cross-domain injection corpus so every injection is a genuinely new topic.
DIVERSE_CORPUS = "data/injection_diverse.json"
NUM_CANDIDATES = 8  # pick the farthest-off-topic of N sampled snippets


def _git_hash() -> str:
    """Short commit the study ran on -- stamped on every row so a resumed
    summary can never silently mix rows from different code versions."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


_GIT_HASH = _git_hash()

# Serializes the (fast) seed-conversation fetch so concurrent worker threads
# don't clobber each other's global random state during Experiment construction.
_FETCH_LOCK = threading.Lock()

# Injection *timings* -> (trigger, extra InjectionConfig kwargs).
#   after_collapse = inject once when it gets stuck   -> "can we rescue it?"
#   early_5        = inject once at round 5 (pre-collapse) -> "can we prevent it?"
#   every_5        = inject every 5 rounds (steady dosing) -> "does a drip keep
#                    it diverse, or just walk it between stuck-topics?"
# Treatment timings -- injected into the (collapsing) self-loop.
TIMINGS: Dict[str, tuple] = {
    "after_collapse": (InjectionTrigger.AFTER_COLLAPSE, {}),
    "early_5": (InjectionTrigger.FIXED_ROUND, {"fixed_round": 5}),
    "every_5": (InjectionTrigger.FIXED_INTERVAL, {"interval": 5}),
}

# Fresh-run baseline (control): inject the same snippet at round 2, into a loop
# that has NOT collapsed yet, so we can compare "how far a big real injection
# moves a fresh loop" against "how far it moves a collapsed one" -- i.e. measure
# the pull (basin) of the attractor, not just the raw displacement. Its own
# condition ("fresh"), not a treatment timing.
FRESH_TIMING = "fresh"
_ALL_TIMINGS: Dict[str, tuple] = {
    **TIMINGS,
    FRESH_TIMING: (InjectionTrigger.FIXED_ROUND, {"fixed_round": 2}),
}

# 5 prompt-types: "none" (pure free generation) + 4 system-prompt strategies.
PROMPTS: Dict[str, Optional[str]] = {
    "none": None,
    "no_repeat": (
        "You are in an open-ended conversation. Each reply must introduce a "
        "genuinely new idea, fact, or question -- never repeat points already "
        "made or keep circling the same topic."
    ),
    "follow_new": (
        "Pay close attention to any new or surprising information in the "
        "latest message and follow where it leads, instead of continuing your "
        "previous train of thought."
    ),
    "curious": (
        "You are an intensely curious conversationalist who loves tangents. "
        "Chase whatever is most interesting or unexpected in the last message, "
        "even if it means changing the subject."
    ),
    "disagree": (
        "Be critical. Each reply should question, challenge, or disagree with "
        "something in the previous message -- never just agree and elaborate."
    ),
}

# Selectable loop models (same aliases as run_longrun.py). Company-served open
# models each resolve their own <NAME>_BASE_URL/_API_KEY.
MODELS = {
    "gpt-4o-mini": (Provider.OPENAI, OpenAIModels.GPT4O_MINI),
    "qwen-2.5-72b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.QWEN_2_5_72B),
    "qwen-2.5-7b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.QWEN_2_5_7B),
    "llama-3.3-70b": (
        Provider.OLLAMA_COMPANY,
        OllamaCompanyModels.LLAMA_3_3_70B,
    ),
    "llama-3.2-3b": (
        Provider.OLLAMA_COMPANY,
        OllamaCompanyModels.LLAMA_3_2_3B,
    ),
}

FIELDS = [
    "model",
    "temp",
    "seed",
    "prompt",
    "condition",  # floor | inject | fresh
    "size",
    "timing",
    "collapse_onset",
    "n_injections",
    "injection_round",
    "injection_distance",
    "recovered",
    "recovered_confirmed",  # recovered AND the judge agrees it's varied
    "judge_score",
    "recovery_round",
    "hold_length",
    # dose-response of the transient (the reframed primary endpoint): how far
    # the injection knocked the loop off the attractor, and for how long.
    "peak_displacement",  # max version-(b) distance from the collapsed window
    "displaced_rounds",   # # post-anchor rounds held above the distance cutoff
    "git_hash",           # code version this row was produced on (#9)
]


def _make_config(
    injection: Optional[InjectionConfig],
    max_iters: int,
    temp: float,
    model_key: str,
    system_prompt: Optional[str],
    model_seed: Optional[int],
    output_dir: str,
) -> ExperimentConfig:
    """Free-generation self-loop config: last-message feeding, large token cap,
    the given system prompt (or none)."""
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
                system_prompt=system_prompt,  # None = pure free generation
                temperature=temp,
                max_tokens=2048,  # large cap so turns finish on their own
                seed=model_seed,
            )
        ],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=max_iters,
        max_total_characters=10**7,
        history_window=1,  # feed only the last message (Maiti-style self-loop)
        output_dir=output_dir,
        injection_config=injection,
    )


def _make_injection(
    size: InjectionSize, timing: str, seed: int
) -> InjectionConfig:
    trigger, extra = _ALL_TIMINGS[timing]
    return InjectionConfig(
        size=size,
        source=InjectionSource.REAL,
        trigger=trigger,
        corpus_path=DIVERSE_CORPUS,
        num_candidates=NUM_CANDIDATES,
        rng_seed=seed,
        **extra,
    )


def _run_one(
    model_key: str,
    temp: float,
    seed: int,
    prompt_name: str,
    condition: str,
    size: Optional[InjectionSize],
    timing: Optional[str],
    injection: Optional[InjectionConfig],
    max_iters: int,
    stochastic: bool,
    make_plots: bool,
    study_root: str,
) -> Dict[str, object]:
    # model_seed None => stochastic sampling. A served vLLM is deterministic
    # when seeded, which freezes it into ignoring injections; seed only for
    # exact reproduction.
    model_seed = None if stochastic else seed
    # Each run lands under study/<model>/<prompt>/ so 900 runs stay browsable.
    output_dir = os.path.join(study_root, model_key, prompt_name)
    os.makedirs(output_dir, exist_ok=True)
    # The seed conversation is fetched in Experiment.__init__ using the global
    # random state, so serialize construction: with threads that keeps each
    # seed's starting conversation deterministic. The slow part (run) is
    # outside the lock and runs concurrently.
    with _FETCH_LOCK:
        random.seed(seed)
        exp = Experiment(
            _make_config(
                injection,
                max_iters,
                temp,
                model_key,
                PROMPTS[prompt_name],
                model_seed,
                output_dir,
            )
        )
    exp.run()
    m = exp.metadata
    rec = m.recovery or {}
    inj = m.injection or {}
    agent_contents = [
        mm.content for mm in exp.result_metrics if isinstance(mm, AgentMetric)
    ]
    # Confirm any flagged recovery with the discourse judge (one call per
    # candidate recovery; no-op when nothing recovered).
    judged = judge_recovery(agent_contents, rec)

    # Dose-response endpoints (#6): magnitude + duration of the transient,
    # derived from the per-round version-(b) distances the recovery evaluator
    # already logged. peak = how far off the attractor it got; displaced_rounds
    # = how many post-anchor rounds stayed above the "moved away" cutoff.
    dists = rec.get("post_injection_distances") or []
    cutoff = RecoveryConfig().distance_cutoff
    peak_displacement = max(dists) if dists else None
    displaced_rounds = sum(1 for d in dists if d >= cutoff)

    # Per-run PNGs into the run's own folder (newest run_* under output_dir).
    if make_plots:
        try:
            dirs = sorted(
                glob.glob(os.path.join(output_dir, "run_*")),
                key=os.path.getmtime,
            )
            if dirs:
                label = (
                    f"{model_key}_{prompt_name}_"
                    f"{(size.value if size else 'floor')}_{timing or ''}"
                )
                plot_run(dirs[-1], label)
        except Exception as e:  # noqa: BLE001
            print(f"[grid] plot skipped: {e}")

    return {
        "model": model_key,
        "temp": temp,
        "seed": seed,
        "prompt": prompt_name,
        "condition": condition,
        "size": size.value if size else "",
        "timing": timing or "",
        "collapse_onset": m.collapse_onset_round,
        "n_injections": len(m.injections or []),
        "injection_round": inj.get("round"),
        "injection_distance": inj.get("distance"),
        "recovered": rec.get("recovered"),
        "recovered_confirmed": judged["recovered_confirmed"],
        "judge_score": judged["judge_score"],
        "recovery_round": rec.get("recovery_round"),
        "hold_length": rec.get("hold_length"),
        "peak_displacement": peak_displacement,
        "displaced_rounds": displaced_rounds,
        "git_hash": _GIT_HASH,
    }


def _worker(spec: Dict[str, object]) -> Dict[str, object]:
    """Run one cell (own process, so per-run random seeding never clashes)."""
    size = InjectionSize(spec["size"]) if spec["size"] else None
    injection = (
        _make_injection(size, str(spec["timing"]), int(spec["seed"]))
        if spec["condition"] in ("inject", "fresh")
        else None
    )
    return _run_one(
        str(spec["model"]),
        float(spec["temp"]),
        int(spec["seed"]),
        str(spec["prompt"]),
        str(spec["condition"]),
        size,
        (str(spec["timing"]) if spec["timing"] else None),
        injection,
        int(spec["max_iters"]),
        bool(spec["stochastic"]),
        bool(spec["make_plots"]),
        str(spec["study_root"]),
    )


def _write(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=120)
    ap.add_argument(
        "--models",
        default="qwen-2.5-7b,qwen-2.5-72b,llama-3.3-70b",
        help="comma-separated model aliases; choices: " + ",".join(MODELS),
    )
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument(
        "--temps",
        type=str,
        default=None,
        help="comma-separated temperatures to sweep (overrides --temp)",
    )
    ap.add_argument(
        "--sizes",
        type=str,
        default=",".join(s.value for s in SIZES),
        help="comma-separated injection sizes; default: " + ",".join(
            s.value for s in SIZES
        ),
    )
    ap.add_argument(
        "--timings",
        type=str,
        default=",".join(TIMINGS),
        help="comma-separated timings; default: " + ",".join(TIMINGS),
    )
    ap.add_argument(
        "--prompts",
        type=str,
        default=",".join(PROMPTS),
        help="comma-separated prompt-types; default: " + ",".join(PROMPTS),
    )
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="no sampling seed (required for the company vLLM, which is "
        "frozen/deterministic when seeded)",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="skip per-run PNG generation (faster)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrent conversations (own process each). vLLM handles "
        "concurrency well; raise if the endpoints + machine can take it.",
    )
    ap.add_argument("--out", default="results/study/summary.csv")
    args = ap.parse_args()

    # The study lives in one folder; each run goes to study/<model>/<prompt>/.
    study_root = os.path.dirname(args.out)
    os.makedirs(study_root, exist_ok=True)

    temps = (
        [float(t) for t in args.temps.split(",")]
        if args.temps
        else [args.temp]
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [InjectionSize(s.strip()) for s in args.sizes.split(",")]
    timings = [t.strip() for t in args.timings.split(",")]
    prompts = [p.strip() for p in args.prompts.split(",")]

    # floor + (sizes x timings) treatments + (sizes) fresh-run baselines
    per_cell = 1 + len(sizes) * len(timings) + len(sizes)
    total = len(temps) * args.seeds * len(models) * len(prompts) * per_cell
    print(
        f"[grid] plan: {len(models)} models x {len(temps)} temps x "
        f"{args.seeds} seeds x {len(prompts)} prompts x "
        f"(1 floor + {len(sizes)}x{len(timings)} inj + {len(sizes)} fresh) "
        f"= {total} runs"
    )

    make_plots = not args.no_plots
    common = dict(
        max_iters=args.max_iters,
        stochastic=args.stochastic,
        make_plots=make_plots,
        study_root=study_root,
    )
    # Build every cell as a picklable spec, then run them concurrently.
    specs: List[Dict[str, object]] = []
    for temp in temps:
        for seed in range(args.seeds):
            for model_key in models:
                for prompt_name in prompts:
                    specs.append(dict(
                        model=model_key, temp=temp, seed=seed,
                        prompt=prompt_name, condition="floor",
                        size=None, timing=None, **common,
                    ))
                    for size in sizes:
                        for timing in timings:
                            specs.append(dict(
                                model=model_key, temp=temp, seed=seed,
                                prompt=prompt_name, condition="inject",
                                size=size.value, timing=timing, **common,
                            ))
                        # fresh-run baseline: same size, injected early into a
                        # not-yet-collapsed loop (the control for displacement).
                        specs.append(dict(
                            model=model_key, temp=temp, seed=seed,
                            prompt=prompt_name, condition="fresh",
                            size=size.value, timing=FRESH_TIMING, **common,
                        ))

    def _key(model, temp, seed, prompt, condition, size, timing) -> tuple:
        return (
            str(model), str(float(temp)), str(int(seed)), str(prompt),
            str(condition), str(size or ""), str(timing or ""),
        )

    # Resume: if the summary already exists, keep its rows and skip cells that
    # are already done -- so re-running after the endpoints refresh continues
    # where it left off, redoing nothing that finished.
    rows: List[Dict[str, object]] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = list(csv.DictReader(f))
        done = {
            _key(r["model"], r["temp"], r["seed"], r["prompt"],
                 r["condition"], r["size"], r["timing"])
            for r in rows
        }
        kept = [
            s for s in specs
            if _key(s["model"], s["temp"], s["seed"], s["prompt"],
                    s["condition"], s["size"], s["timing"]) not in done
        ]
        print(
            f"[grid] resume: {len(specs) - len(kept)} cells already done, "
            f"{len(kept)} left to run"
        )
        specs = kept

    # Interleave by (model, temp) so concurrent workers hit all endpoints AND
    # both temperatures at once. Temperature is the outermost build loop, so
    # without this the whole first temp would finish before the second is ever
    # touched; round-robining the (model, temp) buckets instead means seed-0
    # coverage builds across every model and temperature together -- so a window
    # that stops early still yields balanced coverage, not one temp only.
    by_cell: Dict[tuple, List[Dict[str, object]]] = {}
    for s in specs:
        by_cell.setdefault((str(s["model"]), str(s["temp"])), []).append(s)
    lists = list(by_cell.values())
    specs = [
        lst[i] for i in range(max((len(x) for x in lists), default=0))
        for lst in lists if i < len(lst)
    ]

    if not specs:
        print("[grid] nothing left to run (all cells done)")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, s) for s in specs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[grid] a run failed: {type(e).__name__}: {e}")
            # Write the summary after EVERY run so nothing is lost if the
            # endpoints expire / the laptop closes mid-study.
            _write(rows, args.out)
            print(f"[grid] {i}/{len(specs)} runs done", flush=True)

    print(f"[grid] wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
