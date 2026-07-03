"""Multi-agent runs (Phase 2): do two/three models talking still collapse, and
can memory, injection, or *persona diversity* stop it?

This is the multi-agent counterpart to ``run_grid.py`` (single-agent self-loop).
The engine already supports several agents (each its own model + system prompt)
taking turns round-robin, so this file just wires up the scenarios we care about
and reuses the same scoring / recovery / judge / plotting.

Three levers, each mapped to an existing knob:
  * **number of agents**     -- 2 big models, or 3 (adds the small Qwen).
  * **memory**               -- ``history_window``: 1 = each agent sees only the
                                last message (self-loop style); None = full
                                history (agents remember the whole conversation).
  * **persona diversity**    -- per-agent system prompts. Homogeneous (no
                                persona) vs the clashing Builder/Skeptic/Wanderer
                                trio, after the Diversity paper's "social
                                attributes" idea (doc/Diversity.md): distinct
                                standpoints to resist convergence/collapse.
  * plus optional **injection** (reuses run_grid's every_5 / after_collapse).

Usage::

    poetry run python analysis/run_multiagent.py --stochastic \\
        --out results/multiagent_v1/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from babel_ai.enums import (  # noqa: E402
    AgentSelectionMethod,
    AnalyzerType,
    FetcherType,
    InjectionSize,
)
from babel_ai.experiment import Experiment  # noqa: E402
from babel_ai.recovery import RecoveryConfig  # noqa: E402
from llm_judge import judge_recovery  # noqa: E402
from run_grid import MODELS, _make_injection, _GIT_HASH  # noqa: E402
from run_longrun import plot_run  # noqa: E402
from models import (  # noqa: E402
    AgentConfig,
    AgentMetric,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
)

# Three deliberately clashing personas (the Diversity-paper "social attributes":
# distinct standpoints so the agents don't converge). Assigned round-robin to
# whichever agents a scenario uses.
PERSONAS: Dict[str, str] = {
    "builder": (
        "You are a builder in this conversation. In every reply, propose a "
        "concrete NEW idea, example, or direction that moves things forward. "
        "Do not merely react to what was said -- add something the others have "
        "not considered yet."
    ),
    "skeptic": (
        "You are a skeptic in this conversation. In every reply, challenge or "
        "poke holes in the most recent point: ask for evidence, name a hidden "
        "assumption, or give a counter-example. Never simply agree and "
        "elaborate."
    ),
    "wanderer": (
        "You are a wanderer in this conversation. In every reply, follow the "
        "most interesting or unexpected thread and pull the conversation toward "
        "a fresh angle, analogy, or tangent -- even if it changes the subject."
    ),
}
PERSONA_CYCLE = ["builder", "skeptic", "wanderer"]

_FETCH_LOCK = threading.Lock()

FIELDS = [
    "scenario", "agents", "personas", "memory", "inject", "size",
    "collapse_onset", "n_injections", "recovered", "recovered_confirmed",
    "judge_score", "hold_length", "peak_displacement", "displaced_rounds",
    "git_hash",
]


def _make_config(
    model_keys: List[str],
    personas: Optional[List[str]],
    memory: Optional[int],
    temp: float,
    max_iters: int,
    model_seed: Optional[int],
    injection,
    output_dir: str,
) -> ExperimentConfig:
    """Build a multi-agent config: one AgentConfig per model, each with its own
    persona system prompt (or None), taking turns round-robin. ``memory`` is the
    history_window (1 = last message only; None = full conversation)."""
    agent_configs = []
    for i, mk in enumerate(model_keys):
        provider, model = MODELS[mk]
        persona = None
        if personas:
            persona = PERSONAS[personas[i % len(personas)]]
        agent_configs.append(
            AgentConfig(
                provider=provider,
                model=model,
                system_prompt=persona,
                temperature=temp,
                max_tokens=2048,
                seed=model_seed,
            )
        )
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
        agent_configs=agent_configs,
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=max_iters,
        max_total_characters=10**7,
        history_window=memory,  # 1 = no memory; None = full memory
        output_dir=output_dir,
        injection_config=injection,
    )


# The pilot: one row per scenario. `agents` are model aliases; `personas` None =
# homogeneous, or a list assigned round-robin; `memory` = history_window;
# `inject` = every_5 injection (paragraph) or None.
BIG_TWO = ["llama-3.3-70b", "qwen-2.5-72b"]
THREE = ["llama-3.3-70b", "qwen-2.5-72b", "qwen-2.5-7b"]


def _suite() -> List[Dict[str, object]]:
    return [
        # B1: two big models talking -- do they collapse? (floor + injection)
        dict(scenario="two_floor", agents=BIG_TWO, personas=None,
             memory=1, inject=None),
        dict(scenario="two_inject", agents=BIG_TWO, personas=None,
             memory=1, inject="every_5"),
        # B2: memory -- same, but agents remember the whole conversation
        dict(scenario="two_memory_inject", agents=BIG_TWO, personas=None,
             memory=None, inject="every_5"),
        # B3: three agents together
        dict(scenario="three_inject", agents=THREE, personas=None,
             memory=1, inject="every_5"),
        # C: persona diversity (Diversity paper) -- does it prevent collapse?
        dict(scenario="diverse_floor", agents=BIG_TWO,
             personas=["builder", "skeptic"], memory=1, inject=None),
        dict(scenario="diverse_three_floor", agents=THREE,
             personas=PERSONA_CYCLE, memory=1, inject=None),
        dict(scenario="diverse_inject", agents=THREE,
             personas=PERSONA_CYCLE, memory=1, inject="every_5"),
    ]


def _run_one(
    spec: Dict[str, object], temp: float, max_iters: int, seed: int,
    stochastic: bool, make_plots: bool, study_root: str,
) -> Dict[str, object]:
    model_keys = list(spec["agents"])  # type: ignore[arg-type]
    personas = spec["personas"]
    memory = spec["memory"]
    inject_timing = spec["inject"]
    size = InjectionSize.PARAGRAPH
    injection = (
        _make_injection(size, str(inject_timing), seed)
        if inject_timing else None
    )
    model_seed = None if stochastic else seed
    output_dir = os.path.join(study_root, str(spec["scenario"]))
    os.makedirs(output_dir, exist_ok=True)

    with _FETCH_LOCK:
        random.seed(seed)
        exp = Experiment(_make_config(
            model_keys, personas, memory, temp, max_iters, model_seed,
            injection, output_dir,
        ))
    exp.run()
    m = exp.metadata
    rec = m.recovery or {}
    agent_contents = [
        mm.content for mm in exp.result_metrics if isinstance(mm, AgentMetric)
    ]
    judged = judge_recovery(agent_contents, rec)
    dists = rec.get("post_injection_distances") or []
    cutoff = RecoveryConfig().distance_cutoff
    peak = max(dists) if dists else None
    displaced = sum(1 for d in dists if d >= cutoff)

    if make_plots:
        try:
            dirs = sorted(glob.glob(os.path.join(output_dir, "run_*")),
                          key=os.path.getmtime)
            if dirs:
                plot_run(dirs[-1], str(spec["scenario"]))
        except Exception as e:  # noqa: BLE001
            print(f"[multiagent] plot skipped: {e}")

    return {
        "scenario": spec["scenario"],
        "agents": "+".join(model_keys),
        "personas": "+".join(personas) if personas else "none",
        "memory": "full" if memory is None else memory,
        "inject": inject_timing or "none",
        "size": size.value if injection else "",
        "collapse_onset": m.collapse_onset_round,
        "n_injections": len(m.injections or []),
        "recovered": rec.get("recovered"),
        "recovered_confirmed": judged["recovered_confirmed"],
        "judge_score": judged["judge_score"],
        "hold_length": rec.get("hold_length"),
        "peak_displacement": peak,
        "displaced_rounds": displaced,
        "git_hash": _GIT_HASH,
    }


def _write(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", default="results/multiagent_v1/summary.csv")
    args = ap.parse_args()

    study_root = os.path.dirname(args.out)
    os.makedirs(study_root, exist_ok=True)
    specs = _suite()
    print(f"[multiagent] {len(specs)} scenarios: "
          + ", ".join(str(s["scenario"]) for s in specs))

    # Resume: skip scenarios already in the summary.
    rows: List[Dict[str, object]] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = list(csv.DictReader(f))
        done = {r["scenario"] for r in rows}
        specs = [s for s in specs if str(s["scenario"]) not in done]
        print(f"[multiagent] resume: {len(specs)} scenarios left")

    make_plots = not args.no_plots
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_run_one, s, args.temp, args.max_iters, args.seed,
                      args.stochastic, make_plots, study_root): s
            for s in specs
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[multiagent] {futs[fut]['scenario']} failed: "
                      f"{type(e).__name__}: {e}")
            _write(rows, args.out)
            print(f"[multiagent] {i}/{len(specs)} scenarios done", flush=True)
    print(f"[multiagent] wrote {len(rows)} scenarios to {args.out}")


if __name__ == "__main__":
    main()
