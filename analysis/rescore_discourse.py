"""Re-score a study for DISCOURSE collapse, alongside the topical detector.

The windowed-cosine detector only sees *topical* collapse. As observed
2026-07-03, the diversity prompts (no_repeat / curious) prevent topical collapse
but the models substitute a *discourse* rut -- a fixed conversational move that
the cosine detector scores as "diverse" (Llama: "The concept of X..."; Qwen-7B:
list + "By ...-ing these we can..." + recurring question). Different models rut
in different surface forms, so no cheap regex catches them all; the LLM judge
(llm_judge.judge_window) recognises "same move, drifting topic" semantically.

This walks every run in a study, runs the judge ONCE on a representative late
window (cheap: one call per run), and reports the TOPICAL collapse rate (from
each run's meta.json) next to the DISCOURSE collapse rate (from the judge) --
by prompt and model. The key cell is "topically diverse BUT discourse-collapsed":
the hidden ruts the headline "never collapsed" was missing.

Usage::

    poetry run python analysis/rescore_discourse.py \\
        --study results/study_injection_v1 --window 10 --limit 0
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from llm_judge import COLLAPSE_THRESHOLD, judge_window  # noqa: E402


def _judge_with_retry(window: List[str], tries: int = 5) -> dict:
    """judge_window with exponential backoff on the gpt-4o TPM rate limit
    (30k tokens/min). Returns a parse_error verdict if it never succeeds."""
    delay = 3.0
    for attempt in range(tries):
        try:
            return judge_window(window)
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"  [judge] giving up after {tries} tries: "
                      f"{type(e).__name__}")
                return {"score": None, "repetitive": None,
                        "pattern": "rate_limited"}
            time.sleep(delay)
            delay *= 2
    return {"score": None, "repetitive": None, "pattern": "rate_limited"}


def _agent_turns(run_dir: str) -> List[str]:
    import pandas as pd

    csvs = glob.glob(os.path.join(run_dir, "*.csv"))
    if not csvs:
        return []
    df = pd.read_csv(csvs[0])
    if "agent_id" not in df.columns:
        return []
    agents = df[df["agent_id"].notna()]
    return [str(c) for c in agents["content"].tolist() if str(c).strip()]


def _topical_collapsed(run_dir: str) -> Optional[bool]:
    metas = glob.glob(os.path.join(run_dir, "*_meta.json"))
    if not metas:
        return None
    meta = json.load(open(metas[0]))
    onsets = meta.get("collapse_onsets")
    if onsets is None:
        onset = meta.get("collapse_onset_round")
        return onset is not None
    return len(onsets) > 0


def _model_prompt(run_dir: str, study: str) -> Tuple[str, str]:
    """study/<model>/<prompt>/run_... -> (model, prompt)."""
    rel = os.path.relpath(run_dir, study).split(os.sep)
    model = rel[0] if len(rel) > 0 else "?"
    prompt = rel[1] if len(rel) > 1 else "?"
    return model, prompt


def score_run(run_dir: str, window: int) -> Optional[dict]:
    turns = _agent_turns(run_dir)
    if len(turns) < 4:
        return None
    # a representative late window -- where any sustained rut is established
    w = turns[-window:] if len(turns) > window else turns
    verdict = _judge_with_retry(w)
    score = verdict.get("score")
    return {
        "discourse_score": score,
        "discourse_collapsed": (
            None if score is None else score >= COLLAPSE_THRESHOLD
        ),
        "pattern": verdict.get("pattern"),
        "topical_collapsed": _topical_collapsed(run_dir),
        "n_turns": len(turns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="results/study_injection_v1")
    ap.add_argument("--window", type=int, default=10,
                    help="# of late turns to judge (1 call/run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="judge at most N runs (0 = all); use a small N to "
                    "estimate cost first")
    ap.add_argument("--out", default=None,
                    help="write per-run scores CSV (default: <study>/"
                    "discourse_scores.csv)")
    ap.add_argument("--delay", type=float, default=2.6,
                    help="seconds between calls to stay under the 30k TPM "
                    "gpt-4o rate limit (~23 calls/min)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.study, "discourse_scores.csv")
    cols = ["model", "prompt", "topical_collapsed", "discourse_score",
            "discourse_collapsed", "n_turns", "pattern", "run"]

    # Resume: keep already-scored runs, skip them, write incrementally so a
    # rate-limit hiccup never loses progress.
    rows: List[dict] = []
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            rows = list(csv.DictReader(f))
        done = {r["run"] for r in rows}

    run_dirs = sorted(glob.glob(os.path.join(args.study, "*", "*", "run_*")))
    if args.limit:
        run_dirs = run_dirs[: args.limit]
    todo = [d for d in run_dirs if os.path.basename(d) not in done]
    print(f"[discourse] {len(run_dirs)} runs, {len(done)} already scored, "
          f"{len(todo)} to judge (1 GPT-4o call each, last {args.window} "
          f"turns, {args.delay}s apart)")

    def _flush() -> None:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    for i, d in enumerate(todo, 1):
        model, prompt = _model_prompt(d, args.study)
        res = score_run(d, args.window)
        if res is None:
            continue
        res.update(model=model, prompt=prompt, run=os.path.basename(d))
        rows.append(res)
        _flush()  # save after every run
        print(f"  [{i}/{len(todo)}] {model:<13} {prompt:<11} "
              f"topical={str(res['topical_collapsed']):<5} "
              f"discourse={res['discourse_score']} "
              f"| {str(res['pattern'])[:50]}")
        time.sleep(args.delay)
    print(f"[discourse] wrote {out}")

    # Resumed rows come back from CSV as strings ("True"/"False"/""); coerce the
    # booleans back so the counts below don't treat "False" as truthy.
    def _as_bool(v: object) -> Optional[bool]:
        if isinstance(v, bool) or v is None:
            return v  # type: ignore[return-value]
        s = str(v)
        if s in ("True", "true", "1"):
            return True
        if s in ("False", "false", "0"):
            return False
        return None
    for r in rows:
        r["topical_collapsed"] = _as_bool(r.get("topical_collapsed"))
        r["discourse_collapsed"] = _as_bool(r.get("discourse_collapsed"))

    # ---- side-by-side rates by prompt, then by (prompt, model) ----
    def rate(sub: List[dict], key: str) -> str:
        vals = [r[key] for r in sub if r[key] is not None]
        if not vals:
            return " - "
        return f"{sum(1 for v in vals if v)}/{len(vals)}"

    print("\n=== TOPICAL vs DISCOURSE collapse rate, by prompt ===")
    by_p: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_p[r["prompt"]].append(r)
    print(f"  {'prompt':<11} {'topical':>9} {'discourse':>11} "
          f"{'diverse-but-discourse-stuck':>28}")
    for p, sub in sorted(by_p.items()):
        hidden = [r for r in sub
                  if r["topical_collapsed"] is False
                  and r["discourse_collapsed"]]
        print(f"  {p:<11} {rate(sub, 'topical_collapsed'):>9} "
              f"{rate(sub, 'discourse_collapsed'):>11} "
              f"{f'{len(hidden)}/{len(sub)}':>28}")

    print("\n=== by (prompt, model) ===")
    by_pm: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        by_pm[(r["prompt"], r["model"])].append(r)
    for (p, m), sub in sorted(by_pm.items()):
        print(f"  {p:<11} {m:<14} topical={rate(sub, 'topical_collapsed'):>6} "
              f"discourse={rate(sub, 'discourse_collapsed'):>6}")


if __name__ == "__main__":
    main()
