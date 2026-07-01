"""Calibrate the LLM-judge threshold against human labels.

Two-step workflow:

1. ``build`` — gather a varied set of conversation windows (early vs late turns
   across several runs, plus synthetic clearly-diverse / clearly-collapsed
   anchors), score each with the judge, save them, and print them **without the
   scores** so a human can label collapsed / not *blind*.
2. ``score`` — given the human labels, find the judge-score threshold that best
   matches them and report the judge's accuracy / where collapse begins.

Usage::

    poetry run python analysis/calibrate_judge.py build
    # ...label each window, then:
    poetry run python analysis/calibrate_judge.py score --labels 1,0,1,1,0,...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import List

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from llm_judge import judge_window  # noqa: E402  (analysis/ is on sys.path)

CALIB_FILE = "results/analysis/judge_calibration.json"

# Synthetic anchors so the set always spans both extremes.
_ANCHORS = [
    (
        "anchor_diverse",
        [
            "What's the capital of France?",
            "Paris. Actually I disagree that eggs should be cold -- room "
            "temperature emulsifies better.",
            "Switching gears: here's a quick sort, def f(x): return sorted(x).",
            "That's O(n log n). Different question -- does remote work hurt "
            "team culture?",
        ],
    ),
    (
        "anchor_collapsed",
        [
            "Thank you so much! I'm happy to help with your cooking journey.",
            "Thank you so much! I'm happy to help with your cooking journey.",
            "Thanks again! I'm so glad to help you on your cooking journey.",
            "Thank you! I'm always happy to help with your cooking journey.",
        ],
    ),
]


def _windows_from_run(run_dir: str, w: int = 5) -> List[dict]:
    """One early and one late window from a run."""
    csvs = glob.glob(os.path.join(run_dir, "*.csv"))
    if not csvs:
        return []
    df = pd.read_csv(csvs[0])
    a = df[df["agent_id"].notna()].reset_index(drop=True)
    turns = [str(c) for c in a["content"].tolist()]
    if len(turns) < w + 2:
        return []
    tag = os.path.basename(run_dir)
    tag = re.sub(r"drift_experiment_\d+_\d+_", "", tag)[:40]
    return [
        {"id": f"{tag}|early", "turns": turns[1 : 1 + w]},
        {"id": f"{tag}|late", "turns": turns[-w:]},
    ]


def build(run_dirs: List[str]) -> None:
    items: List[dict] = [
        {"id": i, "turns": t} for i, t in _ANCHORS
    ]
    for d in run_dirs:
        items.extend(_windows_from_run(d))

    print(f"Scoring {len(items)} windows with the judge...", file=sys.stderr)
    for it in items:
        v = judge_window(it["turns"])
        it["score"] = v.get("score")
        it["pattern"] = v.get("pattern")

    os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
    with open(CALIB_FILE, "w") as f:
        json.dump(items, f, indent=2)

    # Present blind (no scores) for human labelling.
    print("\n=== LABEL EACH WINDOW: collapsed (same move repeated) or not ===")
    print("Reply with one 1 (collapsed) / 0 (diverse) per window, in order.\n")
    for k, it in enumerate(items, 1):
        print(f"[{k}] {it['id']}")
        for t in it["turns"]:
            print(f"     - {re.sub(r'\\s+', ' ', t).strip()[:100]}")
        print()


def score(labels: List[int]) -> None:
    items = json.load(open(CALIB_FILE))
    scored = [(it, lab) for it, lab in zip(items, labels) if it.get("score") is not None]
    if len(scored) != len(labels):
        print("warning: some windows had no judge score; aligning by order")

    # Find the threshold (midpoint between adjacent sorted scores) with best
    # agreement: predict collapsed if score >= threshold.
    scores = sorted({it["score"] for it, _ in scored})
    cands = [0.0] + [
        (scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)
    ] + [1.0]
    best_t, best_acc = 0.7, -1.0
    for t in cands:
        acc = sum(
            1 for it, lab in scored if (it["score"] >= t) == bool(lab)
        ) / len(scored)
        if acc > best_acc:
            best_acc, best_t = acc, t

    print(f"\nbest threshold = {best_t:.2f}  (judge agrees with you "
          f"{best_acc * 100:.0f}% of {len(scored)} windows)\n")
    print(f"{'window':<34} {'judge':>6} {'you':>4} {'pred':>5} {'ok':>3}")
    for it, lab in scored:
        pred = int(it["score"] >= best_t)
        ok = "Y" if pred == lab else "x"
        print(f"{it['id']:<34} {it['score']:>6.2f} {lab:>4} {pred:>5} {ok:>3}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument(
        "--runs", nargs="*", default=None,
        help="run dirs; default = a spread of recent batches",
    )
    s = sub.add_parser("score")
    s.add_argument("--labels", required=True, help="comma-separated 1/0")
    args = ap.parse_args()

    if args.cmd == "build":
        runs = args.runs or sorted(
            glob.glob("results/longrun/runset_*/run_*")
            + glob.glob("results/longrun/batch_*/drift_experiment_*")
        )[-8:]
        build(runs)
    else:
        build_labels = [int(x) for x in args.labels.split(",")]
        score(build_labels)


if __name__ == "__main__":
    main()
