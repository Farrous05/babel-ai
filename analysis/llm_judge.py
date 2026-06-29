"""LLM-as-judge collapse signal — catches *discourse-level* collapse.

Our metric detector (windowed cosine + Jaccard) measures *meaning* and *words*.
It misses a collapse where the model repeats the same **conversational move**
every turn (e.g. "enthusiastically affirm the other + build on their idea")
while *varying* both the topic and the wording — because the topic drift keeps
cosine distance high and the synonym/topic variation keeps Jaccard distance
high. A human spots it instantly; so does an LLM asked the right question.

This module slides a window over a run's turns and asks a judge model (separate
from the loop model, for independence) whether the window is stuck in one
conversational pattern, *ignoring topic*. Read-only: scores an existing run.

Usage::

    poetry run python analysis/llm_judge.py <run_dir> [--window 6 --step 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import List, Optional

sys.path.insert(0, "src")

from api.enums import OpenAIModels  # noqa: E402
from api.openai import openai_request  # noqa: E402

JUDGE_MODEL = OpenAIModels.GPT4O

# Calibrated against human labels (2026-06-29, 10 windows, 90% agreement with
# the gpt-4o judge + the rubric below). Clear topical/verbatim collapse scores
# 0.7-0.95; genuinely diverse scores 0.0-0.3; the discourse-borderline (same
# move, drifting topic) sits at ~0.4 and is genuinely fuzzy. Use 0.5 as the
# high-confidence cutoff; treat 0.4-0.5 as "uncertain -> human glance".
COLLAPSE_THRESHOLD = 0.5

_SYSTEM = (
    "You are a strict conversation analyst detecting whether an AI talking to "
    "itself has collapsed into repetition. Score decisively, using the full "
    "0.0-1.0 range -- do not bunch everything in the middle."
)

_PROMPT = """Below are {n} consecutive turns from an AI talking to itself. \
Decide how COLLAPSED they are -- how much the AI is just repeating itself \
rather than genuinely progressing.

IGNORE topic changes. A conversation can hop across many topics and still be \
collapsed. What matters is whether each turn does something STRUCTURALLY NEW:

- PROGRESS (low score): a turn adds a new claim, a disagreement, a question, \
new information, or a real shift in stance.
- REPETITION (high score): a turn reuses the same wording, OR makes the same \
conversational move every time (e.g. "enthusiastically affirm the other + \
build on their idea"), OR restates the same point in new words. Warm framing \
alone ("I'm glad/excited/thrilled...") is NOT progress.

Use the FULL range and be decisive:
- 0.9-1.0: near-verbatim / near-identical turns (the same sentences or \
structure recur with tiny edits), or every turn the exact same move with no \
new substance.
- 0.6-0.8: clearly the same template/move each turn, even though the topic or \
wording varies (discourse-level loop).
- 0.3-0.5: mostly progressing, with some repetition.
- 0.0-0.2: genuinely varied -- different kinds of moves and real new content \
each turn.

Respond with ONLY this JSON (no prose):
{{"score": <float 0.0-1.0>, "repetitive": <true|false>, \
"pattern": "<short description of the repeated move, or 'none'>"}}

TURNS:
{turns}"""


def _parse(raw: str) -> dict:
    """Pull the JSON verdict out of the judge's reply, robustly."""
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {"score": None, "repetitive": None, "pattern": "parse_error"}


def judge_window(
    turns: List[str], model: OpenAIModels = JUDGE_MODEL
) -> dict:
    """Score one window of consecutive turns for pattern-repetition."""
    body = "\n".join(
        f"[{i + 1}] {re.sub(r'\\s+', ' ', t).strip()[:320]}"
        for i, t in enumerate(turns)
    )
    resp = openai_request(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _PROMPT.format(n=len(turns), turns=body),
            },
        ],
        model=model,
        temperature=0.0,
        max_tokens=200,
    )
    return _parse(resp.content)


def judge_run(
    run_dir: str,
    window: int = 6,
    step: int = 3,
    model: OpenAIModels = JUDGE_MODEL,
) -> List[dict]:
    """Slide a window over a run's turns; return a verdict per window."""
    import pandas as pd

    csv = glob.glob(os.path.join(run_dir, "*.csv"))[0]
    df = pd.read_csv(csv)
    agents = df[df["agent_id"].notna()].reset_index(drop=True)
    contents = [str(c) for c in agents["content"].tolist()]

    out: List[dict] = []
    for lo in range(0, max(1, len(contents) - window + 1), step):
        w = contents[lo : lo + window]
        if len(w) < 2:
            break
        v = judge_window(w, model)
        v["rounds"] = f"{lo}-{lo + len(w) - 1}"
        out.append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a drift_experiment_* run directory")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--step", type=int, default=3)
    args = ap.parse_args()

    verdicts = judge_run(args.run_dir, args.window, args.step)
    print(f"{'rounds':>8} {'score':>6} {'repetitive':>11}  pattern")
    scores = []
    for v in verdicts:
        s = v.get("score")
        if isinstance(s, (int, float)):
            scores.append(s)
        print(
            f"{v['rounds']:>8} {str(s):>6} {str(v.get('repetitive')):>11}  "
            f"{str(v.get('pattern'))[:70]}"
        )
    if scores:
        mean = sum(scores) / len(scores)
        frac = sum(1 for s in scores if s >= COLLAPSE_THRESHOLD) / len(scores)
        print(
            f"\nOVERALL: mean score={mean:.2f}, "
            f"{frac * 100:.0f}% of windows collapsed "
            f"(score>={COLLAPSE_THRESHOLD})"
        )


if __name__ == "__main__":
    main()
