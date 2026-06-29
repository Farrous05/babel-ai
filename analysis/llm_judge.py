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

JUDGE_MODEL = OpenAIModels.GPT4O_MINI

_SYSTEM = (
    "You are an expert conversation analyst. You decide whether an AI talking "
    "to itself has collapsed into a repetitive PATTERN."
)

_PROMPT = """Below are {n} consecutive turns from an AI talking to itself.

IGNORE whether the topic changes. A conversation can roam across many topics \
and still be COLLAPSED if every turn makes the SAME conversational move -- for \
example: always "enthusiastically praise the other and build on their idea", \
always restating the same point in new words, or always the same \
question->answer->affirm shape. Genuine progress means each turn does \
something structurally new (a real disagreement, a new kind of contribution, a \
shift in stance), not just a new topic in the same template.

Judge whether these turns are stuck in one repetitive conversational pattern.

Respond with ONLY this JSON (no prose):
{{"score": <float 0.0-1.0, where 1.0 = every turn is the same move>, \
"repetitive": <true|false>, "pattern": "<short description of the repeated \
move, or 'none'>"}}

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
        frac = sum(1 for s in scores if s >= 0.6) / len(scores)
        print(
            f"\nOVERALL: mean score={mean:.2f}, "
            f"{frac * 100:.0f}% of windows judged repetitive (score>=0.6)"
        )


if __name__ == "__main__":
    main()
