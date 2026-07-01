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


def judge_recovery(
    agent_contents: List[str],
    recovery: Optional[dict],
    model: OpenAIModels = JUDGE_MODEL,
) -> dict:
    """Confirm (or reject) a flagged recovery with the discourse judge.

    The cheap recovery criteria (recovery.py) can pass a stretch that is still
    a *discourse* loop -- the model repeating the same conversational move while
    the topic drifts (cosine + Jaccard are blind to it). This runs the LLM judge
    on the flagged hold stretch only (so it costs one call per *candidate*
    recovery, not per round) and returns whether the judge agrees it is
    genuinely varied.

    Validated 2026-06-30 against 23 hand-labelled stretches: at the 0.5 cutoff
    the judge agrees with the human 19/23, catching 16 of 18 false recoveries
    and erring toward "stuck" (it over-flags template storytelling).

    Returns ``{"judge_score": float|None, "recovered_confirmed": bool|None}``.
    ``None`` when there is no recovery to confirm or the stretch is too short.
    """
    if not recovery or not recovery.get("recovered"):
        return {"judge_score": None, "recovered_confirmed": None}
    start = recovery.get("recovery_round")
    hold = recovery.get("hold_length") or 0
    if start is None:
        return {"judge_score": None, "recovered_confirmed": None}
    turns = [
        agent_contents[i]
        for i in range(start, min(start + hold, len(agent_contents)))
        if agent_contents[i] and agent_contents[i].strip()
    ][:10]
    if len(turns) < 3:
        return {"judge_score": None, "recovered_confirmed": None}
    score = judge_window(turns, model).get("score")
    confirmed = score is not None and score < COLLAPSE_THRESHOLD
    return {"judge_score": score, "recovered_confirmed": confirmed}


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


def plot_judge_vs_metric(
    run_dir: str, window: int = 6, step: int = 3
) -> Optional[str]:
    """Two stacked panels sharing the x-axis: the **metric** collapse signal
    (windowed cosine distance, low = collapsed) and the **LLM-judge** collapse
    score (high = collapsed). Collapse-declared (orange) and injection (purple)
    markers on both, so you can see directly where the two signals agree and
    where the judge catches a collapse the metric misses (or vice versa)."""
    import ast
    import json

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.lines import Line2D
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}")
        return None

    csv = glob.glob(os.path.join(run_dir, "*.csv"))[0]
    meta = json.load(open(glob.glob(os.path.join(run_dir, "*_meta.json"))[0]))
    df = pd.read_csv(csv)
    agents = df[df["agent_id"].notna()].reset_index(drop=True)

    # metric: windowed cosine distance per round
    rounds, cosd = [], []
    for i, row in agents.iterrows():
        try:
            a = ast.literal_eval(row["analysis"]) if row.get("analysis") else {}
        except Exception:  # noqa: BLE001
            a = {}
        sw = a.get("semantic_similarity_window")
        rounds.append(i)
        cosd.append(None if sw is None else 1.0 - sw)

    # judge: score per sliding window, plotted at the window's centre round
    jx, jy = [], []
    for v in judge_run(run_dir, window, step):
        s = v.get("score")
        if not isinstance(s, (int, float)):
            continue
        lo, hi = (int(x) for x in v["rounds"].split("-"))
        jx.append((lo + hi) / 2)
        jy.append(s)

    onsets = meta.get("collapse_onsets") or []
    inj = [e["round"] for e in (meta.get("injections") or [])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(rounds, cosd, "-o", ms=3, color="C0")
    ax1.axhline(0.40, ls="--", lw=1.0, color="grey", label="cutoff 0.40")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("metric: windowed\ncosine distance")
    ax1.set_title(
        "Collapse — metric vs LLM judge   (metric: BELOW 0.40 = collapsed)"
    )
    ax2.plot(jx, jy, "-s", ms=4, color="C2")
    ax2.axhline(
        COLLAPSE_THRESHOLD, ls="--", lw=1.0, color="grey",
        label=f"threshold {COLLAPSE_THRESHOLD}",
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("LLM judge\ncollapse score")
    ax2.set_xlabel("agent round")
    ax2.set_title("(judge: ABOVE threshold = collapsed)")

    handles = [
        Line2D([], [], color="orange", lw=1.2, label="collapse declared"),
        Line2D([], [], color="purple", lw=1.2, ls=":", label="injection"),
    ]
    for ax in (ax1, ax2):
        for r in onsets:
            ax.axvline(r, ls="-", lw=0.9, color="orange", alpha=0.6)
        for r in inj:
            ax.axvline(r, ls=":", lw=1.0, color="purple", alpha=0.7)
        ax.grid(True, axis="y", ls=":", alpha=0.3)
        ax.legend(
            handles=ax.get_legend_handles_labels()[0] + handles,
            loc="upper right", fontsize=8,
        )

    fig.tight_layout()
    out = os.path.join(run_dir, "trajectory_judge_vs_metric.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
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
