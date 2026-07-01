"""Long, free-generation runs with automatic repeated injection (pilot).

Runs one long self-loop **per injection size** (word / sentence / paragraph) and
plots each trajectory. Key design choices:

  * **Hand-picked seed topic** — the conversation starts from a fixed
    *non-technical* seed (home cooking, ``data/seed_handpicked.json``). Injection
    text is drawn from the *tech-heavy* ShareGPT corpus, so it is **genuinely
    off-topic** (tech dropped into a cooking chat). Earlier runs seeded from
    ShareGPT itself, so a "random" injection landed back in the same coding
    domain — not off-topic at all.
  * **Automatic injection on collapse** — ``after_collapse`` + ``repeat``:
    re-inject every time the loop (re-)collapses, picking the most off-topic
    real snippet (farthest of 8 candidates) of the requested size.
  * **Free generation** — no brevity system prompt, large token cap (2048), and
    last-message feeding (``history_window=1``), so each turn is a natural-length
    utterance (closer to Maiti/Multi_LLM) and cost stays flat with run length.

Each run logs the per-round version-(b) distance and a trajectory PNG
(version-(b) distance + Jaccard-to-injection + perplexity, injections marked).

Usage::

    poetry run python analysis/run_longrun.py                  # word,sentence,paragraph
    poetry run python analysis/run_longrun.py --sizes sentence --rounds 200
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
from models import (  # noqa: E402
    AgentConfig,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
    InjectionConfig,
)

OUT_DIR = "results/longrun"

# Hand-picked seed topic (non-technical: home cooking), so that injections
# drawn from the tech-heavy ShareGPT corpus are *genuinely* off-topic. Earlier
# runs seeded from ShareGPT itself, so a "random" injection landed back in the
# same coding domain (e.g. "where do I add the javascript") — not off-topic.
SEED_PATH = "data/seed_handpicked.json"
INJECTION_CORPUS = "data/sharegpt_real.json"

SIZES = {
    "word": InjectionSize.WORD,
    "sentence": InjectionSize.SENTENCE,
    "paragraph": InjectionSize.PARAGRAPH,
    "multi_paragraph": InjectionSize.MULTI_PARAGRAPH,  # fixed bigger dose
    "output_sized": InjectionSize.OUTPUT_SIZED,  # match last output length
    "skim_half": InjectionSize.SKIM_HALF,  # replace half the output
}


def _injection(size: str, seed: int) -> InjectionConfig:
    """Real, far-off-topic injection of the given size, dosed automatically on
    each (re-)collapse (after_collapse + repeat). Injection text is sampled
    from the tech corpus — off-topic vs the hand-picked cooking seed."""
    return InjectionConfig(
        size=SIZES[size],
        source=InjectionSource.REAL,
        trigger=InjectionTrigger.AFTER_COLLAPSE,
        repeat=True,
        num_candidates=8,  # pick the most off-topic of 8 samples
        corpus_path=INJECTION_CORPUS,
        rng_seed=seed,
    )


# Selectable loop models. The default OpenAI path is unchanged; ollama_company
# points at the company OpenAI-compatible vLLM/Ollama endpoint (creds in .env).
MODELS = {
    "gpt-4o-mini": (Provider.OPENAI, OpenAIModels.GPT4O_MINI),
    "llama-3.3-70b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.LLAMA_3_3_70B),
    "llama-3-70b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.LLAMA_3_70B),
    "qwen-2.5-72b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.QWEN_2_5_72B),
    "qwen-2.5-7b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.QWEN_2_5_7B),
    "llama-3.2-3b": (Provider.OLLAMA_COMPANY, OllamaCompanyModels.LLAMA_3_2_3B),
}


def _config(
    injection: InjectionConfig,
    rounds: int,
    temp: float,
    seed: int,
    model_keys: List[str] = ("gpt-4o-mini",),
    system_prompt: Optional[str] = None,
) -> ExperimentConfig:
    # One AgentConfig per model_key. A single key => self-loop; two or more =>
    # a multi-agent conversation (round-robin alternates the agents, each fed
    # only the previous turn -- so each model responds to the other's message).
    agents = []
    for mk in model_keys:
        provider, model = MODELS[mk]
        agents.append(
            AgentConfig(
                provider=provider,
                model=model,
                # None = free generation; a minimal frame stops instruct models
                # breaking character ("this conversation just started").
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=2048,  # large cap so turns finish on their own
                # seed < 0 => no sampling seed (stochastic): a served vLLM is
                # deterministic when seeded, even at temp 1.0, which freezes the
                # loop into a byte-identical fixed point; pass None to let temp
                # actually inject stochasticity.
                seed=(None if seed < 0 else seed),
            )
        )
    return ExperimentConfig(
        fetcher_config=FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path=SEED_PATH,  # hand-picked cooking seed
            min_messages=2,
            max_messages=6,
        ),
        analyzer_config=AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=10
        ),
        agent_configs=agents,
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=rounds,
        max_total_characters=10**7,  # don't stop the long run early
        history_window=1,  # feed only the last message (Maiti-style)
        output_dir=OUT_DIR,
        injection_config=injection,
    )


def _run_one(
    size: str, rounds: int, temp: float, seed: int,
    model_keys: List[str] = ("gpt-4o-mini",),
    system_prompt: Optional[str] = None,
) -> str:
    import random

    # Keep the fetched seed conversation + injection snippet reproducible even
    # when the model sampling is stochastic (seed < 0 => no model seed).
    fetch_seed = seed if seed >= 0 else 0
    random.seed(fetch_seed)  # fix the fetched seed conversation
    inj = _injection(size, seed=fetch_seed)
    exp = Experiment(
        _config(inj, rounds, temp, seed, model_keys, system_prompt)
    )
    exp.run()
    m = exp.metadata
    n_inj = len(m.injections)
    print(
        f"[longrun:{size}] done — collapse_onsets={m.collapse_onsets}, "
        f"injections={n_inj} at rounds "
        f"{[e['round'] for e in m.injections]}"
    )
    # find the run dir we just wrote (most recent under OUT_DIR)
    dirs = sorted(
        glob.glob(os.path.join(OUT_DIR, "run_*"))
        + glob.glob(os.path.join(OUT_DIR, "drift_experiment_*")),
        key=os.path.getmtime,
    )
    return dirs[-1] if dirs else ""


def report_latencies(run_dir: str) -> None:
    """Print, per injection, two re-collapse latencies:

      * **re-declared (windowed):** rounds from the injection until the detector
        declares the next collapse -- the robust, smoothed measure.
      * **re-collapsed (raw):** rounds until the raw turn-to-turn distance
        (1 - lexical_similarity) first drops back below the 0.40 cutoff -- the
        faithful "how fast did it snap back", unaffected by window smoothing.

    The two differ because the windowed signal lags (it must flush the post-
    injection diverse turns out of its 10-turn average); the gap is exactly the
    detection latency the old fixed cooldown used to hide.
    """
    import pandas as pd

    csv = glob.glob(os.path.join(run_dir, "*.csv"))[0]
    meta = json.load(open(glob.glob(os.path.join(run_dir, "*_meta.json"))[0]))
    df = pd.read_csv(csv)
    agents = df[df["agent_id"].notna()].reset_index(drop=True)

    raw_dist = []
    for _, row in agents.iterrows():
        a = _parse_analysis(row["analysis"]) if row.get("analysis") else {}
        ll = a.get("lexical_similarity")
        raw_dist.append(None if ll is None else 1.0 - ll)

    onsets = sorted(meta.get("collapse_onsets") or [])
    inj_rounds = sorted(e["round"] for e in (meta.get("injections") or []))
    cutoff = 0.40

    print(f"[latency] {os.path.basename(run_dir)}")
    for r in inj_rounds:
        nxt = next((o for o in onsets if o > r), None)
        win = (nxt - r) if nxt is not None else None
        raw = next(
            (
                i - r
                for i in range(r + 1, len(raw_dist))
                if raw_dist[i] is not None and raw_dist[i] <= cutoff
            ),
            None,
        )
        print(
            f"  injection @r{r:<4} re-collapsed(raw)="
            f"{raw if raw is not None else '—':>4}  "
            f"re-declared(windowed)={win if win is not None else '—':>4}"
        )


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


# Common English words that appear in almost any paragraph -- excluding them is
# what separates *real* parroting ("javascript") from incidental overlap
# ("the", "where") when measuring how much of an injection a turn echoes.
_STOPWORDS = set(
    "a an the of to in on at for and or but is are was were be been being do "
    "does did this that these those i you he she it we they me him her us them "
    "my your his its our their where when how what who why which as by with "
    "from into about over under up down out off no not so if then than too "
    "very can will just add have has had not".split()
)


def _injection_containment(turn: str, injection: str) -> Optional[float]:
    """Fraction of the injection's *content* (non-stopword) tokens that appear
    in the turn. Unlike Jaccard-to-injection (which saturates near 1 because a
    short injection is swamped by a long turn's vocabulary) this stays
    sensitive to real parroting regardless of turn length. Returns ``None`` if
    the injection has no content words to measure against."""
    content = _word_set(injection) - _STOPWORDS
    if not content:
        return None
    return len(content & _word_set(turn)) / len(content)


def plot_run(run_dir: str, label: str) -> List[str]:
    """Write **one standalone PNG per metric** (clearer than a stacked figure
    -- each graph reads on its own). All four share the same x-axis (agent
    round) and the same vertical markers (orange = collapse declared, dotted
    purple = injection) so they line up if you view them side by side:

      * ``..._1a_collapse_windowed`` — the *windowed* (smoothed-over-10)
        cosine distance (primary trigger) + windowed Jaccard (secondary).
        Below the 0.40 cutoff for K=3 rounds = collapse declared; watch it dip
        into the attractor and re-collapse after each injection.
      * ``..._1b_collapse_raw`` — the same turn-to-turn signal at full
        resolution (raw Jaccard, i vs i-1). Drops to 0 = verbatim repeat of
        the previous turn; split out so the spiky raw line doesn't clutter the
        smooth windowed curves (no info lost, just separated).
      * ``..._2_recovery`` — the recovery picture. Two lines: version-(b)
        distance (how far from the old topic) AND signal-(a) (still varying
        turn-to-turn). A real recovery needs BOTH above their bars (0.70 and
        0.40) for >=K rounds, then judge confirmation -- because version-(b)
        alone also rises for a *migration* (jump to a new topic then re-loop).
        Green shading marks rounds passing both checks; the title gives the
        verdict read straight off the two plotted signals.
      * ``..._3_parroting`` — fraction of the injection's *content* (non-
        stopword) words echoed by the turn. Replaces Jaccard-to-injection,
        which saturated near 1 because a short injection is swamped by a long
        turn. High = the model is repeating the injection's distinctive words.
      * ``..._4_perplexity`` — GPT-2 perplexity, the gibberish guard; spikes =
        nonsense.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.lines import Line2D
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}")
        return []

    csv = glob.glob(os.path.join(run_dir, "*.csv"))[0]
    meta = json.load(open(glob.glob(os.path.join(run_dir, "*_meta.json"))[0]))
    df = pd.read_csv(csv)
    agents = df[df["agent_id"].notna()].reset_index(drop=True)

    injections = meta.get("injections") or []
    inj_rounds = [e["round"] for e in injections]
    onsets = meta.get("collapse_onsets") or []
    first_inj = min(inj_rounds) if inj_rounds else None

    rounds, db, cont, ppl, trunc = [], [], [], [], 0
    cos_w, jac_w, jac_raw = [], [], []  # collapse signal (windowed + raw)
    for i, row in agents.iterrows():
        a = _parse_analysis(row["analysis"]) if row.get("analysis") else {}
        rounds.append(i)
        # collapse-detection signal (windowed, turn-to-turn): distance =
        # 1 - similarity, so low = collapsed (matches the 0.40 cutoff).
        sw = a.get("semantic_similarity_window")
        lw = a.get("lexical_similarity_window")
        ll = a.get("lexical_similarity")  # raw turn-to-turn (i vs i-1)
        cos_w.append(None if sw is None else 1.0 - sw)
        jac_w.append(None if lw is None else 1.0 - lw)
        jac_raw.append(None if ll is None else 1.0 - ll)
        # version-(b) is only meaningful once there is an attractor to
        # measure from, i.e. after the first injection -- blank it before.
        d = a.get("version_b_distance")
        db.append(d if (first_inj is not None and i >= first_inj) else None)
        ppl.append(a.get("token_perplexity"))
        # how much of the latest injection's *content* words this turn echoes
        prior = [e for e in injections if e["round"] < i]
        cont.append(
            _injection_containment(row.get("content", ""), prior[-1]["text"])
            if prior
            else None
        )
        # Count a turn as truncated only if it ends *mid-word* (last char is
        # alphanumeric). Emoji / punctuation / list markers are legitimate
        # endings -- this model's free-gen attractor often closes turns with
        # emoji (🌱 ✨) or "!", which an end-punctuation check wrongly flagged.
        content = str(row.get("content", "")).rstrip()
        if content and content[-1].isalnum():
            trunc += 1

    head = (
        f"{label} injection — {len(inj_rounds)} injections, "
        f"{len(onsets)} collapses, {trunc}/{len(rounds)} truncated"
    )
    marker_handles = [
        Line2D([], [], color="orange", lw=1.4, label="collapse declared"),
        Line2D([], [], color="purple", lw=1.4, ls=":", label="injection"),
    ]

    def _mark(ax) -> None:
        # Thin, faint vertical lines so the data curves stay readable even
        # with ~30 collapses+injections over a long run (no info removed --
        # every collapse/injection is still marked, just lightly).
        for r in onsets:
            ax.axvline(r, ls="-", lw=0.7, color="orange", alpha=0.45)
        for r in inj_rounds:
            ax.axvline(r, ls=":", lw=0.8, color="purple", alpha=0.55)
        ax.grid(True, axis="y", ls=":", alpha=0.3)
        ax.set_xlabel("agent round")

    def _save(fig, suffix: str) -> str:
        fig.tight_layout()
        # Write each run's PNGs *into that run's own folder* (next to its CSV /
        # meta / PDF), so one draft's outputs are all together and easy to check.
        out = os.path.join(run_dir, f"trajectory_{label}_{suffix}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out

    written: List[str] = []

    # --- 1a. collapse-detection signal: the WINDOWED (smoothed) distances
    # that actually drive detection. Kept on its own so the two smooth curves
    # are readable without the spiky raw line on top. ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        rounds, cos_w, "-o", ms=3, color="C0",
        label="windowed cosine distance (primary trigger)",
    )
    ax.plot(
        rounds, jac_w, "-s", ms=3, color="C1", alpha=0.8,
        label="windowed Jaccard distance (secondary)",
    )
    ax.axhline(0.40, ls="--", lw=1.0, color="grey", label="cutoff (0.40)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("windowed turn-to-turn\ndistance (smoothed over 10)")
    ax.set_title(
        f"Collapse signal (windowed) — {head}\n"
        "(below cutoff for K=3 rounds = collapse declared)"
    )
    _mark(ax)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles,
              loc="upper right", fontsize=8, ncol=2)
    written.append(_save(fig, "1a_collapse_windowed"))

    # --- 1b. the RAW (un-smoothed) turn-to-turn Jaccard distance: same info
    # at full resolution, so verbatim re-collapse (raw -> 0 = identical word
    # set as the previous turn) is visible instead of averaged away. ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        rounds, jac_raw, "-d", ms=2.5, lw=1.0, color="C4",
        label="raw Jaccard distance (turn i vs i-1)",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("raw turn-to-turn\ndistance (no smoothing)")
    ax.set_title(
        f"Collapse signal (raw) — {head}\n"
        "(drops to 0 = this turn is a verbatim repeat of the previous)"
    )
    _mark(ax)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles,
              loc="lower right", fontsize=8)
    written.append(_save(fig, "1b_collapse_raw"))

    # --- 2. recovery signal: needs BOTH "moved away from the old attractor"
    # AND "still varying turn-to-turn". version-(b) distance alone ALSO rises
    # for a *migration* (the model jumps to a new topic then loops on it), so
    # we overlay signal-(a) (turn-to-turn distance): a genuine recovery keeps
    # both lines above their bars; a migration shows version-(b) high while
    # signal-(a) collapses back down. Green shading = both checks pass that
    # round; a real recovery needs >=K such rounds in a row + judge confirmation.
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(rounds, db, "-o", ms=3, color="C0",
            label="distance from old topic (version-b)")
    ax.plot(rounds, cos_w, "-s", ms=2.5, color="C2", alpha=0.8,
            label="still varying now (turn-to-turn)")
    ax.axhline(0.70, ls="--", lw=1.0, color="C0",
               label="moved-away bar (0.70)")
    ax.axhline(0.40, ls="--", lw=1.0, color="C2",
               label="variety bar (0.40)")
    both = [
        (db[i] is not None and db[i] >= 0.70)
        and (cos_w[i] is not None and cos_w[i] >= 0.40)
        for i in range(len(rounds))
    ]
    ax.fill_between(rounds, 0, 1.05, where=both, color="green", alpha=0.13,
                    label="passes BOTH checks")
    if first_inj is not None:
        ax.axvspan(-0.5, first_inj, color="grey", alpha=0.08,
                   label="pre-injection (no attractor yet)")
    # Verdict from the two plotted signals (independent of any stored flag):
    # the longest run where both checks hold.
    best = cur = 0
    for ok in both:
        cur = cur + 1 if ok else 0
        best = max(best, cur)
    verdict = (
        f"CANDIDATE — both checks held {best} rounds (needs judge)"
        if best >= 5
        else f"NO recovery — both checks held only {best} rounds (need 5)"
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("distance (0 = identical, 1 = unrelated)")
    ax.set_title(
        f"Recovery — {head}\n"
        "real recovery = both lines above bars, >=5 rounds, + judge\n"
        f"{verdict}"
    )
    _mark(ax)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles,
              loc="center right", fontsize=7, ncol=2)
    written.append(_save(fig, "2_recovery"))

    # --- 3. parroting guard (content-word containment of the injection) ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(rounds, cont, "-s", ms=3, color="C2",
            label="injection content-words echoed")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of injection\ncontent-words present")
    ax.set_title(
        f"Parroting — {head}\n"
        "(1 = turn repeats all the injection's distinctive words)"
    )
    _mark(ax)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles,
              loc="lower right", fontsize=8)
    written.append(_save(fig, "3_parroting"))

    # --- 4. gibberish guard (perplexity) ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(rounds, ppl, "-^", ms=3, color="C3", label="GPT-2 perplexity")
    ax.set_ylabel("perplexity")
    ax.set_title(
        f"Gibberish guard — {head}\n(spikes = nonsense; NOT valid for "
        "non-English turns -- GPT-2 is English-only)"
    )
    _mark(ax)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles,
              loc="upper right", fontsize=8)
    written.append(_save(fig, "4_perplexity"))

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument(
        "--temps",
        type=str,
        default=None,
        help="comma-separated temperatures to sweep (overrides --temp); each "
        "temp x size is its own run, e.g. 0.2,0.5,0.7",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sizes",
        type=str,
        default="word,sentence,paragraph",
        help="comma-separated injection sizes; choices: "
        + ",".join(SIZES),
    )
    ap.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="comma-separated model keys; one = self-loop, two+ = multi-agent "
        f"conversation (round-robin). choices: {','.join(MODELS)}",
    )
    ap.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="optional system prompt to frame the self-loop (default none = "
        "free generation). Use to stop instruct models breaking character.",
    )
    args = ap.parse_args()

    # Each launch writes into its own timestamped batch folder so a new run's
    # CSVs and PNGs never mix with an older run's outputs.
    from datetime import datetime

    global OUT_DIR
    OUT_DIR = os.path.join(
        OUT_DIR, f"runset_{datetime.now().strftime('%m%d-%H%M%S')}"
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    model_keys = [m.strip() for m in args.model.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODELS:
            raise SystemExit(f"unknown model {mk!r}; choose from {list(MODELS)}")
    mode = "self-loop" if len(model_keys) == 1 else "multi-agent"
    print(
        f"[longrun] batch {OUT_DIR} | {mode}: {'+'.join(model_keys)}"
    )

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    temps = (
        [float(t) for t in args.temps.split(",")]
        if args.temps
        else [args.temp]
    )
    for size in sizes:
        if size not in SIZES:
            raise SystemExit(
                f"unknown size {size!r}; choose from {list(SIZES)}"
            )
    for temp in temps:
        for size in sizes:
            # label includes temp so the per-metric PNGs for each (temp, size)
            # combination are written side by side without overwriting.
            label = f"{size}_t{temp}"
            run_dir = _run_one(
                size, args.rounds, temp, args.seed, model_keys,
                args.system_prompt,
            )
            if run_dir:
                report_latencies(run_dir)
                for out in plot_run(run_dir, label):
                    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
