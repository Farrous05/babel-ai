# Weekly check-up — babel-ai (injection study) — 2026-07-02

## TL;DR (one paragraph)

The project studies a single LLM fed its own output in a loop, which reliably
**collapses into repetition**, and asks whether **injecting off-topic text**
can knock it out of that rut. This week I (1) hardened the recovery metric so it
can no longer be fooled by topic-hopping, language-switching, or a fixed
conversational "rut"; (2) built a discourse judge and validated it against hand
labels; (3) rebuilt the experiment grid for a proper study (multiple models,
temperatures, seeds, prompts, injection sizes and timings, with concurrency,
resume, and per-run graphs); and (4) ran a **hostile peer-review pass** of the
whole codebase, verified its findings against the source, and fixed the ones
that actually bias results. The most important outcome is a **reframe of the
research question**: rather than chasing "lasting recovery" (a near-certain
negative that the literature already reports), we now measure the **dose–
response of the transient** — how far an injection moves the model off the
attractor and for how long — against a fresh-run control. This is a defensible,
novel contribution and the pipeline is now correct enough to produce it.

---

## 1. Research question & current position

**Core question.** A self-looping LLM collapses into a low-diversity "attractor"
(it keeps saying variations of the same thing). Does an external text injection
push it out, and does the effect **last**? Secondary: does injection **size**,
**timing**, or a steering **system prompt** matter?

**Where we now stand.** Our own earlier data and the reference papers agree that
injections produce only **transient** displacement — the model wanders off the
topic briefly, then slides back into a (often new) rut. A previously reported
"first lasting recovery" was retracted after the metric was tightened.

**The reframe (this week's key decision).** "Smallest injection that produces
*lasting* recovery" is chasing a null result that Kong et al. already
demonstrate (12 intervention strategies fail to restore lasting diversity). The
defensible contribution is a **dose–response characterization of the transient**:
- how far off the attractor an injection knocks the loop (**peak displacement**),
- for how many rounds it stays off (**displaced rounds / time-to-re-collapse**),
- as a function of injection **size**, **timing**, and **content**,
- measured against a **fresh-run baseline** (same injection into a loop that has
  *not* collapsed), so we quantify the **pull (basin) of the attractor**, not
  just raw displacement.

In short: we are **measuring the shape of the attractor's basin**, not claiming
to escape it.

---

## 2. What I did this week

**Metric hardening (recovery).** Added two criteria to the recovery evaluator so
a run only counts as "recovered" if it is genuinely varied, not merely relocated:
- **Anti-migration** — the turn must be diverse *right now* (windowed turn-to-turn
  distance above the collapse cutoff), so jumping from one tight loop to another
  no longer reads as recovery.
- **Same-language** — a turn that switches script (e.g. English → Chinese) is
  drift, not recovery, and it also invalidates the English-based perplexity and
  overlap guards; such turns are excluded. (New dependency-free script detector,
  `src/babel_ai/language.py`.)

**Discourse judge.** Built an LLM judge (`analysis/llm_judge.py`) that flags the
"cheerleader rut" — same conversational move, drifting topic — which the
geometric metrics miss. Validated at 19/23 agreement with hand labels (see §5 for
the caveat the review raised).

**Grid runner rebuilt** (`analysis/run_grid.py`). Free-generation self-loop;
injections drawn from a **cross-domain corpus** with a *farthest-of-8* selection
so every injection is a genuinely new topic (distance ~0.95 vs the old
~0.84 tech-into-tech); sizes (paragraph / output-sized / skim-half) × timings
(after-collapse / early / steady-drip); 3 models (Qwen-7B, Qwen-72B, Llama-3.3-70B);
temperatures and seeds swept; **concurrency** (thread pool over the endpoints),
**resume** (re-run continues where it stopped), and **per-run graphs** identical
to the long-run maker.

**Hostile review + fixes.** Commissioned an adversarial code+method review, then
**verified each finding against the source** before acting. Fixed the ones that
bias results (§3).

---

## 3. Adversarial review — findings verified and fixed

I treated the review as a checklist and confirmed every claim in the code before
fixing. Status of each:

| # | Issue (verified in code) | Action |
|---|--------------------------|--------|
| Seed | Every seed conversation ended on an assistant answer, so the loop's first turn reacted to an *answer* as if a human said it | **Fixed** — trim seed to end on a human turn (`prompt_fetcher.py`), committed |
| 1 | Injected text stayed in the scored window, so collapse/recovery metrics partly measured the *injection* for the whole hold window | **Fixed** — score over a clean, un-mutated copy of the turns (`experiment.py`) |
| 3 | The result aggregator expected columns the runner never wrote (couldn't add up results); no fresh-run baseline | **Fixed** — rewrote aggregator to the real schema; added the **fresh-run baseline** condition |
| 4 | The collapse detector's embedder truncates at ~256 tokens, so on 300–1000-word turns it only "saw" each turn's opening; docs contradicted the code | **Fixed** — chunk-and-mean-pool long turns (`analyzer.py`); corrected the doc |
| 6 | Only a binary "recovered" was reported; the transient's magnitude/length was thrown away | **Fixed** — added **peak displacement** and **displaced rounds** columns (the dose–response endpoints) |
| 9 | A resumed study could silently mix rows from different code versions | **Fixed** — stamp each row with the git commit; each code version gets a clean folder |

**Deliberately NOT treated as bugs / deferred (with reasons):**
- **Timing anchor** (review #2): the reviewer flagged that early/drip injections
  are measured from a not-yet-collapsed point. We keep this on purpose — part of
  the question is how injection affects the loop *even before deep collapse*
  (prevention, not just rescue).
- **Noise arm** (part of review #3): dropped on purpose — random-noise injection
  is Kong et al.'s settled negative; **large *real* injections into a
  single-model loop is the part that is novel to us.**
- **Deferred to before the *full* study, not the pilot:** recalibrating the
  collapse cutoff on the exact deployed signal + a threshold-sensitivity sweep
  (review #5); expanding the judge validation set with a second labeler and
  reporting agreement with confidence intervals (#7); more seeds on one
  pre-registered primary contrast for statistics (#8); matched token-budget so
  the "size" axis is a clean ladder (#10).

---

## 4. Metrics & methods (current definitions)

### 4.0 The two reference models (held fixed across every run)

- **Embedding model — Sentence-BERT `all-MiniLM-L6-v2`**: turns text into a
  vector so we can measure *meaning* similarity by the angle between vectors
  (**cosine similarity**: 1 = identical meaning, 0 = unrelated). Used for the
  turn-to-turn collapse signal. Long turns are split into pieces and averaged
  so the whole turn counts, not just its opening.
- **Reference embedder — OpenAI `text-embedding-3-large`**: a stronger, longer-
  context embedder used only for "how far is this turn from the collapsed
  rut?" (the version-(b) distance below) and for scoring how off-topic an
  injection is.
- **Perplexity model — GPT-2**: a *separate* model that rates how "surprised"
  it is by a turn. High perplexity ⇒ likely gibberish. It is independent of the
  loop model, so it is a fair quality check.

### 4.1 Raw per-turn metrics (scored on every generated turn, stored in the CSV)

| Metric (CSV key) | Range | Plain-language meaning |
|---|---|---|
| `word_count` | ≥ 0 | number of words in the turn |
| `unique_word_count` | ≥ 0 | number of *distinct* words in the turn |
| `coherence_score` | 0–1 | `unique / total` words — vocabulary variety *inside* one turn (low ⇒ the turn repeats itself) |
| `token_perplexity` | ≥ 1 | GPT-2 perplexity — how surprising/garbled the turn is (our gibberish guard) |
| `lexical_similarity` | 0–1 | **word overlap** (Jaccard) vs the *previous* turn — 1 = same words reused |
| `semantic_similarity` | −1–1 | **meaning overlap** (SBERT cosine) vs the *previous* turn — 1 = same meaning |
| `lexical_similarity_window` | 0–1 | mean word overlap vs the previous **W = 10** turns |
| `semantic_similarity_window` | −1–1 | mean meaning overlap vs the previous **W = 10** turns — the **collapse detector's primary input** |
| `version_b_distance` | 0–2 | `1 − cosine` of the turn from the **collapsed window's average vector** (text-embedding-3-large) — "how far from the rut" |

Two useful shorthands derived from the above:
- **Cosine distance** `= 1 − semantic_similarity_window` — turn-to-turn
  *difference* in meaning (high = varied, low = stuck). This is what the
  collapse rule thresholds.
- **Jaccard distance** `= 1 − lexical_similarity_window` — turn-to-turn
  difference in *words* (logged as a secondary signal).

### 4.2 Collapse detection (turns metrics into a yes/no)

**Collapse detection** (`collapse.py`): the primary signal is the **windowed
cosine distance** (1 − mean cosine similarity of the current turn to the previous
W=10 turns, SBERT `all-MiniLM-L6-v2`, now chunk-pooled for long turns). Collapse
is declared when that distance stays **≤ 0.40 for K=3 consecutive rounds** after
a **10-round warm-up**. A hysteresis re-arm at 0.50 prevents false re-collapse
right after an injection. Jaccard (word-overlap) distance is logged as a
secondary signal.

### 4.3 Recovery (the strict "did it genuinely escape?" test)

**Recovery** (`recovery.py`): after the anchor round, a run "recovers" only if,
for **5 consecutive rounds (hold-K = 5)**, the turn passes ALL of:
1. **moved away** — version-(b) distance ≥ 0.70 (far from the collapsed rut);
2. **not gibberish** — GPT-2 perplexity ≤ 150;
3. **not parroting the injection** — word-overlap distance to the injected text
   ≥ 0.5 (it isn't just echoing what we injected);
4. **did not re-collapse** — the round isn't a fresh collapse onset;
5. **diverse right now** — turn-to-turn cosine distance ≥ 0.40 (so *hopping into
   a new rut* doesn't count — it must actually vary turn to turn);
6. **same language** — no script switch (e.g. English → Chinese), which would
   both be drift and break the English-based guards above.

The discourse judge then confirms the streak is genuinely varied and not a fixed
conversational "rut" (see §2). Thresholds are provisional and slated for
calibration (#5/#7).

### 4.4 Dose–response endpoints (new — the reframed primary measures)

- **peak displacement** — the *farthest* the loop got from the collapsed
  attractor after injection (max version-(b) distance). "How hard did the
  injection knock it off?"
- **displaced rounds** — how many post-injection rounds stayed above the
  "moved away" cutoff (0.70). "How long did the knock last before it slid
  back?" (a transient-length / time-to-re-collapse measure.)

Reported by size × timing, with the **fresh-run baseline** (same injection into
a not-yet-collapsed loop) as the control: comparing collapsed-vs-fresh
displacement is how we estimate the **pull (basin depth) of the attractor**.

### 4.5 Injection quality

- **injection distance** — cosine distance (text-embedding-3-large) between the
  injected snippet and the recent conversation; we pick the *farthest of 8*
  sampled snippets so every injection is genuinely off-topic (current corpus
  gives ~0.95, vs ~0.84 for the old same-domain injections).

---

## 5. Reference papers & positioning

- **Maiti et al. (arXiv:2512.06256)** — two LLMs converge into repetition; source
  of our threshold-on-consecutive-low-distance detection rule.
- **Kong, Lai, Piao & Evans (U. Chicago)** — closed-loop multi-LLM systems show
  robust semantic collapse; **twelve** interventions fail to restore lasting
  diversity. This is the strongest argument *against* the old framing and the
  reason for our reframe. If our injections don't produce lasting recovery, that
  is *consistent with the literature*, not a failure.
- **Shumailov et al. (Nature 631, 2024)** — training-time model collapse; a
  *different mechanism*, cited to be explicit we are not conflating the two.

---

## 6. What's next

**Immediate:** relaunch a small **clean pilot** (1–2 models) on the fixed
pipeline to confirm end-to-end, then the fuller grid as endpoint windows allow
(the runner resumes across windows, saving after every run).

**Before the publishable study:** recalibrate the collapse cutoff on the exact
deployed signal + sensitivity sweep (#5); expand judge validation (second
labeler, confidence intervals) (#7); pre-register one primary contrast and run
≥20 seeds there (#8); matched token-budget for the size axis (#10).

**Phase 2:** multi-agent (two models talking, e.g. Llama-70B ↔ Qwen-72B),
including a "you are talking to someone else" framing that is *truthful* in the
multi-agent setting (it is a false premise in the self-loop, so it is held out
of Phase 1).

---

## 7. Open questions for you

1. Do you agree with the **reframe** (dose–response of the transient / basin
   depth) as the primary contribution, rather than "lasting recovery"?
2. For statistics: is one **pre-registered primary contrast** (e.g. peak
   displacement: large real injection vs fresh-run baseline, after collapse) the
   right headline, with the rest as exploratory?
3. Priority of **Phase 2 (multi-agent)** vs. deepening the single-model
   characterization first?
