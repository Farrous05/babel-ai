# Metrics & How We Define Collapse

Reference for the diversity/quality metrics this project uses and the rule that
turns them into a binary "the loop has collapsed" decision. **The point of these
metrics is the project's real goal — characterising the model's *behaviour when
we inject off-topic text* after collapse** (does it move away from the attractor,
parrot the injection, or ignore it and keep repeating?). Collapse detection is
the prerequisite: it tells us *when* to inject and gives us the attractor to
measure distance from. The raw metrics are computed in
[`src/babel_ai/analyzer.py`](../src/babel_ai/analyzer.py) (`SimilarityAnalyzer`)
and stored per turn in the `analysis` column of each run's CSV. The collapse
rule below is implemented in
[`collapse.py`](../src/babel_ai/collapse.py); the constants were calibrated on
early runs (see §5).

---

## 1. The raw metrics (per turn)

Each generated turn is scored against the conversation so far. Two fixed
reference models are used and held constant across every run:

- **Embedding model**: `all-MiniLM-L6-v2` (Sentence-BERT) — for semantic
  similarity.
- **Perplexity reference model**: `gpt2` — a *separate* model from the loop
  model (`gpt-4o-mini`), so perplexity is an independent quality judge.

| Metric (CSV key)            | Range        | One-line meaning |
|-----------------------------|--------------|------------------|
| `word_count`                | ≥ 0          | length of the turn in words |
| `unique_word_count`         | ≥ 0          | number of distinct words in the turn |
| `coherence_score`           | 0 – 1        | `unique_word_count / word_count` — vocabulary variety *within* a turn |
| `token_perplexity`          | ≥ 1 (or ∞)   | GPT-2 perplexity of the turn — how "surprised" GPT-2 is by the text |
| `lexical_similarity`        | 0 – 1        | Jaccard word overlap vs the **previous** turn |
| `semantic_similarity`       | -1 – 1       | Sentence-BERT cosine vs the **previous** turn |
| `lexical_similarity_window` | 0 – 1        | mean Jaccard overlap vs the previous **W** turns |
| `semantic_similarity_window`| -1 – 1       | mean cosine vs the previous **W** turns |

### Detail per metric

**`word_count` / `unique_word_count`** — Plain counts after whitespace
tokenization. Building blocks for `coherence_score`; also useful for spotting
degenerate turns (e.g. a turn that is one word repeated).

**`coherence_score = unique / total`** — Intra-turn vocabulary variety. A turn
that says "yes yes yes yes" scores low; a turn with all distinct words scores
near 1. Note: this measures variety *inside a single turn*, **not** repetition
*across* turns — which is why (see §5) it turned out to be a poor collapse
signal: a model can repeat the same *idea* every turn using fresh words and keep
`coherence_score` high.

**`token_perplexity`** — Each turn is tokenized and scored under GPT-2;
`perplexity = exp(-mean(log p(token)))`. Low = fluent/predictable text; very
high = gibberish. We use it only as a **quality guard** (is the text still
sane?), not as a collapse trigger. Long texts are chunked to GPT-2's context
window before scoring.

**`lexical_similarity` (Jaccard)** — Treat the current turn and the previous
turn as word *sets*; `Jaccard = |intersection| / |union|`. 1 = identical word
sets (heavy verbatim repetition), 0 = no shared words. It is purely surface
overlap — synonyms count as different.

**`semantic_similarity` (cosine)** — Embed the current and previous turn with
Sentence-BERT, take cosine similarity. High (→1) = the two turns *mean* the same
thing even if worded differently; low (→0) = different meaning. Catches
synonym-disguised repetition that Jaccard misses.

**`*_window` variants** — The same Jaccard / cosine, but averaged over the last
`W` turns instead of just the immediately previous one (`analyze_window` in the
config; we use **W = 10**). Windowing smooths out the turn-to-turn alternation
that appears when a loop collapses into an A/B/A/B cycle (see §4), so the
windowed signal is the steadier one.

---

## 2. Derived distances

The metrics above are **similarities** (high = repetitive). For detection we
work with **distances** (high = diverse, low = collapsed), which read more
naturally and match the referenced paper's thresholding rule:

```
jaccard_distance  = 1 - lexical_similarity
cosine_distance   = 1 - semantic_similarity
```

We also distinguish *two ways* of measuring cosine distance:

- **(a) Turn-to-turn / windowed** — distance between recent turns. Shows whether
  the model keeps saying the same thing. Used to *detect* collapse.
- **(b) Distance from the collapsed window** — distance of each post-injection
  turn from the cluster of turns *right before* an injection. **This is the
  central injection-behaviour signal:** it shows the system *moving away from
  the attractor* — recovery is distance (b) growing and staying grown. It is
  logged per round (`version_b_distance` in the CSV) and drives both the
  recovery evaluation and the behaviour analysis
  (ignore / parrot / blend / escape).

---

## 3. What each metric says about collapse

| Signal                         | Collapsed loop looks like… |
|--------------------------------|-----------------------------|
| `semantic_similarity` (cosine) | **high / cosine distance low** ← primary signal |
| `lexical_similarity` (Jaccard) | usually high, but can stay low under synonym variation |
| `coherence_score`              | unreliable — may stay high or even rise |
| `token_perplexity`             | stays in a sane band (guard, not a trigger) |

The headline signal of collapse is **rising semantic similarity = falling
cosine distance** sustained over turns. Perplexity is the guardrail that
distinguishes "collapsed into a stable repeated phrase" (sane, low perplexity)
from "collapsed into gibberish" (which we would *not* count as a clean result).

---

## 4. How we define collapse

We adopt the **threshold-on-consecutive-low-distance** rule (from the referenced
convergence-detection paper), calibrated against hand-labeled runs:

> Compute the distance signal per turn, smoothed over a sliding window **W**.
> Declare the loop **collapsed** at the first turn where the smoothed distance
> has stayed **at or below a cutoff for K consecutive turns**.

Concretely, for this project:

- **Primary distance signal**: **windowed cosine distance**
  = `1 - semantic_similarity_window` (W = 10). Chosen over turn-to-turn because
  collapse here is typically a 2-cycle (the model alternates between two stock
  replies), which makes the turn-to-turn signal *alternate* high/low while the
  windowed average stays stably low. (See §5.)
- **Secondary signal (kept, not a trigger)**: **windowed Jaccard distance**
  = `1 - lexical_similarity_window`. Semantic is the trigger, but lexical is
  tracked and logged alongside it — it corroborates collapse and flags the
  synonym-varied case where lexical stays high while semantic collapses.

### Locked constants (single source of truth)

| Constant | Value | Notes |
|----------|-------|-------|
| Window `W` | **10** | sliding window for the windowed metrics |
| Persistence `K` | **3** | consecutive turns below cutoff to declare collapse (was 5; lowered 2026-06-24 to match Maiti's window-size=3 and to declare collapse sooner, leaving more room to observe the post-injection response) |
| Cosine cutoff | **0.40** | windowed cosine *distance* ≤ 0.40 (windowed cosine *similarity* ≥ 0.60) |
| Jaccard cutoff | **0.40** | windowed Jaccard distance — informational/secondary only |
| Warm-up | **10** | no collapse declared before this round, so the W-turn window has filled and the windowed signal is trustworthy (avoids premature collapse in the unstable first rounds) |
| Re-arm cutoff | **0.50** | hysteresis upper threshold for repeated injection (see below) |

### Re-arming after an injection — hysteresis, not a cooldown

For **repeated** injection (re-inject on each re-collapse), the detector must
decide *when a re-collapse counts*. We use a **Schmitt-trigger / hysteresis**
rule rather than a fixed time cooldown:

- After an injection the detector is **disarmed**: it will not declare a new
  collapse until the windowed cosine distance first rises **above the re-arm
  cutoff (0.50)** — i.e. the loop has visibly left the attractor.
- Only then can a fresh dip below 0.40 (for K rounds) declare the next collapse.

Why: right after an injection the W-turn window still contains pre-injection
collapsed turns, so it reads *low*; a plain detector would re-fire off that stale
dip. The old fix was a fixed `window`-round cooldown, which **hid the true
re-collapse speed** (it forbade re-detection for ~10 rounds regardless). The
hysteresis rule replaces it: a re-collapse only counts if the loop *un-collapsed*
first, with no arbitrary wait — and a contaminated low window simply keeps the
detector disarmed instead of triggering a false re-collapse. The 0.50 cutoff
sits above the 0.40 collapse cutoff (forming the hysteresis band) and within the
observed diverse-start range. Implemented in
[collapse.py](../src/babel_ai/collapse.py) (`rearm`, `rearm_cutoff`); true
re-collapse latency is reported separately from the raw turn-to-turn signal.

The cosine cutoff is **0.40**, not the 0.34 first floated: 0.34 came from the
*turn-to-turn* midpoints, but once the signal moved to *windowed* cosine, the
three runs' collapsed plateaus sat at distance 0.26 / 0.32 / 0.36 and their
diverse starts at 0.47–0.71 — so the cutoff must be ≥ 0.36 (to catch the
highest plateau) and < 0.47 (to not fire on the diverse start). 0.40 sits in
that band. These are the defaults the detector ships with; validating them
against more hand-labeled onsets at scale — and the windowed-vs-consecutive
detector choice — is still open (see
[methodology_notes.md](methodology_notes.md) §2).

When collapse is first declared we record two things:

- **TTC (time-to-collapse / collapse-onset round)** — the round at which
  collapse is declared (the end of the first qualifying streak).
- **Collapse rate** — how fast it got there: the **linear-regression slope of
  semantic similarity rising from round 0 to onset** (equivalently, the rate at
  which cosine distance falls toward the attractor). Higher slope = faster
  collapse.

Optional strong-confirmation check: flag **near-verbatim repeats** (identical or
very-high-Jaccard adjacent turns), which directly catch the verbatim 2-cycle
end-state.

### What does *not* define collapse
- `coherence_score` falling — not used; it is intra-turn and proved unreliable.
- A single low-distance turn — must persist for K turns (avoids one-round blips).
- Gibberish — high perplexity disqualifies a "collapse" from counting as a clean
  result; the guard keeps us measuring genuine repetition, not noise.

---

## 5. How these definitions were calibrated

The constants and the choice of signal above come from real runs of
[`configs/collapse_sharegpt_short.yaml`](../configs/collapse_sharegpt_short.yaml),
analyzed with
[`analysis/calibrate_collapse.py`](../analysis/calibrate_collapse.py). Key
observations that shaped the definition:

1. **Collapse is real and reproducible** — in the grid regime (full history +
   brevity prompt) every run converges to a "social closing" loop (farewells /
   affirmations), often near-verbatim. **Note:** the *shape* of the attractor is
   regime-dependent — with last-message feeding + free generation the model
   instead converges into long, repetitive elaboration (e.g. code tutorials).
   See [observations_log.md](observations_log.md).
2. **A truncation artifact had to be removed first** — with a plain prompt and a
   token cap, turns were cut mid-sentence and the next turn just *continued the
   previous sentence*, turning the self-loop into plain text-continuation. The
   grid regime fixed this with a brevity instruction ("reply in at most 3
   sentences, always finish your final sentence"). **Update:** later experiments
   dropped the brevity prompt for *free generation* with a large token cap (plus
   last-message feeding); the same truncation artifact returns if the cap is too
   low, so the cap must be large enough that turns finish on their own. See
   [observations_log.md](observations_log.md).
3. **Windowed cosine beats turn-to-turn** — because collapse is a 2-cycle,
   turn-to-turn cosine alternates and is noisy across seeds (one run barely
   moved, 0.72 → 0.80), while windowed cosine rose cleanly in *all* runs
   (~0.30 → 0.65–0.74). Hence windowed is the primary detector signal.
4. **Semantic, not lexical, is primary** — one run collapsed *semantically*
   while staying lexically diverse (synonym-varied farewells), so Jaccard alone
   would have missed it. This matches the paper's finding that cosine distance
   gives the clearest signal.
5. **No single absolute cutoff is perfect across seeds** (runs settle at
   windowed similarity 0.64–0.74), but windowing narrows the spread enough for
   the **0.40 cosine-distance cutoff with K = 3** to fire at sensible onsets
   (~round 10–25), with collapse-rate slopes ≈ +0.008–0.013.
