# HANDOFF — babel-ai (injection study)

For the next agent picking up this work. Read this top to bottom **before
touching anything**, then read the four doc files called out in §7.

---

## 0. THE ONE THING YOU MUST NOT FORGET

**This project is an INJECTION study.** Everything else — collapse detection,
embeddings, the LLM judge, multi-agent, temperature sweeps — is *infrastructure
in service of one question*:

> When a self-looping (or multi-agent) LLM **collapses** into repetition, and we
> **inject** off-topic text, does it **recover** — and if so, what is the
> **smallest injection that makes recovery LAST**, and does **real text beat
> random noise**?

If you ever find yourself optimizing the judge, the embeddings, or the plotting
for their own sake, stop. The deliverable is a defensible answer to *smallest
injection that produces lasting recovery, real vs. noise*. Re-read this
paragraph whenever the work drifts.

---

## 1. Current honest status (READ THIS — a prior claim was retracted)

**We do NOT yet have clean evidence of "lasting recovery" anywhere.** An earlier
agent (me) claimed multi-agent + system-prompt produced the "first lasting
recovery." That claim was **re-examined against the data and withdrawn.** The
facts:

- The run it rested on (multi-agent + prompt, temp 0.5, batch
  `batch_20260629_175948`) **never deeply collapsed before injection.** Windowed
  cosine *distance* only dipped to ~0.30 (barely under the 0.40 cutoff) for ~3
  rounds, got injected at round 13, rose to 0.45–0.54 and held ~10 rounds, then
  started dipping again at r25–26. The run ended at round 26 — **too short** to
  tell if recovery lasts. n=1. And it leaned partly on the LLM judge, which we
  no longer trust (see §3).
- The case that **did** clearly, deeply collapse (multi-agent, no prompt, cosine
  ~0.10) showed only a **transient blip then re-collapse** after injection — not
  lasting recovery.

**Therefore the standing result across everything run so far:** injection
produces **transient recovery / displacement at best; "lasting" is unproven.**
This is consistent with Multi-LLM (12 interventions failed to restore lasting
diversity). A negative result is fine and publishable — but only if rigorously
shown, which it isn't yet.

**Why our runs can't answer it yet (the actionable diagnosis):**
1. Runs were often **too short** post-injection (27–30 rounds) — we cut off
   right where re-collapse might happen.
2. Some "collapses" were **shallow/borderline** (dipped just under 0.40, not a
   real attractor) — so "recovery" was meaningless.
3. Everything is **n=1** (single seed). OpenAI's seed is best-effort, and we saw
   the *same* config recover in a 30-round run but never in a 300-round run —
   huge run-to-run variance.

---

## 2. What is built and trustworthy (the instrument)

The pipeline runs end-to-end. Spec steps 1–6 exist:

- **Collapse detector** — [src/babel_ai/collapse.py](src/babel_ai/collapse.py):
  windowed cosine distance ≤ **0.40** for **K=3** consecutive rounds after a
  W=10 warm-up. Has **hysteresis re-arm** (`rearm_cutoff=0.50`): after an
  injection the detector disarms and only re-declares collapse once windowed
  distance climbs back above 0.50 (replaced the old fixed cooldown — the old
  cooldown's "27 re-collapses" were an artifact of blindly re-injecting every 10
  rounds into a loop that never recovered).
- **Injection** — [src/babel_ai/injection.py](src/babel_ai/injection.py):
  size × source. Sizes: WORD / SENTENCE / PARAGRAPH / MULTI_PARAGRAPH /
  OUTPUT_SIZED (≈ model's own last-output length) / SKIM_HALF (keep first half
  of sentences, replace dropped half with equal-size off-topic). Injections are
  built from **whole sentences ending in terminal punctuation** (fixed a bug
  where mid-sentence fragments got *completed* instead of responded to).
- **Recovery eval** — [src/babel_ai/recovery.py](src/babel_ai/recovery.py): four
  criteria over `hold_k=5` consecutive rounds — moved away from collapsed window
  (version-(b) distance ≥ 0.30) AND sane perplexity (GPT-2 ≤ 150) AND not
  parroting the injection (Jaccard ≥ 0.5). **Thresholds are provisional** and
  the `recovered` flag is **generous** (the no-intervention floor sometimes
  "recovers"). Trust the trajectory (mean distance, slope), not the binary flag.
- **Embeddings** — [src/babel_ai/embeddings.py](src/babel_ai/embeddings.py):
  **two embedders, on purpose.**
  - **SBERT `all-MiniLM-L6-v2`** for the **turn-to-turn collapse detector**
    (analyzer.py). We tried switching everything to `text-embedding-3-large`
    (to match Multi-LLM) and **reverted the detector** — large compresses
    same-domain distances and *hurt* turn-to-turn separation (collapsed/diverse
    bands overlapped on recalibration).
  - **`text-embedding-3-large`** (`reference_embedder()`) for **reference-based
    distances only** — version-(b) distance-from-collapsed-window and injection
    distance. **Tested** ([analysis/calibrate_reference_embedder.py](analysis/calibrate_reference_embedder.py),
    2026-06-30): on a run we know stayed collapsed, *both* embedders rank
    off-topic above in-attractor perfectly (AUC 1.000), so SBERT is not weak
    here — but large keeps in-attractor turns *tighter* to the centroid (mean
    0.31 vs SBERT 0.49), the same compression that hurt the detector, which
    here gives cleaner headroom. So large is justified, but the win is modest.
    Don't "unify" these without re-running that calibration.
  - **⚠ The recovery `distance_cutoff=0.30` is too low (same test).** On a run
    that NEVER recovered, 52% of turns (large) / 95% (SBERT) clear 0.30 and look
    "moved away." Clean separation is ~0.7 (best threshold 0.715 large / 0.807
    SBERT). Raise the cutoff toward ~0.65–0.70 before trusting the `recovered`
    flag. Caveat: tested only on one big topical gap (cooking vs tech); the
    related-topic near-miss boundary still needs human-labeled recoveries.
- **Long-run runner** — [analysis/run_longrun.py](analysis/run_longrun.py):
  writes per-metric PNGs into each run's own folder (1a windowed collapse, 1b
  raw collapse, 2 recovery, 3 parroting, 4 perplexity), reports per-injection
  latencies. CLI: `--model` (comma list; 1 = self-loop, 2+ = round-robin
  multi-agent), `--temps`, `--system-prompt`, `--sizes`, `--rounds`, `--seed`.
- **Grid runner / aggregator** — [analysis/run_grid.py](analysis/run_grid.py),
  [analysis/aggregate_grid.py](analysis/aggregate_grid.py).
- **PDF transcripts** — [analysis/pdf_generation.py](analysis/pdf_generation.py):
  marks ALL injections, no cosmetic "SCENE" breaks.

Metrics are verified textbook-correct (Jaccard and cosine checked by hand). ~30
unit tests, suite green except two known-unrelated failures (Azure budget
pricing, flaky topical-chat fetcher).

---

## 3. Things to be skeptical of (don't repeat past mistakes)

- **The LLM judge** ([analysis/llm_judge.py](analysis/llm_judge.py)) is **noisy
  and over-flags** — it called a genuinely diverse control "repetitive." We
  demoted it to a **pre-filter / secondary signal only**. Do NOT base a
  recovery claim on it. It exists because there's a real **discourse-collapse**
  blind spot (model repeats the same *conversational move* while varying topic +
  wording — invisible to cosine *and* Jaccard), but the judge's calibration is
  weak. If you rely on it, average multiple calls and validate against human
  labels first.
- **Three axes of collapse:** TOPICAL (cosine), LEXICAL (Jaccard), DISCOURSE
  (judge/human only). State which axis any claim is about.
- **n=1 is not a result.** Every single-seed claim is variance until replicated.
- **Don't conflate displacement with recovery.** Sustained forced dosing
  *migrates* the attractor (cooking → tech walk over many doses) without ever
  un-collapsing it. Bigger injection escapes farther/faster but doesn't restore
  lasting diversity.

---

## 4. The next experiment (designed, NOT yet approved/run)

Built to fix the §1 diagnosis. **Get the user's OK before launching.**

**Injection grid to answer the actual question:**
- **size × source × seeds** on `gpt-4o-mini` (cheap), then repeat on Llama 3.3
  70B / Qwen 2.5 72B via the company endpoint.
- **Confirm a collapse before injecting** (not a shallow boundary dip) —
  otherwise recovery is meaningless.
- **Long post-injection window** — enough rounds to see if diversity *holds* vs.
  re-collapses. 27–30 is too short.
- **Multiple seeds per cell** (the papers use large N: Maiti 50 seeds, Multi-LLM
  hundreds of rounds × 7 models). Compare average recovery rates, not paired
  single runs.
- **Include the fresh-run baseline** (`run_grid.py --fresh`) — collapsed-injection
  vs fresh-injection is *the* core comparison and is currently missing. Some
  "recovery" may just be the injected text being diverse on its own.
- **Real vs. noise** must be reported per cell.

Headline metrics to report: "how far / how fast does it leave the original
attractor" (version-(b) distance trajectory + slope) and a *durable* recovery
flag, NOT just the generous binary.

---

## 5. Hard constraints (NON-NEGOTIABLE — verbatim)

- **NEVER use temperature 0** (degenerate). Allowed temps: 0.2 / 0.5 / 0.7 / 1.0.
  (0.2 is fine, it is ≠ 0.)
- **The company Ollama/vLLM endpoint is CONFIDENTIAL.** Use env vars only —
  `OLLAMA_COMPANY_BASE_URL` / `OLLAMA_COMPANY_API_KEY`, and per-model
  `LLAMA_3_3_70B_BASE_URL` / `_API_KEY`, `QWEN_2_5_72B_BASE_URL` / `_API_KEY`,
  etc. (resolution in [src/api/company_ollama.py](src/api/company_ollama.py)).
  **Never hardcode or commit any URL or key.** vLLM = one model per endpoint.
- **No Claude as co-author.** Never add `Co-Authored-By: Claude` or any Claude
  collaborator line to commits in this repo. (All commits so far correctly omit
  it.)
- **Keep Jaccard** as a logged secondary signal (don't remove it).
- **This environment cannot `git push`** (no auth). The user pushes from their
  own terminal. Commit locally only when asked; current branch is
  `collapse-injection-pipeline`.

---

## 6. How to run

```bash
poetry env use python3.13      # one-time — torch has no wheels on 3.14
poetry install

# Self-loop long run, one size:
poetry run python analysis/run_longrun.py --sizes paragraph --rounds 300 --seed 0

# Multi-temp:
poetry run python analysis/run_longrun.py --sizes paragraph --temps 0.5,0.7 --rounds 60

# Multi-agent (round-robin over the listed models):
poetry run python analysis/run_longrun.py --model gpt-4o-mini,llama-3.3-70b --rounds 60

# With a system prompt (triggers discourse-collapse regime):
poetry run python analysis/run_longrun.py --system-prompt "..." --rounds 60

# Judge a finished run (secondary signal only):
poetry run python analysis/llm_judge.py <run_dir>
```

Output: timestamped `results/longrun/batch_<ts>/` folders, each run with its CSV,
`_meta.json`, and per-metric PNGs.

---

## 7. Required reading (in this order)

1. [doc/project_summary.md](doc/project_summary.md) — the question, what's built,
   the grid results, and the "infrastructure validated, signal not yet present"
   honest interpretation.
2. [doc/observations_log.md](doc/observations_log.md) — dated empirical findings,
   newest first: size controls *escape* not lasting diversity; hysteresis
   revealed the loop never un-collapses; clean-injection behaviours; the
   discourse blind spot.
3. [doc/methodology_notes.md](doc/methodology_notes.md) — why the method looks
   the way it does, paper alignment (Maiti / Multi-LLM / Shumailov), and the key
   decisions (windowed vs consecutive detector; seed is for reproducibility not
   science; judge is a separate signal but needs human calibration).
4. [doc/metrics_and_collapse.md](doc/metrics_and_collapse.md) — exact metric
   definitions, the collapse rule, the discourse axis + judge, the embedding-model
   split, rearm cutoff, K=3.

Also useful: [doc/mental_map.md](doc/mental_map.md) (how the code is wired),
[doc/how_to_run_experiments.md](doc/how_to_run_experiments.md), and the paper
notes [doc/Maiti_et_al.md](doc/Maiti_et_al.md),
[doc/Multi_LLM.md](doc/Multi_LLM.md), [doc/Shumailov.md](doc/Shumailov.md).

---

## 8. Open / deferred items

- Re-tune the version-(b) recovery distance cutoff for the
  `text-embedding-3-large` scale (it was set under SBERT).
- Multi-agent: split same-agent vs cross-agent distance (turn i vs i−1 is
  cross-agent = convergence; windowed mixes both).
- Seed-replicate the multi-agent cells before any multi-agent claim.
- Detector validation: hand-label runs, check windowed vs consecutive against
  the human (separately for single- vs multi-agent).
- Reduce judge noise (average N calls) — or keep it purely as a pre-filter.
