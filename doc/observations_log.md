# Observations Log

A running, dated log of empirical observations worth remembering — things we
noticed but haven't fully explained yet. Newest first. When an entry is
explained or acted on, note the resolution inline. Companion to
[methodology_notes.md](methodology_notes.md) (decisions) and
[project_summary.md](project_summary.md) (results).

---

## 2026-07-03 — "This conversation has just begun" resets are a no-memory artifact

**Observation.** In the `follow_new` probe (Llama-3.3-70B, `after_collapse`
injection vs its floor), **both** runs ended with the model *denying the
conversation ever happened*:
- inject run, final turn: *"I must correct you — this conversation has just
  started, and I haven't provided any previous information about QAT…"*
- floor run, final turn: *"It seems there's been a misunderstanding. This
  conversation has just begun…"*

Both then re-collapsed straight back into the same topic (MPS / batch size).
It is **not** injection-driven — the no-injection floor does it too.

**Mechanism (why both do it).** A direct consequence of `history_window = 1`
(each turn the model is fed only the *single* previous message, relabelled as
the user). As the loop collapses it degenerates into **self-referential
meta-commentary** — turns like *"Your comprehensive overview of batch size is
thorough…"* that **reference a prior answer**. On the next turn the model sees
only that praise message and has **no access to the "overview" being
referenced** (memory is one message deep), so it concludes *"I never gave that
— this must be a misunderstanding, we just started."* The model is being
*consistent*, not broken: a message referencing invisible history reads as an
error, so it "resets" — then re-collapses.

**Why it matters / prediction to test.** This is exactly what the **memory arm**
(`history_window = None`, the `two_memory` multi-agent scenarios) should change:
with the whole conversation visible the model *cannot* claim "this just started."
**Prediction: full-memory runs eliminate the "conversation just began" resets.**
If they do, it confirms the resets are a memory artifact of the self-loop, not a
property of the models; if they persist, the reset is deeper than context
visibility. Check when the multi-agent suite runs.

**Related:** injection was read as an error too — turn 14 of the inject run
reacted to the injected off-topic block with the same "this is a
misunderstanding" move before snapping back, i.e. the model treats an abrupt
appended injection as noise to recover from, not a thread to follow (why the
`follow_new` prompt did not produce recovery).

---

## 2026-07-01 — Qwen 2.5-7B switches to Chinese mid-run; English metrics go blind

**What.** In 5 of 63 Qwen 2.5-7B CSV files, the model starts producing Chinese-script output mid-run and largely stays there. Discovered by scanning all Qwen CSVs for CJK Unicode characters — this was not flagged anywhere in the pipeline at the time these runs were produced.

**The 5 affected runs:**

| Run | Temp | Condition | First Chinese round | % rows in Chinese |
|-----|------|-----------|--------------------|--------------------|
| longrun batch_20260630_114214, paragraph-real | 1.0 | inject after collapse | 11 | ~59% (3566/6006 rows) |
| longrun batch_20260630_130136, paragraph-real | 1.0 | inject after collapse | 10 | ~44% (36/81 rows) |
| grid inject-sentence-noise (seed 1) | 0.7 | inject after collapse | 41 | ~26% (79/307 rows) |
| grid fresh-sentence-noise (seed 1) | 0.7 | fresh injection | 12 | ~15% (46/297 rows) |
| grid inject-word-real (seed 2) | 0.7 | inject after collapse | 12 | <1% (1 row only) |

**Pattern.**
1. **Temperature predicts onset speed.** At temp 1.0 the switch happens at round 10–11 (immediately after warmup); at temp 0.7 it takes until round 12–41 or is a single-row blip.
2. **Once switched, it stays.** In the t1.0 runs and the sentence-noise run the model locks into Chinese and does not return — the language switch is itself a new attractor, not a one-off slip.
3. **Sentence-noise may be a trigger at t0.7.** Two of the three t0.7 cases are sentence-noise injections. A shuffled-word injection provides no coherent English signal to respond to, which may make Chinese fallback more likely.
4. **The switch is visible in the metrics.** Fully-Chinese turns show `word_count: 1` (the English tokenizer treats the whole sentence as one token), so Jaccard, coherence score, and perplexity all break down silently — the pipeline does not detect or flag this.

**Consequence.** Any "recovery" or "no-recovery" verdict on the rows after the switch is unreliable: cosine distance rises (Chinese text is far from the English cooking centroid), which can spuriously clear the 0.70 recovery cutoff and look like recovery. These rows are exactly the kind of false positives criterion 7 (language guard in `language.py`) was designed to kill — but criterion 7 was not yet active when these runs were produced.

**Resolution.** Criterion 7 added to `recovery.py` (see session 2026-06-30). These specific runs predate that fix and should be treated as pre-fix artifacts; re-run before drawing conclusions from them.

---

## 2026-06-30 — Reference embedder tested: large is justified but modest; the 0.30 recovery cutoff is broken

**Why.** We had switched reference-based distances (version-(b)
distance-from-collapsed-window, injection distance) to
`text-embedding-3-large` calling it "strong" there — on *reasoning* only (its
same-domain compression hurts the turn-to-turn detector but shouldn't hurt
distance-to-a-far-reference). We had never measured it. Did so:
[analysis/calibrate_reference_embedder.py](../analysis/calibrate_reference_embedder.py).

**Setup.** Labeled set with known ground truth, no human labels needed. Run
`batch_20260626_173617` stayed in the Italian-cooking attractor for all 297
turns (the round-11 tech injection was ignored — we read it end-to-end). So:
NEAR (in-attractor) = 40 post-injection cooking turns; FAR (off-topic) = 39
passages (Swift/3D-graphics grid run + ShareGPT snippets + the injection text).
Centroid = collapsed window (rounds 2-11), exactly as recovery.py builds it.
Distance = `1 - cos(text, centroid)` under each embedder.

**Result.**

| | NEAR mean | FAR mean | gap | AUC | % NEAR above 0.30 cutoff |
|---|----|----|----|----|----|
| SBERT all-MiniLM-L6-v2 | 0.486 | 0.935 | +0.450 | **1.000** | **95%** |
| text-embedding-3-large | 0.308 | 0.839 | +0.532 | **1.000** | **52%** |

**Two findings.**
1. **"Strong" holds for large — but SBERT is equally strong at ranking** (both
   AUC 1.000). The earlier implication that we *needed* large for reference
   distance was overstated. The real, narrower justification: large keeps
   in-attractor turns *tighter* to the centroid (0.31 vs 0.49) — the same
   compression that ruined turn-to-turn separation gives cleaner headroom here.
   So keep large, but the advantage is modest, not categorical.
2. **The recovery `distance_cutoff=0.30` is miscalibrated (the bigger finding).**
   On a run that *never recovered*, 52% (large) / 95% (SBERT) of turns clear
   0.30 and would be flagged "moved away." Clean separation sits at ~0.7 (best
   threshold 0.715 large / 0.807 SBERT). This quantifies the long-noted
   "generous recovered flag" — the distance criterion is far too loose and
   inflated past "recovery" counts. **Action: raise cutoff toward ~0.65-0.70.**

**Caveat.** One attractor, and cooking-vs-tech is a *large* topical gap (so AUC
1.0 is partly "easy mode"). The hard regime — a turn that genuinely recovered to
a *related* topic vs. one still circling — is untested, because we have no
human-labeled genuine recovery to calibrate against (nothing clearly recovered).

---

## 2026-06-29 — Detector blind spot: discourse-level collapse is invisible to cosine & Jaccard (LLM-judge added)

**The finding.** Collapse has a third axis our metrics don't cover. Cosine
catches *topical* repetition and Jaccard catches *lexical* repetition, but
neither catches **discourse collapse**: the model repeating the same
*conversational move* every turn ("enthusiastically affirm + build on your
idea") while *varying* the topic and the wording. Topic drift keeps cosine
distance high; synonym/topic variation keeps Jaccard distance high — so both
read "diverse" while a human instantly sees a loop.

**Evidence.** Llama 3.3 + a "keep the conversation going" system prompt:

| run | metric verdict | LLM-judge |
|-----|----------------|-----------|
| system-prompt, 30 rounds (t0.5) | **no collapse** | 0.84 (collapsed) |
| system-prompt, 60 rounds (t0.2) | collapsed only at **round 32** | 0.88 (collapsed from the start) |

So the metric is a **lagging, partial** indicator of discourse collapse: it
fires late (round 32) or not at all, while a human/judge sees it immediately.
Even first-sentence Jaccard missed it (0.83) because the template varies its
words ("glad/thrilled/fascinated", "guanciale/porcini") — only the *speech act*
repeats.

**The tool.** Built `analysis/llm_judge.py` — a gpt-4o-mini judge that slides a
window over a run and scores pattern-repetition *ignoring topic*. Validated:
0.84–0.9 on collapsed runs, ~0.5 on diverse — real separation, but it
**over-flags** (called a genuinely-diverse control "repetitive" at 0.5). So use
the **score with a ~0.7 threshold**, not the model's boolean, and **calibrate
the threshold against a few human labels** (the detector-validation step flagged
open in [methodology_notes.md §2](methodology_notes.md)).

**Correction to a prior claim.** "Llama 3.3 resists collapse / sustains
diversity" (2026-06-29 pilot below) was likely an **artifact**: Llama was
collapsing at the discourse level the whole time, invisible to the metric. The
"%>0.40 sustained" measured *topical* drift while it sat in one conversational
move. Revised view: **both models collapse — gpt-4o-mini topically (visible),
Llama discourse-ally (invisible to the metric).** Neither is more resilient; the
instrument saw one and not the other. Relatedly, the system prompt did **not**
prevent collapse — it *converted* a detectable topical collapse into an
undetected discourse one.

**Implication.** Don't discard the metrics — *complete* them. Cosine/Jaccard are
valid for topical/lexical collapse (regime: base / brevity-prompt); add the
**judge** for discourse collapse (regime: instruct + free-generation + frame);
report all three and state which each catches.

---

## 2026-06-29 — Clean injections expose distinct behaviours; recovery is transient + high-variance; Llama pilot

**Injection-fragment bug (fixed).** The new size-matched conditions
(`output_sized` = inject ≈ the model's own last-output length; `skim_half` =
keep the first half of the output's sentences and *replace* the dropped half
with equal-size off-topic text) trimmed the injection to an exact word count,
leaving a **mid-sentence fragment** — which the self-loop simply *completed*
instead of responding to. Fixed: injections are built from whole sentences and
always end on terminal punctuation (the ShareGPT corpus is code-heavy, so a
period is appended when a snippet ends in `}`/`)`). Verified 0/36 unclean.

**With clean injections, the three conditions behave distinctly** (gpt-4o-mini,
1 seed, read from transcripts): `paragraph` → *captured* (continues the injected
story); `output_sized` → *meta-recognition* ("you've shared a variety of
topics") then re-collapses into a helper loop; `skim_half` → *ignored* (the
retained half re-anchors it to its own topic). **Caveat:** each condition drew a
*different* random snippet (fiction vs. game manual vs. XML), so behaviour is
confounded by injection *content* — not yet attributable to size/structure.

**Recovery is transient and high-variance.** Across the 3×3 (temp × condition,
1 seed) grid every cell briefly crossed the 0.50 re-arm threshold then
re-collapsed — recovery happens but never holds. And the *same* config
(paragraph/0.7/seed0) recovered in a 30-round run yet *never* recovered in the
300-round run (peak 0.27): **huge run-to-run variance** from OpenAI's
best-effort seed. ⇒ condition/temp differences are within the noise at n=1;
**seed replication is required** before any of them can be trusted.

**Next: second model.** Llama 3.3 70B wired via the company OpenAI-compatible
vLLM endpoint (`OLLAMA_COMPANY`). Smoke confirms it **does collapse** in a
self-loop (onset ~round 10). A 9-run pilot (3 conditions × 3 temps × seed 0) is
running to compare its injection behaviour against gpt-4o-mini.

---

## 2026-06-26 — Hysteresis reveals the loop never un-collapses; the old "re-collapse cycles" were an artifact

**Setup.** Same single paragraph run (gpt-4o-mini, temp 0.7, seed 0, 300 rounds,
hand-picked cooking seed now ending on a *human question*), but with the
detector's **hysteresis re-arm** (re-arm only after windowed cosine distance
climbs back above 0.50) replacing the fixed 10-round cooldown. Batch:
`results/longrun/batch_20260626_173617`. Baseline with the old cooldown (same
seed/size): `batch_20260626_145904` (27 injections).

**Observation.** With hysteresis the run produced **1 collapse and 1 injection
(round 11), then nothing for the remaining 289 rounds** — because the model
**never recovered**. After the injection the windowed cosine distance peaked at
**0.267** and never once crossed even the 0.40 collapse cutoff (0/285 rounds
above 0.50). By our own definition the loop was *continuously collapsed* for the
whole run after round 11.

**What the injection actually did.** It was **ignored** (content-word overlap
with the off-topic tech paragraph ≈ 0.03–0.10) and the topic stayed cooking the
entire run. version-(b) distance drifted up (0.10 → ~0.50), but that is
*migration between equally-collapsed attractors* ("Absolutely!…" → "Salute!…" →
"Buon appetito!…" → "Thank you for your heartfelt reflections…"), **not**
restored diversity. Perplexity stayed 10–40 throughout (fluent repetition, never
gibberish).

**Why it matters.**
- The old cooldown's "27 injections / 27 re-collapses" were an **artifact**: it
  blindly re-injected every 10 rounds into a loop that had never recovered. The
  honest picture is **one collapse that never lifts**.
- Strongest confirmation yet of Multi-LLM's "interventions fail to restore
  diversity": here a single paragraph failed to restore *any* diversity, even
  for one turn.
- Separates two questions the old code conflated: **(i) does one injection
  recover the loop?** — no, decisively; **(ii) what does sustained forced dosing
  do?** — it *migrates* the attractor over many doses (the cooking→tech walk in
  the 2026-06-25 batches) without ever un-collapsing it.
- Single dose is **ignored** (stays on topic); only *cumulative* dosing migrates
  the topic. So injection effect is dose-dependent, and "recovery" never occurs.

**Caveat.** Graph note: the version-(b) "recovery" panel draws the 0.40 line,
which is the *turn-to-turn* collapse cutoff, not a recovery threshold for
version-(b); crossing it here means "left the origin," not "un-collapsed."
Relabel pending. One seed.

---

## 2026-06-25 — Injection size controls *escape*, but nothing buys *lasting* diversity

**Setup.** Three long self-loop runs (gpt-4o-mini, temp 0.7, seed 0, 300 rounds,
`history_window=1`, no system prompt, 2048-token cap), one per injection **size**
(word / sentence / paragraph). Seed is a hand-picked *non-technical* topic
(spaghetti carbonara) so injections drawn from the tech ShareGPT corpus are
genuinely off-topic. Injection is automatic on every (re-)collapse
(`after_collapse` + `repeat`, farthest-of-8). Batch:
`results/longrun/batch_20260625_100341/`.

**Observation 1 — size monotonically controls escape from the original
attractor** (version-(b) distance, measured from the *first* collapsed window):
- **word** — slow climb, plateaus ~0.65: only ever a *partial* escape.
- **sentence** — stuck low (~0.3–0.5) for ~40 rounds / several injections, then
  breaks to ~0.9 and holds.
- **paragraph** — jumps to ~1.0 by round ~35 and holds 0.8–1.0: fast, near-total
  displacement.

Bigger injection ⇒ escapes faster, farther, and more durably. Parroting
(content-word containment) scales the same way: paragraph injections produce a
**sawtooth blend** (containment spikes to 0.6–0.85 right after each injection,
then decays to ~0.1–0.3 as the next loop forms); word injections move it only
coarsely (1 content word).

**Observation 2 — no size restores *lasting* diversity.** All three keep
re-collapsing for the whole run (word 30, sentence 29, paragraph 26 collapse
onsets ≈ one every ~10 rounds). Injection never *stops* the loop; it *relocates*
it to an ever-more-distant new attractor. Consistent with Multi-LLM's finding
that interventions fail to restore lasting diversity.

**Why it matters.** This is the core injection-behaviour result for the
single-model self-loop: the spec's "smallest size that recovers" has no clean
answer here because *recovery* (diversity that holds) never happens — what size
buys is *displacement*, not *escape from collapse as a regime*. The right
summary metric is "how far / how fast does it leave the original attractor",
which size clearly controls, rather than a binary recovered yes/no.

**Caveats.** One seed only (seed 0) — needs replication across seeds before
claiming the size ordering is robust. Free-gen attractor here is an upbeat
**emoji/affirmation** style (🌱 ✨ !); this broke the old truncation heuristic
(end-punctuation check) — fixed to flag only mid-word (alphanumeric) endings,
after which truncation is 0–1/296 (no real truncation; max turn 1023 words, well
under the 2048-token cap).

**Status: clean result, pending seed-replication.** Next: same three sizes on a
bigger Ollama model (to discuss); optionally the fresh-run baseline once the
single-model picture is locked.

---

## 2026-06-24 — The *shape* of collapse depends on the input/length regime

**Observation.** When we switched the self-loop from *full-history + brevity
prompt + small token cap* (the 60-round grid setup) to *last-message-only input
(`history_window=1`) + no system prompt + large token cap (2048)*, the
collapsed state changed character entirely:

- **Old regime (grid):** gpt-4o-mini converged into short **"social closing"
  loops** — repeated farewells/affirmations ("Thanks! Take care!"), a short
  A,B,A,B cycle.
- **New regime (long run):** starting from a coding seed, gpt-4o-mini converges
  into **long, repetitive code tutorials** — it keeps elaborating the same
  coding topic at ~500-word turns. Still a *semantic* collapse (the detector
  fires; turns mean the same thing each round), but the attractor is verbose
  technical elaboration, not short farewells.

**Why it matters.**
- The changes we made for faithfulness/practicality (last-message feeding, free
  generation) did **not** just change cost/fidelity — they changed **what
  collapse *is*** for this model. So long-run results are **not directly
  comparable** to the old grid's "smallest size that recovers" numbers.
- It suggests the attractor is **seed-/regime-dependent**: with only the last
  message in context and room to write, the model rides the seed's topic and
  elaborates rather than winding down into closings. Possibly the brevity
  prompt + full history was what pushed it toward "wrap-up" behaviour.

**Status: unexplained — revisit later.** Candidate explanations to test:
- Is it the **last-message feeding** (no long history to "wind down" from), the
  **removed brevity prompt** (no pressure to be short/closing), the **larger cap
  (room to elaborate)**, or the **seed topic** (coding → code)? Disentangle by
  toggling one factor at a time.
- Does it reproduce across **seeds / seed sources** (non-coding seeds)? If a
  Wikipedia/news seed gives a different attractor, the attractor is
  seed-driven; Maiti studied exactly this (seed source × convergence).
- Relates to the windowed-vs-consecutive detector question
  ([methodology_notes.md §2](methodology_notes.md)): a different collapse shape
  may favour a different detector.
