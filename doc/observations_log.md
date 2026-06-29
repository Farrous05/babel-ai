# Observations Log

A running, dated log of empirical observations worth remembering — things we
noticed but haven't fully explained yet. Newest first. When an entry is
explained or acted on, note the resolution inline. Companion to
[methodology_notes.md](methodology_notes.md) (decisions) and
[project_summary.md](project_summary.md) (results).

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
