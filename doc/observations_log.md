# Observations Log

A running, dated log of empirical observations worth remembering — things we
noticed but haven't fully explained yet. Newest first. When an entry is
explained or acted on, note the resolution inline. Companion to
[methodology_notes.md](methodology_notes.md) (decisions) and
[project_summary.md](project_summary.md) (results).

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
