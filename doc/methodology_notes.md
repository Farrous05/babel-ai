# Methodology Notes — paper coherence & key decisions

Why our method looks the way it does, how it lines up with the literature, and
three decisions made on 2026-06-24. Companion to
[metrics_and_collapse.md](metrics_and_collapse.md) (exact metric defs),
[future_work.md](future_work.md) (roadmap), and the source papers in `doc/`.

---

## 1. How our approach lines up with the papers

We have three reference papers in `doc/`:

- **Maiti et al.** ([Maiti_et_al.md](Maiti_et_al.md)) — *"Convergence of Outputs
  When Two LLMs Interact in a Multi-Agentic Setup."* Two **base** models
  (Mistral Nemo, Llama-2-13B), no system prompt, raw text back-and-forth, 25
  turns, temp 0.7 / top-p 0.95, 50-token cap. Metrics: cosine distance, Jaccard
  distance, BLEU, coherence (NPMI). Collapse detected by **thresholding the
  per-step distance between consecutive outputs**, requiring it to stay low for
  **3 consecutive steps**; cutoffs set by comparison with **manually labelled
  runs**. *This is the source of our detection rule.*
- **Multi_LLM** ([Multi_LLM.md](Multi_LLM.md)) — Kong, Lai, Piao & Evans,
  *"Multi-LLM Systems Exhibit Robust Semantic Collapse."* Closed-loop multi-LLM
  systems show **semantic convergence despite lexical variation**, across 7
  model families and 200–1000 rounds. **Twelve** intervention strategies
  (decoding params, prompt design, agent composition, activation engineering,
  RL) **fail** to restore lasting semantic diversity. Frames the question via
  Turing's "injection": can a system *amplify* an injected idea (supercritical)
  or does it fall back to quiescence (subcritical)?
- **Shumailov et al.** ([Shumailov.md](Shumailov.md)) — *"AI models collapse
  when trained on recursively generated data"* (Nature 2024). **Different
  phenomenon:** training-time degradation when a model is retrained on its own
  synthetic output across generations. Thematically related (feedback loops
  degrade models) but **not** the conversational/inference-loop collapse we
  study. Keep for framing/citation; do not conflate.

**Coherence verdict.** Our design is squarely on this frontier:
- Semantic cosine as the **primary** signal, Jaccard secondary, perplexity as a
  quality guard — matches Maiti (cosine clearest) and is *validated* by
  Multi_LLM's "semantic convergence despite lexical variation" (word-overlap
  alone would miss it).
- Our injection→recovery experiment is exactly Maiti's stated #1 future
  direction ("external intervention to help a conversation escape low-diversity
  regions") and Multi_LLM's Turing supercritical/subcritical question.
- **Expectation-setting:** Multi_LLM found 12 interventions failed. So runs
  labelled "ignore" (injection had no lasting effect) are **consistent with the
  literature**, not a bug — a negative/￬"hard to escape" result is itself
  publishable.
- **Known deviations from Maiti:** we use one *instruct* model in a self-loop
  (not two base models), a brevity system prompt, a 150-token cap (not 50), and
  60 rounds (Multi_LLM uses 200–1000 — ours are short). And our detector uses a
  *windowed* distance, not a consecutive one (see Decision 1).

---

## 2. Decision 1 — windowed vs. consecutive collapse distance

**What the code does today.** The collapse trigger is the **windowed** cosine
distance: `1 - mean(cosine(current turn, each of the previous W=10 turns))`,
required below 0.40 for **K=5** consecutive rounds (after a 10-round warm-up).
See [analyzer.py](../src/babel_ai/analyzer.py) `_analyze_semantic_similarity`
and [collapse.py](../src/babel_ai/collapse.py).

**How that differs from Maiti.** Maiti thresholds the distance between
**consecutive** outputs (turn *i* vs *i−1*) and requires 3-in-a-row. We deviate
to a windowed average.

**Why we deviated.** Maiti's two *base* models collapse to a single repeated
phrase (a fixed point), so consecutive distance drops to ~0 and stays — a clean
signal. Our single *instruct* model tends to settle into a **2-cycle**
(alternating between two different stock closings), so consecutive distance
*zigzags* and never cleanly drops, even though the loop is clearly stuck.
Averaging over a 10-turn window flattens the ping-pong and exposes the
stuck-ness. (Documented in [metrics_and_collapse.md §5](metrics_and_collapse.md).)

**The honest caveat & the plan.** This was calibrated on only ~3 runs, so it is
lightly tested, and the windowing may be compensating for *our* setup choices
(instruct model + system prompt) rather than being fundamentally required. We
already log **both** the consecutive and the windowed distance. **Resolution:**
hand-label a sample of runs and check *which detector agrees with the human* —
separately for single-agent and for the multi-agent mode (which is Maiti-like
and may collapse to a clean fixed point where consecutive distance works again).
Pick the detector with evidence rather than by assumption. Tracked in
future_work §E.

---

## 3. Decision 3 — recovery/behaviour judging: metrics for scale, LLM-judge **plus** human for validation

The automated labels (recovered flag; ignore/parrot/gibberish/blend/escape in
[injection_behavior.py](../analysis/injection_behavior.py)) are **mechanical
thresholds** on the three metrics — objective and scalable, but blind to
nuance. They are the right tool for the big automated pass over many runs.

**Decision:** validate them on a *sample* with **both an LLM-as-judge and a
human check** — not to replace the metrics, but to confirm the thresholds label
runs the way a person would (continue / parrot / ignore / blend / escape). This
mirrors Maiti, who set their cutoffs *against manually labelled runs*. The spec
already lists an LLM-judge/NLI as the planned upgrade "if perplexity proves too
blunt." Tracked in future_work §C.

---

## 4. Decision 6 — sampling seed is for reproducibility, not for the science

We thread a request-level sampling `seed` (AgentConfig.seed). **The papers do
not use sampling seeds, and we should not rely on them scientifically.**

- **Why it was added:** to make real-vs-noise a *paired* comparison (same
  trajectory up to injection, only the injected text differs) — tighter signal
  from fewer runs.
- **Why that's weak:** OpenAI's seed is **best-effort**, so the two runs don't
  actually stay identical; the pairing we were buying doesn't fully materialize.
- **What the papers do instead:** **large N.** Maiti ran 50 seeds; Multi_LLM ran
  hundreds of rounds across 7 models, comparing *distributions/averages*. With
  enough runs you don't need pairing.

**Decision:** keep the seed only for **reproducibility/debugging** (re-running an
exact run to inspect it). For the science, **lean on large N** (cheap on the
company Ollama), comparing average recovery rates — like the papers. Note: local
Ollama can be made *truly* deterministic if exact reproduction is ever wanted,
unlike the hosted OpenAI API. Tracked in future_work §D.
