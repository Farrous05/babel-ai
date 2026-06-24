# References & Inspiration

This work builds directly on three papers. We summarise each and note how it
shapes our methods. (Full texts are available at the links — please read them
there rather than relying on any local copy.)

## 1. Maiti et al. — *Convergence of Outputs When Two Large Language Models Interact in a Multi-Agentic Setup*
arXiv:2512.06256 (2025).

Two independently-trained LLMs respond to each other for many turns from a seed
sentence; most conversations start coherent and then **converge into
repetition**. They quantify this with cosine distance, Jaccard distance, BLEU,
and a coherence (NPMI) change, and **detect collapse by thresholding the
per-step distance** — flagging convergence when the distance stays low for a
few consecutive steps (window size 3), with cutoffs calibrated against
hand-labelled runs.

**How it shapes us:** this is the source of our collapse-detection rule
(threshold-on-consecutive-low-distance) and our metric choices (cosine as the
primary signal, Jaccard secondary, perplexity as a quality guard). Their #1
stated future direction — *external intervention to help a conversation escape
low-diversity regions* — is exactly our injection→recovery experiment. They
also motivate the multi-agent extension and the seed-source axis. See
[methodology_notes.md](methodology_notes.md) §1–2 for where we follow them and
where we deliberately deviate (windowed vs. consecutive distance).

## 2. Kong, Lai, Piao & Evans — *Multi-LLM Systems Exhibit Robust Semantic Collapse*
Knowledge Lab, University of Chicago (preprint).

Closed-loop multi-LLM systems exhibit **semantic collapse — systematic
convergence in meaning despite apparent lexical variation** — across model
families and 200–1000-round simulations. Twelve intervention strategies
(decoding parameters, prompt design, agent composition, activation engineering,
RL) **fail to restore lasting semantic diversity**. They frame the question via
Turing's "injection": can a system *amplify* an injected idea (supercritical) or
fall back to quiescence (subcritical)?

**How it shapes us:** it validates using **semantic cosine** as the primary
signal (lexical variation alone hides the convergence), frames our injection
study as the supercritical/subcritical question, and sets expectations — if our
injections often fail to produce lasting recovery, that is *consistent with the
literature*, not a bug.

## 3. Shumailov et al. — *AI models collapse when trained on recursively generated data*
Nature 631 (2024).

Training a model on its own (and predecessors') generated data across
generations causes **"model collapse"**: the tails of the distribution
disappear and outputs converge to a low-variance point estimate.

**How it shapes us:** important framing, but a **different mechanism** —
training-time degradation across generations, not conversational/inference-loop
repetition. We cite it to be explicit that our "collapse" is the
Maiti/Multi-LLM kind (within a single conversation), distinct from Shumailov's
training-time collapse, so the two are not conflated.
