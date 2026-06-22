# babel-ai — Mental Map

A practical map of how this project is wired, so you can navigate it without
reading every file. For *running* it, see the bottom section.

---

## 1. What it does (one paragraph)

babel-ai studies **"LLM drift"**: what happens when a language model keeps
talking *with no fresh human input* — either to itself or to another model.
You give it a starting conversation, it loops the model on its own output for
N turns, and it measures how the text changes over time (repetition, semantic
similarity, perplexity, vocabulary variety). The research question: *do LLMs
need external input to stay coherent?*

---

## 2. The core loop (the one thing to understand)

Everything is orchestrated by **`Experiment`**. One run is:

```
  ┌─────────────┐
  │   FETCHER   │  pick a SEED conversation
  │ (data/web)  │  e.g. 2 real ShareGPT messages
  └──────┬──────┘
         │  seed messages
         ▼
  ┌─────────────────────────────────────────────┐
  │              INTERACTION LOOP                 │  (repeat until limit)
  │                                               │
  │   pick next AGENT  ──►  agent.generate()      │
  │        ▲                      │               │
  │        │                      ▼               │
  │   round-robin        append response to       │
  │   selection          message history          │
  │                                               │
  │   stop when: max_iterations OR max_total_chars│
  └──────────────────────┬───────────────────────┘
                         │  full message list
                         ▼
  ┌─────────────┐
  │  ANALYZER   │  score every message (similarity, perplexity, ...)
  └──────┬──────┘
         ▼
   results/<timestamp>.csv   +   <timestamp>_meta.json
```

Key insight on the "self-loop": even with one model, `Agent._define_msg_tree`
relabels the history as alternating `user`/`assistant` so the model thinks
it's replying to a partner. That's how a single model "talks to itself."

---

## 3. Module map

```
src/
├── main.py                      ENTRY POINT. Parse CLI, load YAML configs,
│                                run experiments (parallel via asyncio).
│
├── babel_ai/                    THE EXPERIMENT ENGINE
│   ├── experiment.py            Experiment: orchestrates the loop above,
│   │                            saves CSV + meta JSON.
│   ├── agent.py                 Agent: one provider+model+params. Builds the
│   │                            message tree and calls the LLM. Also holds
│   │                            round_robin_agent_selection().
│   ├── prompt_fetcher.py        The 4 fetchers (seed sources) — see §4.
│   ├── analyzer.py              SimilarityAnalyzer: the drift metrics — see §5.
│   └── enums.py                 FetcherType / AnalyzerType / AgentSelectionMethod
│                                + their factory mappings.
│
├── api/                         LLM PROVIDER LAYER
│   ├── llm_interface.py         LLMInterface.generate_response(): retries,
│   │                            backoff, budget tracking. Provider-agnostic.
│   ├── enums.py                 Provider + per-provider model enums
│   │                            (OpenAIModels, AnthropicModels, ...). << model
│   │                            names live here; only listed models are usable.
│   ├── openai.py / anthropic.py / azure_openai.py / ollama.py
│   │                            One request function per provider. Each loads
│   │                            .env and returns an LLMResponse.
│   └── budget.py               BudgetTracker: prices token usage per request,
│                                writes logs/budget_tracking.json.
│
└── models/                      DATA CONTRACTS (Pydantic)
    ├── configs.py               ExperimentConfig / AgentConfig / FetcherConfig
    │                            / AnalyzerConfig. The YAML maps onto these,
    │                            and they VALIDATE everything (see §6).
    ├── metrics.py               Metric / FetcherMetric / AgentMetric /
    │                            AnalysisResult / ExperimentMetadata (CSV rows).
    └── api.py                   LLMResponse (content + token counts).
```

Supporting dirs: `configs/` (YAML experiment definitions), `notebooks/`
(analysis & visualization), `analysis/pdf_generation.py` (report export),
`tests/` (unit + integration), `data/` (datasets — **gitignored**, not shipped).

---

## 4. Fetchers (where the seed comes from)

All implement `BasePromptFetcher.get_conversation() -> List[{role, content}]`.
Chosen by `fetcher_config.fetcher` in YAML.

| `fetcher`              | Source                          | Needs data file? |
|------------------------|---------------------------------|------------------|
| `random`               | Reddit + random-word web APIs   | No (internet)    |
| `sharegpt`             | ShareGPT JSON dump              | Yes — `data_path`|
| `topical_chat`         | Alexa Topical-Chat `.jsonl`     | Yes — 2 paths    |
| `infinite_conversation`| dir of `conversation_*.json`    | Yes — `data_path`|

⚠️ The `sharegpt` fetcher expects each record to have an **`items`** key
(`[{"from", "value"}]`). Real ShareGPT dumps use `conversations` — you must
rename the key. (That's what `data/sharegpt_real.json` already is.)

---

## 5. The drift metrics (what `SimilarityAnalyzer` outputs)

Computed per message, stored in the `analysis` column of the CSV:

| Metric                       | Meaning                                              |
|------------------------------|------------------------------------------------------|
| `word_count`                 | length of the message                                |
| `coherence_score`            | unique words / total words (vocab variety; ↓ = repetitive) |
| `token_perplexity`           | GPT-2 perplexity (↑ = more surprising/less fluent)   |
| `lexical_similarity`         | Jaccard overlap vs *previous* message                |
| `semantic_similarity`        | Sentence-BERT cosine vs *previous* message (↑ = repeating itself) |
| `*_similarity_window`        | same, but averaged over `analyze_window` messages    |

Models used: `all-MiniLM-L6-v2` (semantic) and `gpt2` (perplexity). These
download on first run. **Rising semantic_similarity over turns = drift toward
repetition**, the headline signal.

Only `similarity` exists today; add new ones via `AnalyzerType` + a subclass.

---

## 6. Config = the control surface

You never edit Python to run an experiment — you write a YAML that becomes an
`ExperimentConfig`. Shape (see `configs/openai_sharegpt.yaml`):

```yaml
fetcher_config:   { fetcher, data_path?, min_messages?, max_messages?, category? }
analyzer_config:  { analyzer, analyze_window }
agent_configs:    [ { provider, model, system_prompt?, temperature?, max_tokens?, ... } ]
agent_selection_method: round_robin
max_iterations:   <int>     # stop after this many total messages
max_total_characters: <int> # OR stop after this many chars
output_dir:       results
```

Pydantic enforces rules in `models/configs.py`, e.g.:
- `model` must be a real entry in that provider's enum (`api/enums.py`).
- `random` fetcher *requires* `category`; `sharegpt` *requires* `data_path` /
  `min_messages` / `max_messages` and *forbids* the others.
- `max_messages >= min_messages`, temperature in [0,2], etc.

So most "it won't start" errors are config validation, not deep bugs.

---

## 7. Gotchas discovered (real, current)

1. **Python version**: needs 3.13. On 3.14, `torch 2.6` has no wheels and
   `poetry install` silently installs nothing. Pin with
   `poetry env use python3.13`.
2. **Dead model names**: `api/enums.py` only shipped old GPT-4 *preview*
   snapshots that 404 on current keys. `gpt-4o-mini` / `gpt-4o` were added.
3. **Shipped `configs/test_config.yaml` doesn't run out of the box** — it
   points at `data/sharegpt_sample.json` (not in repo) and a dead model.
4. **Cost shows $0** for `gpt-4o-mini` because it isn't in `budget.py`'s
   pricing table — usage is real, just unpriced.

---

## 8. How to run (quick reference)

```bash
poetry env use python3.13      # one-time, avoids the torch/3.14 trap
poetry install                 # pulls torch/transformers (~GB)
cp .env_example .env           # add OPENAI_API_KEY (or other provider keys)

# No dataset needed:
poetry run python src/main.py configs/openai_random.yaml

# Real ShareGPT seed:
poetry run python src/main.py configs/openai_sharegpt.yaml

# Flags: --debug (verbose), --sequential (don't parallelize multiple configs)
```

Output lands in `results/` as `drift_experiment_<ts>.csv` + `_meta.json`.
Open the CSV's `analysis` column (a dict per row) to read the drift metrics,
or use the `notebooks/` for plots.

---

## 9. Where to extend

| Want to add...        | Touch                                                        |
|-----------------------|-------------------------------------------------------------|
| A new model           | the relevant enum in `src/api/enums.py`                      |
| A new provider        | `api/<provider>.py` + `Provider` enum wiring in `api/enums.py`|
| A new seed source     | subclass `BasePromptFetcher` + `FetcherType`                |
| A new drift metric    | subclass `Analyzer` + `AnalyzerType` (+ fields in `AnalysisResult`) |
| A new turn-taking rule| add to `AgentSelectionMethod` (only `round_robin` today)    |
```
