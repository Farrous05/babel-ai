# babel-ai — collapse & recovery in self-looping LLMs

> A research fork of [babel-ai](https://github.com/) — the upstream framework
> studies long-term behaviour and drift of LLMs talking to themselves. **This
> fork takes a specific direction:** characterising how a self-looping LLM
> *collapses* into repetition, and what happens when we *inject* off-topic text
> to knock it out.

## What this fork studies

An LLM fed its own output in a loop converges into repetition (a "collapse").
We:

1. **Detect collapse** online — windowed cosine distance below a cutoff for K
   consecutive rounds (calibrated; see [doc/metrics_and_collapse.md](doc/metrics_and_collapse.md)).
2. **Inject** off-topic text once collapse is reached — varying **size** (word
   / sentence / paragraph) and **source** (real text vs. shuffled noise), with
   an `<end>` marker and the injected span excluded from scoring. Injection can
   fire **once** or **repeatedly** (re-inject on each re-collapse, or on a fixed
   interval).
3. **Measure the response** with the spec's metrics (version-(b) cosine distance
   from the collapsed window, Jaccard distance to the injection, perplexity) and
   label the behaviour: *ignore / parrot / gibberish / blend / escape*.

The approach is grounded in three papers (Maiti et al.; Kong/Lai/Piao/Evans;
Shumailov et al.) — see [doc/references.md](doc/references.md) for how each
shapes our methods.

## Quick Start

### Installation

**Prerequisites:** Python 3.13+ and Poetry.

```bash
git clone <repository-url>
cd babel_ai
poetry install
poetry shell
```

**Environment** (for external providers): copy `.env_example` to `.env` and add
your API keys.

### Running an experiment

```bash
# Single run from a YAML config
poetry run python src/main.py configs/collapse_sharegpt.yaml

# Temperature-sweep grid (size × source × temperature × seeds)
poetry run python analysis/run_grid.py --seeds 3 --temps 0.5,0.7,1.0

# Long run with repeated injection (two cadences, free generation)
poetry run python analysis/run_longrun.py --mode both --rounds 150
```

Results are written per-run under `results/` (CSV conversation + JSON metadata
+ PDF), which is gitignored.

### Analysis

```bash
# Recovery-rate tables (by size × source, per temperature)
poetry run python analysis/aggregate_grid.py --csv results/grid/grid_sweep_no_t0.csv

# Injection-behaviour analysis (ignore/parrot/gibberish/blend/escape)
poetry run python analysis/injection_behavior.py
```

### Tests

```bash
poetry run pytest tests/unit_tests/
```

## Key components

- **Collapse detector** ([src/babel_ai/collapse.py](src/babel_ai/collapse.py)) —
  online windowed-cosine threshold rule; logs onset round + rate.
- **Injection module** ([src/babel_ai/injection.py](src/babel_ai/injection.py)) —
  size × source, `<end>` marker, span-tagging, distance, far-off-topic sampling.
- **Recovery evaluation** ([src/babel_ai/recovery.py](src/babel_ai/recovery.py)) —
  did it move away from the attractor, stay sane, and not parrot the injection?
- **Experiment orchestrator** ([src/babel_ai/experiment.py](src/babel_ai/experiment.py)) —
  self-loop (last-message feeding), online scoring, one-shot/repeated injection.
- **Analysis** ([analysis/](analysis/)) — grid sweep, aggregation, behaviour
  analysis, long-run runner, calibration.
- **Providers** ([src/api/](src/api/)) — OpenAI, Anthropic, Azure OpenAI, Ollama
  (incl. an OpenAI-compatible endpoint for multi-agent runs).

## Project structure

```
babel_ai/
├── src/
│   ├── babel_ai/          # core: experiment, collapse, injection, recovery, analyzer
│   ├── api/               # LLM provider interfaces
│   ├── models/            # Pydantic configs + metric models
│   └── main.py            # YAML-config experiment runner
├── analysis/              # grid sweep, aggregation, behaviour analysis, long-run
├── configs/               # experiment configurations (YAML)
├── tests/                 # unit tests
├── doc/                   # documentation (metrics, methodology, references)
└── data/                  # datasets (not included)
```

## Documentation

- **[Metrics & collapse](doc/metrics_and_collapse.md)** — metric definitions and
  the exact collapse rule.
- **[Methodology notes](doc/methodology_notes.md)** — paper coherence and key
  design decisions.
- **[Observations log](doc/observations_log.md)** — dated empirical findings.
- **[References](doc/references.md)** — the papers this work builds on.
- **[How to run experiments](doc/how_to_run_experiments.md)** ·
  **[Contributing](doc/contributing.md)** · **[Codebase map](doc/mental_map.md)**

## Style

Black / isort (79-char), Pydantic configs, Poetry, pytest.

## License

[LICENSE](LICENSE)
