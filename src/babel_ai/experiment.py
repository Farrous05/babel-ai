"""LLM Drift Experiment.

This module implements an experiment to analyze long-term
behavior of Large Language Models when they operate in a
self-loop without external input.
"""

import json
import logging
import random
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd

from babel_ai.agent import Agent
from babel_ai.analyzer import Analyzer
from babel_ai.collapse import CollapseDetector
from babel_ai.enums import AgentSelectionMethod, InjectionTrigger
from babel_ai.injection import apply_injection, load_corpus
from babel_ai.prompt_fetcher import BasePromptFetcher
from babel_ai.recovery import evaluate_recovery, version_b_distances
from models import (
    AgentMetric,
    AnalysisResult,
    ExperimentConfig,
    ExperimentMetadata,
    FetcherMetric,
    Metric,
)

logger = logging.getLogger(__name__)


class Experiment:
    """Main class for running LLM drift experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        use_notebook_tqdm: bool = False,
    ):

        # set uuid
        self.uuid = uuid4()
        logger.info(
            f"Initializing Experiment {self.uuid} with config: {config}"
        )
        if use_notebook_tqdm:
            logger.info("Using notebook tqdm")
        else:
            logger.info("Using standard tqdm")

        # save configs
        self.config = config
        self.max_iterations = config.max_iterations
        self.max_total_characters = config.max_total_characters
        self.use_notebook_tqdm = use_notebook_tqdm

        # create output directory
        if config.output_dir is None:
            self.output_dir = Path.cwd() / "results"
            logger.warning(
                f"Experiment {self.uuid} "
                f"No output directory specified, using {self.output_dir}"
            )
        else:
            self.output_dir = Path(config.output_dir)
            logger.info(
                f"Experiment {self.uuid} "
                f"Using output directory {self.output_dir}"
            )

        # create metadata
        self.metadata = ExperimentMetadata(
            timestamp=datetime.now(),
            config=config,
        )

        # create analyzer
        self.analyzer = Analyzer.create_analyzer(
            analyzer_type=self.config.analyzer_config.analyzer,
            analyze_window=self.config.analyzer_config.analyze_window,
        )

        # online collapse detector (calibrated defaults, see
        # babel_ai/collapse.py and doc/metrics_and_collapse.md)
        self.collapse_detector = CollapseDetector()

        # post-collapse injection setup (spec step 3). Disabled unless an
        # injection_config is supplied; injection fires once, after collapse.
        self.injection_config = self.config.injection_config
        # rounds at which an injection fired (supports repeated dosing); the
        # last one is tracked for the re-collapse cooldown.
        self._injection_rounds: List[int] = []
        self._last_injection_round: Optional[int] = None
        # collapse onsets captured during the loop (each rearm clears the
        # detector's own onset, so we record them here); rate of the first.
        self._collapse_onsets: List[int] = []
        self._first_collapse_rate: Optional[float] = None
        self._injection_corpus: List[str] = []
        self._injection_rng = random.Random(
            self.injection_config.rng_seed if self.injection_config else None
        )
        if self.injection_config is not None:
            corpus_path = (
                self.injection_config.corpus_path
                or self.config.fetcher_config.data_path
            )
            if not corpus_path:
                raise ValueError(
                    "injection_config requires corpus_path (or a fetcher "
                    "data_path) to source injection text from"
                )
            self._injection_corpus = load_corpus(corpus_path)

        # create prompt fetcher
        # TODO: this is a hack to get the fetcher kwargs
        # we should consider something more elegant
        fetcher_kwargs = {
            k: v
            for k, v in self.config.fetcher_config.model_dump(
                exclude={"fetcher"}
            ).items()
            if v is not None
        }
        self.prompt_fetcher = BasePromptFetcher.create_fetcher(
            fetcher_type=self.config.fetcher_config.fetcher,
            **fetcher_kwargs,
        )

        # create agents
        self.agents = [
            Agent(agent_config) for agent_config in self.config.agent_configs
        ]

        # create agent selection method
        self.agent_selection_method = AgentSelectionMethod(
            self.config.agent_selection_method
        ).get_generator(self.agents)

        # set up results list
        self.result_metrics: List[Metric] = []

        # keep track of message history for the experiment
        self.messages: List[Dict[str, str]] = (
            self.prompt_fetcher.get_conversation()
        )

        # update metadata
        self.metadata.num_fetcher_messages = len(self.messages)

    def run(
        self,
        output_dir: Optional[Path] = None,
    ) -> List[Metric]:
        """Run the drift experiment."""
        # Produces self.result_metrics from
        # fetched starting point and with the
        # agents.
        self.run_interaction_loop()
        # Analyzes the self.result_metrics and adds
        # the analysis to the metrics.
        self._analyze_response(self.result_metrics)
        # Saves the results to a CSV file
        # and metadata to a JSON file.
        self._save_results_to_csv(
            metrics=self.result_metrics,
            metadata=self.metadata,
            output_dir=output_dir,
        )
        logger.info(
            f"Experiment {self.uuid} completed with "
            f"{len(self.result_metrics)} metrics"
        )
        return self.result_metrics

    def run_interaction_loop(
        self,
    ) -> List[Metric]:
        """Run the drift experiment with the given initial messages.

        Args:
            initial_messages: List of message dictionaries to start with

        Returns:
            List of Metric objects containing experiment results

        Raises:
            ValueError: If messages are not properly formatted
        """

        logger.info(
            f"Experiment {self.uuid} "
            f"Running interaction loop with {len(self.messages)} messages"
        )

        for i, message in enumerate(self.messages):
            self.result_metrics.append(
                FetcherMetric(
                    iteration=i,
                    timestamp=datetime.now(),
                    role=message["role"],
                    content=message["content"],
                    fetcher_type=self.config.fetcher_config.fetcher,
                    fetcher_config=self.config.fetcher_config,
                )
            )
            logger.debug(
                f"Experiment {self.uuid} " f"Added fetcher metric {i}"
            )

        logger.info(
            f"Experiment {self.uuid} "
            f"Added {len(self.result_metrics)} fetcher metrics"
        )

        # keep track of total characters
        self.total_characters = sum(
            len(msg["content"]) for msg in self.messages
        )

        # continue until max iterations or max total characters is reached
        iteration = len(
            self.result_metrics
        )  # Start from after fetcher metrics
        # round index for the collapse detector: 0 = first self-loop turn
        agent_round = 0
        while self._should_continue_generation():

            # log iteration
            logger.info(
                f"Experiment {self.uuid} "
                f"Iteration {iteration} of {self.max_iterations} "
            )

            # select next agent
            agent = next(self.agent_selection_method)

            # log iteration
            logger.debug(
                f"Experiment {self.uuid} "
                f"Iteration {iteration} "
                f"Generating response for agent {agent.id}"
            )

            # generate response. The model is fed only the most recent
            # `history_window` messages (default 1 = just the previous output,
            # the Maiti/Multi_LLM self-loop); None feeds the full history.
            # Scoring/analysis below still uses the full conversation.
            hw = self.config.history_window
            model_input = self.messages if hw is None else self.messages[-hw:]
            response = agent.generate_response(model_input)

            # log response
            logger.debug(
                f"Experiment {self.uuid} "
                f"Iteration {iteration} "
                f"Response: {response}"
            )

            # add response to conversation history and update total characters
            self.total_characters += len(response)

            self.messages.append(
                {
                    "role": str(agent.id),
                    "content": response,
                }
            )

            # add response to results, scoring this turn online (so the
            # collapse detector can react while the loop is still running)
            agent_metric = AgentMetric(
                iteration=iteration,
                timestamp=datetime.now(),
                role=str(agent.id),
                content=response,
                agent_id=str(agent.id),
                agent_config=agent.config,
            )
            agent_metric.analysis = self.analyzer.analyze(
                [msg["content"] for msg in self.messages]
            )
            self.result_metrics.append(agent_metric)

            # feed this round to the collapse detector
            self.collapse_detector.update_from_analysis(
                agent_round, agent_metric.analysis
            )

            # capture each collapse onset *before* a repeat injection can
            # rearm() the detector and clear it. The collapse rate is recorded
            # at the first collapse (well-defined; later re-collapses reuse the
            # same growing semantic history).
            onset = self.collapse_detector.onset_round
            if onset is not None and onset not in self._collapse_onsets:
                self._collapse_onsets.append(onset)
                if self._first_collapse_rate is None:
                    self._first_collapse_rate = (
                        self.collapse_detector.collapse_rate()
                    )

            # inject when the configured trigger fires (possibly repeatedly)
            if self.injection_config is not None and self._should_inject(
                agent_round
            ):
                self._apply_injection(agent_round)
                # for repeated after-collapse dosing, re-arm the detector so a
                # *new* collapse must form before the next injection.
                if (
                    self.injection_config.trigger
                    == InjectionTrigger.AFTER_COLLAPSE
                    and self.injection_config.repeat
                ):
                    self.collapse_detector.rearm(agent_round)

            # log agent metric
            logger.debug(
                f"Experiment {self.uuid} "
                f"Iteration {iteration} "
                f"Added agent metric {iteration}"
            )

            iteration += 1
            agent_round += 1

        # save total iterations/characters to metadata
        self.metadata.total_characters = self.total_characters
        self.metadata.num_iterations_total = iteration

        # record collapse detection results (TTC + rate). For repeated dosing
        # the detector is rearmed after each injection, so we report the FIRST
        # onset (back-compat) plus the full list of onsets.
        self.metadata.collapse_onsets = self._collapse_onsets
        self.metadata.collapse_onset_round = (
            self._collapse_onsets[0] if self._collapse_onsets else None
        )
        self.metadata.collapse_rate = self._first_collapse_rate
        if self._collapse_onsets:
            logger.info(
                f"Experiment {self.uuid} collapsed at rounds "
                f"{self._collapse_onsets} (first rate "
                f"{self.metadata.collapse_rate})"
            )
        else:
            logger.info(f"Experiment {self.uuid} did not collapse")

        # evaluate recovery (spec step 4) / baselines (spec step 5). Both the
        # recovery eval and the per-round version-(b) column anchor at the same
        # round: the injection round for an intervention / fresh-run baseline,
        # or the collapse-onset round for the no-intervention floor.
        anchor_round: Optional[int] = None
        injection_text = ""
        if self._injection_rounds and self.metadata.injection is not None:
            # anchor recovery/version-(b) at the FIRST injection; the per-round
            # version-(b) column then covers the whole multi-injection run.
            anchor_round = self.metadata.injection["round"]
            injection_text = self.metadata.injection["text"]
        elif self.injection_config is None and self._collapse_onsets:
            # no-intervention baseline (the floor): does the collapsed loop
            # drift away from the attractor on its own? Anchor at the first
            # onset; empty injection_text -> not-parroting trivially satisfied.
            anchor_round = self._collapse_onsets[0]

        if anchor_round is not None:
            self._evaluate_recovery(
                anchor_round=anchor_round, injection_text=injection_text
            )
            self._log_version_b_distances(anchor_round)

        return self.result_metrics

    def _log_version_b_distances(self, anchor_round: int) -> None:
        """Promote the spec version-(b) distance to a per-round CSV column.

        Computes each agent turn's cosine distance from the collapsed-window
        centroid (the ``window`` turns up to and including ``anchor_round``)
        and stores it on that turn's ``analysis.version_b_distance``, so it
        lands in the run CSV alongside the other per-round metrics. Covers the
        whole trajectory (pre- and post-anchor), not just post-injection.
        """

        agent_metrics = [
            m for m in self.result_metrics if isinstance(m, AgentMetric)
        ]
        contents = [m.content for m in agent_metrics]
        window = self.collapse_detector.config.window
        win_lo = max(0, anchor_round - window + 1)
        collapsed_window = [
            c for c in contents[win_lo : anchor_round + 1] if c.strip()
        ]
        if not collapsed_window:
            return

        distances = version_b_distances(contents, collapsed_window)
        for metric, distance in zip(agent_metrics, distances):
            if metric.analysis is not None:
                metric.analysis.version_b_distance = distance

    def _should_inject(self, agent_round: int) -> bool:
        """Whether the configured injection trigger fires this round.

        - ``fixed_round``: once, at the configured round.
        - ``fixed_interval``: every ``interval`` rounds (steady dosing).
        - ``after_collapse``: when collapse is declared; once by default, or on
          each *re-collapse* when ``repeat`` is set. Re-collapse cadence is now
          governed by the detector's **hysteresis** re-arm (it can only declare
          a new collapse after the loop has visibly left the attractor), so no
          time-based cooldown is needed here -- ``collapsed`` simply cannot be
          True again until a genuine new collapse has formed.
        """
        cfg = self.injection_config
        trigger = cfg.trigger

        if trigger == InjectionTrigger.FIXED_ROUND:
            return (
                not self._injection_rounds and agent_round == cfg.fixed_round
            )

        if trigger == InjectionTrigger.FIXED_INTERVAL:
            return agent_round > 0 and agent_round % cfg.interval == 0

        if trigger == InjectionTrigger.AFTER_COLLAPSE:
            if not self.collapse_detector.collapsed:
                return False
            if not cfg.repeat:
                return not self._injection_rounds
            return True

        return False

    def _evaluate_recovery(
        self, anchor_round: int, injection_text: str
    ) -> None:
        """Evaluate recovery relative to an anchor round and log to metadata.

        For an intervention the anchor is the injection round; for the
        no-intervention floor it is the collapse-onset round.
        """

        agent_metrics = [
            m for m in self.result_metrics if isinstance(m, AgentMetric)
        ]
        result = evaluate_recovery(
            agent_contents=[m.content for m in agent_metrics],
            analyses=[m.analysis for m in agent_metrics],
            injection_round=anchor_round,
            injection_text=injection_text,
            window=self.collapse_detector.config.window,
            # a round on which the detector re-declared collapse cannot count as
            # recovery (criterion 5: moved away AND did not re-collapse).
            recollapse_rounds=[
                r for r in self._collapse_onsets if r > anchor_round
            ],
        )
        self.metadata.recovery = asdict(result)
        logger.info(
            f"Experiment {self.uuid} recovery: recovered="
            f"{result.recovered} round={result.recovery_round} "
            f"hold={result.hold_length}"
        )

    def _apply_injection(self, round_idx: int) -> None:
        """Apply the one-shot post-collapse injection (spec step 3).

        Appends marker + injection text to the last output *in the message
        history the model sees*, but leaves the scored ``AgentMetric.content``
        as the original output -- so the injected span is excluded from scoring
        automatically. The tagged ``InjectionEvent`` is logged to metadata.
        """

        window = self.collapse_detector.config.window
        agent_outputs = [
            m.content
            for m in self.result_metrics
            if isinstance(m, AgentMetric)
        ]
        window_texts = agent_outputs[-window:]

        last_content = self.messages[-1]["content"]
        new_content, event = apply_injection(
            last_content=last_content,
            size=self.injection_config.size,
            source=self.injection_config.source,
            corpus=self._injection_corpus,
            window_texts=window_texts,
            round_idx=round_idx,
            use_marker=self.injection_config.use_marker,
            marker=self.injection_config.marker,
            rng=self._injection_rng,
            num_candidates=self.injection_config.num_candidates,
        )

        # The model reads the injection on its next turn; the AgentMetric for
        # this round keeps the original content, so scoring excludes the span.
        self.messages[-1]["content"] = new_content
        self.total_characters += len(new_content) - len(last_content)
        event_dict = asdict(event)
        self.metadata.injections.append(event_dict)
        if self.metadata.injection is None:
            self.metadata.injection = event_dict  # first one (back-compat)
        self._injection_rounds.append(round_idx)
        self._last_injection_round = round_idx
        logger.info(
            f"Experiment {self.uuid} applied injection #"
            f"{len(self._injection_rounds)} at round {round_idx}"
        )

    def _analyze_response(self, metrics: List[Metric]) -> AnalysisResult:
        """Analyze the response of the last agent."""

        logger.info(
            f"Experiment {self.uuid} "
            f"Analyzing response for {len(metrics)} metrics"
        )

        content = [metric.content for metric in metrics]

        for i, metric in enumerate(metrics):

            # Agent turns are already scored online inside the interaction
            # loop; only fill in metrics that were not analyzed yet (the
            # fetcher seed messages). Keeps the CSV output identical.
            if metric.analysis is not None:
                continue

            # log analysis
            logger.info(
                f"Experiment {self.uuid} "
                f"Analyzing response for {i} of {len(metrics)} metrics"
            )

            metric.analysis = self.analyzer.analyze(content[: i + 1])

        return metrics

    def _should_continue_generation(self) -> bool:
        """Check if generation should continue based on configured limits."""
        logger.debug(
            f"Experiment {self.uuid} "
            f"Checking if generation should continue"
        )
        logger.debug(
            f"Experiment {self.uuid} "
            f"Max iterations: {self.max_iterations}"
            f"Max total characters: {self.max_total_characters}"
            f"Total characters: {self.total_characters}"
            f"Number of iterations: {len(self.messages)}"
        )
        # Check iteration limit
        if len(self.messages) >= self.max_iterations:
            logger.info(
                f"Experiment {self.uuid} "
                "Max iterations reached: "
                f"{len(self.messages)} >= {self.max_iterations}"
            )
            return False
        # Check character limit
        if self.total_characters >= self.max_total_characters:
            logger.info(
                f"Experiment {self.uuid} "
                "Max total characters reached: "
                f"{self.total_characters} >= {self.max_total_characters}"
            )
            return False
        return True

    def _save_results_to_csv(
        self,
        metrics: List[Metric],
        metadata: ExperimentMetadata,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save experiment results to a CSV file and metadata to JSON.

        This method handles different types of metrics (Metric, FetcherMetric,
        AgentMetric) and flattens all nested structures into a single CSV file.
        It also saves comprehensive metadata to a JSON file.

        Args:
            metrics: List of Metric objects from the experiment
            metadata: ExperimentMetadata object containing experiment info
        """
        logger.info(
            f"Experiment {self.uuid} " f"Saving results to {output_dir}"
        )

        # Convert metrics to DataFrame
        df = pd.DataFrame([metric.to_dict() for metric in metrics])

        # Use current working directory if no output directory specified
        output_dir = output_dir or self.output_dir

        # One folder per experiment, with a human-readable run name. Uses the
        # "run_" prefix; discovery (graphical_analysis, run_longrun) accepts
        # both this and the legacy "drift_experiment_" prefix.
        base_filename = self._run_name(metadata)
        run_dir = output_dir / base_filename
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / f"{base_filename}.csv"
        meta_path = run_dir / f"{base_filename}_meta.json"
        pdf_path = run_dir / f"{base_filename}.pdf"

        logger.debug(
            f"Experiment {self.uuid} "
            f"Saving csv to {csv_path}"
            f"Saving metadata to {meta_path}"
        )

        df.to_csv(csv_path, index=False)

        # Save metadata
        metadata_dict = metadata.model_dump()

        with open(meta_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        logger.info(f"Saved experiment results to {csv_path}")
        logger.info(f"Saved experiment metadata to {meta_path}")

        # Best-effort conversation PDF in the same folder (never fatal).
        self._generate_pdf(csv_path, meta_path, pdf_path)

    def _run_name(self, metadata: ExperimentMetadata) -> str:
        """Descriptive, filesystem-safe run name (with timestamp for
        uniqueness): model, temperature, and intervention condition."""

        def sanitize(text: str) -> str:
            return re.sub(r"[^0-9A-Za-z.-]+", "-", str(text)).strip("-")

        def short_model(value: str) -> str:
            # Strip the HF org prefix ("Qwen/Qwen2.5-7B-Instruct" -> the
            # readable model name) so the folder isn't the name doubled.
            return sanitize(value.split("/")[-1])

        agents = self.config.agent_configs
        if len(agents) == 1:
            model_part = short_model(agents[0].model.value)
        else:
            model_part = "multiagent-" + "-vs-".join(
                short_model(a.model.value) for a in agents
            )
        temp_part = f"t{agents[0].temperature}"
        # Seed in the name for provenance (which run came from which seed);
        # "noseed" when sampling is stochastic. Recoverable from meta.json too.
        seed_part = (
            f"s{agents[0].seed}" if agents[0].seed is not None else "noseed"
        )

        inj = self.config.injection_config
        if inj is None:
            cond = "no-injection"
        else:
            # Distinguish the three injection timings so folder names don't
            # collide: rescue (after collapse), early5 (once at round 5),
            # dose5 (repeated every 5 rounds).
            kind = {
                "AFTER_COLLAPSE": "rescue",
                "FIXED_ROUND": "early5",
                "FIXED_INTERVAL": "dose5",
            }.get(inj.trigger.name, inj.trigger.name.lower())
            # source only when it's not the default real snippet (keeps the
            # common case clean; flags a noise control if ever used)
            src = "" if inj.source.name == "REAL" else f"-{inj.source.value}"
            cond = f"{kind}-{inj.size.value}{src}"

        ts = metadata.timestamp.strftime("%m%d-%H%M%S")
        return f"run_{model_part}_{temp_part}_{seed_part}_{cond}_{ts}"

    def _generate_pdf(
        self, csv_path: Path, meta_path: Path, pdf_path: Path
    ) -> None:
        """Generate the theater-script PDF for this run (best-effort).

        ``pdf_generation`` lives in ``analysis/`` (not on the src path), so we
        add it lazily and never let a PDF failure break the run.
        """
        try:
            import sys

            analysis_dir = str(
                Path(__file__).resolve().parents[2] / "analysis"
            )
            if analysis_dir not in sys.path:
                sys.path.insert(0, analysis_dir)
            from pdf_generation import TheaterScriptPDFGenerator

            TheaterScriptPDFGenerator().generate_pdf(
                str(csv_path), str(meta_path), str(pdf_path)
            )
            logger.info(f"Saved conversation PDF to {pdf_path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"PDF generation skipped: {e}")
