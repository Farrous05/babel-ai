"""Reference embedder for reference-based distance metrics.

``reference_embedder()`` returns a LOCAL bge-large-en-v1.5 by default (see the
note there); the ``OpenAIEmbedder`` below is the legacy text-embedding-3-large
backend, kept as an opt-in fallback (REFERENCE_EMBEDDER=openai). Both expose an
``encode()`` with the same signature/return as the sentence-transformers model
(a ``torch.Tensor``), so ``analyzer.py``, ``injection.py`` and ``recovery.py``
keep working unchanged (``cos_sim``, ``.mean(dim=0)``, ``.item()`` all apply).

Why switch from SBERT (all-MiniLM-L6-v2):
  * **No truncation** — SBERT reads only the first 256 tokens; our free-gen
    turns are 300-1000 words, so SBERT only "saw" each turn's opening.
    ``text-embedding-3-large`` takes 8191 tokens, covering whole turns.
  * **Matches the reference paper** (comparability with Multi-LLM).

Embeddings are cached in-memory per process (deterministic within a run and
avoids re-billing identical text). The client is built lazily, so importing
this module needs no API key.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Union

import torch
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# text-embedding-3-large accepts 8191 tokens; cap chars defensively (~4 chars/
# token) so a pathologically long turn can't error the request.
_MAX_CHARS = 30000


class OpenAIEmbedder:
    """SBERT-compatible embedder backed by an OpenAI embedding model."""

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        self.model = model
        self._cache: Dict[str, list] = {}
        self._client = None

    def _client_or_build(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("OpenAI embedder ready: %s", self.model)
        return self._client

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = True,
        **_: object,
    ) -> torch.Tensor:
        """Embed text(s); returns a 1-D tensor for a str, 2-D for a list --
        matching ``SentenceTransformer.encode``."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        # OpenAI rejects empty input; substitute a space. Cap length.
        norm = [((t if t and t.strip() else " ")[:_MAX_CHARS]) for t in items]

        missing = [t for t in dict.fromkeys(norm) if t not in self._cache]
        for i in range(0, len(missing), 256):  # batch the API calls
            batch = missing[i : i + 256]
            resp = self._client_or_build().embeddings.create(
                model=self.model, input=batch
            )
            for t, d in zip(batch, resp.data):
                self._cache[t] = d.embedding

        out = torch.tensor(
            [self._cache[t] for t in norm], dtype=torch.float32
        )
        return out[0] if single else out


# Shared reference embedder for *reference-based* distances (version-(b) =
# distance-from-collapsed-window, and injection distance). It is NOT used for the
# turn-to-turn collapse detector -- that stays on SBERT MiniLM (analyzer.py).
#
# LOCAL as of 2026-07-20: was OpenAI text-embedding-3-large, but that key is out
# of quota and, more importantly, a paid network call inside the 24h harvest is a
# single point of failure (it 429'd and broke every run in a smoke test). bge-
# large-en-v1.5 is local, free, already cached, already the dataset builder's
# embedder, and competitive with text-embedding-3-large on MTEB. It auto-uses
# CUDA on the harvest node.
#
# ONE CAVEAT: the RecoveryConfig cutoffs were calibrated on text-embedding-3-
# large's distance scale. bge-large's cosine distances differ, so the recovery
# cutoff needs a re-check (open item; recovery only drops the ~2% "recovered"
# runs, so the harvest is unblocked regardless). To go back to OpenAI, set
# REFERENCE_EMBEDDER=openai (needs a funded OPENAI_API_KEY).
_REFERENCE_MODEL = "BAAI/bge-large-en-v1.5"
_REFERENCE = None


def reference_embedder():
    """Lazy singleton LOCAL embedder for reference-based distance metrics.

    Returns a SentenceTransformer (or the OpenAIEmbedder if REFERENCE_EMBEDDER=
    openai). Both expose ``encode(texts, convert_to_tensor=True) -> Tensor`` (1-D
    for a str, 2-D for a list), so callers in recovery.py / injection.py are
    unchanged.
    """
    global _REFERENCE
    if _REFERENCE is None:
        if os.getenv("REFERENCE_EMBEDDER", "local").lower() == "openai":
            _REFERENCE = OpenAIEmbedder("text-embedding-3-large")
        else:
            from sentence_transformers import SentenceTransformer

            _REFERENCE = SentenceTransformer(_REFERENCE_MODEL)
            logger.info("Local reference embedder ready: %s", _REFERENCE_MODEL)
    return _REFERENCE
