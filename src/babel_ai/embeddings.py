"""OpenAI text-embedding backend (drop-in for the SBERT semantic model).

The Multi-LLM paper embeds text with OpenAI ``text-embedding-3-large``; we match
that. This exposes an ``encode()`` with the same signature/return as the
sentence-transformers model (a ``torch.Tensor``), so ``analyzer.py``,
``injection.py`` and ``recovery.py`` keep working unchanged (``cos_sim``,
``.mean(dim=0)``, ``.item()`` all still apply).

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
