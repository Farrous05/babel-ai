"""Company Ollama / vLLM endpoint interface (OpenAI-compatible).

The company inference endpoints (Ollama and vLLM) expose an OpenAI-compatible
API, so we talk to them with the OpenAI client pointed at the endpoint's base
URL plus a Bearer key -- the provider's recommended path.

Credentials come from the environment (never hard-coded). vLLM serves **one
model per endpoint**, so each model has its own URL + key. Each model looks for
env vars named after the model, falling back to a shared default:

    <MODEL>_BASE_URL / <MODEL>_API_KEY   per-model endpoint (preferred)
    OLLAMA_COMPANY_BASE_URL / _API_KEY   shared fallback (single endpoint)

where ``<MODEL>`` is the OllamaCompanyModels member name, e.g.::

    LLAMA_3_3_70B_BASE_URL=https://<llama-endpoint>/v1/
    LLAMA_3_3_70B_API_KEY=<llama key>
    QWEN_2_5_72B_BASE_URL=https://<qwen-endpoint>/v1/
    QWEN_2_5_72B_API_KEY=<qwen key>

Clients are built lazily and cached per endpoint, so importing this module (and
running other experiments or the test suite) works fine with no creds set.
"""

import logging
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from api.enums import OllamaCompanyModels
from models.api import LLMResponse

logger = logging.getLogger(__name__)

load_dotenv()

# Cache one OpenAI client per distinct base URL.
_CLIENTS: Dict[str, OpenAI] = {}


def _creds_for(model: OllamaCompanyModels) -> "tuple[str, str]":
    """Resolve (base_url, api_key) for a model: per-model env vars first
    (``<MODEL_NAME>_BASE_URL`` / ``_API_KEY``), then the shared
    ``OLLAMA_COMPANY_*`` fallback."""
    prefix = model.name  # e.g. "LLAMA_3_3_70B"
    base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv(
        "OLLAMA_COMPANY_BASE_URL"
    )
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv(
        "OLLAMA_COMPANY_API_KEY"
    )
    if not base_url or not api_key:
        raise ValueError(
            f"No endpoint configured for {model.name}. Set "
            f"{prefix}_BASE_URL and {prefix}_API_KEY (or the shared "
            "OLLAMA_COMPANY_BASE_URL / OLLAMA_COMPANY_API_KEY) in your .env."
        )
    return base_url, api_key


def _client(model: OllamaCompanyModels) -> OpenAI:
    """Lazily build/cache the OpenAI-compatible client for a model's endpoint."""
    base_url, api_key = _creds_for(model)
    if base_url not in _CLIENTS:
        logger.info(
            "Initializing client for %s at %s", model.name, base_url
        )
        _CLIENTS[base_url] = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENTS[base_url]


def company_ollama_request(
    messages: list,
    model: OllamaCompanyModels = OllamaCompanyModels.GPT_OSS_20B,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> LLMResponse:
    """Send a chat request to the company OpenAI-compatible endpoint.

    Mirrors ``openai_request``'s signature/return so it drops into
    ``LLMInterface`` unchanged.
    """
    logger.info(
        "Sending request to company Ollama endpoint with model %s, "
        "temperature %s, max_tokens %s",
        model.value,
        temperature,
        max_tokens,
    )

    content = ""
    usage = None
    for attempt in range(3):
        try:
            response = _client(model).chat.completions.create(
                model=model.value,
                messages=messages,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
        except Exception as e:
            logger.error("Error in company Ollama request: %s", str(e))
            raise

        msg = response.choices[0].message
        content = msg.content
        usage = response.usage
        # Some models return empty `content` while putting text in a reasoning
        # field, or return empty on a confusing self-loop turn. Fall back to the
        # reasoning field, then retry, so an empty turn can't corrupt the loop.
        if not content or not content.strip():
            content = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or content
            )
        if content and content.strip():
            break
        logger.warning(
            "Empty content from %s (attempt %d/3); retrying",
            model.value,
            attempt + 1,
        )

    if not content or not content.strip():
        logger.error("Still empty after retries from %s", model.value)

    return LLMResponse(
        content=content or "",
        input_token_count=usage.prompt_tokens if usage else 0,
        output_token_count=usage.completion_tokens if usage else 0,
    )
