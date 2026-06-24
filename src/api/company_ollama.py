"""Company Ollama / vLLM endpoint interface (OpenAI-compatible).

The company inference endpoints (Ollama and vLLM) expose an OpenAI-compatible
API, so we talk to them with the OpenAI client pointed at the endpoint's base
URL plus a Bearer key -- the provider's recommended path.

Credentials come from the environment (never hard-coded):

    OLLAMA_COMPANY_BASE_URL   e.g. https://<your-endpoint>/v1/
    OLLAMA_COMPANY_API_KEY    your endpoint API key

The client is created lazily on first request, so importing this module (and
running OpenAI/Anthropic experiments or the test suite) works fine even when
the company credentials are not set. Remember: with the Ollama framework you
must `pull` the model on the endpoint before it can be used.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from api.enums import OllamaCompanyModels
from models.api import LLMResponse

logger = logging.getLogger(__name__)

load_dotenv()

_CLIENT: Optional[OpenAI] = None


def _client() -> OpenAI:
    """Lazily build the OpenAI-compatible client from env credentials."""
    global _CLIENT
    if _CLIENT is None:
        base_url = os.getenv("OLLAMA_COMPANY_BASE_URL")
        api_key = os.getenv("OLLAMA_COMPANY_API_KEY")
        if not base_url or not api_key:
            raise ValueError(
                "Company Ollama endpoint not configured. Set "
                "OLLAMA_COMPANY_BASE_URL (e.g. https://<endpoint>/v1/) and "
                "OLLAMA_COMPANY_API_KEY in your .env."
            )
        logger.info("Initializing company Ollama client at %s", base_url)
        _CLIENT = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENT


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

    try:
        response = _client().chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        content = response.choices[0].message.content
        usage = response.usage
        return LLMResponse(
            content=content,
            input_token_count=usage.prompt_tokens if usage else 0,
            output_token_count=usage.completion_tokens if usage else 0,
        )
    except Exception as e:
        logger.error("Error in company Ollama request: %s", str(e))
        raise
