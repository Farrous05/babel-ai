"""Smoke-test the company Ollama / vLLM endpoint.

Run this once after setting OLLAMA_COMPANY_BASE_URL and OLLAMA_COMPANY_API_KEY
in .env to confirm connectivity and that the model is pulled/served before
launching a real experiment.

Usage::

    poetry run python analysis/check_ollama_endpoint.py
    poetry run python analysis/check_ollama_endpoint.py --model gpt-oss:120b
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from api.company_ollama import company_ollama_request  # noqa: E402
from api.enums import OllamaCompanyModels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=OllamaCompanyModels.GPT_OSS_20B.value,
        help="served model ID to test",
    )
    ap.add_argument("--prompt", default="Say this is a test.")
    args = ap.parse_args()

    try:
        model = OllamaCompanyModels(args.model)
    except ValueError:
        print(
            f"Model '{args.model}' is not in OllamaCompanyModels. Add it to "
            f"src/api/enums.py first. Known: "
            f"{[m.value for m in OllamaCompanyModels]}"
        )
        sys.exit(1)

    print(f"Testing {model.value} ...")
    try:
        resp = company_ollama_request(
            messages=[{"role": "user", "content": args.prompt}],
            model=model,
            temperature=0.2,
            max_tokens=64,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {e}")
        print(
            "Checklist: (1) OLLAMA_COMPANY_BASE_URL ends in /v1/, "
            "(2) OLLAMA_COMPANY_API_KEY is set, (3) the model is pulled on "
            "the endpoint."
        )
        sys.exit(1)

    print("OK. Response:")
    print(f"  {resp.content!r}")
    print(
        f"  tokens in/out: {resp.input_token_count}/"
        f"{resp.output_token_count}"
    )


if __name__ == "__main__":
    main()
