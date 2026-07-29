"""In-process HuggingFace provider — run a local fine-tuned model inside babel-ai.

WHY THIS EXISTS
The <end>-token model is a HuggingFace directory with a NEW special token.
Ollama cannot serve it (GGUF only, and the added token does not survive
conversion). Serving it with vLLM works, but that means a second GPU job that has
to stay alive AND stay reachable over the network -- when that job was cancelled
mid-run, the experiment went with it.

Loading the model IN-PROCESS removes the server entirely: babel-ai's own job holds
the weights. No port, no base_url, no second job to babysit.

WHAT IT GIVES US
babel-ai's Experiment loop, similarity analyzer, collapse detector, recovery
evaluation and per-run PDF all work unchanged -- so the trained model gets the
SAME instrumentation the 10k harvest had. That is what turns "it escaped once"
into "it escapes N% of its own collapses", with plots.

<end> VISIBILITY  (decode with skip_special_tokens=False)
<end> is registered as a SPECIAL token, and the default decode strips special
tokens out of the returned string. The model still fires -- but the marker is
deleted from the text, so the transcript reads as if it never happened.
Observed directly against the served model, same prompt:
    default                   -> " The '1000' in the title likely refers to..."
    skip_special_tokens=False -> "<end> The '1000' in the title likely refers to..."
Without the flag, every escape is invisible in the CSV and the analysis.

USAGE (config yaml):
    agent_configs:
      - provider: "local_hf"
        model: "/u/fash/internship-hpc-repo-template/models/end-token-A-qwen3b-full"
The model string is a filesystem PATH (or any HF id).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from models.api import LLMResponse

logger = logging.getLogger(__name__)

# One cached (tokenizer, model) per path. Loading a 3B takes ~30s and babel builds
# a fresh Agent per run, so without this cache every run would reload the weights.
# The LOCK matters: babel-ai runs the experiments of a batch in parallel threads,
# so without it 8 runs all miss the empty cache at once and each loads its own
# copy of the weights -- 8x the GPU memory for one model.
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _tokenizer_kwargs(path: str) -> dict:
    """Cross-version compatibility for the tokenizer we trained.

    The model was SAVED by transformers 5.x, which writes the added <end> token
    into tokenizer_config.json as ``extra_special_tokens: ["<end>"]`` -- a LIST.
    babel-ai's venv runs transformers 4.57, which expects that field to be a DICT
    and does ``special_tokens.keys()`` on it, so loading dies with

        AttributeError: 'list' object has no attribute 'keys'

    Passing the field explicitly overrides what is in the file: we blank the
    v5-style list and re-declare <end> through ``additional_special_tokens``,
    which both versions understand. Verified under 4.57: vocab 151666 and
    <end> keeps id 151665, exactly as under 5.x -- so this changes nothing about
    the model, only how the file is parsed.

    We deliberately do NOT rewrite tokenizer_config.json: every measured result
    in this project came from loading that directory under 5.x, and editing the
    trained artifact to suit a second environment risks all of them.
    """
    cfg_path = os.path.join(path, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        logger.info("v5-style extra_special_tokens %s -> 4.x form", extra)
        return {"extra_special_tokens": {},
                "additional_special_tokens": list(extra)}
    return {}


def _load(path: str):
    if path in _CACHE:
        return _CACHE[path]
    with _CACHE_LOCK:
        if path in _CACHE:           # another thread loaded it while we waited
            return _CACHE[path]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local HF model in-process: %s", path)
        tok = AutoTokenizer.from_pretrained(path, **_tokenizer_kwargs(path))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # NO device_map="auto" here. device_map is an accelerate feature, and
        # babel-ai's venv does not have accelerate -- with it, every request dies
        # with "Using a `device_map` ... requires `accelerate`". We do not need
        # it either: device_map exists to SHARD a model across several GPUs, and
        # a 3B in bf16 is ~6GB, so it fits on one H200 many times over. Loading
        # normally and moving the whole model to the GPU needs no extra package.
        # (The internship-repo scripts DO use device_map -- they run inside the
        # container, which has accelerate. This difference is environment-only.)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16
        ).to(device).eval()
        _CACHE[path] = (tok, model, torch)
        logger.info("Loaded %s on %s (vocab %d, <end> id %s)", path, device,
                    len(tok), tok.convert_tokens_to_ids("<end>"))
        return _CACHE[path]


def local_hf_request(
    messages: list,
    model=None,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> LLMResponse:
    """Generate in-process. Mirrors the other providers' signature/return."""
    path = getattr(model, "value", None) or os.getenv("LOCAL_HF_MODEL")
    if not path:
        raise ValueError(
            "local_hf provider needs a model path (config `model:` field or "
            "LOCAL_HF_MODEL env var)."
        )
    tok, mdl, torch = _load(path)

    ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    x = torch.tensor([ids], device=mdl.device)
    # Pass the mask explicitly. We send ONE un-padded sequence, so every token is
    # real and the mask is all ones -- which is exactly what generate() falls back
    # to. But it cannot INFER that: this model's pad token (151643) is also listed
    # in eos_token_id, so transformers gives up and warns
    #   "attention mask is not set and cannot be inferred ... pad token is same as
    #    eos token"
    # Supplying it makes the intent explicit and silences a warning that would
    # otherwise be indistinguishable from a real padding bug in the logs.
    attn = torch.ones_like(x)
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        out = mdl.generate(
            x,
            attention_mask=attn,
            max_new_tokens=max_tokens or 512,
            do_sample=temperature > 0,
            temperature=temperature or None,
            top_p=top_p,
            # Every sampling knob is passed EXPLICITLY, even where the value looks
            # like a default. The trained model ships a generation_config.json
            # (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) and
            # anything we leave out is silently taken from that file -- so an
            # unset knob is not "the default", it is "whatever training saved".
            #
            # top_k=20 is DELIBERATE, not a leftover. Explicit does not mean off:
            # this was briefly set to 0 (no truncation), which let every one of
            # 151,666 tokens compete at each step and wrecked generation -- replies
            # switching language mid-conversation ("Die Uberblick...", "殣，中国..."),
            # and turns collapsing to a single ".". Every result that ever worked
            # (the turn-12 escape, AUC 1.000, 97% cross-model) was measured with
            # top_k=20 inherited from that file, so 20 is the value to reproduce.
            top_k=20,
            repetition_penalty=1.0,   # do NOT discourage repetition -- collapse
            #                            is the phenomenon we are studying
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

    new = out[0][len(ids):]
    # keep <end> in the text (see the module docstring)
    text = tok.decode(new, skip_special_tokens=False)

    # CUT at <|im_start|>, do not merely delete it. Sometimes the model finishes
    # its reply and, instead of stopping, opens the NEXT speaker's turn:
    #     "<|im_start|>user\nThe current status of the data collection is..."
    # generate() only halts on eos (<|im_end|>/<|endoftext|>), so nothing stops it
    # writing both sides of the conversation. Everything from that marker on is the
    # model impersonating its partner and must be dropped -- deleting just the tag
    # would leave the fake user turn glued onto the model's own reply, which then
    # enters the history and contaminates every later turn. Observed in 2 of 8 runs.
    #
    # This is the cost of skip_special_tokens=False: <end> survives, but so does
    # every other special token, so they have to be handled by hand. CUT at any
    # marker that means "a new turn / tool call started" -- <|im_start|> (next
    # speaker) and <tool_call> (Qwen's tool token, which the model leaks in
    # degenerate escapes; observed in the v4 live test). Everything from the
    # EARLIEST such marker on is not part of this reply. <end> is deliberately NOT
    # in this list -- it is ours to keep.
    cuts = [text.find(m) for m in ("<|im_start|>", "<tool_call>")]
    cuts = [c for c in cuts if c != -1]
    if cuts:
        cut = min(cuts)
        logger.warning("model began a new turn/tool-call mid-reply; truncating "
                       "(dropped %d chars)", len(text) - cut)
        text = text[:cut]
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    text = text.strip()

    return LLMResponse(
        content=text,
        input_token_count=max(len(ids), 1),
        output_token_count=max(len(new), 1),
    )
