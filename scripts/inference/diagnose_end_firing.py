"""Why does <end> never appear in the babel-ai runs? Three-way diagnostic.

THE PUZZLE
The offline eval scores P(<end>) ~ 1.0 on collapsed windows (AUC 1.000). In
babel-ai the model self-conversed into near-verbatim repetition -- windowed
similarity climbing 0.61 -> 0.94 -- and fired <end> exactly ZERO times in 16 runs.
Those two facts cannot both describe the same model behaviour, so something
between them differs.

WHAT THIS SEPARATES
We take held-out rows the model was MEASURED on (known-firing positives) and push
them through the real babel-ai provider, changing one thing at a time:

  A. local_hf_request() + babel's system prompt   <- exactly what the runs did
  B. local_hf_request() + NO system prompt        <- exactly what training saw
  C. raw P(<end>) scoring, both prompts           <- exactly what the eval did

  <end> in NEITHER A nor B  -> the pipeline is eating the token (plumbing bug),
                               and both live runs measured nothing
  <end> in B but not A      -> the SYSTEM PROMPT suppresses firing; fix is one
                               line of yaml
  <end> in BOTH             -> neither; real conversations differ from the frozen
                               training windows in some other way

C is the control on A/B: it reads the probability directly off the model instead
of sampling, so it says whether the model WANTED to fire even when no <end> token
came out. A high P(<end>) with no <end> in the text is decisive evidence of a
plumbing or sampling problem rather than a model problem.

Run inside babel-ai's venv (that is the environment under test).
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/dais/u/fash/babel-ai/src")

DATA = "/u/fash/internship-hpc-repo-template/data/method_a_v3_heldout.jsonl"
BABEL_SYS = "You are a helpful assistant participating in a conversation."
N = 12


def main() -> None:
    import torch

    from api.local_hf import _load, local_hf_request

    path = "/u/fash/internship-hpc-repo-template/models/end-token-A-qwen3b-full"
    tok, mdl, _ = _load(path)
    end_id = tok.convert_tokens_to_ids("<end>")
    print(f"<end> id = {end_id}, vocab = {len(tok)}", flush=True)

    rows = [json.loads(x) for x in open(DATA) if x.strip()]
    pos = [r for r in rows if r.get("label_type") == "positive"][:N]
    print(f"testing {len(pos)} held-out POSITIVES "
          f"(rows the offline eval scored ~1.0)\n", flush=True)

    def p_end(messages):
        """Probability of <end> as the next token -- the eval's own method."""
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=False)
        x = torch.tensor([ids], device=mdl.device)
        with torch.no_grad():
            logits = mdl(x).logits[0, -1]
        return torch.softmax(logits.float(), -1)[end_id].item()

    class FakeModel:            # local_hf_request reads model.value
        value = path

    hits_a = hits_b = 0
    pa_sum = pb_sum = 0.0
    for i, r in enumerate(pos):
        msgs = r["messages"]
        with_sys = [{"role": "system", "content": BABEL_SYS}] + msgs

        # C: does the model WANT to fire, under each prompt?
        pa, pb = p_end(with_sys), p_end(msgs)
        pa_sum += pa
        pb_sum += pb

        # A/B: does <end> actually come out of the real provider?
        ta = local_hf_request(messages=with_sys, model=FakeModel(),
                              temperature=0.7, max_tokens=60).content
        tb = local_hf_request(messages=msgs, model=FakeModel(),
                              temperature=0.7, max_tokens=60).content
        a, b = "<end>" in ta, "<end>" in tb
        hits_a += a
        hits_b += b

        print(f"[{i:>2}] A sys P={pa:.3f} fired={a}  |  B nosys P={pb:.3f} "
              f"fired={b}", flush=True)
        print(f"     A: {ta[:90]!r}", flush=True)
        print(f"     B: {tb[:90]!r}", flush=True)

    n = len(pos)
    print(f"\n{'=' * 70}")
    print(f"A  babel system prompt : fired {hits_a}/{n}   mean P(<end>) "
          f"{pa_sum / n:.3f}")
    print(f"B  no system prompt    : fired {hits_b}/{n}   mean P(<end>) "
          f"{pb_sum / n:.3f}")
    print(f"{'=' * 70}")
    if hits_a == 0 and hits_b == 0 and max(pa_sum, pb_sum) / n > 0.5:
        print("VERDICT: model WANTS to fire but no <end> reaches the text "
              "-> PLUMBING BUG in the provider/decode path.")
    elif hits_b > hits_a:
        print("VERDICT: the SYSTEM PROMPT suppresses firing -> drop "
              "system_prompt from the livetest yaml and re-run.")
    elif hits_a and hits_b:
        print("VERDICT: pipeline and prompt are both fine -> the live "
              "conversation differs from frozen training windows some other way.")
    else:
        print("VERDICT: inconclusive -- read the per-row output above.")


if __name__ == "__main__":
    main()
