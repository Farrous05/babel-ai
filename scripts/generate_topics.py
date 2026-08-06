"""Build a large TOPIC BANK: real ShareGPT subjects -> standalone paragraphs.

WHAT A TOPIC IS: the paragraph grafted after <end> in a positive example
("...loop... <end> A comet is a dirty snowball left over from the birth of the
solar system..."). It teaches the model to PIVOT to something new.

WHY SCALE: the existing 60 topics were all hand-written. A 10k harvest yields
thousands of positives, so each topic would be reused ~100x and the model would
memorize 60 recitations instead of learning "switch to something unrelated".

WHY SHAREGPT SUBJECTS INSTEAD OF A HAND-PICKED DOMAIN LIST: a hardcoded list makes
the author's biases the topic distribution. ShareGPT's 94k questions are already a
real, diverse subject list written by actual people about things they cared about.
So we take the SUBJECT of a real question and write a paragraph about it.

WHY NOT JUST USE SHAREGPT'S ANSWERS: they are answer-shaped ("Sure! Here's how
you'd do that...") and reference a question that won't be there. Grafted after
<end>, that would teach the model to answer a question nobody asked. So we
re-prompt: describe the SUBJECT, standalone.

DISJOINT BY ID from the seed pool: a run seeded from subject X must never be
handed a "fresh" topic that is also about X. We exclude every conversation id
already used as a seed.

Usage (needs local Ollama; OPENAI_BASE_URL points at it):
  python scripts/generate_topics.py --sharegpt FULL.json --exclude data/seeds_10k.json \
      --n 2000 --model mixtral:8x7b-instruct-v0.1-q4_K_M --out data/topics_big.json
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

RNG = random.Random(0)

PROMPT = """Below is a message someone once sent to an AI assistant.

--- MESSAGE ---
{question}
--- END MESSAGE ---

Identify the SUBJECT of that message, then write ONE self-contained paragraph
ABOUT THAT SUBJECT.

HARD REQUIREMENTS:
- Do NOT answer the message. Do NOT address anyone. Do NOT mention "the question",
  "the message", or "you". A reader who never saw the message must find your
  paragraph complete and natural on its own.
- 70-90 words, one paragraph, no title, no preamble, no bullet points.
- Factual, specific and vivid: real names, numbers, mechanisms -- not generic filler.
- Write it as interesting standalone prose, the way a good encyclopedia or essayist
  would open a description of the subject.

Respond with JSON only: {{"paragraph": "..."}}"""


def turns_of(conv):
    return conv.get("items") or conv.get("conversations") or []


def first_human(conv):
    for m in turns_of(conv):
        if m.get("from") == "human" and (m.get("value") or "").strip():
            return m["value"]
    return ""


def one(client, model, question):
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.9,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT.format(question=question[:1200])}],
        )
        p = json.loads(r.choices[0].message.content).get("paragraph", "").strip()
        # reject the failure mode this prompt is designed to avoid: an answer.
        low = p.lower()
        if len(p.split()) < 40 or low.startswith(("sure", "certainly", "yes,", "of course")):
            return None
        return p
    except Exception:
        return None


def dedupe(items, embedder_id, threshold):
    """Drop near-duplicates by embedding cosine -- ShareGPT has many questions on
    the same subject, so without this "10k topics" is really a few hundred wearing
    different hats.

    Greedy: keep an item only if it is below `threshold` against everything kept so
    far. Embeddings are L2-normalised, so cosine == dot product, and we compare
    against a PREALLOCATED buffer -- re-slicing a growing tensor each iteration
    would copy ~40MB x 20k times and take hours.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(embedder_id)
    emb = m.encode(items, convert_to_tensor=True, batch_size=128,
                   show_progress_bar=False, normalize_embeddings=True)
    n, d = emb.shape
    buf = torch.empty((n, d), device=emb.device, dtype=emb.dtype)
    keep, k = [], 0
    for i in range(n):
        if k and (buf[:k] @ emb[i]).max().item() >= threshold:
            continue
        buf[k] = emb[i]
        k += 1
        keep.append(i)
    return [items[i] for i in keep], len(items) - len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharegpt", required=True, help="the full ShareGPT dump")
    ap.add_argument("--exclude", default=None,
                    help="seed pool json; its ids are held out so topics and seeds "
                         "never share a subject")
    ap.add_argument("--n", type=int, default=2000, help="target AFTER dedupe")
    ap.add_argument("--model", default="mixtral:8x7b-instruct-v0.1-q4_K_M",
                    help="small/fast is fine -- volume is the point, not nuance")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--embedder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--dedupe-threshold", type=float, default=0.88)
    ap.add_argument("--overgen", type=float, default=1.5,
                    help="oversample: dedupe + answer-rejection remove many")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    used = set()
    if args.exclude:
        used = {c.get("id") for c in json.load(open(args.exclude))}
        print(f"excluding {len(used)} ids already used as seeds", flush=True)

    data = json.load(open(args.sharegpt))
    pool = [(c.get("id"), first_human(c)) for c in data
            if c.get("id") not in used and len(first_human(c).split()) >= 5]
    RNG.shuffle(pool)
    want = int(args.n * args.overgen)
    pool = pool[:want]
    print(f"{len(pool)} disjoint subjects -> generating (target {args.n} after dedupe)",
          flush=True)

    from openai import OpenAI
    client = OpenAI()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        raw = [p for p in ex.map(lambda q: one(client, args.model, q[1]), pool) if p]
    raw = list(dict.fromkeys(raw))
    print(f"  generated {len(raw)} paragraphs (rejected answers/short)", flush=True)

    kept, dropped = dedupe(raw, args.embedder, args.dedupe_threshold)
    kept = kept[:args.n]
    print(f"  dedupe (cos>={args.dedupe_threshold}) dropped {dropped} -> {len(kept)} kept")

    # same shape as injection_diverse.json: the builder reads items[0].value
    json.dump([{"items": [{"value": t}]} for t in kept],
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(kept)} topics -> {args.out}")


if __name__ == "__main__":
    main()
