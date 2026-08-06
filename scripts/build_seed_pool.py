"""Build a genre-balanced ShareGPT seed pool for the diverse-collapse harvest.

The babel self-loop already seeds from a short ShareGPT opening ("human -> gpt
-> human") and then loops. Left alone it samples uniformly, so the coding-heavy
ShareGPT mix would reproduce the corpus's topic skew. Here we classify each
opening by genre (keyword rules), then DOWNSAMPLE the big genres so coding is
capped and creative/casual/real-world are boosted -- "no single genre dominates".

Output: a ShareGPT-format JSON (list of {id, items}) of SHORT seeds (each trimmed
to its first `human, gpt, human` and ending on a human turn), ready to drop in as
the fetcher's data_path. Prints the raw and balanced genre distribution.

Usage:
  python scripts/build_seed_pool.py IN_sharegpt.json OUT_seeds.json
"""
from __future__ import annotations

import json
import re
import sys
import random
from collections import Counter, defaultdict

RNG = random.Random(0)

# Flattened target shares -- coding capped, diverse genres boosted (user choice).
TARGET = {
    "coding":     0.22,
    "creative":   0.18,
    "casual":     0.15,
    "knowledge":  0.15,
    "real_world": 0.12,
    "business":   0.09,
    "writing":    0.09,
}

# First match wins; order matters (identify code and creative before generic).
GENRE_RULES = [
    ("creative", r"\b(story|poem|comic|character|dialogue|scene|novel|fanfic|"
                 r"lyrics|rap|joke|screenplay|rewrite the following|write a "
                 r"(short )?(story|poem|song))\b"),
    ("coding",   r"\b(python|javascript|typescript|java\b|c\+\+|c#|rust|kernel|"
                 r"code|function|script|api|sql|html|css|git|regex|compile|"
                 r"array|arraylist|linux|ubuntu|terminal|algorithm|npm|docker|"
                 r"node|react|wsdl|jax|netfilter|def |import |print\()\b"),
    ("business", r"\b(marketing|customer|company|sales|product launch|revenue|"
                 r"business|startup|brand|strategy|seo|ecommerce|invoice|"
                 r"stakeholder|kpi)\b"),
    ("real_world", r"\b(explain|what is|what are|difference between|history|"
                   r"science|energy|fossil|farming|meat|climate|health|physics|"
                   r"biology|economy|politics|philosophy|nutrition|the negatives"
                   r" of)\b"),
    ("casual",   r"\b(how are you|tell me about yourself|what'?s your name|"
                 r"do you feel|your favou?rite|hi\b|hello|hey\b|let'?s chat|"
                 r"good morning)\b"),
    ("writing",  r"\b(write|summari[sz]e|rephrase|verbose|essay|email|paragraph|"
                 r"bullet points|outline|blog|more detail|more verbose)\b"),
    ("knowledge", r"\b(learn|learning|teach|tutorial|guide|how (do|to)|"
                  r"language|dogme|arduino|meaning of|steps to)\b"),
]


def first_human(items):
    for m in items:
        if m.get("from") == "human":
            return m.get("value") or ""
    return ""


def classify(text):
    t = text.lower()
    for genre, pat in GENRE_RULES:
        if re.search(pat, t):
            return genre
    return "knowledge"  # generic fallback bucket


def short_seed(items):
    """Trim to the first human->gpt->human (ends on human); else first human."""
    out = []
    for m in items:
        if m.get("from") not in ("human", "gpt"):
            continue
        out.append(m)
        if len(out) >= 3 and out[-1]["from"] == "human":
            break
    while out and out[-1]["from"] != "human":
        out.pop()
    return out


def turns_of(conv):
    """The full ShareGPT dump keys turns as 'conversations'; our 400-conversation
    sample keys them as 'items'. Accept either, so one builder serves both."""
    return conv.get("items") or conv.get("conversations") or []


def main():
    src, dst = sys.argv[1], sys.argv[2]
    # --max-pool: keep EVERY non-coding seed and cap coding to CODING_CAP of the
    # total. Hitting exact per-genre shares wastes seeds (the rarest genre caps
    # the whole pool at ~59); for a 10k harvest we want every seed we can get,
    # and coding-dominance is the only imbalance actually worth correcting.
    max_pool = "--max-pool" in sys.argv
    # --target N: cap the pool size. 10k seeds for a 10k harvest = ~1 run per seed,
    # so the collapse data never re-treads a subject.
    # --skip N: skip the first N usable conversations. Used to carve a DISJOINT set
    # for the topic bank, so a grafted topic can never be about the same subject a
    # run was seeded from.
    def _flag(name, cast=int):
        if name in sys.argv:
            return cast(sys.argv[sys.argv.index(name) + 1])
        return None
    target = _flag("--target")
    skip = _flag("--skip") or 0

    data = json.load(open(src))
    by_genre = defaultdict(list)
    seen = 0
    for conv in data:
        seed = short_seed(turns_of(conv))
        if not seed:                       # no usable human opening
            continue
        seen += 1
        if seen <= skip:                   # reserved for the other (disjoint) set
            continue
        g = classify(first_human(seed))
        by_genre[g].append({"id": conv.get("id"), "items": seed})

    raw = {g: len(v) for g, v in by_genre.items()}
    print("raw genre counts:", dict(sorted(raw.items(), key=lambda x: -x[1])))

    pool = []
    final = {}
    if max_pool:
        CODING_CAP = 0.25
        non_coding = [(g, v) for g, v in by_genre.items() if g != "coding"]
        n_non = sum(len(v) for _, v in non_coding)
        # coding <= CAP*(coding + n_non)  =>  coding <= CAP/(1-CAP) * n_non
        picks = {}
        for g, v in non_coding:
            RNG.shuffle(v)
            picks[g] = v
        cod = by_genre.get("coding", [])
        RNG.shuffle(cod)
        picks["coding"] = cod[:int(CODING_CAP / (1 - CODING_CAP) * n_non)]
        # scale down to --target while PRESERVING the genre proportions (and so
        # the coding cap): proportional subsampling can't skew the mix.
        total = sum(len(v) for v in picks.values())
        scale = min(1.0, target / total) if target else 1.0
        for g, v in picks.items():
            take = v if scale >= 1.0 else v[:max(1, round(len(v) * scale))]
            pool.extend(take)
            final[g] = len(take)
        N = len(pool)
    else:
        # largest pool size N such that TARGET[g]*N <= available for every genre
        N = min(int(len(by_genre[g]) / TARGET[g]) for g in TARGET if by_genre.get(g))
        for g, share in TARGET.items():
            want = round(share * N)
            avail = by_genre.get(g, [])
            RNG.shuffle(avail)
            take = avail[:want]
            pool.extend(take)
            final[g] = len(take)
    RNG.shuffle(pool)

    json.dump(pool, open(dst, "w"), indent=1)
    tot = len(pool)
    print(f"\nbalanced pool size: {tot}  (binding N={N})")
    print("balanced genre distribution:")
    for g, c in sorted(final.items(), key=lambda x: -x[1]):
        print(f"  {g:11s} {c:3d}  ({100*c/tot:4.1f}%)")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
