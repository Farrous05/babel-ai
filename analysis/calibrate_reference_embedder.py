"""Test whether text-embedding-3-large is actually "strong" for the
*reference-based* distance (version-(b) distance-from-collapsed-window), vs SBERT.

We claimed large is strong here on reasoning alone; this measures it. The task
the reference embedder must do well: given the centroid of a collapsed attractor,
put turns that are STILL in that attractor NEAR (low distance) and genuinely
OFF-TOPIC text FAR (high distance), with a clean gap. Whichever embedder gives
the cleaner separation (higher AUC, bigger gap) is the right one for recovery.py.

Labeled set (objective, no human labels needed) — built from a run we have
already read end-to-end:
  * NEAR  (label 0, "in attractor"): post-injection turns of the cooking run
    `batch_20260626_173617`, which we know stayed on Italian food for all 297
    turns (the round-11 tech injection was ignored).
  * FAR   (label 1, "off topic"): turns from an unrelated Swift/3D-graphics grid
    run + the off-topic injection texts + ShareGPT corpus snippets.

The collapsed-window centroid is the cooking turns up to & incl. the injection,
exactly as recovery.py builds it. We compute 1 - cos(text, centroid) under both
embedders and compare NEAR vs FAR separation.

Usage::
    poetry run python analysis/calibrate_reference_embedder.py
"""

from __future__ import annotations

import glob
import json
import random
import sys

sys.path.insert(0, "src")

COOKING_RUN = "results/longrun/batch_20260626_173617"
SWIFT_RUN_GLOB = "results/grid/*inj-paragraph-real*"
CORPUS = "data/sharegpt_real.json"
N_PER_GROUP = 40
RNG = random.Random(0)


def _agent_turns(run_glob: str) -> list:
    import pandas as pd

    csv = glob.glob(f"{run_glob}/*.csv")
    if not csv:
        csv = glob.glob(f"{run_glob}/*/*.csv")
    df = pd.read_csv(csv[0])
    ag = df[df["agent_id"].notna()].reset_index(drop=True)
    return [str(c) for c in ag["content"].tolist() if str(c).strip()]


def _injection_round(run_glob: str) -> int:
    meta = json.load(open(glob.glob(f"{run_glob}/*/*_meta.json")[0]))
    return meta["injections"][0]["round"]


def _corpus_snippets(n: int) -> list:
    """A few off-topic passages straight from the injection source corpus."""
    try:
        data = json.load(open(CORPUS))
    except Exception:  # noqa: BLE001
        return []
    texts = []
    records = data if isinstance(data, list) else data.get("conversations", [])
    for rec in records:
        items = rec.get("items") or rec.get("conversations") or []
        for it in items:
            v = (it.get("value") or "").strip()
            if 200 < len(v) < 1500:
                texts.append(v)
    RNG.shuffle(texts)
    return texts[:n]


def _score(embedder, label: str, window, near, far):
    """Print NEAR/FAR distance stats + AUC for one embedder."""
    from sentence_transformers.util import cos_sim

    centroid = embedder.encode(window, convert_to_tensor=True).mean(dim=0)

    def dists(texts):
        emb = embedder.encode(texts, convert_to_tensor=True)
        out = []
        for i in range(len(texts)):
            sim = float(cos_sim(emb[i], centroid).item())
            out.append(1.0 - max(-1.0, min(1.0, sim)))
        return out

    dn, dfar = dists(near), dists(far)

    def stats(xs):
        xs = sorted(xs)
        mean = sum(xs) / len(xs)
        return mean, xs[0], xs[len(xs) // 2], xs[-1]

    mn, mn_lo, mn_med, mn_hi = stats(dn)
    mf, mf_lo, mf_med, mf_hi = stats(dfar)

    # AUC = P(far ranked higher than near) over all pairs (1.0 = perfect).
    wins = ties = 0
    for a in dfar:
        for b in dn:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    auc = (wins + 0.5 * ties) / (len(dfar) * len(dn))

    # Best single threshold separating the two groups (Youden-style scan).
    cand = sorted(set(dn + dfar))
    best_t, best_acc = 0.0, 0.0
    for t in cand:
        acc = (
            sum(1 for x in dn if x < t) + sum(1 for x in dfar if x >= t)
        ) / (len(dn) + len(dfar))
        if acc > best_acc:
            best_acc, best_t = acc, t

    print(f"\n=== {label} ===")
    print(
        f"  NEAR (in-attractor)  mean={mn:.3f}  "
        f"[min {mn_lo:.3f} | med {mn_med:.3f} | max {mn_hi:.3f}]"
    )
    print(
        f"  FAR  (off-topic)     mean={mf:.3f}  "
        f"[min {mf_lo:.3f} | med {mf_med:.3f} | max {mf_hi:.3f}]"
    )
    print(f"  gap (mean_far - mean_near) = {mf - mn:+.3f}")
    print(f"  separation AUC             = {auc:.3f}   (1.0 = perfect)")
    print(f"  best threshold             = {best_t:.3f}  (acc {best_acc:.0%})")
    print(
        f"  at the current recovery cutoff 0.30: "
        f"{sum(1 for x in dn if x >= 0.30) / len(dn):.0%} of NEAR turns "
        f"FALSELY look 'recovered'"
    )
    return {"auc": auc, "gap": mf - mn, "near_mean": mn, "far_mean": mf}


def main() -> None:
    inj = _injection_round(COOKING_RUN)
    cooking = _agent_turns(COOKING_RUN)
    window = [c for c in cooking[max(0, inj - 9) : inj + 1] if c.strip()]
    near_pool = cooking[inj + 1 :]
    near = RNG.sample(near_pool, min(N_PER_GROUP, len(near_pool)))

    swift = _agent_turns(SWIFT_RUN_GLOB)
    inj_texts = [
        e["text"]
        for e in json.load(
            open(glob.glob(f"{COOKING_RUN}/*/*_meta.json")[0])
        )["injections"]
    ]
    far = (
        RNG.sample(swift, min(20, len(swift)))
        + _corpus_snippets(18)
        + inj_texts
    )
    far = far[:N_PER_GROUP]

    print(
        f"Collapsed window: {len(window)} cooking turns (rounds "
        f"{max(0, inj - 9)}-{inj}).  NEAR={len(near)} cooking turns, "
        f"FAR={len(far)} off-topic passages."
    )

    from babel_ai.embeddings import OpenAIEmbedder
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    large = OpenAIEmbedder("text-embedding-3-large")

    r_sbert = _score(sbert, "SBERT all-MiniLM-L6-v2", window, near, far)
    r_large = _score(large, "OpenAI text-embedding-3-large", window, near, far)

    print("\n--- verdict ---")
    better = "large" if r_large["auc"] > r_sbert["auc"] else "SBERT"
    print(
        f"  AUC: SBERT {r_sbert['auc']:.3f}  vs  large {r_large['auc']:.3f} "
        f"-> {better} separates better"
    )
    print(
        f"  gap: SBERT {r_sbert['gap']:+.3f}  vs  large {r_large['gap']:+.3f}"
    )


if __name__ == "__main__":
    main()
