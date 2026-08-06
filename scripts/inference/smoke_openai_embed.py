"""Smoke test: can a COMPUTE node reach the OpenAI embedding API?

The `internet` job proved outbound HTTPS to HuggingFace/Cloudflare. This proves
the *specific* path we need for the recovery metric + topic distance:
text-embedding-3-large via babel's reference_embedder(). Checks:
  1. the API call succeeds from the compute node (key + network + firewall),
  2. the vector is the right size (3072 dims for text-embedding-3-large),
  3. the geometry is sane: a paraphrase pair is MORE similar than an unrelated
     pair (so the embedder is actually returning meaningful vectors, not junk).
Exit code 0 = PASS, non-zero = FAIL.
"""
import socket
import sys

import torch
import torch.nn.functional as F

from babel_ai.embeddings import reference_embedder

print(f"host: {socket.gethostname()}", flush=True)

emb = reference_embedder()                       # OpenAIEmbedder("text-embedding-3-large")
texts = [
    "The cat sat quietly on the warm windowsill.",     # A
    "A feline rested calmly on the sunny ledge.",       # A' (paraphrase of A)
    "Quantum chromodynamics describes the strong force between quarks.",  # B (unrelated)
]
vecs = emb.encode(texts, convert_to_tensor=True)
dim = vecs.shape[-1]
sim_para = F.cosine_similarity(vecs[0], vecs[1], dim=0).item()   # A vs A'
sim_unrel = F.cosine_similarity(vecs[0], vecs[2], dim=0).item()  # A vs B

print(f"embedding dim: {dim}", flush=True)
print(f"cos(paraphrase pair)  A~A' = {sim_para:.3f}", flush=True)
print(f"cos(unrelated pair)   A~B  = {sim_unrel:.3f}", flush=True)

ok_dim = dim == 3072
ok_geom = sim_para > sim_unrel
print(f"dim==3072: {ok_dim} | paraphrase>unrelated: {ok_geom}", flush=True)
if ok_dim and ok_geom:
    print("SMOKE TEST PASS", flush=True)
    sys.exit(0)
print("SMOKE TEST FAIL", flush=True)
sys.exit(1)
