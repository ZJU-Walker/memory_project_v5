"""Diagnostic (no training): how distinguishable are the v5 sentence encodings, keys and values
for the two side variants of the inspect sentence, under the ckpt-999 stage-A parameters?
CPU-only; runs on a fast-NFS host."""
import os, sys, time, json
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MEMORY_PROJECT_ROOT", "/iris/u/kewalk/memory_project_v5")
import numpy as np, jax, jax.numpy as jnp
import openpi.training.config as _config, openpi.models.model as _model
import openpi.models.tokenizer as _tok
from openpi.shared import download
import sentencepiece

params_dir = sys.argv[1]
cfg = _config.get_config("pi05_yam_mem_v5_stageA")
t0 = time.time()
params = _model.restore_params(params_dir, restore_type=np.ndarray)
model = cfg.model.load(params); model.eval()
print("model loaded in %.0fs" % (time.time() - t0), flush=True)
sp = sentencepiece.SentencePieceProcessor(model_file="/iris/u/kewalk/memory_project_v5/v35/cache/openpi/big_vision/paligemma_tokenizer.model")
sentences = {
    "inspect_L": "inspect both bins: banana left, grey pepper box right",
    "inspect_R": "inspect both bins: banana right, grey pepper box left",
    "open_lids": "open both lids",
    "close": "close both lids and reset arms",
    "wait_L": "wait; target bin is left",
    "wait_R": "wait; target bin is right",
}
L = cfg.model.memory_v5_sentence_len
toks, mask = [], []
for s in sentences.values():
    ids = sp.encode(s.lower().strip() + "\n")
    row = np.zeros(L, np.int32); m = np.zeros(L, bool); row[:len(ids)] = ids; m[:len(ids)] = True
    toks.append(row); mask.append(m)
toks = jnp.asarray(np.stack(toks)); mask = jnp.asarray(np.stack(mask))
e = model.v5_encode_sentence(toks, mask)
k, v = model.v5_sentence_intent(e)
k = k[:, 0]; v = v[:, 0]
names = list(sentences)
def cos(a, b): return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-9))
print("\npairwise cosine of encodings e / keys k / values v:")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"  {names[i]:10s} vs {names[j]:10s}  e={cos(e[i], e[j]):.4f}  k={cos(k[i], k[j]):.4f}  v={cos(v[i], v[j]):.4f}")
# what a bank holding [open_lids, inspect_L, close] returns for a query aligned to each key, and
# whether the inspect_L vs inspect_R difference survives a 3-sentence bank
state = model.memory_semantic.init_state(2)
order = ["open_lids", "inspect_L", "close"]
for name in order:
    i = names.index(name); j = names.index("inspect_R") if name == "inspect_L" else i
    kk = jnp.stack([k[i], k[j]])[:, None]; vv = jnp.stack([v[i], v[j]])[:, None]
    state, aux = model.v5_semantic_write(state, kk, vv, jnp.asarray([True, True]))
iL, iR = names.index("inspect_L"), names.index("inspect_R")
readL = model.memory_semantic.read_key(state, jnp.stack([k[iL], k[iR]])[:, None])[:, 0]
print("\nbank A holds [open, inspect_L, close]; bank B holds [open, inspect_R, close]")
print("  read(bank A, key inspect_L) vs value inspect_L cos=%.4f" % cos(readL[0], v[iL]))
print("  read(bank B, key inspect_R) vs value inspect_R cos=%.4f" % cos(readL[1], v[iR]))
print("  read(bank A) vs read(bank B) cos=%.4f  (1.0 = the side is invisible to the reader)" % cos(readL[0], readL[1]))
print("  |read A - read B| / |read A| = %.4f" % float(jnp.linalg.norm(readL[0] - readL[1]) / (jnp.linalg.norm(readL[0]) + 1e-9)))
