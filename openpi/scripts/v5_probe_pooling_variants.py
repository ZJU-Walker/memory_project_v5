"""Diagnostic (CPU, no training): side-distinguishability of candidate sentence poolings on the
frozen blocks-0..8 token states, with a trained stage-A checkpoint. For the two inspect-sentence
side variants (and the other phase sentences) reports the cosine of the pooled vectors."""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MEMORY_PROJECT_ROOT", "/iris/u/kewalk/memory_project_v5")
import numpy as np, jax, jax.numpy as jnp
import openpi.training.config as _config, openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
import sentencepiece

params_dir = sys.argv[1]
cfg = _config.get_config("pi05_yam_mem_v5_stageA")
t0 = time.time()
model = cfg.model.load(_model.restore_params(params_dir, restore_type=np.ndarray)); model.eval()
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
names = list(sentences); L = cfg.model.memory_v5_sentence_len
toks = np.zeros((len(names), L), np.int32); mask = np.zeros((len(names), L), bool); pieces = []
for i, s in enumerate(sentences.values()):
    ids = sp.encode(s.lower().strip() + "\n"); toks[i, :len(ids)] = ids; mask[i, :len(ids)] = True
    pieces.append([sp.id_to_piece(t) for t in ids])
print("inspect_L pieces:", pieces[0])
toks = jnp.asarray(toks); mask = jnp.asarray(mask)

# token states from the same text-only pass v5_encode_sentence uses (pre-pooling)
b, n = toks.shape
depth = model.PaliGemma.llm.module.configs[0].depth
emb = model.PaliGemma.llm(jnp.where(mask, toks, 0), method="embed")
cache = model._v32_empty_cache(b, n, emb.dtype)
attn = model._pad_attention_columns(make_attn_mask(mask, jnp.zeros(mask.shape, jnp.int32)), n)
pos = jnp.maximum(jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1, 0)
(h, _), _ = model.PaliGemma.llm([emb, None], mask=attn, positions=pos, kv_cache=cache, cache_position=0,
                                active_layers=jnp.arange(depth) <= model.memory_layer, apply_final_norm=False)
h = np.asarray(h.astype(jnp.float32)); e0 = np.asarray(emb.astype(jnp.float32)); m = np.asarray(mask)
def cosine(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
def report(label, vecs):
    iL, iR = names.index("inspect_L"), names.index("inspect_R")
    others = [cosine(vecs[iL], vecs[names.index(k)]) for k in ("open_lids", "close", "wait_L")]
    print(f"  {label:34s} cos(inspect_L, inspect_R)={cosine(vecs[iL], vecs[iR]):.4f}   cos(inspect_L, open/close/wait_L)={others[0]:.3f}/{others[1]:.3f}/{others[2]:.3f}")
# global mean token state over all valid tokens of all sentences (proxy for a fixed centering vector)
valid = h[m]; mu = valid.mean(0); sd = valid.std(0) + 1e-6
def mean_pool(x): return np.stack([x[i][m[i]].mean(0) for i in range(b)])
def last_tok(x): return np.stack([x[i][m[i]][-1] for i in range(b)])
def side_positions(i):
    return [j for j, pc in enumerate(pieces[i]) if pc.strip("▁") in ("left", "right")]
def side_states(x):
    out = []
    for i in range(b):
        ps = side_positions(i)
        sel = [x[i][j] for j in ps] if ps else [x[i][m[i]].mean(0)]
        while len(sel) < 2:
            sel.append(sel[-1])
        out.append(np.concatenate(sel[:2]))
    return out
print("\nlayer-8 token states (what the v5 encoder pools):")
report("mean (current v5)", mean_pool(h))
report("last token", last_tok(h))
report("mean, centered (h - mu)", mean_pool(h - mu))
report("mean, standardized ((h-mu)/sd)", mean_pool((h - mu) / sd))
report("max over tokens", np.stack([h[i][m[i]].max(0) for i in range(b)]))
report("states at the side words, concat", side_states(h))
report("side-word states, centered", side_states(h - mu))
report("side-word states, standardized", side_states((h - mu) / sd))
# standardized mean with per-sentence statistics only (no reference set): degenerate (mean of centered = 0)?
report("mean, per-sentence standardized", mean_pool(np.stack([(h[i] - h[i][m[i]].mean(0)) / (h[i][m[i]].std(0) + 1e-6) for i in range(b)])))
print("\nraw token embeddings (layer 0):")
report("mean", mean_pool(e0))
report("states at the side words, concat", side_states(e0))
# anisotropy: how much of the token-state norm is the common direction
proj = (valid @ (mu / np.linalg.norm(mu)))
print("\nanisotropy: mean token norm %.1f, projection on the common direction %.1f (%.0f%%)" % (np.linalg.norm(valid, axis=1).mean(), proj.mean(), 100 * proj.mean() / np.linalg.norm(valid, axis=1).mean()))
