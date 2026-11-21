# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Recompute the `label` output of Perch 2.0 from `spatial_embedding` and the prototype head weights. Usage: python -I recompute.py <savedmodel_dir>"""
import sys, numpy as np, tensorflow as tf
d = sys.argv[1]
m = tf.saved_model.load(d)
rng = np.random.default_rng(0)
t = np.arange(160000) / 32000
x = np.stack([0.1 * rng.standard_normal(160000), 0.3 * np.sin(2 * np.pi * 3000 * t) * (np.sin(2*np.pi*2*t) > 0) + 0.01 * rng.standard_normal(160000)]).astype(np.float32)
out = {k: v.numpy() for k, v in m.signatures["serving_default"](inputs=tf.constant(x)).items()}
ck = tf.train.load_checkpoint(d + "/variables/variables")
g = lambda i: ck.get_tensor(f"_tf_var_leaves/{i}/.ATTRIBUTES/VARIABLE_VALUE")
bias, pk, P = g(494), g(495), g(496)
eps = 1e-5
U = P / (np.linalg.norm(P, axis=1, keepdims=True) + eps)            # [C,1536,4]
s = out["spatial_embedding"]                                          # [B,16,4,1536]
u = s / (np.linalg.norm(s, axis=-1, keepdims=True) + eps)
sims = np.einsum("btfd,cdk->btfck", u, U, optimize=True).max(axis=(1, 2))   # [B,C,4]
label = (sims * np.maximum(pk, 0)).sum(-1) + bias
print("max|label - protopnet(spatial)|:", np.abs(label - out["label"]).max(), "label range", out["label"].min(), out["label"].max())
print("embedding == mean(spatial)?", np.abs(out["embedding"] - s.mean(axis=(1, 2))).max())
print("frac protop_kernel<=0:", (pk <= 0).mean(), "bias range", bias.min(), bias.max())
