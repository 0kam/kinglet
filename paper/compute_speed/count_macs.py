# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Exact multiply-accumulates and parameters per 5-s window, counted on the distributed models.

Teacher: the Perch 2.0 CPU SavedModel (config PERCH_DIR) is one XlaCallModule (a jax2tf export). XLA lowers it to
HLO (`experimental_get_compiler_ir`, stage "hlo"), whose instructions keep the Flax module path in their metadata
(frontend / backbone / heads_protopnet_logits). MACs = every convolution (output elements x kernel elements per
output feature) and dot (output elements x contracted size); the frontend's RFFT is listed, not counted.
Parameters = the module's 497 weight arguments, BatchNorm running statistics excluded (an argument whose first
non-reshape consumer is the BatchNorm `sub` (mean) or the `add` feeding `rsqrt` (variance)).
Students: the released trunk ONNX in RELEASE_ASSETS (log-mel + backbone + embedding Gemm), same rule on Conv /
MatMul / Gemm with shapes from one onnxruntime pass; the class head is counted from the headpack (full 14,795 rows)
and the Japanese pack. With the FFT frontend (fftfront_v1) the power spectrogram comes from numpy's rfft outside the
graph (listed, not counted, as the teacher's RFFT) and the power-input trunk ONNX starts at the mel filterbank:
"fft_frontend" per tier.
Student parameters for Table 1 come from the checkpoints (PyTorch parameters, BN statistics excluded; see
figures/build_figure_data.py); here the trunk ONNX initializers are split into mel constants, embedding Gemm and
the backbone with BatchNorm folded into the convolutions (one shift per channel instead of scale + shift).

Usage: python compute_speed/count_macs.py [--out FILE]   ->  results/count_macs_v2.json + a table
(count_macs_v2 = the conv-DFT trunks of count_macs_v1 plus the FFT-frontend trunks)
"""

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PERCH_DIR, RELEASE_ASSETS, RESULTS  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
VERSION = "v0.3.2"
POWER = "v0.3.3"  # power-input trunks of the FFT frontend (fftfront_v1), exported from the v0.3.2 checkpoints
TIER_ROW = {"b": "Kinglet B", "s": "Kinglet S", "xs": "Kinglet XS"}
INSTR = re.compile(r"^\s*(?:ROOT )?%(\S+) = (\w+)\[([\d,]*)\]\S* (\S+?)\((.*?)\)(?:, (.*))?$")
OPERAND = re.compile(r"\w+\[([\d,]*)\]\S* %([\w.-]+)")


def prod(s) -> int:
    return int(np.prod([int(d) for d in s if d != ""])) if len(s) else 1


def perch_hlo(batch: int) -> str:
    import tensorflow as tf

    model = tf.saved_model.load(str(PERCH_DIR))  # keep a reference: the signature does not own the variables
    f = tf.function(lambda x: model.signatures["serving_default"](inputs=x), jit_compile=True)
    return f.experimental_get_compiler_ir(tf.TensorSpec([batch, 160000], tf.float32))(stage="hlo")


def part(op_name: str) -> str:
    return ("label_head" if "heads_" in op_name else "frontend" if "/frontend/" in op_name
            else "backbone" if "/backbone/" in op_name else "other")


def count_hlo(txt: str) -> dict:
    """MACs per model part, the FFTs, and parameters per part (BN statistics separate) of the main computation."""
    main = txt[txt.index("\n%main.") :]
    main = main[: main.index("\n}\n")]
    ins = {}
    for line in main.splitlines()[1:]:
        m = INSTR.match(line)
        if m:
            name, _, shape, op, args, attrs = m.groups()
            meta = re.search(r'op_name="([^"]*)"', attrs or "")
            ins[name] = {"op": op, "shape": shape.split(","), "operands": OPERAND.findall(args),
                         "attrs": attrs or "", "op_name": meta.group(1) if meta else ""}
    macs, ffts = collections.Counter(), []
    for v in ins.values():
        if v["op"] == "convolution":
            (_, _), (kshape, _) = v["operands"]
            klab = re.search(r"dim_labels=\w+_(\w+)->", v["attrs"]).group(1)
            k = [int(d) for d in kshape.split(",")]
            macs[part(v["op_name"])] += prod(v["shape"]) * prod(k) // k[klab.index("o")]
        elif v["op"] == "dot":
            (lshape, _), _ = v["operands"]
            dims = re.search(r"lhs_contracting_dims=\{([\d,]*)\}", v["attrs"]).group(1)
            ls = lshape.split(",")
            macs[part(v["op_name"])] += prod(v["shape"]) * prod([ls[int(d)] for d in dims.split(",")])
        elif v["op"] == "fft":
            ffts.append({"shape": v["shape"], "attrs": v["attrs"].split(", metadata")[0]})
    users = collections.defaultdict(list)
    for name, v in ins.items():
        for _, src in v["operands"]:
            users[src].append(name)

    def real_users(name):  # first consumers that are not layout ops
        out = []
        for u in users[name]:
            out += real_users(u) if ins[u]["op"] in ("reshape", "broadcast", "convert", "transpose") else [u]
        return out

    params, stats, roles = collections.Counter(), collections.Counter(), collections.Counter()
    for name, v in ins.items():
        if v["op"] != "parameter" or v["shape"][-1] == "160000":  # weights only, not the waveform
            continue
        us = real_users(name)
        u = ins[us[0]]
        role = ("mean" if u["op"] == "subtract" else "var" if u["op"] == "add" and any(
            ins[w]["op"] == "rsqrt" for w in real_users(us[0])) else "") if "BatchNorm" in u["op_name"] else ""
        where = next((p for p in (part(ins[w]["op_name"]) for w in us) if p != "other"), "other")
        (stats if role else params)[where] += prod(v["shape"])
        roles[role] += 1
    n_bn = sum(v["op"] == "rsqrt" and "BatchNorm" in v["op_name"] for v in ins.values())
    assert roles["mean"] == roles["var"] == n_bn, (roles, n_bn)
    # weight preprocessing that does not depend on the input runs once per call, whatever the batch size
    dep = {n for n, v in ins.items() if v["op"] == "parameter" and v["shape"][-1] == "160000"}
    per_call = collections.Counter()
    for name, v in ins.items():  # HLO text is in topological order
        if any(src in dep for _, src in v["operands"]):
            dep.add(name)
        elif v["op"] not in ("parameter", "constant", "reshape", "broadcast", "convert", "transpose", "iota"):
            per_call[f"{part(v['op_name'])}:{v['op']}"] += prod(v["shape"])
    return {"macs": dict(macs), "fft": ffts, "params": dict(params), "bn_stats": dict(stats), "n_batchnorm": n_bn,
            "per_call_weight_ops_elements": {k: n for k, n in per_call.items() if n >= 1e6}}


def frontend_node(name: str) -> bool:
    """Trunk graphs: backbone and embedding head live under /model/, the frontend (/mel/... in the waveform trunk,
    top-level nodes in the power-input trunk) outside it."""
    return not name.startswith("/model/")


def count_onnx(path: Path) -> dict:
    """MACs per part of a trunk ONNX (waveform or power input) for one window (shapes from one onnxruntime pass)."""
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(path))
    g = m.graph
    init = {i.name: list(i.dims) for i in g.initializer}
    nodes = [n for n in g.node if n.op_type in ("Conv", "MatMul", "Gemm")]
    have = {o.name for o in g.output}
    for name in {x for n in nodes for x in (n.input[0], n.input[1], n.output[0])} - have - set(init):
        g.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.zeros((1, 160000) if inp.name == "waveform" else (1, 501, 513), np.float32)
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp.name: x})))
    shape = lambda x: list(out[x].shape) if x in out else init[x]  # noqa: E731
    macs = collections.Counter()
    for n in nodes:
        where = "embedding_head" if n.op_type == "Gemm" else "frontend" if frontend_node(n.name) else "backbone"
        if n.op_type == "Conv":
            k = prod(shape(n.input[1])[1:])  # (Cout, Cin / groups, kh, kw)
        else:
            trans_a = any(a.name == "transA" and a.i for a in n.attribute)
            k = shape(n.input[0])[-2 if trans_a else -1]
        macs[where] += prod(shape(n.output[0])) * k
    feat = out["feature"].shape[1]
    return {"macs": dict(macs), "feature_dim": feat}


def student_params(path: Path) -> dict:
    """Trunk ONNX initializers: mel constants (frontend), embedding Gemm, backbone (BatchNorm folded)."""
    import onnx

    g = onnx.load(str(path)).graph
    init = {i.name: int(np.prod(i.dims)) for i in g.initializer}
    used = collections.defaultdict(set)
    for n in g.node:
        for x in n.input:
            if x in init:
                used[x].add(frontend_node(n.name))
    mel = sum(init[x] for x, w in used.items() if w == {True})
    emb = sum(init[x] for n in g.node if n.op_type == "Gemm" for x in n.input[1:] if x in init)
    return {"onnx_initializers": sum(init.values()), "mel_constants": mel, "embedding_head": emb,
            "backbone_bn_folded": sum(init.values()) - mel - emb}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=RESULTS / "count_macs_v2.json")
    args = ap.parse_args()

    res = {"perch_dir": "kagglehub google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"}
    h1, h2 = count_hlo(perch_hlo(1)), count_hlo(perch_hlo(2))
    assert all(h2["macs"][k] == 2 * v for k, v in h1["macs"].items()), (h1["macs"], h2["macs"])  # per window
    assert h1["params"] == h2["params"]
    import tensorflow as tf

    stored = sum(int(np.prod(s)) for n, s in tf.train.list_variables(str(PERCH_DIR / "variables/variables"))
                 if n.startswith("_tf_var_leaves"))
    assert stored == sum(h1["params"].values()) + sum(h1["bn_stats"].values()), stored
    res["teacher"] = h1 | {"stored_values": stored}
    res["teacher"]["macs"]["total"] = sum(h1["macs"].values())
    res["students"] = {}
    for tier in ("b", "s", "xs"):
        pack = np.load(RELEASE_ASSETS / f"kinglet-{tier}-{VERSION}.pack-jp.npz")["W"].shape[0]
        variants = {}
        for name, path in (("conv_dft", RELEASE_ASSETS / f"kinglet-{tier}-{VERSION}.onnx"),
                           ("fft_frontend", RELEASE_ASSETS / f"kinglet-{tier}-{POWER}.power.onnx")):
            r = count_onnx(path)
            d = r["feature_dim"]
            r["macs"] |= {"class_head_full": 14795 * d, "class_head_jp_pack": pack * d}
            trunk = r["macs"]["frontend"] + r["macs"]["backbone"] + r["macs"]["embedding_head"]
            # timed paths: exact fold (default on computers) and compact JP pack (microcontrollers)
            r["macs"] |= {"total_exact": trunk + r["macs"]["class_head_full"], "total_pack": trunk + pack * d}
            r["params"] = student_params(path) | {"class_head_full": 14795 * (d + 1),
                                                  "class_head_jp_pack": pack * (d + 1), "jp_pack_rows": pack}
            variants[name] = r
        fft = variants["fft_frontend"]
        assert fft["macs"]["backbone"] == variants["conv_dft"]["macs"]["backbone"]
        fft["fft"] = [{"shape": [1, 501, 513], "fft_length": 1024,
                       "where": "numpy rfft outside the graph (kinglet.power_spectrogram); listed, not counted"}]
        res["students"][tier] = variants["conv_dft"] | {"fft_frontend": fft}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))

    t = res["teacher"]
    print("GMACs per 5-s window (exact, distributed graphs)")
    print(f"{'model':12} {'frontend':>9} {'backbone':>9} {'emb head':>9} {'class head':>11} {'JP pack':>8} "
          f"{'total exact':>12} {'total pack':>11}")
    print(f"{'Perch 2.0':12} {t['macs']['frontend'] / 1e9:9.3f} {t['macs']['backbone'] / 1e9:9.3f} {0:9.3f} "
          f"{t['macs']['label_head'] / 1e9:11.3f} {'-':>8} {t['macs']['total'] / 1e9:12.3f} {'-':>11}")
    for tier, r in [(t, r) for t, r in res["students"].items()] + [(t + "_fft", r["fft_frontend"])
                                                                    for t, r in res["students"].items()]:
        m = r["macs"]
        row = TIER_ROW[tier.removesuffix("_fft")] + (" FFT" if tier.endswith("_fft") else "")
        print(f"{row:12} {m['frontend'] / 1e9:9.3f} {m['backbone'] / 1e9:9.3f} "
              f"{m['embedding_head'] / 1e9:9.3f} {m['class_head_full'] / 1e9:11.3f} {m['class_head_jp_pack'] / 1e9:8.4f} "
              f"{m['total_exact'] / 1e9:12.3f} {m['total_pack'] / 1e9:11.3f}")
    print(f"teacher FFT: {t['fft']}")
    print(f"teacher params: {t['params']} (+ BN statistics {t['bn_stats']}; stored values {t['stored_values']:,})")
    for tier, r in res["students"].items():
        print(TIER_ROW[tier], r["params"], "FFT frontend:", r["fft_frontend"]["params"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
