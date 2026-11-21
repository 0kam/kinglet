# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Export a distilled student as ONNX: waveform-in (preprocessing included) or power-in (body only).

The waveform-in graph is: waveform (B, 160000) float32 in [-1, 1] at 32 kHz mono
  -> log-mel (reflect padding + periodic Hann + mel filterbank, numerically matching the torchaudio
     MelSpectrogram used in training). --mel-impl picks how the power spectrogram is computed inside
     the graph: conv-DFT (default, fastest of the three on x86 in ONNX Runtime; 0.526 GMACs per window),
     framing + GEMM with the DFT matrix, or the ONNX STFT op (slow in ORT CPU 1.24)
  -> normalize (cfg mu/sd) -> backbone
  -> outputs "embedding" (B, 1536; the projection into Perch 2.0 embedding space)
     and "logits" (B, 317 pre-sigmoid for BCE checkpoints; B, 318 for softmax-distilled v0.2
     checkpoints, where scores = softmax over all 318 and the last channel is background/"other";
     classes = <out>.classes.csv, 317 rows either way).
--mel-impl power exports only the body: input "power" (B, 501, 513), the power spectrogram, which the
caller computes with an FFT (kinglet.power_spectrogram, numpy rfft; fftfront_v1). Everything after it
(mel filterbank, log, normalization, backbone) is the same graph as the other implementations.

Verification: on real audio the ONNX outputs must match the torch pipeline
(evaluate.accuracy.score_student.Distilled) within tolerance (embedding cosine, logit max abs diff). The audio
is the first eight WABAD test windows (DATA_ROOT/wabad/test.parquet, 5-s window files sorted by path).

Usage:
  python release/export_onnx.py --ckpt DISTILL_ROOT/g3conf_v1/efficientnet_lite0_full/best.pt --split \\
      --out kinglet-s.onnx
  python release/export_onnx.py --ckpt DISTILL_ROOT/g2xcl_v1/mobilenetv4_conv_small_full/best.pt --split \\
      --mel-impl power --out kinglet-xs.power.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT  # noqa: E402
from evaluate.accuracy.score_student import Distilled  # noqa: E402
from teacher.embed_perch import decode, windows  # noqa: E402
from train.backbones import MEL, SR  # noqa: E402


class Body(torch.nn.Module):
    """Everything after the power spectrogram (torch-exportable): mel -> log -> norm -> model."""

    def __init__(self, d: Distilled, split: bool = False):
        super().__init__()
        import torchaudio

        fb = torchaudio.functional.melscale_fbanks(MEL["n_fft"] // 2 + 1, 0.0, SR / 2,
                                                   MEL["n_mels"], SR, norm=None, mel_scale="htk")
        self.register_buffer("melfb", fb)  # (513, 128)
        self.register_buffer("mu", torch.tensor(float(d.cfg["mel_mu"])))
        self.register_buffer("sd", torch.tensor(float(d.cfg["mel_sd"])))
        self.model = d.model
        self.split = split

    def forward(self, power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = (power @ self.melfb).transpose(1, 2)  # (B, 501, 513) -> (B, 128, 501)
        x = ((torch.log(mel + 1e-6) - self.mu) / self.sd).unsqueeze(1)
        h, emb, logits = self.model(x)
        return (emb, h) if self.split else (emb, logits)


def gemm_prefix() -> "onnx.ModelProto":
    """waveform -> power via framing (Gather) + one GEMM with the DFT matrix. Numerically identical
    to conv-DFT; faster on some platforms (e.g. ARM), slower on others - benchmark per target."""
    import onnx
    from onnx import TensorProto as TP
    from onnx import helper

    n_fft, hop, t = MEL["n_fft"], MEL["hop_length"], 501
    win = torch.hann_window(n_fft).numpy()
    k = 2 * np.pi * np.arange(n_fft)[None, :] * np.arange(n_fft // 2 + 1)[:, None] / n_fft
    w = (np.concatenate([np.cos(k), -np.sin(k)]).astype(np.float32) * win).T.copy()  # (1024, 1026)
    idx = (np.arange(t)[:, None] * hop + np.arange(n_fft)[None, :]).astype(np.int64)
    nodes = [
        helper.make_node("Pad", ["waveform", "pads"], ["padded"], mode="reflect"),
        helper.make_node("Gather", ["padded", "idx"], ["frames"], axis=1),
        helper.make_node("MatMul", ["frames", "W"], ["z"]),
        helper.make_node("Mul", ["z", "z"], ["z2"]),
        helper.make_node("Slice", ["z2", "s0", "s1", "sax"], ["re2"]),
        helper.make_node("Slice", ["z2", "s1", "s2", "sax"], ["im2"]),
        helper.make_node("Add", ["re2", "im2"], ["power"]),
    ]
    graph = helper.make_graph(
        nodes, "gemm_prefix",
        [helper.make_tensor_value_info("waveform", TP.FLOAT, ["batch", 5 * SR])],
        [helper.make_tensor_value_info("power", TP.FLOAT, ["batch", t, n_fft // 2 + 1])],
        initializer=[helper.make_tensor("pads", TP.INT64, [4], [0, n_fft // 2, 0, n_fft // 2]),
                     helper.make_tensor("idx", TP.INT64, list(idx.shape), idx.ravel()),
                     helper.make_tensor("W", TP.FLOAT, list(w.shape), w.ravel()),
                     *[helper.make_tensor(n, TP.INT64, [1], [v])
                       for n, v in (("s0", 0), ("s1", n_fft // 2 + 1), ("s2", n_fft + 2), ("sax", 2))]])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def stft_prefix() -> "onnx.ModelProto":
    """waveform (B, 160000) -> power spectrogram (B, 501, 513) via the ONNX STFT op (opset 17)."""
    import onnx
    from onnx import TensorProto as TP
    from onnx import helper

    n_fft, hop = MEL["n_fft"], MEL["hop_length"]
    win = torch.hann_window(n_fft).numpy()  # periodic Hann, as in torchaudio
    nodes = [
        helper.make_node("Pad", ["waveform", "pads"], ["padded"], mode="reflect"),
        helper.make_node("Unsqueeze", ["padded", "ax2"], ["sig"]),
        helper.make_node("STFT", ["sig", "step", "window"], ["spec"], onesided=1),
        helper.make_node("Mul", ["spec", "spec"], ["sq"]),
        helper.make_node("ReduceSum", ["sq", "ax3"], ["power"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "stft_prefix",
        [helper.make_tensor_value_info("waveform", TP.FLOAT, ["batch", 5 * SR])],
        [helper.make_tensor_value_info("power", TP.FLOAT, ["batch", 501, n_fft // 2 + 1])],
        initializer=[
            helper.make_tensor("pads", TP.INT64, [4], [0, n_fft // 2, 0, n_fft // 2]),
            helper.make_tensor("ax2", TP.INT64, [1], [2]),
            helper.make_tensor("ax3", TP.INT64, [1], [3]),
            helper.make_tensor("step", TP.INT64, [], [hop]),
            helper.make_tensor("window", TP.FLOAT, [n_fft], win),
        ])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    return m


class ConvMel(torch.nn.Module):
    """log-mel as exportable ops: reflect pad -> DFT as Conv1d -> power -> mel matmul -> log."""

    def __init__(self, mu: float, sd: float):
        super().__init__()
        import torchaudio

        n_fft, hop, n_mels = MEL["n_fft"], MEL["hop_length"], MEL["n_mels"]
        self.pad = n_fft // 2
        win = torch.hann_window(n_fft)
        # DFT angles in float64 from the exact integer phase n*k mod n_fft: float32 angles (as in the v0.1-v0.3.2
        # trunks) carry up to 3e-4 rad of error, which leaks ~1e-10 of a frame's energy into empty bins and moves
        # log-mel cells of loud recordings with empty bands by up to 2.5 (fftfront_v1)
        nk = (torch.arange(n_fft)[None, :] * torch.arange(n_fft // 2 + 1)[:, None]) % n_fft
        k = 2 * torch.pi * nk.double() / n_fft
        self.dft = torch.nn.Conv1d(1, 2 * (n_fft // 2 + 1), n_fft, hop, bias=False)
        self.dft.weight.data = torch.cat([torch.cos(k), -torch.sin(k)]).float().unsqueeze(1) * win
        self.dft.weight.requires_grad_(False)
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 0.0, SR / 2, n_mels, SR,
                                                   norm=None, mel_scale="htk")
        self.register_buffer("melfb", fb.T)  # (n_mels, n_fft//2+1)
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("sd", torch.tensor(sd))
        self.n_bins = n_fft // 2 + 1

    def forward(self, a: torch.Tensor) -> torch.Tensor:  # (B, 160000) -> (B, 1, 128, 501)
        x = torch.nn.functional.pad(a.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        z = self.dft(x)
        power = z[:, : self.n_bins] ** 2 + z[:, self.n_bins :] ** 2
        mel = self.melfb @ power
        return ((torch.log(mel + 1e-6) - self.mu) / self.sd).unsqueeze(1)


class Wrapped(torch.nn.Module):
    """split=False: (embedding, logits) as deployed through v0.2 (single-file model).
    split=True: (embedding, feature) — the trunk of a v0.3 split deployment, where a regional
    head pack (kinglet build-pack) turns the trunk feature into scores."""

    def __init__(self, d: Distilled, split: bool = False):
        super().__init__()
        self.mel = ConvMel(d.cfg["mel_mu"], d.cfg["mel_sd"])
        self.model = d.model
        self.split = split

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, emb, logits = self.model(self.mel(waveform))
        return (emb, h) if self.split else (emb, logits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mel-impl", choices=["conv", "gemm", "stft", "power"], default="conv",
                    help="power spectrogram inside the graph: conv-DFT (fastest in-graph on x86), "
                         "framing+GEMM (try on ARM), the ONNX STFT op (slow in ORT CPU 1.24), or none: "
                         "'power' exports the body only, fed by kinglet.power_spectrogram (numpy FFT)")
    ap.add_argument("--split", action="store_true",
                    help="export the trunk only: outputs (embedding, feature); scores come from a "
                         "head pack applied outside the graph (v0.3 split deployment)")
    args = ap.parse_args()

    import onnx
    from onnx.compose import merge_models

    d = Distilled(args.ckpt, "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_names = ["embedding", "feature" if args.split else "logits"]
    if args.mel_impl == "conv":
        w = Wrapped(d, args.split).eval()
        torch.onnx.export(w, torch.zeros(2, 5 * SR), str(args.out), dynamo=False, opset_version=17,
                          input_names=["waveform"], output_names=out_names,
                          dynamic_axes={"waveform": {0: "batch"}})
    else:
        body = Body(d, args.split).eval()
        tmp = args.out.with_suffix(".body.onnx")
        torch.onnx.export(body, torch.zeros(2, 501, MEL["n_fft"] // 2 + 1), str(tmp), dynamo=False,
                          opset_version=17, input_names=["power"], output_names=out_names,
                          dynamic_axes={"power": {0: "batch"}})
        if args.mel_impl == "power":
            tmp.replace(args.out)
        else:
            body_m = onnx.load(str(tmp))
            pre = gemm_prefix() if args.mel_impl == "gemm" else stft_prefix()
            pre.ir_version = body_m.ir_version  # merge_models requires matching IR versions
            merged = merge_models(pre, body_m, io_map=[("power", "power")])
            onnx.save(merged, str(args.out))
            tmp.unlink()

    if not args.split:  # classes csv next to the model (split: the head pack carries the codes)
        import pandas as pd

        from teacher.embed_perch import SPECIES_LIST

        sl = pd.read_csv(SPECIES_LIST).drop_duplicates("ebird_species_code").set_index("ebird_species_code")
        codes = [c["species"] for c in d.cfg["classes"]]
        pd.DataFrame({"index": range(len(codes)), "ebird_code": codes,
                      "scientific_name": [sl["ebird_sciname"].get(c, "") for c in codes],
                      "japanese_name": [sl["japanese_name"].get(c, "") for c in codes]}).to_csv(
            args.out.with_suffix(".classes.csv"), index=False)

    # verify on real audio: ONNX (ort) vs the torch eval pipeline (torchaudio mel, Distilled path)
    import onnxruntime as ort
    import pandas as pd

    paths = sorted(pd.read_parquet(DATA_ROOT / "wabad/test.parquet")["path"].unique())[:8]
    x = np.stack([windows(decode(str(p), 5), 5.0)[1][0] for p in paths])
    with torch.no_grad():
        xt = torch.from_numpy(x)
        m = ((torch.log(d.mel.cpu()(xt) + 1e-6) - d.cfg["mel_mu"]) / d.cfg["mel_sd"]).unsqueeze(1)
        h_ref, emb_ref, log_ref = d.model(m)
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    if args.mel_impl == "power":
        from kinglet import power_spectrogram

        emb, second = sess.run(None, {"power": power_spectrogram(x)})
    else:
        emb, second = sess.run(None, {"waveform": x})
    ref = h_ref if args.split else log_ref
    cos = torch.nn.functional.cosine_similarity(torch.from_numpy(emb), emb_ref).numpy()
    dsec = np.abs(second - ref.numpy()).max()
    mb = args.out.stat().st_size / 1e6
    name2 = "feature" if args.split else "logit"
    print(f"{args.out.name}: {mb:.1f} MB; verify on {len(x)} real windows: "
          f"emb cos min {cos.min():.6f}, {name2} max|diff| {dsec:.4f}")
    assert cos.min() > 0.9999 and dsec < 0.05, "ONNX does not match the torch pipeline"
    print("OK")


if __name__ == "__main__":
    main()
