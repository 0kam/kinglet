# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Export Kinglet XS as the int8 TFLite model of the microcontroller (Grove Vision AI V2, Ethos-U55).

Deploy graph = trunk (normalized log-mel (1, 1, 128, 501) -> feature) + the compact Japanese output layer
(318 rows = 317 species + "other", no sound-group rows) + softmax. Only public audio is used: the int8 ranges are
calibrated on 512 focal validation windows (8 validation parts spread over the training sources x 64 windows);
the parity check uses 2,048 WABAD windows evenly spaced over the manifest (never used for calibration).

Three stages; the TFLite converter needs its own environment (litert-torch 0.8.0 requires torch < 2.10; see
mcu/BUILD.md):

  stage pack (main environment): the compact output layer, kinglet.pack.build_pack on the focal-only output
    layer bundle (<ckpt dir>/headpack_focal.npz, evaluate/species_lists/export_headpack.py --focal-only) for the
    317 Japanese species with no sound groups -> pack_jp_focal.npz
  stage prep (main environment, GPU optional):
    - deploy_state.pt {backbone state_dict, W, b, codes, mel_mu, mel_sd}
    - calib.npz: the 512 normalized log-mel calibration windows (fp16)
    - parity.npz: the 2,048 WABAD windows (fp16) + fp32 torch reference probabilities
  stage convert (converter environment): rebuild the graph, pt2e static int8 (per-channel weights, calibration =
    calib.npz), export TFLite, parity against the fp32 probabilities -> parity_results.json

Usage:
  python mcu/export_int8.py --stage pack
  python mcu/export_int8.py --stage prep
  <converter env>/bin/python mcu/export_int8.py --stage convert
Output: DISTILL_ROOT/xs_pub_deploy/{pack_jp_focal.npz, deploy_state.pt, calib.npz, parity.npz,
        kinglet-xs_pub-jp-int8.tflite, parity_results.json}; then Vela (mcu/BUILD.md).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT, DISTILL_ROOT, jp_list  # noqa: E402

TAG = "xs_pub"
OUT = DISTILL_ROOT / f"{TAG}_deploy"
CKPT = DISTILL_ROOT / "g2xcl_v1/mobilenetv4_conv_small_full/best.pt"  # Kinglet XS
FIRE_THR = 0.1


class Deploy(torch.nn.Module):
    """Trunk + folded regional head + softmax: the graph that runs on the NPU."""

    def __init__(self, backbone: str, n_rows: int):
        super().__init__()
        import timm

        self.net = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=1).eval()
        with torch.no_grad():
            d = self.net(torch.zeros(1, 1, 128, 501)).shape[1]
        self.head = torch.nn.Linear(d, n_rows)

    def forward(self, x: torch.Tensor):
        e = self.net(x)  # the embedding feature is exposed as a second output
        return torch.softmax(self.head(e), 1), e


def build_focal_pack(ckpt: Path, dst: Path) -> None:
    """Compact Japanese output layer (ridge "other" row, no sound-group rows) from the focal-only bundle."""
    from kinglet.pack import build_pack

    species = [{"ebird_code": r["species"], "scientific_name": "", "vocab_index": r["perch_idx"]} for r in jp_list()]
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build_pack(ckpt.parent / "headpack_focal.npz", species, dst, groups={}, compact=True)
    print(json.dumps(rep, indent=1, default=str))


def focal_mels(n_parts: int, per_part: int, mu: float, sd: float) -> np.ndarray:
    """Normalized log-mels of focal val windows, spread over the public training sources."""
    from train.train_distill import SOURCES, parts_index

    parts = [p for p in parts_index(0, SOURCES) if p["val"].sum() >= per_part]
    wins = []
    for p in parts[:: len(parts) // n_parts][:n_parts]:
        rows = np.flatnonzero(p["val"])
        rows = rows[np.linspace(0, len(rows) - 1, per_part).astype(int)]
        m = np.asarray(np.load(p["npy"], mmap_mode="r")[rows], np.float32)
        wins.append(((m - mu) / sd)[:, None].astype(np.float16))
    return np.concatenate(wins)


def wabad_mels(n: int, mel, mu: float, sd: float, dev: str) -> np.ndarray:
    """Normalized log-mels of n WABAD windows evenly spaced over the manifest (one 5 s file each)."""
    import pandas as pd

    from teacher.embed_perch import decode, windows

    paths = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")["path"]
    w = np.stack([windows(decode(p, 0), 5.0)[1][0] for p in paths.iloc[np.linspace(0, len(paths) - 1, n).astype(int)]])
    m = torch.cat([(torch.log(mel(torch.from_numpy(w[i:i + 64]).to(dev)) + 1e-6) - mu) / sd
                   for i in range(0, n, 64)])
    return m.unsqueeze(1).cpu().numpy().astype(np.float16)


def prep(ckpt: Path, pack_path: Path) -> None:
    import torchaudio

    from train.backbones import MEL, SR
    from train.train_distill import Student

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    pack = np.load(pack_path, allow_pickle=True)
    W, b = pack["W"].astype(np.float32), pack["b"].astype(np.float32)

    student = Student(cfg["backbone"], cfg["n_out"], head=cfg.get("head", "full"))
    student.load_state_dict(state["model"])
    deploy = Deploy(cfg["backbone"], len(W))
    deploy.net.load_state_dict(student.backbone.state_dict())
    deploy.head.weight.data = torch.from_numpy(W)
    deploy.head.bias.data = torch.from_numpy(b)
    deploy = deploy.to(dev).eval()

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL).to(dev)
    calib = focal_mels(8, 64, cfg["mel_mu"], cfg["mel_sd"])
    parity = wabad_mels(2048, mel, cfg["mel_mu"], cfg["mel_sd"], dev)

    dead = b < -1e8  # dead rows (bias -1e9, species without a class) would blow up the int8 logit scale:
    if dead.any():   # clamp below the observed real-logit floor (still ~0 prob after softmax)
        with torch.no_grad():
            lg = deploy.head(deploy.net(torch.from_numpy(calib[:64]).float().to(dev)))
        floor = float(lg[:, ~torch.from_numpy(dead).to(dev)].min())
        W[dead], b[dead] = 0.0, floor - 10.0
        deploy.head.weight.data = torch.from_numpy(W).to(dev)
        deploy.head.bias.data = torch.from_numpy(b).to(dev)
        print(f"clamped {int(dead.sum())} dead rows to logit {floor - 10.0:.1f}")
    with torch.no_grad():
        ref = np.concatenate([deploy(torch.from_numpy(parity[i:i + 64]).float().to(dev))[0].cpu().numpy()
                              for i in range(0, len(parity), 64)])

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": cfg["backbone"], "net": deploy.net.state_dict(), "W": W, "b": b,
                "codes": pack["codes"], "mel_mu": cfg["mel_mu"], "mel_sd": cfg["mel_sd"]},
               OUT / "deploy_state.pt")
    np.savez(OUT / "calib.npz", mels=calib)
    np.savez(OUT / "parity.npz", mels=parity, ref=ref)
    print(f"prep: calib {calib.shape}, parity {parity.shape}, ref {ref.shape} -> {OUT}")


def convert() -> None:
    import litert_torch
    from litert_torch.quantize import pt2e_quantizer, quant_config
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    state = torch.load(OUT / "deploy_state.pt", map_location="cpu", weights_only=False)
    deploy = Deploy(state["backbone"], len(state["W"]))
    deploy.net.load_state_dict(state["net"])
    deploy.head.weight.data = torch.from_numpy(state["W"])
    deploy.head.bias.data = torch.from_numpy(state["b"])
    deploy.eval()

    calib = np.load(OUT / "calib.npz")["mels"]
    sample = (torch.zeros(1, 1, 128, 501),)
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True))
    exported = torch.export.export_for_training(deploy, sample).module()
    prepared = prepare_pt2e(exported, quantizer)
    move_exported_model_to_eval(prepared)
    with torch.no_grad():  # the exported graph is fixed at batch 1
        for i in range(len(calib)):
            prepared(torch.from_numpy(calib[i:i + 1]).float())
    converted = convert_pt2e(prepared, fold_quantize=False)
    edge = litert_torch.convert(converted, sample,
                                quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer))
    dst = OUT / f"kinglet-{TAG}-jp-int8.tflite"
    edge.export(str(dst))
    print(f"exported {dst} ({dst.stat().st_size / 1e6:.2f} MB)")

    p = np.load(OUT / "parity.npz")
    mels, ref = p["mels"], p["ref"]
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(dst))
    it.allocate_tensors()
    i_d = it.get_input_details()[0]
    outs = it.get_output_details()
    o_d = next(d for d in outs if d["shape"][-1] == len(ref[0]))  # probs; the other is the embedding
    for d in outs:
        print(f"output {d['name']}: shape {d['shape'].tolist()} dtype {d['dtype'].__name__} "
              f"quant {d['quantization']}")
    i_s, i_z = i_d["quantization"]  # int8 input; output may be int8 or already float
    o_s, o_z = o_d["quantization"]
    probs = np.empty_like(ref)
    for i in range(len(mels)):
        q = np.clip(np.round(mels[i:i + 1].astype(np.float32) / i_s + i_z), -128, 127)
        it.set_tensor(i_d["index"], q.astype(np.int8))
        it.invoke()
        o = it.get_tensor(o_d["index"])[0]
        probs[i] = (o.astype(np.float32) - o_z) * o_s if o.dtype == np.int8 else o
    sp, sp_ref = probs[:, :-1], ref[:, :-1]  # species rows ("other" is last)
    fire, fire_ref = sp > FIRE_THR, sp_ref > FIRE_THR
    res = {
        "tflite_mb": round(dst.stat().st_size / 1e6, 2),
        "n_parity_windows": len(mels),
        "top1_agreement": round(float((probs.argmax(1) == ref.argmax(1)).mean()), 4),
        "mean_abs_dprob": round(float(np.abs(sp - sp_ref).mean()), 6),
        "max_abs_dprob": round(float(np.abs(sp - sp_ref).max()), 4),
        "fire_flag_disagree": round(float((fire != fire_ref).mean()), 6),
        "window_fire_disagree": round(float((fire.any(1) != fire_ref.any(1)).mean()), 4),
    }
    print(f"input quantization: scale {i_s!r}, zero point {i_z} (mcu/gen_mel_assets.py --in-scale / --in-zp)")
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "parity_results.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["pack", "prep", "convert"], required=True)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--pack", type=Path, default=OUT / "pack_jp_focal.npz",
                    help="compact output layer (W, b, codes), written by --stage pack")
    args = ap.parse_args()
    if args.stage == "pack":
        build_focal_pack(args.ckpt, args.pack)
    elif args.stage == "prep":
        prep(args.ckpt, args.pack)
    else:
        convert()


if __name__ == "__main__":
    main()
