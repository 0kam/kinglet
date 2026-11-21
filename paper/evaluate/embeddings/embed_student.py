# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Extract the teacher-aligned 1536-d embeddings of a distilled student for an eval manifest.

Windows and decode follow common.scoring.infer (5 s windows at 32 kHz); the output npz mirrors
embed_perch parts so probe scripts can treat teacher and student embeddings identically:
  <out>/part-0000.npz: ids, window_idx, t_start, emb float16 (N, 1536)
                       (+ feat float16 (N, d), the trunk feature under the heads, with --feature)

With --tflite the embeddings take the microcontroller path instead (mcuemb_v2): host log-mel ->
int8 TFLite trunk -> the int8 feature tensor the device logs -> kinglet.embedding_from_feature with
the projection of the released trunk ONNX (--trunk-onnx). The checkpoint then only supplies the mel
normalization. The device's float32 mel (CMSIS-DSP) is not emulated (the golden vectors of mcu/ cover it).

Usage:
  python evaluate/embeddings/embed_student.py --ckpt DISTILL_ROOT/g3conf_v1/regnety_016_full/best.pt \
      --manifest DATA_ROOT/wabad/test.parquet --id-col window_id --out DISTILL_ROOT/emb_wabad_B
  python evaluate/embeddings/embed_student.py --ckpt DISTILL_ROOT/g2xcl_v1/mobilenetv4_conv_small_full/best.pt \
      --tflite DISTILL_ROOT/xs_pub_deploy/kinglet-xs_pub-jp-int8.tflite \
      --trunk-onnx RELEASE_ASSETS/kinglet-xs-v0.3.2.onnx \
      --manifest DATA_ROOT/wabad/test.parquet --id-col window_id --out DISTILL_ROOT/emb_wabad_XS_int8pub
(DATA_ROOT, DISTILL_ROOT, RELEASE_ASSETS: the locations set in paper/config.yaml.)
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from teacher.embed_perch import SR, decode, windows  # noqa: E402
from train.backbones import MEL  # noqa: E402
from train.train_distill import Student  # noqa: E402

CHUNK = 256
BATCH = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--id-col", default="window_id")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--decoders", type=int, default=4)
    ap.add_argument("--tflite", type=Path, help="int8 deploy model (mcu/export_int8.py); needs --trunk-onnx")
    ap.add_argument("--trunk-onnx", type=Path, help="released trunk ONNX holding the feature -> embedding layer")
    ap.add_argument("--feature", action="store_true", help="also save the trunk feature (packfit_v1)")
    args = ap.parse_args()
    assert bool(args.tflite) == bool(args.trunk_onnx), "--tflite and --trunk-onnx go together"
    assert not (args.feature and args.tflite), "--feature is for the fp32 checkpoint path"
    if (args.out / "DONE").exists():
        print("already done")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    model = Student(cfg["backbone"], cfg.get("n_out", len(cfg["classes"])),
                    head=cfg.get("head", "full")).to(dev).eval()
    model.load_state_dict(state["model"])
    import torchaudio  # external mel + normalization

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL).to(dev)

    if args.tflite:
        import tensorflow as tf

        it = tf.lite.Interpreter(model_path=str(args.tflite), num_threads=8)
        it.allocate_tensors()
        i_d = it.get_input_details()[0]
        f_d = next(d for d in it.get_output_details() if d["shape"][-1] == model.emb_head.in_features)
        (i_s, i_z), (f_s, f_z) = i_d["quantization"], f_d["quantization"]
        assert i_d["dtype"] == f_d["dtype"] == np.int8, "expected an int8 deploy model"
        print(f"int8 feature: scale {f_s:.4g}, zero point {f_z}")

    man = pd.read_parquet(args.manifest)
    rows = list(zip(man[args.id_col].astype(str), man[args.path_col]))
    pool, prefetch = ThreadPoolExecutor(args.decoders), ThreadPoolExecutor(1)
    load = lambda chunk: list(pool.map(lambda r: decode(r[1], 0), chunk))  # noqa: E731
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    nxt, meta, embs, feats = prefetch.submit(load, chunks[0]), [], [], []
    for k, chunk in enumerate(chunks):
        audio = nxt.result()
        if k + 1 < len(chunks):
            nxt = prefetch.submit(load, chunks[k + 1])
        wins = []
        for (rid, _), a in zip(chunk, audio):
            assert a is not None, f"decode failed: {rid}"
            starts, w = windows(a, args.hop)
            meta += [(rid, i, s) for i, s in enumerate(starts)]
            wins += list(w)
        for b in range(0, len(wins), BATCH):
            x = torch.from_numpy(np.stack(wins[b:b + BATCH])).to(dev)
            with torch.no_grad():
                x = ((torch.log(mel(x) + 1e-6) - cfg["mel_mu"]) / cfg["mel_sd"]).unsqueeze(1)
                if args.tflite:
                    for q in np.clip(np.round(x.cpu().numpy() / i_s + i_z), -128, 127).astype(np.int8):
                        it.set_tensor(i_d["index"], q[None])
                        it.invoke()
                        embs.append(it.get_tensor(f_d["index"]).copy())
                    continue
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    feat, emb, _ = model(x)
            embs.append(emb.float().cpu().numpy().astype(np.float16))
            if args.feature:
                feats.append(feat.float().cpu().numpy().astype(np.float16))
        if k % 20 == 0 or k == len(chunks) - 1:
            print(f"{min((k + 1) * CHUNK, len(rows))}/{len(rows)} recordings, {len(meta)} windows", flush=True)

    emb = np.concatenate(embs)
    if args.tflite:  # int8 features (N, d), as logged on the device -> Perch space on the host
        from kinglet import embedding_from_feature

        emb = embedding_from_feature(args.trunk_onnx, emb, f_s, f_z, normalize=False).astype(np.float16)  # raw, as mcuemb_v2
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "part-0000.tmp.npz",
             ids=np.array([m[0] for m in meta]),
             window_idx=np.array([m[1] for m in meta], np.int32),
             t_start=np.array([m[2] for m in meta], np.float32),
             emb=emb, **({"feat": np.concatenate(feats)} if args.feature else {}))
    (args.out / "part-0000.tmp.npz").rename(args.out / "part-0000.npz")
    (args.out / "DONE").touch()
    print(f"saved {len(meta)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
