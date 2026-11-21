# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""opensetcurve_v1 / regionlists_v1: one decoding pass over the eight benchmarks for every model that
needs full-vocabulary outputs.

Per set (windows exactly as common.scoring.infer / embed_student.py: same decode, 5-s windows):
  feat_<set>_<tier>.npz  trunk feature h (N, d) f16 of the released v0.3 tiers B/S/XS (bf16 autocast on
                         the GPU, as embed_student.py --feature); exact folds and packs are computed from h
  teacher_<set>.npz      Perch 2.0: log of the full-softmax normaliser logZ (N,) f32 and the logits of the
                         listed classes (Japan 317, Germany, Costa Rica lists; columns = `vidx`) f16
WABAD student features already exist (packfit_v1/feat_wabad_*), so WABAD runs the teacher only.

  python evaluate/species_lists/opensetcurve_extract.py --part students   # BirdSet 7 (GPU, ~2 GB)
  python evaluate/species_lists/opensetcurve_extract.py --part teacher    # all 8 (Perch GPU build)
  python evaluate/species_lists/opensetcurve_extract.py --part teacher --perch-cpu --sets WABAD   # CPU build
  (teacher_chain.sh runs the teacher part on the GPU build, then opensetcurve.py.)
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA_ROOT, DISTILL_ROOT, PERCH_DIR, PERCH_GPU_DIR, RESULTS, jp_list  # noqa: E402
from common.scoring import SETS  # noqa: E402
from evaluate.species_lists.openset_public import PUBLIC_SETS  # noqa: E402
from teacher.embed_perch import SR, decode, windows  # noqa: E402

OUT = DISTILL_ROOT / "opensetcurve_v1"
TIERS = {"B": "g3conf_v1/regnety_016_full", "S": "g3conf_v1/efficientnet_lite0_full",
         "XS": "g2xcl_v1/mobilenetv4_conv_small_full"}
CHUNK, BATCH_TF, BATCH = 512, 32, 128


def read(path: str, max_s: float) -> np.ndarray | None:
    """embed_perch.decode without the ffmpeg process when the file is already mono 32 kHz (same samples to
    1e-8 on the benchmark ogg files; spawning ffmpeg per file dominates the run time)."""
    info = sf.info(path)
    if info.samplerate != SR or info.channels != 1:
        return decode(path, max_s)
    a, _ = sf.read(path, dtype="float32", frames=int(max_s * SR) if max_s > 0 else -1)
    return a


def listed_vidx() -> np.ndarray:
    """Perch indices of every list this run evaluates (teacher columns)."""
    jp = [c["perch_idx"] for c in jp_list()]
    reg = [pd.read_csv(RESULTS / f"regionlist_{r}.csv")["vocab_index"].tolist() for r in ("de", "cr")]
    return np.array(sorted({int(i) for i in jp + reg[0] + reg[1] if i >= 0}))


class Teacher:
    def __init__(self, vidx: np.ndarray, cpu: bool):
        import tensorflow as tf

        if cpu:  # CPU build of the same weights (teacher/embed_perch.py)
            tf.config.set_visible_devices([], "GPU")
        for g in tf.config.list_physical_devices("GPU"):  # do not reserve the whole GPU
            tf.config.experimental.set_memory_growth(g, True)
        self.tf, self.vidx = tf, vidx
        self.fn = tf.saved_model.load(str(PERCH_DIR if cpu else PERCH_GPU_DIR)).signatures[
            "serving_default"]

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logz, sel = [], []
        for i in range(0, len(x), BATCH_TF):
            b = x[i : i + BATCH_TF]
            nb = len(b)
            if nb < BATCH_TF:  # fixed XLA batch: pad the tail
                b = np.pad(b, ((0, BATCH_TF - nb), (0, 0)))
            lab = self.fn(inputs=self.tf.constant(b))["label"].numpy()[:nb].astype(np.float64)
            m = lab.max(1, keepdims=True)
            logz.append((np.log(np.exp(lab - m).sum(1, keepdims=True)) + m)[:, 0])
            sel.append(lab[:, self.vidx])
        return np.concatenate(logz).astype(np.float32), np.concatenate(sel).astype(np.float16)


class Students:
    def __init__(self):
        import torch
        import torchaudio
        from teacher.embed_perch import SR
        from train.backbones import MEL
        from train.train_distill import Student

        self.torch, self.models = torch, {}
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        for tier, ck in TIERS.items():
            st = torch.load(DISTILL_ROOT / ck / "best.pt", map_location="cpu", weights_only=False)
            cfg = st["cfg"]
            m = Student(cfg["backbone"], cfg["n_out"], head=cfg.get("head", "full")).to(self.dev).eval()
            m.load_state_dict(st["model"])
            self.models[tier] = (m, cfg["mel_mu"], cfg["mel_sd"])
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL).to(self.dev)

    def __call__(self, x: np.ndarray) -> dict:
        torch, out = self.torch, {t: [] for t in self.models}
        for b in range(0, len(x), BATCH):
            with torch.no_grad():
                lm = torch.log(self.mel(torch.from_numpy(x[b : b + BATCH]).to(self.dev)) + 1e-6)
                for t, (m, mu, sd) in self.models.items():
                    with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                        feat = m(((lm - mu) / sd).unsqueeze(1))[0]
                    out[t].append(feat.float().cpu().numpy().astype(np.float16))
        return {t: np.concatenate(v) for t, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=PUBLIC_SETS)
    ap.add_argument("--part", choices=("students", "teacher"), required=True)
    ap.add_argument("--decoders", type=int, default=16)
    ap.add_argument("--perch-cpu", action="store_true", help="teacher on the CPU build (GPU busy)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    vidx = listed_vidx()
    teach = args.part == "teacher"
    model = Teacher(vidx, args.perch_cpu) if teach else Students()
    for set_name in args.sets:
        done = [OUT / f"teacher_{set_name}.npz"] if teach else [OUT / f"feat_{set_name}_{t}.npz" for t in TIERS]
        if (not teach and set_name == "WABAD") or all(p.exists() for p in done):
            print(set_name, "skipped (done, or WABAD features from packfit_v1)")
            continue
        manifest, id_col, off_col, max_s, hop, _ = SETS[set_name]
        man = pd.read_parquet(DATA_ROOT / manifest)
        rows = list(zip(man[id_col].astype(str), man["path"]))
        pool, prefetch = ThreadPoolExecutor(args.decoders), ThreadPoolExecutor(1)
        load = lambda ch: list(pool.map(lambda r: read(r[1], max_s), ch))  # noqa: E731
        chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
        nxt, ids, widx, logz, sel, feats = prefetch.submit(load, chunks[0]), [], [], [], [], []
        for k, ch in enumerate(chunks):
            audio = nxt.result()
            if k + 1 < len(chunks):
                nxt = prefetch.submit(load, chunks[k + 1])
            wins = []
            for (rid, _), a in zip(ch, audio):
                assert a is not None, f"decode failed: {rid}"
                starts, w = windows(a, hop)
                ids += [rid] * len(starts)
                widx += list(range(len(starts)))
                wins += list(w)
            x = np.stack(wins).astype(np.float32)
            if teach:
                lz, sl = model(x)
                logz.append(lz)
                sel.append(sl)
            else:
                feats.append(model(x))
            if k % 20 == 0 or k == len(chunks) - 1:
                print(f"[{set_name}] {min((k + 1) * CHUNK, len(rows))}/{len(rows)} recordings, {len(ids)} windows",
                      flush=True)
        ids, widx = np.array(ids), np.array(widx, np.int32)
        if not teach:
            for t in TIERS:
                np.savez(OUT / f"feat_{set_name}_{t}.tmp.npz", ids=ids, window_idx=widx,
                         feat=np.concatenate([f[t] for f in feats]))
                os.replace(OUT / f"feat_{set_name}_{t}.tmp.npz", OUT / f"feat_{set_name}_{t}.npz")
        else:
            np.savez(OUT / f"teacher_{set_name}.tmp.npz", ids=ids, window_idx=widx, vidx=vidx,
                     logz=np.concatenate(logz), logits=np.concatenate(sel))
            os.replace(OUT / f"teacher_{set_name}.tmp.npz", OUT / f"teacher_{set_name}.npz")
        print(f"[{set_name}] saved {len(ids)} windows", flush=True)


if __name__ == "__main__":
    main()
