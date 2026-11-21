# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Distil Perch 2.0 into small CNN students (Appendix S1).

Inputs per part (rows aligned; teacher/embed_perch.py, teacher/prepare_mels.py,
teacher/make_softmax_targets.py --full):
  <distill_root>/<source>/part-NNNN.npz   emb (N, 1536) f32 teacher embeddings, ids (recording ids), ...
  <distill_root>/<source>/part-NNNN.npy   log-mels (N, 128, 501) f16
  <softmax_full>/<source>/part-NNNN.npy   teacher softmax over the whole Perch 2.0 vocabulary (N, 14795) f16
Sources (SOURCES): the Japanese manifest (emb_train60), BirdSet XCM (emb_xcm60) and BirdSet XCL (emb_xcl60).
Training rows = manifest split "pool", validation rows = split "val".

Student: backbones.build(name) -> embedding head (Linear to 1536) + class head (Linear to 14,795).
Loss: (1 - cosine(student_emb, teacher_emb)) + soft-label cross-entropy against the teacher softmax.
Model selection: best validation cosine similarity (best validation cross-entropy for clsonly).

Several students can be trained in one run on the same batches: pass --backbone spec1 spec2 ...;
checkpoints then go to <out>/<spec>/ (with "@" replaced by "_"), with a single spec to <out>/ directly.
A spec is name@head with head one of
  full     (default) both losses train the whole network,
  detach   the class head sees a stop-gradient copy of the trunk feature (the trunk trains on the
           embedding loss only; loss ablation),
  clsonly  no embedding loss: the embedding head stays at its random init (loss ablation;
           model selection by validation cross-entropy).
A head may end in _sN (e.g. name@full_s1): torch.manual_seed(N) right before that student is built,
so its initialisation does not depend on its position in --backbone (repeats). Without the suffix
the shared initialisation stream is used. All students of one run see the same batches, whose order
is set by --seed alone.

Shuffling: parts in random order, rows through a 20k-row RAM buffer.

Usage:
  python train_distill.py --out <distill_root>/g3conf_v1 \\
      --backbone regnety_016@full efficientnet_lite0@full --epochs 15 --resume
  python train_distill.py --out <tmp> --backbone mobilenetv4_conv_small --parts 1 --epochs 1   # smoke test
Output <out>[/<spec>]/: best.pt, last.pt (backbone + heads + mel statistics + config), log.txt
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT, DISTILL_ROOT  # noqa: E402
from train.backbones import build  # noqa: E402

SOURCES = {"emb_train60": DATA_ROOT / "manifest.parquet",
           "emb_xcm60": DATA_ROOT / "birdset/train_manifest.parquet",
           "emb_xcl60": DATA_ROOT / "birdset/xcl_manifest.parquet"}
LOGIT_W = 1.0
BUFFER = 20_000


class Student(torch.nn.Module):
    """Input: a normalised log-mel (B, 1, 128, 501). Output: (trunk feature, embedding, class logits).

    head: "full" / "detach" / "clsonly" have the same structure; the stop-gradient (losses()) or the
    dropped embedding loss (main()) is applied in training."""

    def __init__(self, backbone: str, n_classes: int, t_emb_dim: int = 1536, head: str = "full"):
        super().__init__()
        assert head in ("full", "detach", "clsonly"), f"unknown head variant {head}"
        self.backbone = build(backbone)
        self.backbone.eval()  # batch-1 probe must not hit BatchNorm in training mode (mobilenetv4)
        with torch.no_grad():  # actual output dim (timm num_features is pre-head for some models)
            d = self.backbone(torch.zeros(1, 1, 128, 501)).shape[1]
        self.backbone.train()
        self.emb_head = torch.nn.Linear(d, t_emb_dim)
        self.head = head
        self.logit_head = torch.nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return h, self.emb_head(h), self.logit_head(h)


def parts_index(n_parts_cap: int, sources: dict, soft_dir: Path | None = None) -> list[dict]:
    """One entry per part: paths + boolean train/val row masks (manifest split by rec_id).
    soft_dir: teacher softmax targets, soft_dir/<source>/part-NNNN.npy aligned with the npz rows."""
    out = []
    for src, man_path in sources.items():
        man = pd.read_parquet(man_path)
        split = dict(zip(man["rec_id"].astype(str), man["split"]))
        npzs = sorted((DISTILL_ROOT / src).glob("part-*.npz"))[: n_parts_cap or None]
        for npz in npzs:
            if not npz.with_suffix(".npy").exists():
                continue
            ids = np.load(npz)["ids"]
            sp = np.array([split[i] for i in ids.tolist()])
            entry = {"npz": npz, "npy": npz.with_suffix(".npy"), "train": sp == "pool", "val": sp == "val"}
            if soft_dir is not None:
                entry["soft"] = soft_dir / src / (npz.stem + ".npy")
                assert entry["soft"].exists(), f"missing softmax targets {entry['soft']}"
            out.append(entry)
    assert out, "no aligned parts found (run teacher/embed_perch.py and teacher/prepare_mels.py first)"
    return out


def load_rows(part: dict, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(mels f16, teacher emb f32, teacher targets, row ids) for the split's rows of one part.
    The targets are the softmax targets when the part has them, else the npz teacher logits."""
    m = part[which]
    z = np.load(part["npz"])
    mel = np.load(part["npy"], mmap_mode="r")[: len(m)]
    if "soft" in part:
        lg = np.asarray(np.load(part["soft"], mmap_mode="r")[: len(m)][m])
    else:
        lg = z["logits"][m]
    mel, emb, ids = np.ascontiguousarray(mel[m]), z["emb"][m], z["ids"][m]
    return mel, emb, lg, ids


def mel_stats(parts: list[dict]) -> tuple[float, float]:
    xs = np.concatenate([np.load(p["npy"], mmap_mode="r")[:2000].astype(np.float32).ravel()
                         for p in parts[:3]])
    return float(xs.mean()), float(xs.std())


def batches(parts: list[dict], rng: np.random.Generator, bs: int):
    """Shuffle-buffer iterator over train rows of all parts; the next part is prefetched on a
    thread so the disk read overlaps the GPU step. Every train row is yielded exactly once."""
    buf_m, buf_e, buf_l = [], [], []

    def pop(final):
        m = np.concatenate(buf_m) if len(buf_m) > 1 else buf_m[0]
        e = np.concatenate(buf_e) if len(buf_e) > 1 else buf_e[0]
        lg = np.concatenate(buf_l) if len(buf_l) > 1 else buf_l[0]
        buf_m.clear(), buf_e.clear(), buf_l.clear()  # drop the pieces now: RAM peak, not ~2x the buffer
        idx = rng.permutation(len(m))
        end = len(m) if final else len(m) - BUFFER
        for i in range(0, end, bs):
            j = idx[i : min(i + bs, end)]
            if len(j) < 2:  # a 1-row batch breaks BatchNorm-after-pool models (e.g. mobilenetv4)
                continue
            yield m[j], e[j], lg[j]
        keep = idx[end:]
        buf_m[:], buf_e[:], buf_l[:] = [m[keep]], [e[keep]], [lg[keep]]

    ex = ThreadPoolExecutor(1)
    order = rng.permutation(len(parts))
    fut = ex.submit(load_rows, parts[order[0]], "train")
    for k in range(len(order)):
        mel, emb, lg, _ = fut.result()
        if k + 1 < len(order):
            fut = ex.submit(load_rows, parts[order[k + 1]], "train")
        buf_m.append(mel), buf_e.append(emb), buf_l.append(lg)
        if sum(len(x) for x in buf_m) >= 2 * BUFFER:
            yield from pop(False)
    yield from pop(True)
    ex.shutdown()


def prep(mel, emb_t, lg_t, mu, sd, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (torch.from_numpy(mel).float().unsqueeze(1).to(device, non_blocking=True) - mu) / sd
    te = torch.from_numpy(emb_t).to(device)
    tl = torch.from_numpy(lg_t.astype(np.float32)).to(device)
    return x, te, tl


def losses(model, x, te, tl, detach_head: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(1 - mean cosine to the teacher embedding, soft-label cross-entropy to the teacher softmax)."""
    h, pe, pl = model(x)
    if detach_head:  # the class head is a probe on the trunk feature: its gradient never reaches the trunk
        pl = model.logit_head(h.detach())
    cos = 1 - F.cosine_similarity(pe, te).mean()
    return cos, -(tl * F.log_softmax(pl, 1)).sum(1).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", nargs="+", default=["mobilenetv4_conv_small"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--parts", type=int, default=0, help="cap parts per source (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="data order (and shared init stream)")
    ap.add_argument("--softmax-full", type=Path, default=DISTILL_ROOT / "softmax_full",
                    help="directory with the teacher's whole-vocabulary softmax targets "
                         "(<dir>/<source>/part-NNNN.npy, (rows, 14795) fp16); the class head width "
                         "follows the files")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>[/<spec>]/last.pt when present (full resume with "
                         "optimizer/scheduler state when the checkpoint has it, else weights-only "
                         "with a fresh schedule over the remaining epochs); fresh start otherwise")
    args = ap.parse_args()

    init_seed = {}  # spec -> per-student torch init seed (name@head_sN)

    def parse_spec(b: str) -> tuple[str, str, str]:
        name, _, head = b.partition("@")
        head, _, s = head.partition("_s")
        if s:
            init_seed[b] = int(s)
        return b, name, head or "full"

    specs = [parse_spec(b) for b in args.backbone]
    outs = {b: (args.out if len(args.backbone) == 1 else args.out / b.replace("@", "_"))
            for b in args.backbone}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.txt").open("a")

    def say(*a):
        print(*a, flush=True)
        print(*a, file=log, flush=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)  # leave headroom for other processes

    classes = json.loads((DISTILL_ROOT / "emb_train60/classes.json").read_text())
    parts = parts_index(args.parts, SOURCES, soft_dir=args.softmax_full)
    mu, sd = mel_stats(parts)
    n_train = int(sum(p["train"].sum() for p in parts))
    say(f"{len(parts)} parts, {n_train} train windows, {int(sum(p['val'].sum() for p in parts))} val "
        f"windows, mel mu/sd {mu}/{sd}, device {device}, backbones {args.backbone}")
    n_out = int(np.load(parts[0]["soft"], mmap_mode="r").shape[1])  # whole teacher vocabulary

    def make_student(spec, b, hd):
        if spec in init_seed:
            torch.manual_seed(init_seed[spec])
        return Student(b, n_out, head=hd).to(device)

    models = {spec[0]: make_student(*spec) for spec in specs}
    detach_arm = {spec: hd == "detach" for spec, _, hd in specs}
    clsonly = {spec: hd == "clsonly" for spec, _, hd in specs}
    res = {b: torch.load(outs[b] / "last.pt", map_location="cpu", weights_only=False)
           for b in models} if args.resume and all((outs[b] / "last.pt").exists() for b in models) else {}
    start_ep = min(r["epoch"] for r in res.values()) if res else 0
    full = res and all("opt" in r for r in res.values())
    for b, m in models.items():
        if res:
            m.load_state_dict(res[b]["model"])
    opts = {b: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4) for b, m in models.items()}
    if full:  # resume the original one-cycle schedule
        steps_total = max(1, n_train // args.batch) * args.epochs
    else:  # fresh run, or weights-only resume: a new schedule over the remaining epochs
        steps_total = max(1, n_train // args.batch) * (args.epochs - start_ep)
    # weights-only resume: cap the fresh schedule at roughly where the original one left off
    max_lr = args.lr if not res or full else args.lr * (1 - start_ep / args.epochs)
    scheds = {b: torch.optim.lr_scheduler.OneCycleLR(opts[b], max_lr=max_lr, total_steps=steps_total,
                                                     pct_start=0.05) for b in models}
    if res:
        for b in models:
            if full:
                opts[b].load_state_dict(res[b]["opt"])
                scheds[b].load_state_dict(res[b]["sched"])
        say(f"resumed from epoch {start_ep} ({'full' if full else 'weights-only, fresh schedule'})")
    for b, m in models.items():
        say(f"  {b}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params (with heads)")
    run_state = args.out / "run_state.pt"
    rng = np.random.default_rng(args.seed + start_ep)
    best = {b: res[b].get("best", res[b]["val_cos"]) if res else -float("inf") for b in models}
    step = 0
    if full and run_state.exists():
        st = torch.load(run_state, weights_only=False)
        if st["epoch"] == start_ep:
            rng.bit_generator.state = st["rng"]
            step = st["step"]
    for ep in range(start_ep, args.epochs):
        for m in models.values():
            m.train()
        t0, seen = time.time(), 0
        s_cos, s_ce = {b: 0.0 for b in models}, {b: 0.0 for b in models}
        for mel, emb_t, lg_t in batches(parts, rng, args.batch):
            x, te, tl = prep(mel, emb_t, lg_t, mu, sd, device)
            for b, m in models.items():
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    cos, ce = losses(m, x, te, tl, detach_arm[b])
                opts[b].zero_grad(set_to_none=True)
                # clsonly: cos is logged but not trained, so emb_head gets no grad (AdamW skips it)
                (LOGIT_W * ce if clsonly[b] else cos + LOGIT_W * ce).backward()
                opts[b].step()
                if scheds[b].last_epoch < scheds[b].total_steps - 1:  # the sched's own budget
                    scheds[b].step()                                  # (differs after a resume)
                s_cos[b] += cos.item() * len(x)
                s_ce[b] += ce.item() * len(x)
            step += 1
            seen += len(x)
        for m in models.values():
            m.eval()
        v_cos, v_ce = {b: 0.0 for b in models}, {b: 0.0 for b in models}
        v_n = 0
        with torch.no_grad():
            for p in parts:
                if not p["val"].any():
                    continue
                mel, emb_t, lg_t, _ = load_rows(p, "val")
                for i in range(0, len(mel), args.batch):
                    s = slice(i, i + args.batch)
                    x, te, tl = prep(mel[s], emb_t[s], lg_t[s], mu, sd, device)
                    for b, m in models.items():
                        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                            cos, ce = losses(m, x, te, tl, detach_arm[b])
                        v_cos[b] += cos.item() * len(x)
                        v_ce[b] += ce.item() * len(x)
                    v_n += len(x)
        for b, m in models.items():
            val_sim = 1 - v_cos[b] / v_n
            score = -v_ce[b] / v_n if clsonly[b] else val_sim  # model-selection criterion
            say(f"epoch {ep + 1}/{args.epochs} [{b}]: train cos {1 - s_cos[b] / seen:.4f} "
                f"ce {s_ce[b] / seen:.4f} | val cos {val_sim:.4f} ce {v_ce[b] / v_n:.4f} | "
                f"{seen} rows, {time.time() - t0:.0f}s, lr {scheds[b].get_last_lr()[0]:.1e}")
            _, spec_name, spec_hd = dict((s[0], s) for s in specs)[b]
            cfg = {"backbone": spec_name, "head": spec_hd,
                   "mel_mu": mu, "mel_sd": sd, "classes": classes, "n_out": n_out,
                   "lr": args.lr, "batch": args.batch, "epochs": args.epochs, "seed": args.seed,
                   "detach_head": detach_arm[b], "init_seed": init_seed.get(b),
                   "softmax_targets": str(args.softmax_full), "softmax_full": True}
            state = {"model": m.state_dict(), "cfg": cfg, "epoch": ep + 1, "val_cos": val_sim,
                     "best": max(best[b], score)}
            if score > best[b]:
                best[b] = score
                torch.save(state, outs[b] / "best.pt")
            torch.save(state | {"opt": opts[b].state_dict(), "sched": scheds[b].state_dict()},
                       outs[b] / "last.pt")
        torch.save({"epoch": ep + 1, "step": step, "rng": rng.bit_generator.state}, run_state)
    for b in models:
        say(f"best {'-val ce' if clsonly[b] else 'val cos'} [{b}] {best[b]:.4f} -> {outs[b] / 'best.pt'}")


if __name__ == "__main__":
    main()
