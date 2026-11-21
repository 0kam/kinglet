#!/bin/bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# xsvar_v1 + xsabl_v1 (Appendix S1, Table S8): run-to-run variability of XS and the loss ablation.
# 1) One shared-batch run with data order --seed 1 (the released XS, g2xcl_v1 @full, used --seed 0):
#    full_s1 / full_s2 = the released recipe with init seeds 1 / 2 (same data order),
#    clsonly_s1 = no embedding loss, same init and batches as full_s1.
# 2) Scoring on the public benchmarks (WABAD + BirdSet 8) for the three new students and for
#    g2xcl_v1 @detach (the variant with the class loss detached from the backbone).
# 3) WABAD embeddings + trunk features; ridge map trunk feature -> Perch embedding fitted on the focal
#    validation windows (train/ridge_emb_map.py).
# 4) Embedding readouts -> results/xsvar_v1_embprobe.json, results/xsvar_v1_embcompat.json, and the
#    summary results/xsvar_v1_summary.json (Table S8).
# Re-runnable: training resumes from last.pt, scoring reuses its caches, extraction skips finished dirs.
# Smoke test (env overrides): OUT=<tmp> EPOCHS=1 TRAIN_EXTRA="--parts 1" SETS=POW MANIFEST=<small WABAD subset>
#   RIDGE_EXTRA="--parts 1" READOUTS=0 TAG=smoke bash run_xsvar.sh
set -eu
PAPER="$(cd "$(dirname "$0")/.." && pwd)"
ROOT=$(PYTHONPATH="$PAPER" python -c "from common import config; print(config.DATA_ROOT)")
DD=$(PYTHONPATH="$PAPER" python -c "from common import config; print(config.DISTILL_ROOT)")
OUT=${OUT:-$DD/xsvar_v1}
EPOCHS=${EPOCHS:-15}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
SETS=${SETS:-WABAD POW HSN NES PER SNE SSW UHH NBP}
MANIFEST=${MANIFEST:-$ROOT/wabad/test.parquet}
RIDGE_EXTRA=${RIDGE_EXTRA:-}
READOUTS=${READOUTS:-1}
TAG=${TAG:-xsvar}
M=mobilenetv4_conv_small
cd "$PAPER"

echo "=== training $M full_s1 full_s2 clsonly_s1 (shared batches, --seed 1, $EPOCHS ep) $(date) ==="
python train/train_distill.py --out "$OUT" --backbone $M@full_s1 $M@full_s2 $M@clsonly_s1 \
    --softmax-full "$DD/softmax_full" --epochs "$EPOCHS" --seed 1 --resume $TRAIN_EXTRA

# XS0 = the released XS, XSdet = g2xcl_v1 @detach (neither retrained)
declare -A CK=([XS0]=$DD/g2xcl_v1/${M}_full/best.pt [XSdet]=$DD/g2xcl_v1/${M}_detach/best.pt
               [XSs1]=$OUT/${M}_full_s1/best.pt [XSs2]=$OUT/${M}_full_s2/best.pt
               [XScls]=$OUT/${M}_clsonly_s1/best.pt)
NEW="XSs1 XSs2 XScls XSdet"

for N in $NEW; do
  echo "=== scoring $N on $SETS $(date) ==="
  python evaluate/accuracy/score_student.py --ckpt "${CK[$N]}" --name "distill_${TAG}_$N" --sets $SETS --decoders 8
done

for N in $NEW; do
  echo "=== WABAD embeddings + trunk features: $N $(date) ==="
  python evaluate/embeddings/embed_student.py --ckpt "${CK[$N]}" --manifest "$MANIFEST" \
      --out "$OUT/emb_wabad_$N" --feature --decoders 8
done

echo "=== ridge map trunk feature -> Perch embedding (focal val windows) $(date) ==="
RC=() RF=() RO=()
for N in XS0 $NEW; do
  RC+=("${CK[$N]}")
  RO+=("$OUT/emb_wabad_${N}_ridge")
  if [ "$N" = XS0 ]; then RF+=("$DD/packfit_v1/feat_wabad_XS"); else RF+=("$OUT/emb_wabad_$N"); fi
done
python train/ridge_emb_map.py --ckpt "${RC[@]}" --feat "${RF[@]}" --out "${RO[@]}" $RIDGE_EXTRA

if [ "$READOUTS" = 1 ]; then
  echo "=== own-space readouts (retrieval k=1/5, linear probe) $(date) ==="
  python evaluate/embeddings/probe_wabad_emb.py --out results/xsvar_v1_embprobe.json --models teacher XS XSs1 XSs2 XSdet \
      XS0_feat XSs1_feat XSs2_feat XScls_feat XSdet_feat \
      XS0_ridge XSs1_ridge XSs2_ridge XScls_ridge XSdet_ridge
  echo "=== cross-space readouts (field cos, teacher-trained probe, cross-retrieval) $(date) ==="
  python evaluate/embeddings/compat_wabad_emb.py --out results/xsvar_v1_embcompat.json --models XS XSs1 XSs2 XSdet XScls \
      XS0_ridge XSs1_ridge XSs2_ridge XScls_ridge XSdet_ridge
fi

echo "=== summary $(date) ==="
PYTHONPATH="$PAPER" python - "$TAG" "$SETS" "${CK[XS0]}" "${CK[XSs1]}" "${CK[XSs2]}" "${CK[XScls]}" "${CK[XSdet]}" <<'PY'
import json, sys
import numpy as np
import torch
from common import config
tag, sets, *cks = sys.argv[1], sys.argv[2].split(), *sys.argv[3:]
dd = config.DISTILL_ROOT
# teacher (Perch 2.0) cmAP per benchmark, from the teacher scoring runs
teacher = json.load(open(dd / "perch_teacher_birdset7.json")) | {"WABAD": json.load(open(dd / "perch_teacher_wabad.json")),
                                                                 "NBP": json.load(open(dd / "perch_teacher_nbp.json"))}
names = {"XS0": "distill_g2r_full"} | {n: f"distill_{tag}_{n}" for n in ("XSs1", "XSs2", "XScls", "XSdet")}
unused = [s for s in ("NES", "PER", "SNE", "SSW", "UHH") if s in sets]  # BirdSet sets not used in development
# all benchmarks not used in development (POW/HSN are reported separately)
unused_all = [s for s in ("WABAD", "NES", "PER", "SNE", "SSW", "UHH", "NBP") if s in sets]
out = {}
for (n, res), ck in zip(names.items(), cks):
    r = json.load(open(config.RESULTS / f"{res}_results.json"))
    st = torch.load(ck, map_location="cpu", weights_only=False)
    row = {s: round(r[s]["cmAP"], 4) for s in sets if s in r}
    ratio = lambda ss: round(float(np.mean([r[s]["cmAP"] / teacher[s]["cmAP"] for s in ss])), 4) if ss and all(s in r for s in ss) else None  # noqa: E731
    row["ratio5_mean"], row["ratio_unused_mean"] = ratio(unused), ratio(unused_all)
    row |= {"epoch": st["epoch"], "val_cos": round(st["val_cos"], 4), "best": round(st["best"], 4)}
    out[n] = row
    print(n, row)
seeds = [out[n] for n in ("XS0", "XSs1", "XSs2")]
for k in sets + ["ratio5_mean", "ratio_unused_mean"]:
    v = [s[k] for s in seeds if s.get(k) is not None]
    if v:
        print(f"seeds {k}: mean {np.mean(v):.4f} sd {np.std(v, ddof=1):.4f} range {max(v) - min(v):.4f}")
json.dump(out, open(config.RESULTS / f"{tag}_v1_summary.json", "w"), indent=1)
PY
echo "=== ALL DONE $(date) ==="
