# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""mcuboard_v1: WABAD windows through the Grove Vision AI V2's own log-mel + network.

The paper's int8 retention numbers (ptqpub_v1, mcuemb_v2) come from the int8 TFLite model in the
host TFLite interpreter on a host fp32 log-mel. This runs the same WABAD windows through the board
instead: the BIRD_PCM_TEST firmware (mcu/firmware/bird_pdm, `make BIRD_PCM_TEST=1`) receives int16
PCM over the console UART in place of the PDM stream and runs the unchanged cv_bird_run.

Windows = every 4th of the 2,048 ptqpub_v1 parity windows (export_int8.wabad_mels: evenly
spaced over the WABAD manifest), 512 windows, two arms:
  exact32k  the 160,000 samples of the 5-s window at 32 kHz, as the host analysis saw them
            (isolates the board's numerics: same input as the host reference)
  mcu31k    what the board's PDM path would hand the model: window + the first 0.12 s of the next
            window (mcurate.extended), 32,000 -> 31,250 Hz = 160,000 samples, read as 32 kHz
Both arms are int16-quantized on the host; the host references use the same int16 audio.

Stages (main environment for prep / analyze; an environment with pyserial for run / log, see mcu/BUILD.md):
  python mcu/board_pcm_test.py prep      # windows + host fp32 / int8 references
  python mcu/board_pcm_test.py run [--port P]   # board outputs (resumable)
  python mcu/board_pcm_test.py analyze   # -> results/mcuboard_v1.json
  python mcu/board_pcm_test.py log --seconds 75 --log F   # UART log of the production firmware
Files: DISTILL_ROOT/xs_pub_deploy/mcuboard_v1/{windows.npz, host.npz, board_<arm>.npz, board_run.log}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_ROOT, DISTILL_ROOT, RELEASE_ASSETS, RESULTS  # noqa: E402

DEPLOY = DISTILL_ROOT / "xs_pub_deploy"
OUT = DEPLOY / "mcuboard_v1"
TFLITE = DEPLOY / "kinglet-xs_pub-jp-int8.tflite"
TRUNK_ONNX = RELEASE_ASSETS / "kinglet-xs-v0.3.2.onnx"  # released XS trunk (embedding from the int8 feature)
PORT = "/dev/ttyUSB0"  # the board's USB serial (CH340) console
ARMS = ("exact32k", "mcu31k")
N_PARITY, STRIDE = 2048, 4  # 512 windows
WIN, N_CLS, N_FEAT, N_MELS, N_FRAMES = 160_000, 318, 1280, 128, 501
N_MEL_DUMP = 8  # windows per arm whose int8 model input the board also dumps
FIRE_THR = 0.1


def fnv1a(b: bytes) -> int:
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------------------- prep
def host_models():
    """(fp32 Deploy graph, int8 TFLite interpreter, mel fn) - the ptqpub_v1 graph and converter output."""
    import tensorflow as tf
    import torch
    import torchaudio

    from mcu.export_int8 import Deploy
    from train.backbones import MEL, SR

    st = torch.load(DEPLOY / "deploy_state.pt", map_location="cpu", weights_only=False)
    deploy = Deploy(st["backbone"], len(st["W"]))
    deploy.net.load_state_dict(st["net"])
    deploy.head.weight.data = torch.from_numpy(st["W"])
    deploy.head.bias.data = torch.from_numpy(st["b"])
    deploy.eval()
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=SR, **MEL)

    def mel(pcm16: np.ndarray) -> np.ndarray:  # (n, 160000) int16 -> (n, 1, 128, 501) normalized log-mel
        with torch.no_grad():
            x = torch.from_numpy(pcm16.astype(np.float32) / 32768.0)
            return ((torch.log(melspec(x) + 1e-6) - st["mel_mu"]) / st["mel_sd"]).unsqueeze(1).numpy()

    it = tf.lite.Interpreter(model_path=str(TFLITE))
    it.allocate_tensors()
    return deploy, it, mel


def tflite_run(it, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 input (n, 1, 128, 501) -> (probs (n, 318) float, feature (n, 1280) int8)."""
    i_d = it.get_input_details()[0]
    outs = it.get_output_details()
    p_d = next(d for d in outs if d["shape"][-1] == N_CLS)
    f_d = next(d for d in outs if d["shape"][-1] == N_FEAT)
    probs, feats = [], []
    for x in q:
        it.set_tensor(i_d["index"], x[None])
        it.invoke()
        o = it.get_tensor(p_d["index"])[0]
        s, z = p_d["quantization"]
        probs.append((o.astype(np.float32) - z) * s if o.dtype == np.int8 else o.astype(np.float32))
        feats.append(it.get_tensor(f_d["index"])[0].copy())
    return np.stack(probs), np.stack(feats)


def quantize_input(it, m: np.ndarray) -> np.ndarray:
    s, z = it.get_input_details()[0]["quantization"]
    return np.clip(np.round(m / s + z), -128, 127).astype(np.int8)


def prep() -> None:
    import pandas as pd
    import torch
    import torchaudio

    from mcu.mcurate import EXTRA, MCU_SR, SR, extended
    from teacher.embed_perch import decode, windows

    man = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    parity_rows = np.linspace(0, len(man) - 1, N_PARITY).astype(int)  # = export_int8.wabad_mels
    rows = parity_rows[::STRIDE]
    paths = man["path"].iloc[rows].tolist()
    x = np.stack([windows(decode(p, 0), 5.0)[1][0] for p in paths])
    ext, case = zip(*(extended(a, p) for a, p in zip(x, paths)))
    r = torchaudio.functional.resample(torch.from_numpy(np.stack(ext)), SR, MCU_SR).numpy()
    assert r.shape == (len(rows), WIN) and EXTRA == 3840
    to16 = lambda a: np.clip(np.round(a * 32768.0), -32768, 32767).astype(np.int16)  # noqa: E731
    pcm = {"exact32k": to16(x), "mcu31k": to16(r)}
    clip = {k: float((np.abs(v.astype(np.int32)) >= 32767).mean()) for k, v in pcm.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "windows.npz", rows=rows, parity_idx=np.arange(0, N_PARITY, STRIDE),
             window_id=man["window_id"].iloc[rows].to_numpy(), neighbour=np.array(case), **pcm)

    from kinglet import embedding_from_feature

    deploy, it, mel = host_models()
    f_s, f_z = next(d for d in it.get_output_details() if d["shape"][-1] == N_FEAT)["quantization"]
    host = {}
    for arm in ARMS:
        m = mel(pcm[arm])
        q = quantize_input(it, m)
        p8, f8 = tflite_run(it, q)
        with torch.no_grad():
            p32, f32 = zip(*(deploy(torch.from_numpy(m[i:i + 64])) for i in range(0, len(m), 64)))
        host |= {f"{arm}_in_q": q[:, 0], f"{arm}_p_int8": p8, f"{arm}_feat_int8": f8,
                 f"{arm}_emb_int8": embedding_from_feature(TRUNK_ONNX, f8, f_s, f_z).astype(np.float32),
                 f"{arm}_p_fp32": torch.cat(p32).numpy(),
                 f"{arm}_emb_fp32": embedding_from_feature(TRUNK_ONNX, torch.cat(f32).numpy()).astype(np.float32)}
        print(f"{arm}: host int8 vs fp32 top-1 agreement "
              f"{(p8.argmax(1) == host[f'{arm}_p_fp32'].argmax(1)).mean():.4f}", flush=True)
    np.savez(OUT / "host.npz", feat_scale=f_s, feat_zp=f_z, **host)
    print(f"prep: {len(rows)} windows, neighbour {pd.Series(case).value_counts().to_dict()}, "
          f"clipped samples {clip} -> {OUT}")


# ----------------------------------------------------------------------------------------- run
def run(port: str, arms: list[str], limit: int) -> None:
    import serial

    w = np.load(OUT / "windows.npz")
    log = open(OUT / "board_run.log", "a")
    # opening the port resets the board; 15 s covers a 320 KB window at 921,600 baud (3.5 s) + inference
    ser = serial.Serial(port, 921600, timeout=15)

    def line() -> str:
        s = ser.read_until(b"\n").decode("ascii", "replace").rstrip("\r\n")
        if s and not s.startswith(("[probs]", "[emb]", "[mel]")):
            log.write(f"{time.time():.3f} {s}\n")
            log.flush()
        return s

    def wait_ready(timeout: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = line()
            if s == "[pcm] ready":
                return True
            if "golden" in s:
                print(s, flush=True)
        return False

    if not wait_ready(30):  # no reset on open: a BIRD_PCM_TEST board then already waits for "PCM"
        print("no '[pcm] ready' after opening the port; assuming the test firmware is waiting", flush=True)
    for arm in arms:
        dst = OUT / f"board_{arm}.npz"
        n = len(w[arm]) if limit <= 0 else min(limit, len(w[arm]))
        if dst.exists():
            res = dict(np.load(dst))
        else:
            res = {"probs": np.zeros((len(w[arm]), N_CLS), np.float32),
                   "feat": np.zeros((len(w[arm]), N_FEAT), np.int8),
                   "mel": np.zeros((N_MEL_DUMP, N_MELS, N_FRAMES), np.int8),
                   "cycles": np.zeros((len(w[arm]), 2), np.int64),
                   "done": np.zeros(len(w[arm]), bool), "resends": np.zeros(len(w[arm]), np.int32)}
        t_start = time.time()
        for i in np.flatnonzero(~res["done"][:n]):
            payload = w[arm][i].astype("<i2").tobytes()
            h = fnv1a(payload)
            mode = b"1" if i < N_MEL_DUMP else b"0"
            for attempt in range(4):
                ser.write(b"PCM" + mode + payload)
                head = line()
                while head and not head.startswith("[pcmres"):
                    head = line()
                if not head:  # lost bytes: the board still waits for data -> pad until it answers
                    for _ in range(6000):  # <= 384 KB of padding, more than one window
                        if ser.in_waiting:
                            break
                        ser.write(bytes(64))
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("board silent (is the BIRD_PCM_TEST firmware flashed?)")
                    wait_ready(30)
                    res["resends"][i] += 1
                    continue
                f = head.split()
                if len(f) != 8:  # "[pcmres k] run failed"
                    res["resends"][i] += 1
                    assert wait_ready(30), "board did not come back after a failed run"
                    continue
                probs_hex, emb_hex = line(), line()
                mel_hex = line() if mode == b"1" else ""
                ok = int(f[3], 16) == h and probs_hex.startswith("[probs] ") and emb_hex.startswith("[emb] ")
                if ok:
                    res["probs"][i] = np.frombuffer(bytes.fromhex(probs_hex[8:]), ">f4")
                    res["feat"][i] = np.frombuffer(bytes.fromhex(emb_hex[6:]), np.int8)
                    res["cycles"][i] = int(f[5]), int(f[7])
                    if mode == b"1":
                        res["mel"][i] = np.frombuffer(bytes.fromhex(mel_hex[6:]), np.int8).reshape(N_MELS, N_FRAMES)
                    res["done"][i] = True
                else:
                    res["resends"][i] += 1
                assert wait_ready(30), "board did not come back after a window"
                if ok:
                    break
            k = int(res["done"][:n].sum())
            if k % 16 == 0 or k == n:
                np.savez(dst, **res)
                el = time.time() - t_start
                print(f"{arm}: {k}/{n} windows, {el:.0f} s, resends {int(res['resends'].sum())}", flush=True)
        np.savez(dst, **res)
    ser.close()


def uart_log(port: str, seconds: float, dst: Path) -> None:
    """Timestamped UART log of the production firmware (opening the port resets the board) and the
    window period from the host clock, to check a re-flashed board against mcu/BUILD.md."""
    import serial

    ser, t0, wins = serial.Serial(port, 921600, timeout=1), time.time(), []
    with open(dst, "w") as f:
        while time.time() - t0 < seconds:
            s = ser.readline().decode("ascii", "replace").rstrip()
            if s:
                t = time.time() - t0
                f.write(f"{t:8.3f} {s}\n")
                print(f"{t:8.3f} {s}", flush=True)
                wins += [t] if s.startswith("[win ") else []
    if len(wins) > 2:
        d = np.diff(wins[1:])  # the first window follows the golden checks
        print(f"{len(wins)} windows, period {d.mean():.3f} s ({d.min():.3f}-{d.max():.3f}) -> {dst}")


# ------------------------------------------------------------------------------------- analyze
def agreement(p: np.ndarray, ref: np.ndarray) -> dict:
    sp, sr = p[:, :-1], ref[:, :-1]  # species rows ("other" is last), as the export_int8 parity
    fire, fire_r = sp > FIRE_THR, sr > FIRE_THR
    return {"top1_agreement": round(float((p.argmax(1) == ref.argmax(1)).mean()), 4),
            "window_fire_disagree": round(float((fire.any(1) != fire_r.any(1)).mean()), 4),
            "fire_flag_disagree": round(float((fire != fire_r).mean()), 6),
            "max_abs_dprob": round(float(np.abs(sp - sr).max()), 4),
            "mean_abs_dprob": float(f"{np.abs(sp - sr).mean():.3g}"),
            "fire_rate_ref": round(float(fire_r.any(1).mean()), 4)}


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def dist(c: np.ndarray) -> dict:
    return {"mean": round(float(c.mean()), 4), "p10": round(float(np.percentile(c, 10)), 4),
            "min": round(float(c.min()), 4)}


def retrieval_queries(queries: dict, rows: np.ndarray) -> dict:
    """Query-by-example readout (k = 1) for the 512 windows: each labelled species of a query window
    (species filter of probe_wabad_emb: >= 10 positive windows, >= 2 recordings) ranks the stored
    WABAD gallery (all 60,158 windows minus the query's own recording) by cosine; mean AP over
    (window, species) pairs, per query embedding and gallery."""
    import pandas as pd
    from sklearn.metrics import average_precision_score

    from evaluate.embeddings.probe_wabad_emb import MIN_POS, load_emb

    meta = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")
    recs = meta["window_id"].str.rsplit("_", n=1).str[0].to_numpy()
    lab = meta["labels"].tolist()
    sp_all = sorted({c for x in lab for c in x})
    Y = {c: np.zeros(len(meta), bool) for c in sp_all}
    for i, labs in enumerate(lab):
        for c in labs:
            Y[c][i] = True
    species = {c for c in sp_all if Y[c].sum() >= MIN_POS and len(np.unique(recs[Y[c]])) >= 2}
    pairs = [(j, c) for j, r in enumerate(rows) for c in lab[r] if c in species
             and (Y[c] & (recs != recs[r])).any()]
    out = {"n_pairs": len(pairs), "n_query_windows": len({j for j, _ in pairs})}
    ids = meta["window_id"].tolist()
    for gname in ("XS", "XS_int8pub"):
        g = load_emb(gname, ids)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        for qname, q in queries.items():
            s = (q / np.linalg.norm(q, axis=1, keepdims=True)) @ g.T  # (512, 60158)
            aps = [average_precision_score(Y[c][recs != recs[rows[j]]], s[j][recs != recs[rows[j]]])
                   for j, c in pairs]
            out[f"{qname}_query__{gname}_gallery_mAP"] = round(float(np.mean(aps)), 4)
    return out


def analyze() -> None:
    import pandas as pd

    w, host = np.load(OUT / "windows.npz"), np.load(OUT / "host.npz")
    from kinglet import embedding_from_feature

    f_s, f_z = float(host["feat_scale"]), int(host["feat_zp"])
    res = {"run": "mcuboard_v1", "model": TFLITE.name, "firmware": "bird_pdm BIRD_PCM_TEST=1",
           "windows": "every 4th of the 2,048 ptqpub_v1 WABAD parity windows",
           "neighbour_share_mcu31k": pd.Series(w["neighbour"]).value_counts(normalize=True).round(4).to_dict()}

    # bug check 1: the selected windows are the ptqpub_v1 parity windows (stored fp32 reference probs,
    # computed from fp16 log-mels of the float audio) and the stored WABAD embeddings (float audio)
    par = np.load(DEPLOY / "parity.npz")["ref"][w["parity_idx"]]
    from evaluate.embeddings.probe_wabad_emb import load_emb

    ids = pd.read_parquet(DATA_ROOT / "wabad/test.parquet")["window_id"].tolist()
    stored = {k: load_emb(k, ids)[w["rows"]] for k in ("XS", "XS_int8pub", "teacher")}
    res["bugcheck_window_identity"] = {
        "host_fp32_exact32k_vs_parity_ref": agreement(host["exact32k_p_fp32"], par),
        "cos_host_fp32_vs_stored_XS": dist(cos(host["exact32k_emb_fp32"], stored["XS"])),
        "cos_host_int8_vs_stored_XS_int8pub": dist(cos(host["exact32k_emb_int8"], stored["XS_int8pub"]))}

    # how sensitive the int8 path is to input perturbations far below the fp32 graph's: the same windows
    # as float audio (as the stored embeddings were made) vs int16 audio (as the board receives them)
    from teacher.embed_perch import decode, windows

    _, it, mel = host_models()
    k = 32
    xf = np.stack([windows(decode(p, 0), 5.0)[1][0] for p in
                   pd.read_parquet(DATA_ROOT / "wabad/test.parquet")["path"].iloc[w["rows"][:k]]])
    qf = quantize_input(it, mel(xf * 32768.0))
    ef = embedding_from_feature(TRUNK_ONNX, tflite_run(it, qf)[1], f_s, f_z)
    res["int8_sensitivity_float_vs_int16_audio_first32"] = {
        "input_bytes_identical_frac": round(float((qf[:, 0] == host["exact32k_in_q"][:k]).mean()), 4),
        "cos_host_int8_emb": dist(cos(ef, host["exact32k_emb_int8"][:k])),
        "cos_float_audio_host_int8_vs_stored_XS_int8pub": dist(cos(ef, stored["XS_int8pub"][:k]))}
    print(json.dumps(res, indent=1), flush=True)

    queries = {}
    for arm in ARMS:
        if not (OUT / f"board_{arm}.npz").exists():
            continue
        b = np.load(OUT / f"board_{arm}.npz")
        done = b["done"]
        n = int(done.sum())
        if n == 0:
            continue
        bp, bf = b["probs"][done], b["feat"][done]
        be = embedding_from_feature(TRUNK_ONNX, bf, f_s, f_z)
        h8p, h8f, h8e = host[f"{arm}_p_int8"][done], host[f"{arm}_feat_int8"][done], host[f"{arm}_emb_int8"][done]
        h32p, h32e = host[f"{arm}_p_fp32"][done], host[f"{arm}_emb_fp32"][done]
        dq = np.abs(bf.astype(np.int16) - h8f.astype(np.int16))
        r = {"n_windows": n, "resends": int(b["resends"].sum()),
             "board_ms": {"mel_median": round(float(np.median(b["cycles"][done, 0])) / 4e5, 1),
                          "npu_cpu_median": round(float(np.median(b["cycles"][done, 1])) / 4e5, 1),
                          "npu_cpu_max": round(float(b["cycles"][done, 1].max()) / 4e5, 1)},
             "board_vs_host_int8": agreement(bp, h8p),
             "board_vs_host_fp32": agreement(bp, h32p),
             "host_int8_vs_host_fp32": agreement(h8p, h32p),
             "feature_bytes_board_vs_host_int8": {
                 "identical_frac": round(float((dq == 0).mean()), 4), "within1_frac": round(float((dq <= 1).mean()), 4),
                 "max_abs_dq": int(dq.max()), "mean_abs_dq": round(float(dq.mean()), 4),
                 "windows_all_identical": int((dq.max(1) == 0).sum())},
             "emb_cos": {"board_vs_host_int8": dist(cos(be, h8e)), "board_vs_host_fp32": dist(cos(be, h32e)),
                         "host_int8_vs_host_fp32": dist(cos(h8e, h32e))},
             "emb_cos_to_teacher": {k: round(float(cos(e, stored["teacher"][done]).mean()), 4)
                                    for k, e in (("board", be), ("host_int8", h8e), ("host_fp32", h32e))}}
        # diagnostic: the board's own int8 log-mel through the host int8 network splits the board-host
        # gap into spectrogram (mel input bytes) and network kernels (Ethos-U vs TFLite on equal input)
        k = int(min(N_MEL_DUMP, done[:N_MEL_DUMP].sum()))
        if k:
            bm, hm = b["mel"][:k], host[f"{arm}_in_q"][:k]
            dm = np.abs(bm.astype(np.int16) - hm.astype(np.int16))
            p_bm, f_bm = tflite_run(it, bm[:, None])
            r["mel_input_diag_first_windows"] = {
                "n": k, "identical_frac": round(float((dm == 0).mean()), 5),
                "within1_frac": round(float((dm <= 1).mean()), 5), "max_abs_dq": int(dm.max()),
                "board_vs_host_net_on_board_mel_max_abs_dprob": round(float(np.abs(bp[:k] - p_bm).max()), 4),
                "board_vs_host_net_on_board_mel_max_abs_dfeat": int(np.abs(bf[:k].astype(np.int16) - f_bm).max()),
                "board_vs_host_net_on_board_mel_feat_identical_frac": round(float((bf[:k] == f_bm).mean()), 4)}
        res[arm] = r
        queries |= {f"{arm}_board": (be, w["rows"][done]), f"{arm}_host_int8": (h8e, w["rows"][done]),
                    f"{arm}_host_fp32": (h32e, w["rows"][done])}
        print(arm, json.dumps(r, indent=1), flush=True)

    for arm in ARMS:  # retrieval per arm (queries share the same rows within an arm)
        q = {k.removeprefix(f"{arm}_"): v[0] for k, v in queries.items() if k.startswith(arm)}
        if q:
            res[arm]["retrieval_k1_query"] = retrieval_queries(q, queries[f"{arm}_board"][1])
            print(arm, res[arm]["retrieval_k1_query"], flush=True)
    dst = RESULTS / "mcuboard_v1.json"
    dst.write_text(json.dumps(res, indent=1))
    print(f"-> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["prep", "run", "analyze", "log"])
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--limit", type=int, default=0, help="run: only the first N windows per arm (smoke test)")
    ap.add_argument("--seconds", type=float, default=75.0, help="log: how long to record")
    ap.add_argument("--log", type=Path, default=OUT / "fw_prod_uart_after.log", help="log: output file")
    args = ap.parse_args()
    if args.stage == "prep":
        prep()
    elif args.stage == "run":
        run(args.port, args.arms, args.limit)
    elif args.stage == "log":
        uart_log(args.port, args.seconds, args.log)
    else:
        analyze()


if __name__ == "__main__":
    main()
