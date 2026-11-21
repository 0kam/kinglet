# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Measure the Grove Vision AI V2 power draw with a Nordic Power Profiler Kit II (PPK2).

Wiring (board powered BY the PPK2, USB-C unplugged):
  PPK2 VOUT -> Grove connector pin 3 (GROVE_5V_IN, feeds buck U6, Imax 600 mA)
  PPK2 GND  -> Grove connector pin 4 (GND)
The USB VBUS path is diode-ORed, so nothing back-feeds the PC. Without UART the
inference bursts are identified from the current trace itself (5 s window period).

Run in an environment with ppk2-api (mcu/BUILD.md); the paper's three runs:
  python mcu/measure_power.py --seconds 60 --out power_xspub_5v_60s_r1.npz   (r2, r3 likewise)
"""

import argparse
import time

import numpy as np
from ppk2_api.ppk2_api import PPK2_API

PPK2_SPS = 100_000  # fixed ADC rate of the PPK2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--mv", type=int, default=5000, help="source voltage (PPK2 max 5000)")
    ap.add_argument("--port", default=None, help="PPK2 serial port (default: autodetect)")
    ap.add_argument("--out", default=None, help="save 1 ms-averaged trace .npz here")
    args = ap.parse_args()

    port = args.port
    if port is None:
        devs = PPK2_API.list_devices()
        assert devs, "no PPK2 found (check USB + port permissions)"
        # the PPK2 exposes two CDC ports; the lower-numbered one is the API port
        port = sorted(d if isinstance(d, str) else d[0] for d in devs)[0]
    print(f"PPK2 on {port}, sourcing {args.mv} mV for {args.seconds:.0f} s")

    ppk2 = PPK2_API(port)
    # A previous run leaves binary samples in the pipe, which break the text metadata
    # read: stop and drain first. The first metadata request after a completed run still
    # comes back prefixed with stale bytes (seen every time), so retry it.
    ppk2.ser.timeout = 0.2
    for attempt in range(3):
        ppk2.stop_measuring()
        while ppk2.ser.read(65536):
            pass
        try:
            ppk2.get_modifiers()
            break
        except UnicodeDecodeError:
            if attempt == 2:
                raise
    ppk2.set_source_voltage(args.mv)
    ppk2.use_source_meter()
    ppk2.toggle_DUT_power("ON")
    ppk2.start_measuring()

    # Drain the port with blocking reads and decode after the run. The tty read buffer
    # holds only ~10 ms of the 400 kB/s stream, so decoding per sample in the loop
    # (seen: 70 kS/s) or sleeping 10 ms between reads (98 kS/s) drops samples, which
    # compresses the time axis of the trace.
    ppk2.ser.timeout = 0.05
    raw = []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        raw.append(ppk2.ser.read(65536))
    elapsed = time.time() - t0
    ppk2.stop_measuring()
    ppk2.toggle_DUT_power("OFF")

    samples, _ = ppk2.get_samples(b"".join(raw))  # microamps
    ua = np.asarray(samples, dtype=np.float32)
    sps = len(ua) / elapsed
    ma = ua / 1000.0
    v = args.mv / 1000.0

    print(f"\n{len(ua)} samples in {elapsed:.1f} s ({sps / 1000:.1f} kS/s, PPK2 nominal {PPK2_SPS // 1000})")
    if sps < 0.995 * PPK2_SPS:
        print("WARNING: samples were dropped - burst timings below are not trustworthy")
    print(f"mean {ma.mean():.2f} mA ({v * ma.mean():.1f} mW)  median {np.median(ma):.2f} mA")
    print(f"p1 {np.percentile(ma, 1):.2f}  p99 {np.percentile(ma, 99):.2f}  max {ma.max():.2f} mA")

    # 1 ms bins for burst analysis + saved trace
    per_ms = PPK2_SPS // 1000
    trace = ua[:len(ua) // per_ms * per_ms].reshape(-1, per_ms).mean(axis=1) / 1000.0  # mA per 1 ms

    # smooth 10 ms, threshold halfway between baseline and peak, merge gaps <500 ms
    sm = np.convolve(trace, np.ones(10) / 10, mode="same")
    base = np.median(sm)
    thr = base + 0.5 * (sm.max() - base)
    mask = sm > thr
    idx = np.flatnonzero(mask)
    segs = []
    if len(idx):
        splits = np.flatnonzero(np.diff(idx) > 500)
        for part in np.split(idx, splits + 1):
            segs.append((part[0], part[-1] + 1))
    print(f"\nbaseline {base:.2f} mA, burst threshold {thr:.2f} mA, {len(segs)} bursts")
    for i, (s, e) in enumerate(segs):
        seg = trace[s:e]
        print(f"  burst {i}: t={s / 1000:.2f}s dur={e - s}ms mean {seg.mean():.2f} mA peak {seg.max():.2f} mA")
    # window bursts only: the inrush and golden-check bursts end within ~3 s of power-on,
    # and the first recorded window cannot finish before 5 s
    starts = [s for s, _ in segs if s > 5000]
    if len(starts) > 1:
        period = np.diff(starts).mean() / 1000.0
        steady = trace[starts[0]:starts[-1]].mean()  # whole windows only
        print(f"  window period {period:.3f} s (expected 5.120 s = 25 transfers of 0.2048 s)")
        print(f"  steady-state mean over {len(starts) - 1} windows: {steady:.2f} mA ({v * steady:.1f} mW)")

    if args.out:
        np.savez_compressed(args.out, trace_ma=trace.astype(np.float32), mv=args.mv,
                            mean_ma=ma.mean(), sps=sps)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
