# -*- coding: utf-8 -*-
"""
Derivation note -- D4b: the error floor as three additive, independently
computable channels.  Completes D4 and turns the closure report's Sec. 6.3
statement ("floor and margin are independent damage channels") into a formula.

    RMSE_min^2 = RMSE_bias^2(v*)            transit / look-ahead bias   [D2 model]
               + sum_j w_j (L*b(m*_j))^2    grid-lock offset            [D4 Prop C]
               + (L*s_res)^2                residual vote spread        [measured]

Every term comes from primitives (path geometry, the engine's delay chain, and
the open-loop vote transfer function).  Nothing is fitted to floor data.

Ground truth: the six rotation cases and the four square-rotation h-sweep cases,
tau = 433 ms, at each case's own published v*.

Outputs: d4b_results_2026-08-29.json
"""
import importlib.util, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = next(
    p for p in [os.path.join(HERE, "..", "..", "data"),
                os.path.join(HERE, "..", "closure_pack")]
    if os.path.isdir(p))
DELTA, L = 45.0, 2.0


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d2 = _load("d2mod", "d2_corner_noise_shift_2026-08-29.py")
d4 = _load("d4mod", "d4_gridlock_floor_2026-08-29.py")


CASES = {   # name: (n, side, rot_deg)  matching the published rotation test
    "sq_rot0":    (4, 20.0, 45.0),
    "sq_rot22.5": (4, 20.0, 67.5),
    "hex_rot0":   (6, 10.0, 0.0),
    "hex_rot7.5": (6, 10.0, 7.5),
    "tri_rot0":   (3, 17.3205, 0.0),
    "tri_rot30":  (3, 17.3205, 30.0),
}

if __name__ == "__main__":
    ms, b0, s = d4.load_transfer()
    S_RES = float(np.mean(s))
    SIG = 2.0                                   # in-cell dither, deg (see report)
    b = d4.dither_smooth(ms, b0, SIG)

    rot = json.load(open(os.path.join(PACK, "phase1_rotation_results_2026-07-18.json")))
    hs = json.load(open(os.path.join(PACK, "phase3_hB_tri_results_2026-07-19.json")))["h_sweep"]

    out = []
    print(f"channels at tau = 433 ms   (s_res = {S_RES:.2f} deg, sigma_m = {SIG:.1f} deg)")
    print(f"{'case':13s} {'v*':>6} {'bias':>7} {'lock':>7} {'noise':>7} {'total':>7} "
          f"{'obs':>7} {'err':>8}")

    def report(name, n, side, rot_deg, mis, vstar, obs):
        path = d2.regular_polygon(n, side, rot_deg)
        bias = float(np.mean([d2.lap_rmse_bias(path, vstar, 0.433,
                                               start_arc=k*path.circ/3) for k in range(3)]))
        lens = [path.circ/n]*n
        w = np.array(lens)/sum(lens)
        e = []
        for m in mis:
            mm = ((m + DELTA/2) % DELTA) - DELTA/2
            _, bs = d4.lock_offset(mm, ms, b)
            e.append(L*math.radians(abs(bs)))
        lock = float(np.sqrt(np.sum(w*np.array(e)**2)))
        noise = L*math.radians(S_RES)
        tot = math.sqrt(bias**2 + lock**2 + noise**2)
        out.append(dict(case=name, vstar=vstar, bias=bias, lock=lock, noise=noise,
                        total=tot, obs=obs, rel=(tot-obs)/obs))
        print(f"{name:13s} {vstar:6.3f} {bias:7.3f} {lock:7.3f} {noise:7.3f} {tot:7.3f} "
              f"{obs:7.3f} {100*(tot-obs)/obs:+7.1f}%")

    for name, (n, side, rd) in CASES.items():
        c = rot["cases"][name]
        t = c["taus"]["433"]
        report(name, n, side, rd, c["grid_misalign_deg"], t["vstar"], min(t["rmses"]))

    for k, v in hs.items():
        m = float(k)
        t = v["taus"]["433"]
        report(f"sq_rot{m:.3f}", 4, 20.0, 45.0+m, [m]*4, t["vstar"], min(t["rmses"]))

    errs = np.array([r["rel"] for r in out])
    print(f"\nmean |err| = {100*np.mean(np.abs(errs)):.1f}%   bias = {100*np.mean(errs):+.1f}%")
    json.dump(dict(sigma_m=SIG, s_res=S_RES, rows=out),
              open("d4b_results_2026-08-29.json", "w"), indent=2, default=float)
    print("-> d4b_results_2026-08-29.json")
