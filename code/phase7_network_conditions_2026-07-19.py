# -*- coding: utf-8 -*-
"""Network-realistic conditions in the independent simulator. 2026-07-19
Two sharp corollaries of Theorem 2 (mean-delay composition), PRE-REGISTERED:

C1 (delay jitter): per-update delay tau ~ U[tau_m - a, tau_m + a].
   Prediction: v* depends only on the MEAN tau_m (first-moment additivity).
   tau_m = 0.5, a in {0, 0.2, 0.4}  ->  v* = 1.5/(0.5+0.30) = 1.875 for ALL a.

C2 (command dropout): each control update lost with prob p (hold extends).
   Renewal-age formula: ZOH mean delay = T_ctrl*(1+p)/(2*(1-p)).
   p=0:   c = 0.050+0.25 = 0.300 -> v* = 1.875
   p=0.3: c = 0.0929+0.25 = 0.343 -> v* = 1.779
   p=0.5: c = 0.150+0.25 = 0.400 -> v* = 1.667
Independent simulator (no the reference engine engine code): L=3, R=15, T_ctrl=0.1, T_f=0.25,
Gaussian heading noise 2 deg, MC=8."""
import os
import numpy as np, json

DT, DUR = 0.01, 120.0
L, R = 3.0, 15.0
T_CTRL, T_F, SIGMA = 0.10, 0.25, np.radians(2.0)
MC = 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase7_network_results_2026-07-19.json")

def simulate(v, tau_m, jitter_a, drop_p, seed):
    rng = np.random.default_rng(seed)
    n = int(DUR/DT); ce = int(T_CTRL/DT)
    pos = np.array([R, 0.0]); heading = np.pi/2; cmd = heading
    hist = [pos.copy()]; errs = np.empty(n)
    for k in range(n):
        if k % ce == 0 and rng.random() >= drop_p:
            tau_k = tau_m + (rng.uniform(-jitter_a, jitter_a) if jitter_a > 0 else 0.0)
            dl = max(0, int(round(tau_k/DT)))
            dp = hist[max(0, len(hist)-1-dl)]
            th = np.arctan2(dp[1], dp[0])
            aim = R*np.array([np.cos(th + L/R), np.sin(th + L/R)])
            d = aim - dp
            cmd = np.arctan2(d[1], d[0]) + rng.normal(0, SIGMA)
        dh = (cmd - heading + np.pi) % (2*np.pi) - np.pi
        heading += (DT/T_F)*dh
        pos = pos + v*DT*np.array([np.cos(heading), np.sin(heading)])
        hist.append(pos.copy())
        errs[k] = abs(np.hypot(*pos) - R)
    return float(np.sqrt(np.mean(errs**2)))

def vstar(tau_m, jitter_a, drop_p, pred):
    speeds = np.round(np.linspace(0.5*pred, 1.7*pred, 13), 3)
    rm = [float(np.mean([simulate(v, tau_m, jitter_a, drop_p, 1000+7*i)
                          for i in range(MC)])) for v in speeds]
    i = int(np.argmin(rm)); vs = float(speeds[i])
    if 1 <= i <= len(speeds)-2:
        c = np.polyfit(speeds[i-1:i+2], rm[i-1:i+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if speeds[i-1] <= vf <= speeds[i+1]: vs = float(vf)
    return vs

if __name__ == "__main__":
    import time; t0 = time.time()
    out = {}
    print("C1: delay jitter (tau_mean=0.5; prediction: v* = 1.875 for ALL a)")
    for a in [0.0, 0.2, 0.4]:
        pred = 1.875
        vs = vstar(0.5, a, 0.0, pred)
        out[f"jitter_a{a}"] = vs
        print(f"  a={a}: v*={vs:.3f}  err={100*(vs-pred)/pred:+.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    print("C2: command dropout (renewal-age prediction)")
    for p_, pred in [(0.3, 1.779), (0.5, 1.667)]:
        vs = vstar(0.5, 0.0, p_, pred)
        out[f"drop_p{p_}"] = vs
        print(f"  p={p_}: v*={vs:.3f}  pred={pred:.3f}  err={100*(vs-pred)/pred:+.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("Saved:", OUT, flush=True)
