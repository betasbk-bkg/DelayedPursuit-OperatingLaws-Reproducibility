"""
this study -- FIRST TRANSFER TEST: independent implementation. 2026-07-19

Question: does v* = (L/2)/(tau + T_ctrl/2 + T_filter) hold in a pure-pursuit
tracker that shares NOTHING with the the reference engine engine?

Deliberately different from simulation_main.py in every implementation choice:
  - dt = 0.01 s (engine: 1/60)
  - control update every T_ctrl = 0.10 s (engine WIN: 0.30)
  - steering: first-order lag on HEADING ANGLE, time constant T_f = 0.25 s
    (engine: exponential lerp on the velocity VECTOR, T_s = 1/12 s)
  - noise: Gaussian white heading noise sigma = 2 deg per control update
    (engine: 150-agent vote mixture with trolls/slow/other)
  - no quantization (continuum limit), no crowd, no vector averaging
  - parameters: L = 3.0 m (engine 2.0), circle R = 15 m (engine 10),
    tau in {0.3, 0.8} s (engine grid differs), duration 120 s (engine 65)

PRE-REGISTERED zero-parameter predictions:
  c = T_ctrl/2 + T_f = 0.05 + 0.25 = 0.30 s
  v*(tau=0.3) = (L/2)/(0.3+0.30) = 1.5/0.60 = 2.500 m/s
  v*(tau=0.8) = (L/2)/(0.8+0.30) = 1.5/1.10 = 1.364 m/s
Also the V-curve: RMSE(v) ~ (L/R)|L/2 - v*tau_tot| = 0.2*|1.5 - v*tau_tot|.

Pass criterion: v* within ~5%; V-curve slopes within ~15%.
"""
import os
import numpy as np

DT      = 0.01
T_CTRL  = 0.10
T_F     = 0.25
L       = 3.0
R       = 15.0
DUR     = 120.0
SIGMA   = np.radians(2.0)
MC      = 10

def simulate(v, tau, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(DUR/DT)
    ctrl_every = int(T_CTRL/DT)
    delay_steps = int(tau/DT)
    pos = np.array([R, 0.0])
    heading = np.pi/2            # tangent at start
    cmd = heading
    hist = [pos.copy()]
    errs = []
    for k in range(n_steps):
        if k % ctrl_every == 0:
            idx = max(0, len(hist)-1-delay_steps)
            dp = hist[idx]
            # closest point on circle + lookahead L along arc
            th = np.arctan2(dp[1], dp[0])
            aim_th = th + L/R
            aim = R*np.array([np.cos(aim_th), np.sin(aim_th)])
            d = aim - dp
            cmd = np.arctan2(d[1], d[0]) + rng.normal(0, SIGMA)
        # first-order heading lag toward cmd
        dh = (cmd - heading + np.pi) % (2*np.pi) - np.pi
        heading += (DT/T_F)*dh
        pos = pos + v*DT*np.array([np.cos(heading), np.sin(heading)])
        hist.append(pos.copy())
        errs.append(abs(np.hypot(*pos) - R))
    return float(np.sqrt(np.mean(np.array(errs)**2)))

def vstar_sweep(tau, speeds):
    rmses = []
    for v in speeds:
        vals = [simulate(v, tau, 1000+7*i) for i in range(MC)]
        rmses.append(float(np.mean(vals)))
    i = int(np.argmin(rmses))
    vs = speeds[i]
    if 1 <= i <= len(speeds)-2:
        c = np.polyfit(speeds[i-1:i+2], rmses[i-1:i+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if speeds[i-1] <= vf <= speeds[i+1]:
                vs = float(vf)
    return vs, rmses

if __name__ == "__main__":
    C_PRED = T_CTRL/2 + T_F
    print(f"independent implementation: c_pred = {C_PRED:.3f} s")
    for tau in [0.3, 0.8]:
        pred = (L/2)/(tau + C_PRED)
        speeds = list(np.round(np.linspace(0.4*pred, 1.8*pred, 15), 3))
        vs, rmses = vstar_sweep(tau, speeds)
        print(f"\ntau={tau}: v*_obs = {vs:.3f}  v*_pred = {pred:.3f}  "
              f"err = {100*(vs-pred)/pred:+.1f}%")
        # V-curve slope check on the rising side
        te = tau + C_PRED
        print("  v, RMSE_obs, RMSE_pred=(L/R)|L/2-v*te|:")
        for v, rm in zip(speeds, rmses):
            print(f"    {v:6.3f}  {rm:6.4f}  {(L/R)*abs(L/2-v*te):6.4f}")
