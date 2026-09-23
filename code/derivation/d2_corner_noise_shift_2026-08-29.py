# -*- coding: utf-8 -*-
"""
Derivation note -- D2: the corner amplitude as a noise-shifted bias optimum
(open problems (i) "resolving the corner amplitude" and (v) "the triangle's
residual +10-12%").

Claim under test
----------------
The corner optimum is NOT the zero of the deterministic bias (D1/Lemma A shows
that zero is alpha-independent at u = 1/2 in the linearised transit, and drifts
only weakly in the exact one).  It is the minimum of

    RMSE^2(v) = RMSE_bias^2(v)          [noise-free lap, exact geometry, 0 param]
              + sigma_th^2 * T_eff * v^2 * tau * M(u)      [aggregation noise]
              + floor^2                                     [v-independent]

with the SAME two noise constants that close the circle's dither regime in the
paper (sigma_th = 0.1501 rad measured independently; T_eff = 0.528 s), and
M(u) = (1+sin u)/cos u the delayed-OU variance inflation.  Nothing is fitted to
polygon v* data.

If true this explains, with no new constant:
  * why the noise-free transit amplitude A0(alpha) = u*/cos(alpha/2) RISES with
    alpha (0.51 -> 0.60, measured in D1) yet the OBSERVED amplitude is flat at
    0.4989 +/- 0.0105 -- the noise term shifts the optimum down by an
    alpha-dependent amount, because the bias well gets shallower as alpha grows;
  * the sign and size of the residual per polygon, including the triangle.

Ground truth compared against: the published alpha sweep (dodecagon/octagon/
pentagon, side 12 m) and the rotation test (square/hexagon/triangle), both run
with the full 150-agent 8-direction engine.

Outputs: d2_results_2026-08-29.json
"""
import json, math, time
import numpy as np

DT      = 1.0/60.0
WIN     = 0.3
VI      = int(round(WIN/DT))
SMOOTH  = 0.2
T_S     = DT/SMOOTH
L       = 2.0
SIG_TH  = 0.1501        # rad, measured aggregate direction noise (independent)
T_EFF   = 0.528         # s, fitted on the CIRCLE dither surface only
DUR     = 65.0          # engine run length


class Path:
    """closed polyline, arc-length parameterised (same construction as the engine)"""
    def __init__(self, pts):
        self.A = np.array(pts[:-1], float)
        self.V = np.array(pts[1:], float) - self.A
        self.len = np.linalg.norm(self.V, axis=1)
        self.L2 = np.einsum('ij,ij->i', self.V, self.V)
        self.cum = np.concatenate([[0], np.cumsum(self.len)])
        self.circ = float(self.cum[-1])

    def closest(self, p):
        t = np.clip(np.einsum('ij,ij->i', p-self.A, self.V)/self.L2, 0, 1)
        Q = self.A + t[:, None]*self.V
        d = np.linalg.norm(p-Q, axis=1)
        i = int(np.argmin(d))
        return float(self.cum[i]+t[i]*self.len[i]), float(d[i])

    def at(self, a):
        a = a % self.circ
        i = max(min(int(np.searchsorted(self.cum, a))-1, len(self.A)-1), 0)
        return self.A[i] + np.clip((a-self.cum[i])/self.len[i], 0, 1)*self.V[i]


def regular_polygon(n, side, rot_deg=0.0):
    R = side/(2*math.sin(math.pi/n))
    a = np.linspace(0, 2*math.pi, n, endpoint=False) + math.radians(rot_deg)
    P = [[R*math.cos(x), R*math.sin(x)] for x in a]
    return Path(P+[P[0]])


def lap_rmse_bias(path, v, tau_s, start_arc=0.0, dur=DUR):
    """Noise-free, quantiser-free lap: pure deterministic bias RMSE.
    Same primitives as the engine (ZOH at WIN, pure delay, velocity lerp)."""
    delay_f = int(round(tau_s/DT))
    pos = path.at(start_arc).copy()
    s0, _ = path.closest(pos)
    nxt = path.at(s0+0.05) - pos
    cur = nxt/np.linalg.norm(nxt)
    vel = cur*v
    hist = [pos.copy()]
    nf = int(dur/DT)
    acc = 0.0; cnt = 0
    for f in range(nf):
        if f % VI == 0:
            pd = hist[max(0, len(hist)-1-delay_f)]
            sd, _ = path.closest(pd)
            dd = path.at(sd+L) - pd
            n = np.linalg.norm(dd)
            if n > 1e-12:
                cur = dd/n
        vel = vel + SMOOTH*(cur*v - vel)
        pos = pos + vel*DT
        hist.append(pos.copy())
        _, d = path.closest(pos)
        acc += d*d; cnt += 1
    return math.sqrt(acc/cnt)


def M_of_u(u):
    u = min(u, math.pi/2 - 1e-3)
    return (1.0 + math.sin(u))/math.cos(u)


def predict_ustar(path, tau_s, speeds, floor=0.0, mc_starts=3, noise=True,
                  sig=SIG_TH, teff=T_EFF):
    """zero-parameter prediction of u* = v* tau_tot / L"""
    tau_tot = tau_s + WIN/2 + T_S
    J = []
    for v in speeds:
        b2 = np.mean([lap_rmse_bias(path, v, tau_s, start_arc=k*path.circ/mc_starts)**2
                      for k in range(mc_starts)])
        u = v*tau_tot/L
        nz = (sig**2)*teff*(v**2)*tau_s*M_of_u(u) if noise else 0.0
        J.append(b2 + nz + floor**2)
    J = np.array(J)
    i = int(np.argmin(J)); vs = speeds[i]
    if 1 <= i <= len(speeds)-2:
        c = np.polyfit(speeds[i-1:i+2], J[i-1:i+2], 2)
        if c[0] > 0:
            vv = -c[1]/(2*c[0])
            if speeds[i-1] <= vv <= speeds[i+1]:
                vs = vv
    return vs*tau_tot/L, J


# ------------------------------ observed data --------------------------------
# published alpha sweep (side 12 m, random start) and rotation test (random start)
OBS = {   # name: (n_sides, side, rot_deg, alpha_deg, {tau_ms: u_tot observed})
    "dodecagon": (12, 12.0, 0.0, 30.0,  {100: 0.5484, 433: 0.5314, 1000: 0.4696}),
    "octagon":   (8,  12.0, 0.0, 45.0,  {100: 0.4621, 433: 0.4951, 1000: 0.4807}),
    "pentagon":  (5,  12.0, 0.0, 72.0,  {100: 0.4150, 433: 0.4259, 1000: 0.3808}),
    "hexagon":   (6,  10.0, 0.0, 60.0,  {100: 0.442,  433: 0.425,  1000: 0.417}),
    "square":    (4,  20.0, 45.0, 90.0, {100: 0.354,  433: 0.364,  1000: 0.370}),
    "triangle":  (3,  17.3205, 0.0, 120.0, {100: 0.276, 433: 0.300, 1000: 0.262}),
}

if __name__ == "__main__":
    t0 = time.time()
    speeds = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0])
    res = {}
    print(f"{'case':11s} {'alpha':>5} {'tau':>5} {'obs':>7} {'pred(N)':>8} {'pred(0)':>8} "
          f"{'A_obs':>6} {'A_N':>6} {'A_0':>6}  {'err(N)':>7}")
    for name, (n, side, rot, alpha, taus) in OBS.items():
        path = regular_polygon(n, side, rot)
        ca = math.cos(math.radians(alpha)/2)
        res[name] = dict(alpha=alpha, cos=ca, rows=[])
        for tms, uobs in taus.items():
            tau = tms/1000.0
            uN, _ = predict_ustar(path, tau, speeds, noise=True)
            u0, _ = predict_ustar(path, tau, speeds, noise=False)
            row = dict(tau_ms=tms, u_obs=uobs, u_pred_noise=uN, u_pred_clean=u0,
                       A_obs=uobs/ca, A_pred_noise=uN/ca, A_pred_clean=u0/ca,
                       rel_err=(uN-uobs)/uobs)
            res[name]["rows"].append(row)
            print(f"{name:11s} {alpha:5.0f} {tms:5d} {uobs:7.4f} {uN:8.4f} {u0:8.4f} "
                  f"{uobs/ca:6.3f} {uN/ca:6.3f} {u0/ca:6.3f}  {100*(uN-uobs)/uobs:+6.1f}%")
    json.dump(res, open("d2_results_2026-08-29.json", "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> d2_results_2026-08-29.json")
