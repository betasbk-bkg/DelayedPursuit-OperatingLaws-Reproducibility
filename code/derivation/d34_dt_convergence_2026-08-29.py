# -*- coding: utf-8 -*-
"""
Derivation note -- D34: the intercept offset is the flow's time step.

D33 (risk 1) ruled out estimator bias: applying the same estimator to a density
built ANALYTICALLY from the exact orbit recovers the exact edge to -0.04 deg,
while the flow sits +0.32 deg above Eq. (12).  The remaining suspect is the
flow's own discretisation -- the fold is a tangency, where a first-order
integrator displaces the turning point.

Test: refine dt and extrapolate.  If the gap goes to zero, Eq. (12) is exact to
the numerical floor and the open problem closes.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0); D = math.radians(45.0); K = 5.0


def edge(u, dt, grid, T=1200.0):
    M, _ = d10.flow(u, k=K, T=T, dt=dt, mgrid=grid)
    ctr, rho = d10.density(M, nbin=360)
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+1.5) & (rho > 0)
    cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
    return float(-cf[1]/cf[0])


if __name__ == "__main__":
    grid = d19.sawtooth(45.0)
    DTS = (2e-3, 1e-3, 5e-4, 2.5e-4)
    print("flow edge minus Eq.(12), versus time step")
    print(f"{'u':>5} " + " ".join(f"{'dt='+format(d,'.1e'):>9}" for d in DTS)
          + f" {'extrap dt->0':>13}")
    rows = []
    for u in (0.30, 0.50, 0.71, 0.90):
        eq = d19.m_s_closed(u, K, D)*DEG
        vals = [edge(u, dt, grid) - eq for dt in DTS]
        ex = vals[-1] + (vals[-1] - vals[-2])          # first-order Richardson
        rows.append(dict(u=u, dts=list(DTS), vals=vals, extrap=ex))
        print(f"{u:5.2f} " + " ".join(f"{v:9.2f}" for v in vals) + f" {ex:13.2f}")
    e = [abs(r["extrap"]) for r in rows]
    e0 = [abs(r["vals"][0]) for r in rows]
    print(f"\nmean |offset|: {np.mean(e0):.2f} deg at dt = 2e-3  ->  "
          f"{np.mean(e):.2f} deg extrapolated to dt -> 0")
    json.dump(rows, open(os.path.join(HERE, "d34_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("-> d34_results_2026-08-29.json")
