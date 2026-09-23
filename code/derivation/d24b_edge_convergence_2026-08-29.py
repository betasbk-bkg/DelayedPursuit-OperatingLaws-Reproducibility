# -*- coding: utf-8 -*-
"""
D24b: convergence of the caustic-edge estimator.

D24's window sweep showed the offset of Eq. (12) collapses from ~2.2 deg (8 deg
fit window) to ~0.3 deg (2 deg window) and stops growing with k -- the signature
of a fit contaminated by the boost-phase background far from the fold, not of a
missing term in the derivation.  This refines the estimate: longer orbits, finer
bins, windows down to 1 deg, and checks whether the residual converges to zero
within the bin resolution.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0)
US = [0.30, 0.50, 0.71]

def edge(ctr, rho, width):
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+width) & (rho > 0)
    if sel.sum() < 4: return float("nan")
    cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
    return float(-cf[1]/cf[0])

if __name__ == "__main__":
    out = []
    for delta_deg, k in [(45.0, 3.0), (45.0, 5.0), (22.5, 10.0)]:
        grid = d19.sawtooth(delta_deg); D = math.radians(delta_deg)
        print(f"\nDelta = {delta_deg}, k = {k}  (kD/2 = {k*D/2:.2f})")
        print(f"  {'nbin':>6} {'binw':>6} " +
              " ".join(f"{w:>7.1f}deg" for w in (1.0, 1.5, 2.0, 3.0)))
        for nbin in (180, 360, 720):
            binw = delta_deg/nbin
            res = {w: [] for w in (1.0, 1.5, 2.0, 3.0)}
            for u in US:
                M, _ = d10.flow(u, k=k, T=2000.0, mgrid=grid)
                ctr, rho = d10.density(M, nbin=nbin)
                pred = d19.m_s_closed(u, k, D)*DEG
                for w in res:
                    res[w].append(edge(ctr, rho, w) - pred)
            row = {str(w): float(np.nanmean(v)) for w, v in res.items()}
            out.append(dict(delta=delta_deg, k=k, nbin=nbin, binw=binw, offsets=row))
            print(f"  {nbin:6d} {binw:6.3f} " +
                  " ".join(f"{row[str(w)]:+10.2f}" for w in (1.0, 1.5, 2.0, 3.0)))
    best = [r["offsets"]["1.0"] for r in out if r["nbin"] == 720]
    print(f"\nfinest grid, 1 deg window: offsets " +
          ", ".join(f"{b:+.2f}" for b in best) + " deg")
    json.dump(out, open(os.path.join(HERE, "d24b_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("-> d24b_results_2026-08-29.json")
