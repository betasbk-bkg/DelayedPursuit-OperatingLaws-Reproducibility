# -*- coding: utf-8 -*-
"""
D16: c versus k = R/L, and the chattering threshold.

D15 showed c ~ 2.32 independently of the snap half-width and of the return
region -- including the pure sawtooth, where an exact single-sweep calculation
gives c = 0.  The single-sweep calculation assumed the orbit crosses one cell
boundary per period.  It is valid only while the phase advances monotonically,
i.e. while the fold cannot be reached inside a cell:

    k * Delta / 2  <  1        (non-chattering)

For the published circle k*Delta/2 = 5*0.785/2 = 1.96 > 1, so the orbit chatters
across boundaries and the single-sweep map is simply the wrong topology.

Sharp prediction: c should collapse to ~0 below k = 2/Delta = 2.55 and jump to
~2.3 above it.  This scans k directly.
Note: T = 900 (shorter integration than D10/D11) to keep the six-k scan
affordable; the k = 5 row reproduces D11 (c = 2.32) and validates the shortening.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d10", os.path.join(HERE, "d10_delayed_flow_2026-08-29.py"))
d10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d10)
DELTA = 45.0
D_RAD = math.radians(DELTA)

if __name__ == "__main__":
    grid = d10.b_table()
    US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]
    print(f"threshold: k_crit = 2/Delta = {2/D_RAD:.2f}")
    print(f"{'k':>6} {'k*D/2':>7} {'regime':>16} {'c':>7} {'intercept':>10} {'R2 mean':>8}")
    rows = []
    for k in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        us, ms, r2s = [], [], []
        for u in US:
            M, _ = d10.flow(u, k=k, T=900.0, mgrid=grid)
            ctr, rho = d10.density(M, nbin=180)
            e, r2, pk = d10.caustic_edge(ctr, rho)
            if not math.isnan(e):
                us.append(u); ms.append(e); r2s.append(r2)
        if len(us) < 4:
            print(f"{k:6.1f} {k*D_RAD/2:7.2f}  no usable caustic")
            continue
        cf = np.polyfit(us, ms, 1)
        c = cf[0]/(math.degrees(1.0)/k)
        reg = "chattering" if k*D_RAD/2 > 1 else "single sweep"
        rows.append(dict(k=k, kD2=k*D_RAD/2, c=float(c), intercept=float(cf[1]),
                         r2=float(np.mean(r2s)), u=us, m_s=ms))
        print(f"{k:6.1f} {k*D_RAD/2:7.2f} {reg:>16} {c:7.2f} {cf[1]:10.2f} "
              f"{np.mean(r2s):8.3f}")
    json.dump(dict(k_crit=2/D_RAD, rows=rows),
              open(os.path.join(HERE, "d16_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d16_results_2026-08-29.json")
