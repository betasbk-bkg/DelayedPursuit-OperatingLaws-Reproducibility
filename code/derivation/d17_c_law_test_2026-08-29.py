# -*- coding: utf-8 -*-
"""
D17: pre-registered test of the c law found in D16.

D16 (k = 1.5 ... 4) plus D11 (k = 5) give
    k     1.5   2.0   2.5   3.0   4.0   5.0
    c     1.00  1.15  1.34  1.57  1.94  2.32
which is linear in k, and rewriting the edge shift as
    m_s(u) = m_s(0) + u * (Delta/2 + a/k)   <=>   c = k*Delta/2 + a
gives a = 0.357, 0.365, 0.358, 0.392, 0.369, 0.357 -- constant to +-0.02.

So the delay advances the caustic edge by HALF A CELL per unit delay margin,
independently of the geometry, plus a small k-dependent correction; and the
apparent "c = 2" of the two-channel argument is a coincidence of the published
geometry, where k*Delta/2 = 5*(pi/4)/2 = 1.96.

PRE-REGISTERED PREDICTION (written before running):
    k = 6.5  ->  c = 6.5*0.3927 + 0.37 = 2.92
    k = 8.0  ->  c = 8.0*0.3927 + 0.37 = 3.51
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d10", os.path.join(HERE, "d10_delayed_flow_2026-08-29.py"))
d10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d10)
D_RAD = math.radians(45.0)
A_FIT = 0.37

if __name__ == "__main__":
    grid = d10.b_table()
    US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]
    print(f"{'k':>6} {'c predicted':>12} {'c observed':>11} {'err':>8} {'a implied':>10}")
    rows = []
    for k in [6.5, 8.0]:
        us, ms = [], []
        for u in US:
            M, _ = d10.flow(u, k=k, T=900.0, mgrid=grid)
            ctr, rho = d10.density(M, nbin=180)
            e, r2, pk = d10.caustic_edge(ctr, rho)
            if not math.isnan(e):
                us.append(u); ms.append(e)
        cf = np.polyfit(us, ms, 1)
        c = cf[0]/(math.degrees(1.0)/k)
        pred = k*D_RAD/2 + A_FIT
        rows.append(dict(k=k, c_obs=float(c), c_pred=float(pred),
                         a_implied=float(c - k*D_RAD/2), intercept=float(cf[1])))
        print(f"{k:6.1f} {pred:12.2f} {c:11.2f} {100*(c-pred)/pred:+7.1f}% "
              f"{c - k*D_RAD/2:10.3f}")
    json.dump(dict(a_fit=A_FIT, rows=rows),
              open(os.path.join(HERE, "d17_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d17_results_2026-08-29.json")
