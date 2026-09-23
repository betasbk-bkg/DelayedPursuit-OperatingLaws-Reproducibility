# -*- coding: utf-8 -*-
"""
D14: decisive test for D13's null result.

D13's piecewise-linear return map gives a u-INDEPENDENT caustic edge (c = 0),
while the flow driven by the true b(m) gives c = 2.32.  Two explanations:
  (a) the return-map algebra is wrong, or
  (b) the piecewise-linear caricature itself destroys the effect, i.e. the delay
      dependence is carried by the CURVATURE of b(m) near the snap-region end.
Test: run the SAME numerical flow (D10) but feed it the caricature b instead of
the closed-form b.  If c collapses to ~0, explanation (b) is confirmed and the
return map is vindicated as an exact solution of the wrong caricature.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d10", os.path.join(HERE, "d10_delayed_flow_2026-08-29.py"))
d10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d10)

DELTA, M_C, K = 45.0, 19.5, 5.0


def caricature_table(n=2001, b_e=0.0):
    ms = np.linspace(-DELTA/2, DELTA/2, n)
    bs = np.where(np.abs(ms) <= M_C, -ms,
                  -np.sign(ms)*(M_C - (np.abs(ms)-M_C)*(M_C-b_e)/(DELTA/2-M_C)))
    return np.radians(ms), np.radians(bs), math.radians(DELTA)


if __name__ == "__main__":
    res = {}
    for tag, grid in [("closed-form b", d10.b_table()),
                      ("caricature b_e=0", caricature_table(b_e=0.0)),
                      ("caricature b_e=5", caricature_table(b_e=5.0))]:
        us, ms = [], []
        print(f"\n{tag}")
        for u in [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]:
            M, _ = d10.flow(u, mgrid=grid)
            ctr, rho = d10.density(M, nbin=180)
            e, r2, pk = d10.caustic_edge(ctr, rho)
            us.append(u); ms.append(e)
            print(f"   u={u:4.2f}  m_s={e:+7.2f}  R^2={r2:.3f}  peak={pk:+6.2f}")
        cf = np.polyfit(us, ms, 1)
        c = cf[0]/(math.degrees(1.0)/K)
        print(f"   -> c = {c:.2f}   (intercept {cf[1]:+.2f} deg)")
        res[tag] = dict(u=us, m_s=ms, c=float(c), intercept=float(cf[1]))
    json.dump(res, open(os.path.join(HERE, "d14_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d14_results_2026-08-29.json")
