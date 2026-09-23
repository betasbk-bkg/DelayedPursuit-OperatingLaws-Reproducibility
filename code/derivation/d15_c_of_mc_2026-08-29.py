# -*- coding: utf-8 -*-
"""
D15: does c depend on the snap half-width m_c?

D14 showed c = 2.32 is identical for the exact b and for piecewise-linear
caricatures with different return slopes -- so c does not see the return region's
shape.  But an exact calculation in the PERFECT SAWTOOTH limit (m_c = Delta/2,
no return region at all) gives a u-INDEPENDENT edge:

    z0 = Delta/2 - u/k  (period closure),  fold at z* = (1-u)/k
    m_s = -k Delta^2/8 - 1/(2k)          <- u cancels exactly, so c = 0

Both cannot hold unless c depends on m_c.  Prediction: c -> 0 as m_c -> Delta/2,
and since D9 gives m_c = Delta/2 - a_acc, c would be a function of the crowd
composition.  This scans m_c directly.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d10", os.path.join(HERE, "d10_delayed_flow_2026-08-29.py"))
d10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d10)
DELTA, K = 45.0, 5.0


def caricature(m_c, n=2001, b_e=0.0):
    ms = np.linspace(-DELTA/2, DELTA/2, n)
    if m_c >= DELTA/2 - 1e-9:
        bs = -ms
    else:
        bs = np.where(np.abs(ms) <= m_c, -ms,
                      -np.sign(ms)*(m_c - (np.abs(ms)-m_c)*(m_c-b_e)/(DELTA/2-m_c)))
    return np.radians(ms), np.radians(bs), math.radians(DELTA)


if __name__ == "__main__":
    US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]
    print(f"{'m_c':>7} {'a_acc':>7} {'c':>7} {'intercept':>10} {'m_s(0.71)':>10}")
    rows = []
    for m_c in [22.5, 22.0, 21.0, 19.5, 17.0, 14.0, 10.0]:
        us, ms = [], []
        for u in US:
            M, _ = d10.flow(u, mgrid=caricature(m_c))
            ctr, rho = d10.density(M, nbin=180)
            e, r2, pk = d10.caustic_edge(ctr, rho)
            us.append(u); ms.append(e)
        cf = np.polyfit(us, ms, 1)
        c = cf[0]/(math.degrees(1.0)/K)
        rows.append(dict(m_c=m_c, a_acc=DELTA/2-m_c, c=float(c),
                         intercept=float(cf[1]), m_s=ms, u=us))
        print(f"{m_c:7.1f} {DELTA/2-m_c:7.1f} {c:7.2f} {cf[1]:10.2f} "
              f"{ms[US.index(0.71)]:10.2f}")

    print("\nsawtooth-limit closed form  m_s = -k*Delta^2/8 - 1/(2k):")
    D = math.radians(DELTA)
    print(f"   predicted {math.degrees(-K*D*D/8 - 1/(2*K)):+.2f} deg   "
          f"measured at m_c=22.5: {rows[0]['intercept']:+.2f} deg (intercept), "
          f"c = {rows[0]['c']:.2f}")

    good = [r for r in rows if r["m_c"] < 22.4]
    A = np.array([DELTA/2 - r["m_c"] for r in good])
    C = np.array([r["c"] for r in good])
    print("\nc versus a_acc = Delta/2 - m_c:")
    for a, c in zip(A, C):
        print(f"   a_acc = {a:5.1f} deg   c = {c:6.2f}   c/a_acc = {c/a:6.3f}")
    json.dump(dict(rows=rows), open(os.path.join(HERE, "d15_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d15_results_2026-08-29.json")
