# -*- coding: utf-8 -*-
"""
Derivation note -- D21: closing the accuracy gap of the closed-form
sigma_theta by carrying the boundary return of b(m).

D20 gave two evaluations of  sigma^2 = (1/Delta) int_0^Delta [b(m(T))^2 + s^2] dT:
  (a) pure sawtooth b = -m            -> a genuine formula, but 15.7% error,
      because the sawtooth keeps |b| = 22.5 deg at the cell boundary where the
      true mixture has already collapsed to ~5 deg;
  (b) the true closed-form b, s       -> 7.0%, but a numerical quadrature.

This closes the gap while staying elementary.  b is represented by its two-point
linearisation -- the closed form of D9 EVALUATED at m_c and at Delta/2, no fit:

    |m| <= m_c :   b = -m                                   (exact: the snap)
    |m| >  m_c :   b = -sign(m) [ m_c - (|m| - m_c) * (m_c - b_e)/(Delta/2 - m_c) ]

with m_c = Delta/2 - a_acc (D9, exact) and b_e = |b_closed(Delta/2)|.

Because m(T) is a quadratic in T on each phase (Sec. 11.2) and b is piecewise
LINEAR in m, b(m(T)) is piecewise quadratic in T, so b^2 is piecewise quartic:
the integral stays exact polynomial arithmetic.  The only new work is locating
the times at which the orbit crosses +-m_c, which are roots of quadratics.

Outputs: d21_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


d12 = _load("d12", "d12_sigma_closed_2026-08-29.py")
d20 = _load("d20", "d20_sigma_analytic_2026-08-29.py")
DEG = math.degrees(1.0)
P = np.polynomial.polynomial


def quad_roots_in(c, lo, hi, target):
    """times in [lo, hi] where c0 + c1 T + c2 T^2 = target"""
    a, b, cc = c[2], c[1], c[0] - target
    out = []
    if abs(a) < 1e-15:
        if abs(b) > 1e-15:
            out = [-cc/b]
    else:
        d = b*b - 4*a*cc
        if d >= 0:
            r = math.sqrt(d)
            out = [(-b - r)/(2*a), (-b + r)/(2*a)]
    return sorted(t for t in out if lo - 1e-12 <= t <= hi + 1e-12)


def seg_integral(c, t0, t1, m_c, b_e, delta):
    """exact integral of b(m(T))^2 over [t0, t1], b the two-piece caricature."""
    mid = c[0] + c[1]*(t0+t1)/2 + c[2]*((t0+t1)/2)**2
    slope = (m_c - b_e)/(delta/2 - m_c)
    if abs(mid) <= m_c:
        bp = np.array([-c[0], -c[1], -c[2]])                    # b = -m
    else:
        sgn = 1.0 if mid > 0 else -1.0
        # b = -sgn*[ m_c + slope*m_c - slope*|m| ] with |m| = sgn*m
        const = -sgn*(m_c + slope*m_c)
        bp = np.array([const + slope*c[0], slope*c[1], slope*c[2]])
    sq = P.polymul(bp, bp)
    I = P.polyint(sq)
    return float(P.polyval(t1, I) - P.polyval(t0, I))


def sigma_2piece(u, k, delta, m_c, b_e, s_bar_deg):
    cA, cB, th = d20.orbit_polys(u, k, delta)
    total = 0.0
    for c, t0, t1 in [(cA, 0.0, th), (cB, th, delta)]:
        cuts = [t0, t1]
        for tgt in (m_c, -m_c):
            cuts += quad_roots_in(c, t0, t1, tgt)
        cuts = sorted(set(round(x, 12) for x in cuts))
        for i in range(len(cuts)-1):
            if cuts[i+1] - cuts[i] > 1e-12:
                total += seg_integral(c, cuts[i], cuts[i+1], m_c, b_e, delta)
    return math.degrees(math.sqrt(total/delta + math.radians(s_bar_deg)**2))


if __name__ == "__main__":
    K, DELTA_DEG = 5.0, 45.0
    DELTA = math.radians(DELTA_DEG)
    A_ACC = 3.0
    m_c = math.radians(DELTA_DEG/2 - A_ACC)
    b_e = abs(math.radians(d12.b_s_closed(DELTA_DEG/2)[0]))
    ms_grid = np.linspace(-DELTA_DEG/2, DELTA_DEG/2, 361)
    s_bar = float(np.sqrt(np.mean([d12.b_s_closed(v)[1]**2 for v in ms_grid])))
    print(f"two-point linearisation of the closed-form b (no fit):")
    print(f"    m_c = Delta/2 - a_acc = {math.degrees(m_c):.2f} deg")
    print(f"    b_e = |b_closed(Delta/2)| = {math.degrees(b_e):.2f} deg")
    print(f"    return slope = {(m_c-b_e)/(DELTA/2-m_c):.2f},  s_bar = {s_bar:.3f} deg")

    d8 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d8_results_2026-08-29.json")))["sigma_vs_u"]}
    d20r = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d20_results_2026-08-29.json")))["k5"]}

    print(f"\n{'u':>6} {'measured':>9} {'(a) sawtooth':>13} {'(a2) 2-piece':>13} "
          f"{'(b) quadrature':>15}")
    rows = []
    for u in [0.40, 0.50, 0.60, 0.71, 0.80, 0.90]:
        meas = d8[u]["sigma"]
        saw = d20r[u]["sawtooth"]
        two = sigma_2piece(u, K, DELTA, m_c, b_e, s_bar)
        mix = d20r[u]["mixture"]
        rows.append(dict(u=u, measured=meas, sawtooth=saw, two_piece=two, mixture=mix,
                         rel_two=(two-meas)/meas))
        print(f"{u:6.2f} {meas:9.3f} {saw:13.3f} {two:13.3f} {mix:15.3f}")
        print(f"{'':6} {'':9} {100*(saw-meas)/meas:+12.1f}% "
              f"{100*(two-meas)/meas:+12.1f}% {100*(mix-meas)/meas:+14.1f}%")
    for key, tag in [("sawtooth", "(a)  sawtooth      "),
                     ("two_piece", "(a2) two-piece     "),
                     ("mixture", "(b)  quadrature    ")]:
        e = [abs((r[key]-r["measured"])/r["measured"]) for r in rows]
        print(f"    {tag} mean |err| = {100*np.mean(e):5.1f}%   max {100*max(e):5.1f}%")
    op = [r for r in rows if 0.6 <= r["u"] <= 0.8]
    print(f"    (a2) at the published operating margins: "
          f"{100*min(abs(r['rel_two']) for r in op):.1f}-"
          f"{100*max(abs(r['rel_two']) for r in op):.1f}%")
    json.dump(dict(m_c_deg=math.degrees(m_c), b_e_deg=math.degrees(b_e),
                   s_bar=s_bar, rows=rows),
              open(os.path.join(HERE, "d21_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d21_results_2026-08-29.json")
