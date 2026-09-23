# -*- coding: utf-8 -*-
"""
Derivation note -- D20: sigma_theta in closed form.

Sec. 11 gives the orbit itself, not just its caustic edge, so the density never
has to be written down: rho is the invariant measure of a periodic orbit, and any
average against it is a TIME average over one period T_p = Delta.

    sigma_theta^2 = E_rho[ b^2 + s^2 ] = (1/Delta) * int_0^Delta [ b(m(T))^2 + s(m(T))^2 ] dT

with the orbit of Sec. 11.2 (z = Delta/2 - T, one forward crossing per period):

    phase A, T in [0, theta_d]   (jump in flight, J = 1)
        m_A(T) = -Delta/2 + alpha T + (k/2) T^2 ,   alpha = (1-u) + k Delta/2
    phase B, T in [theta_d, Delta]   (J = 0)
        m_B(T) = m1 + beta (T - theta_d) + (k/2)(T^2 - theta_d^2) ,
        beta = (1-u) - k Delta/2 ,   m1 = m_A(theta_d)

Both phases are quadratics in T, so for the pure sawtooth (b = -m) the integral
of b^2 is the integral of a quartic polynomial -- elementary and exact:

    sigma_theta^2 = (1/Delta) [ int_0^{theta_d} m_A^2 dT + int_{theta_d}^{Delta} m_B^2 dT ]
                    + <s^2>                                              (20.1)

Two evaluations are reported:
  (a) SAWTOOTH: (20.1) with b = -m and a constant residual spread -- a genuine
      closed form, no quadrature, no simulation;
  (b) EXACT MIXTURE: the same analytic orbit m(T) but the true closed-form b, s
      of D9/D12 -- one 1-D quadrature, still no simulation of crowd or vehicle.

Both are compared with the engine's measured sigma_theta(u) and with D12, which
needed the flow integrated numerically.

Outputs: d20_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


d12 = _load("d12", "d12_sigma_closed_2026-08-29.py")
DEG = math.degrees(1.0)


def orbit_polys(u, k, delta):
    """(cA, cB, theta_d) with m(T) = c0 + c1 T + c2 T^2 on each phase, radians."""
    th = u/k
    alpha = (1.0-u) + k*delta/2.0
    beta = (1.0-u) - k*delta/2.0
    cA = np.array([-delta/2.0, alpha, k/2.0])
    m1 = cA[0] + cA[1]*th + cA[2]*th*th
    c0 = m1 - beta*th - (k/2.0)*th*th
    cB = np.array([c0, beta, k/2.0])
    return cA, cB, th


def poly_int_sq(c, a, b):
    """exact integral of (c0 + c1 T + c2 T^2)^2 from a to b."""
    p = np.polynomial.polynomial.polymul(c, c)
    P = np.polynomial.polynomial.polyint(p)
    return float(np.polynomial.polynomial.polyval(b, P) -
                 np.polynomial.polynomial.polyval(a, P))


def sigma_sawtooth(u, k, delta, s_bar_deg):
    """closed form (20.1), degrees."""
    cA, cB, th = orbit_polys(u, k, delta)
    I = poly_int_sq(cA, 0.0, th) + poly_int_sq(cB, th, delta)
    return math.degrees(math.sqrt(I/delta)) if I > 0 else 0.0, \
           math.sqrt(math.degrees(math.sqrt(I/delta))**2 + s_bar_deg**2)


def sigma_exact_mixture(u, k, delta, n=20001):
    """analytic orbit + true closed-form b, s (one quadrature)."""
    cA, cB, th = orbit_polys(u, k, delta)
    TA = np.linspace(0.0, th, max(int(n*th/delta), 200))
    TB = np.linspace(th, delta, n)
    mA = cA[0] + cA[1]*TA + cA[2]*TA**2
    mB = cB[0] + cB[1]*TB + cB[2]*TB**2
    m = np.concatenate([mA, mB])
    T = np.concatenate([TA, TB])
    md = np.degrees(m)
    md = ((md + delta*DEG/2) % (delta*DEG)) - delta*DEG/2      # wrap into the cell
    vals = np.array([sum(x*x for x in d12.b_s_closed(v)) for v in md])
    return math.sqrt(np.trapezoid(vals, T)/delta)


if __name__ == "__main__":
    out = {}
    K, DELTA = 5.0, math.radians(45.0)
    ms_grid = np.linspace(-22.5, 22.5, 361)
    S = np.array([d12.b_s_closed(v)[1] for v in ms_grid])
    s_bar = float(np.sqrt(np.mean(S**2)))
    print(f"residual spread, rms over the cell: s_bar = {s_bar:.3f} deg")

    d8 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d8_results_2026-08-29.json")))["sigma_vs_u"]}
    d12r = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d12_results_2026-08-29.json")))["sigma_vs_u"]}

    print("\nsigma_theta(u) at k = 5, Delta = 45 deg   [engine reference in col 2]")
    print(f"{'u':>6} {'measured':>9} {'(a) sawtooth':>13} {'err':>7} "
          f"{'(b) mixture':>12} {'err':>7} {'D12 flow':>9} {'err':>7}")
    rows = []
    for u in [0.40, 0.50, 0.60, 0.71, 0.80, 0.90]:
        meas = d8[u]["sigma"]
        saw_b, saw = sigma_sawtooth(u, K, DELTA, s_bar)
        mix = sigma_exact_mixture(u, K, DELTA)
        flow = d12r[u]["sigma_pred"]
        rows.append(dict(u=u, measured=meas, sawtooth=saw, mixture=mix, flow=flow,
                         rel_saw=(saw-meas)/meas, rel_mix=(mix-meas)/meas,
                         rel_flow=(flow-meas)/meas))
        print(f"{u:6.2f} {meas:9.3f} {saw:13.3f} {100*(saw-meas)/meas:+6.1f}% "
              f"{mix:12.3f} {100*(mix-meas)/meas:+6.1f}% {flow:9.3f} "
              f"{100*(flow-meas)/meas:+6.1f}%")
    out["k5"] = rows
    for tag, key in [("(a) sawtooth", "rel_saw"), ("(b) mixture", "rel_mix"),
                     ("D12 flow", "rel_flow")]:
        e = [abs(r[key]) for r in rows]
        print(f"    {tag:14s} mean |err| = {100*np.mean(e):5.1f}%   max {100*max(e):5.1f}%")

    print("\nattenuation against the open-loop sweep (closed form 10.994 deg)")
    print(f"{'u':>6} {'measured':>9} {'mixture':>9}")
    for r in rows:
        print(f"{r['u']:6.2f} {r['measured']/10.9456:9.3f} {r['mixture']/10.994:9.3f}")
    op = [r for r in rows if 0.6 <= r["u"] <= 0.8]
    print(f"    at the published operating margins u = 0.655-0.753: "
          f"mixture error {100*min(abs(r['rel_mix']) for r in op):.1f}-"
          f"{100*max(abs(r['rel_mix']) for r in op):.1f}%")
    out["attenuation_at_071"] = [r for r in rows if r["u"] == 0.71][0]["mixture"]/10.994

    print("\nsecond grid (Delta = 22.5 deg, k = 10) -- same k*Delta/2 = 1.96")
    for u in [0.50, 0.71]:
        mix = sigma_exact_mixture(u, 10.0, math.radians(22.5))
        print(f"    u={u:4.2f}   sigma = {mix:.3f} deg")
        out.setdefault("grid16", []).append(dict(u=u, sigma=mix))

    json.dump(out, open(os.path.join(HERE, "d20_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d20_results_2026-08-29.json")
