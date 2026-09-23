# -*- coding: utf-8 -*-
"""
Derivation note -- D35: the u-dependence of the caustic broadening.

D33 (risk 2) broadened the analytic caustic by a CONSTANT sigma_m = s_bar, which
follows from the quasi-static gain (a persistent angular error q costs y = L q,
so the aim angle moves by q).  It improved the low-margin end a lot and
overshot the high-margin end, i.e. the width must fall with u.

The quasi-static gain is only the DC value.  Linearising the loop about the
orbit, with the residual vote spread entering as an angle disturbance eta:

    delta_w' = k (eta - delta_w(T - theta_d)),      delta_m' = -k delta_z(T - theta_d)
    delta_z  = eta - delta_w(T - theta_d)

In frequency, delta_Z = eta * j*omega / (j*omega + k e^{-j omega theta_d}) -- it
vanishes at DC, which cancels the integrator in delta_m -- and

    delta_M / eta  =  -k e^{-j omega theta_d} / (j omega + k e^{-j omega theta_d})
    |delta_M / eta| = k / |j omega + k e^{-j omega theta_d}|                (35.1)

which is 1 at DC (recovering s_bar) and rolls off above.  The residual spread is
injected once per vote, so in the time units of the flow (T = omega_0 t) the
sample interval is dT = WIN * omega_0 = (WIN L / (tau_tot R)) * u, giving a
Nyquist limit that itself moves with u.  Hence

    sigma_m(u)^2 = s_bar^2 * (dT/pi) * Integral_0^{omega_N} |35.1|^2 d omega   (35.2)

with the DC check: if the transfer were flat, (35.2) returns s_bar exactly.

Outputs: d35_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d12 = _load("d12", "d12_sigma_closed_2026-08-29.py")
d20 = _load("d20", "d20_sigma_analytic_2026-08-29.py")
d33 = _load("d33", "d33_three_risks_2026-08-29.py")

DEG = math.degrees(1.0)
K, DELTA_DEG = 5.0, 45.0
D = math.radians(DELTA_DEG)
L, R, WIN, TAU_TOT = 2.0, 10.0, 0.3, 0.6663
US = (0.40, 0.50, 0.60, 0.71, 0.80, 0.90)


def sigma_m(u, s_bar, k=K, n=20000):
    """Eq. (35.2): the broadening width, derived, no fitting."""
    th = u/k
    dT = (WIN*L/(TAU_TOT*R))*u          # vote interval in flow time units
    wN = math.pi/dT
    w = np.linspace(1e-9, wN, n)
    den = 1j*w + k*np.exp(-1j*w*th)
    Tf = np.abs(k/den)**2
    I = np.trapezoid(Tf, w)
    return s_bar*math.sqrt(dT*I/math.pi), dT, wN


if __name__ == "__main__":
    grid = np.linspace(-DELTA_DEG/2, DELTA_DEG/2, 361)
    s_bar = float(np.sqrt(np.mean([d12.b_s_closed(v)[1]**2 for v in grid])))
    vals = np.array([sum(x*x for x in d12.b_s_closed(v)) for v in grid])
    d8 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d8_results_2026-08-29.json")))["sigma_vs_u"]}
    d21 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d21_results_2026-08-29.json")))["rows"]}

    print(f"s_bar = {s_bar:.3f} deg   (the DC / quasi-static width)")
    print(f"\n{'u':>5} {'sigma_m(u)':>11} {'dT':>7} {'measured':>9} {'sharp':>8} "
          f"{'const width':>12} {'derived width':>14}")
    rng = np.random.default_rng(11)
    rows = []
    for u in US:
        sm_u, dT, wN = sigma_m(u, s_bar)
        ms = d33.analytic_samples(u, K, D, n=200000)
        out = {}
        for tag, width in (("const", s_bar), ("derived", sm_u)):
            pert = ms + rng.normal(0.0, width, ms.size)
            pert = ((pert + DELTA_DEG/2) % DELTA_DEG) - DELTA_DEG/2
            out[tag] = math.sqrt(float(np.mean(np.interp(pert, grid, vals))))
        meas = d8[u]["sigma"]
        sharp = d21[u]["two_piece"]
        rows.append(dict(u=u, sigma_m=sm_u, dT=dT, measured=meas, sharp=sharp,
                         const=out["const"], derived=out["derived"],
                         rel_sharp=(sharp-meas)/meas,
                         rel_const=(out["const"]-meas)/meas,
                         rel_derived=(out["derived"]-meas)/meas))
        print(f"{u:5.2f} {sm_u:11.3f} {dT:7.3f} {meas:9.3f} "
              f"{100*(sharp-meas)/meas:+7.1f}% {100*(out['const']-meas)/meas:+11.1f}% "
              f"{100*(out['derived']-meas)/meas:+13.1f}%")

    for tag, key in (("sharp (no broadening)", "rel_sharp"),
                     ("constant width s_bar ", "rel_const"),
                     ("derived width (35.2)", "rel_derived")):
        e = 100*np.mean([abs(r[key]) for r in rows])
        print(f"    {tag:22s} mean |err| = {e:5.1f}%")
    json.dump(dict(s_bar=s_bar, rows=rows),
              open(os.path.join(HERE, "d35_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d35_results_2026-08-29.json")
