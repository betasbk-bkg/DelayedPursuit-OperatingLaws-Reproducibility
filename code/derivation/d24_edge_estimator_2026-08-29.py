# -*- coding: utf-8 -*-
"""
Derivation note -- D24: is the intercept offset of Eq. (12) physics or an
estimator artefact?  (Open problem (ii) of manuscript v5, Sec. 11.4.)

Eq. (12) reproduces dm_s/du to 0.1-2.5% but sits below the measured edge by a
constant 0.25 / 1.04 / 2.21 deg at k = 3 / 4 / 5.  Flat in u, growing with k.

Where the estimator could bias it.  Near the fold the snap-region orbit gives
    m' = k (t - t*)    exactly,   so   m - m_s = k (t - t*)^2 / 2
    => rho_fold(m) = A / sqrt(m - m_s)   and   rho^-2 linear in m -- exact.
But the density is NOT rho_fold alone.  The boost phase (Sec. 11.2 phase A)
sweeps the same m range once more at high speed, adding a smooth BACKGROUND.
Fitting rho^-2 linearly over a window then extrapolates a biased zero, and the
bias grows with the background, i.e. with k.

Three estimators are compared against the same flow:
    E1  rho^-2 linear, 8 deg window            (what D10/D19 used)
    E2  the same, window swept 2..12 deg       (bias should scale with window)
    E3  rho = A/sqrt(m - m_s) + B, three parameters, background absorbed

If E3 removes the offset, Eq. (12) is exact and the open problem closes.

Outputs: d24_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0)
US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]


def edge_linear(ctr, rho, width):
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+width) & (rho > 0)
    if sel.sum() < 4:
        return float("nan")
    cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
    return float(-cf[1]/cf[0])


def edge_bg(ctr, rho, width):
    """rho = A/sqrt(m - m_s) + B"""
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+width) & (rho > 0)
    if sel.sum() < 5:
        return float("nan")
    x, y = ctr[sel], rho[sel]
    m0 = edge_linear(ctr, rho, width)
    if math.isnan(m0):
        m0 = x[0] - 1.0

    def f(m, A, ms, B):
        return A/np.sqrt(np.maximum(m - ms, 1e-9)) + B
    try:
        p, _ = curve_fit(f, x, y, p0=[y[0]*math.sqrt(max(x[0]-m0, 0.5)), m0, y[-1]],
                         maxfev=20000)
        return float(p[1])
    except Exception:
        return float("nan")


if __name__ == "__main__":
    out = {"cases": []}
    print("estimator comparison against Eq. (12), pure sawtooth flow")
    for delta_deg, ks in [(45.0, [3.0, 4.0, 5.0]), (22.5, [6.0, 8.0, 10.0])]:
        grid = d19.sawtooth(delta_deg)
        D = math.radians(delta_deg)
        for k in ks:
            offs = {"E1_w8": [], "E3_bg": []}
            sweep = {w: [] for w in (2.0, 4.0, 8.0, 12.0)}
            for u in US:
                M, _ = d10.flow(u, k=k, T=900.0, mgrid=grid)
                ctr, rho = d10.density(M, nbin=180)
                pred = d19.m_s_closed(u, k, D)*DEG
                offs["E1_w8"].append(edge_linear(ctr, rho, 8.0) - pred)
                offs["E3_bg"].append(edge_bg(ctr, rho, 8.0) - pred)
                for w in sweep:
                    sweep[w].append(edge_linear(ctr, rho, w) - pred)
            row = dict(delta=delta_deg, k=k, kD2=k*D/2,
                       off_E1=float(np.nanmean(offs["E1_w8"])),
                       off_E3=float(np.nanmean(offs["E3_bg"])),
                       spread_E3=float(np.nanstd(offs["E3_bg"])),
                       window_sweep={str(w): float(np.nanmean(v)) for w, v in sweep.items()})
            out["cases"].append(row)
            print(f"  Delta={delta_deg:5.1f}  k={k:4.1f}  kD/2={k*D/2:.2f}")
            print(f"    E1 (linear, 8 deg)   offset = {row['off_E1']:+6.2f} deg")
            print(f"    E2 window sweep      " +
                  "  ".join(f"{w:.0f}deg:{row['window_sweep'][str(w)]:+5.2f}"
                            for w in (2.0, 4.0, 8.0, 12.0)))
            print(f"    E3 (with background) offset = {row['off_E3']:+6.2f} deg "
                  f"(u-spread {row['spread_E3']:.2f})")

    e1 = [abs(r["off_E1"]) for r in out["cases"]]
    e3 = [abs(r["off_E3"]) for r in out["cases"]]
    print(f"\nmean |offset|:  E1 = {np.mean(e1):.2f} deg,  E3 = {np.mean(e3):.2f} deg")
    print(f"max  |offset|:  E1 = {max(e1):.2f} deg,  E3 = {max(e3):.2f} deg")
    json.dump(out, open(os.path.join(HERE, "d24_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d24_results_2026-08-29.json")
