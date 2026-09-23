#!/usr/bin/env python3
"""
d131 -- the two plant-class constants, integrated to convergence, and the sign of the corner term.

WHY THIS SCRIPT EXISTS.  d38 derived u* = argmin J(u) for both plant classes with an explicit Euler
scheme at h = 1e-3 that also snaps the delay to the step grid (nd = round(u/h)).  Both are O(h), so
the minimiser it reports -- 0.499496 and 0.306034 -- is quoted to six digits but resolved to three.
Refining d38's own scheme moves it (heading 0.4995 -> 0.49838 -> 0.49848 at h = 2.5e-4, 5e-5), which
is the signature of a discretisation artefact rather than a converged value.

WHAT THIS SCRIPT DOES.
  (1) Integrates the same two linearised corner transits by the method of steps with a dense-output
      RK45 on every smooth piece: the delay is evaluated by interpolation instead of being snapped to
      a grid, the corner's delta is an exact state jump, the energy J = int e^2 dx is carried as an
      extra state (no quadrature error), and every discontinuity of the forcing -- the corner, the
      preview window, and their images one and more delays later -- is an integration breakpoint.
  (2) Reports the same computation at two tolerances and two integration lengths, so the digits that
      are converged can be read off rather than assumed.
  (3) Records the sign question.  d38's CODE adds the bracket [H(x-u-x_c) - H(x-x_c)]; d38's own
      docstring and the manuscript print it with a minus.  With the minus, J(u) is monotone
      increasing over the whole stable range -- there is no interior optimum at all -- so the printed
      sign contradicts the result the code produces.  Both are computed here and written out.
  (4) Reproduces d38's Euler scheme at three step sizes, so the drift toward the converged value is
      in the record.

RESULT (see the JSON):  u*_heading = 0.4984,  u*_curvature = 0.3078.

Run:    python d131_corner_energy_converged_2026-09-23.py
Writes: d131_corner_energy_converged_2026-09-23.json
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

XC = 4.0                      # corner position, as in d38
XEND = 60.0


def g_prev(x):
    """The look-ahead point's view of the corner: nonzero only one look-ahead before it."""
    return (1.0 - (XC - x)) if (XC - 1.0) < x < XC else 0.0


def step_fn(x, u):
    """psi_p(x-u) - psi_p(x) for a unit corner at XC."""
    return (1.0 if (x - u) > XC else 0.0) - (1.0 if x > XC else 0.0)


class History:
    """Piecewise dense output of the solution so far, queried at any earlier x."""

    def __init__(self, ncomp):
        self.segs = []
        self.ncomp = ncomp

    def add(self, sol):
        self.segs.append(sol)

    def at(self, x):
        if x <= 0:
            return np.zeros(self.ncomp)
        for s in self.segs:
            if s.t[0] - 1e-12 <= x <= s.t[-1] + 1e-12:
                return s.sol(min(max(x, s.t[0]), s.t[-1]))[:self.ncomp]
        last = self.segs[-1]
        return last.sol(last.t[-1])[:self.ncomp]


def breakpoints(u, xend):
    pts = {0.0, xend}
    k = 1
    while k * u < xend:
        pts.add(k * u)
        k += 1
    for k in range(0, 9):                       # the corner and the preview window, one delay at a time
        for b in (XC - 1.0 + k * u, XC + k * u):
            if 0.0 < b < xend:
                pts.add(b)
    return sorted(pts)


def energy(u, plant, sign=+1.0, rtol=1e-10, atol=1e-12, xend=XEND):
    """J(u) = int_0^xend e^2 dx.  `sign` multiplies the Heaviside bracket of the heading-servo plant."""
    ncomp = 1 if plant == "heading" else 2
    hist = History(ncomp)

    if plant == "heading":
        def rhs(x, y):
            return [-hist.at(x - u)[0] + g_prev(x - u) + sign * step_fn(x, u), y[0] ** 2]
        y = np.zeros(2)
    else:
        def rhs(x, y):
            d = hist.at(x - u)
            return [y[1], 2.0 * (-d[0] - d[1] + g_prev(x - u)), y[0] ** 2]
        y = np.zeros(3)

    bps = breakpoints(u, xend)
    for a, b in zip(bps[:-1], bps[1:]):
        if b - a < 1e-14:
            continue
        sol = solve_ivp(rhs, (a, b), y, method="RK45", dense_output=True,
                        rtol=rtol, atol=atol, max_step=(b - a))
        hist.add(sol)
        y = sol.y[:, -1].copy()
        if plant != "heading" and abs(b - XC) < 1e-12:
            y[1] -= 1.0                          # the corner's delta kicks the heading error by -alpha
    return float(y[-1])


def argmin_J(plant, lo, hi, sign=+1.0, rtol=1e-10, xend=XEND):
    r = minimize_scalar(lambda u: energy(u, plant, sign, rtol=rtol, xend=xend),
                        bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    return float(r.x), float(r.fun)


def euler_reference(plant_d38, lo, hi, hs=(1e-3, 2.5e-4, 5e-5)):
    """d38's own scheme at several step sizes, for the record."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "d38_corner_energy_two_plants_2026-08-29.py")
    spec = importlib.util.spec_from_file_location("d38", path)
    d38 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d38)
    out = []
    for h in hs:
        r = minimize_scalar(lambda u: d38.transit_energy(u, plant_d38, h=h),
                            bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
        out.append({"h": h, "u_star": float(r.x), "J": float(r.fun)})
    return out


def main():
    out = {"what": "converged plant-class constants; supersedes d38's Euler values",
           "method": "method of steps, dense-output RK45 per smooth piece, delay by interpolation, "
                     "energy carried as a state, discontinuities as breakpoints"}

    # (3) the sign of the bracket, heading servo
    grid = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
    out["sign_study"] = {
        "equation": "e'(x) = -e(x-u) + g(x-u) + s*[H(x-u-x_c) - H(x-x_c)]",
        "u_grid": grid,
        "J_plus_as_in_d38_code": [energy(u, "heading", +1.0) for u in grid],
        "J_minus_as_printed": [energy(u, "heading", -1.0) for u in grid],
    }
    Jm = out["sign_study"]["J_minus_as_printed"]
    out["sign_study"]["minus_is_monotone_increasing"] = all(b > a for a, b in zip(Jm[:-1], Jm[1:]))

    # (1)-(2) the constants
    for plant, lo, hi, old in (("heading", 0.30, 0.80, 0.499496), ("curvature", 0.15, 0.50, 0.306034)):
        rows = []
        for rtol, xend in ((1e-9, XEND), (1e-11, XEND), (1e-10, 45.0), (1e-10, 30.0)):
            u, J = argmin_J(plant, lo, hi, +1.0, rtol=rtol, xend=xend)
            rows.append({"rtol": rtol, "xend": xend, "u_star": u, "J": J})
        us = [r["u_star"] for r in rows]
        out[plant] = {
            "u_star_converged": rows[1]["u_star"],
            "J_min": rows[1]["J"],
            "spread_over_settings": max(us) - min(us),
            "settings": rows,
            "d38_euler_value": old,
            "shift_pct_vs_d38": 100.0 * (rows[1]["u_star"] / old - 1.0),
            "d38_scheme_refined": euler_reference(
                "heading_servo" if plant == "heading" else "curvature_command", lo, hi),
        }
        print("%-10s converged u* = %.6f   (d38 Euler %.6f, %+.2f%%)"
              % (plant, rows[1]["u_star"], old, out[plant]["shift_pct_vs_d38"]))

    out["reported"] = {"heading": round(out["heading"]["u_star_converged"], 4),
                       "curvature": round(out["curvature"]["u_star_converged"], 4)}
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "d131_corner_energy_converged_2026-09-23.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("sign as printed gives a monotone J:", out["sign_study"]["minus_is_monotone_increasing"])
    print("wrote", os.path.basename(dst))


if __name__ == "__main__":
    main()
