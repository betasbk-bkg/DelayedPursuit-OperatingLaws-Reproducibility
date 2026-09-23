#!/usr/bin/env python3
"""
d38 -- The corner-transit energy minimum, derived for BOTH plant classes.

CONTEXT
  Manuscript §VII derives that a speed-proportional-gain plant (Ackermann; and,
  as d36 shows, Nav2 RPP) has an induced loop lag L/(2v) that contributes
  exactly L/2 to the compensation budget at any speed, self-cancelling the
  mechanism so that NO interior optimum exists.  d36 confirms exactly that on
  the third-party controller -- on a CIRCLE.

  But d36 also finds that on a POLYGON the RPP optimum comes back, at
  u* ~ 0.308 (alpha -> 0 limit).  The self-cancellation argument is about the
  quasi-static bias, which is the whole story on a circle and is NOT the whole
  story at a corner: the corner leaves a transient whose energy has a genuine
  minimum in u.  This script derives that minimum for both plants.

METHOD
  Linearise about the path.  x = s/L (arc length in lookahead units),
  e_hat = e/L, u = v*tau_tot/L, corner exterior angle alpha = 1 (linear, so the
  response scales with alpha and drops out of argmin).

  Carrot lateral offset seen at x, for a path of curvature kappa_p:
      y/L = -e_hat - psi + g(x),
      g(x) = int_0^1 (1-sigma) * L*kappa_p(x+sigma) dsigma
  For a corner at x_c, L*kappa_p = alpha*delta(x-x_c), so
      g(x) = 1 - (x_c - x)   for x_c-1 < x < x_c,  else 0
  i.e. the carrot sees the corner one lookahead early -- this is the preview.

  (A) HEADING-SERVO plant   theta = kernel * theta_cmd,  theta_cmd aims at the carrot
          e_hat'(x) = -e_hat(x-u) + g(x-u) - [H(x-u-x_c) - H(x-x_c)]
      free equation  e' + e(x-u) = 0  ->  lam + exp(-lam u) = 0
      stability boundary u = pi/2                    (matches the published limit)

  (B) CURVATURE-COMMAND plant  omega = v*kappa,  kappa = 2y/d^2   (Nav2 RPP)
          e_hat''(x) = 2[-e_hat(x-u) - e_hat'(x-u) + g(x-u)] - delta(x-x_c)
      free equation  e'' + 2e'(x-u) + 2e(x-u) = 0  ->  lam^2 + 2exp(-lam u)(lam+1) = 0
      stability boundary omega^4-4omega^2-4=0, u_c = arctan(omega)/omega = 0.520494

  Lap RMSE^2 = n_corners * L^3 * J(u) / P  with  J(u) = int e_hat^2 dx, so the
  speed enters ONLY through u and  u* = argmin J(u)  is a pure number.

PREDICTION TESTED
  (A) argmin J = 1/2        (Theorem 1's constant for smooth/small-angle courses)
  (B) argmin J = 0.308      (d36's alpha -> 0 measurement on Nav2 RPP)

Run:    python d38_corner_energy_two_plants_2026-08-29.py
Writes: d38_results_2026-08-29.json
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar


def g_preview(x, xc):
    """Carrot's view of the corner: nonzero only one lookahead before it."""
    return (1.0 - (xc - x)) if (xc - 1.0) < x < xc else 0.0


def transit_energy(u, plant, xc=4.0, xend=60.0, h=1.0e-3):
    """
    Integrate the linearised corner transit and return J(u) = int e_hat^2 dx.
    Method of steps; the corner's delta is applied as an exact jump.
    """
    n = int(xend / h)
    nd = int(round(u / h))
    e = np.zeros(n + 1)
    ep = np.zeros(n + 1)          # only used by the curvature-command plant
    ic = int(round(xc / h))

    for i in range(n):
        x = i * h
        j = i - nd
        ed = e[j] if j >= 0 else 0.0
        epd = ep[j] if j >= 0 else 0.0
        gd = g_preview(x - u, xc) if x - u > 0 else 0.0

        if plant == "heading_servo":
            # psi(x) = -e(x-u) + g(x-u) + [psi_p(x-u) - psi_p(x)]
            #        = -e(x-u) + g(x-u) + [H(x-u-xc) - H(x-xc)]
            step = (1.0 if (x - u) > xc else 0.0) - (1.0 if x > xc else 0.0)
            e[i + 1] = e[i] + h * (-ed + gd + step)
        else:
            # e'' = 2[-e(x-u) - e'(x-u) + g(x-u)];  delta at xc kicks psi = e'
            epp = 2.0 * (-ed - epd + gd)
            ep[i + 1] = ep[i] + h * epp
            if i == ic:
                ep[i + 1] -= 1.0          # heading error jumps by -alpha
            e[i + 1] = e[i] + h * ep[i + 1]

        if not np.isfinite(e[i + 1]) or abs(e[i + 1]) > 1e8:
            return np.inf
    return float(np.trapezoid(e * e, dx=h))


def argmin_J(plant, lo, hi):
    r = minimize_scalar(lambda u: transit_energy(u, plant),
                        bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x), float(r.fun)


def main():
    out = {}

    # ---- sanity: the two free-equation stability limits ---------------------
    w = np.sqrt(2 + 2 * np.sqrt(2))
    out["stability_limits"] = {
        "heading_servo_u": float(np.pi / 2),
        "curvature_command_u": float(np.arctan(w) / w),
    }

    for plant, lo, hi, target, label in [
        ("heading_servo", 0.05, 1.45, 0.5, "Theorem 1 constant 1/2"),
        ("curvature_command", 0.05, 0.50, 0.308, "d36 alpha->0 measurement 0.308"),
    ]:
        u_star, J_min = argmin_J(plant, lo, hi)
        # curve for the record
        us = np.linspace(lo, hi, 25)
        curve = [transit_energy(float(x), plant) for x in us]
        out[plant] = {
            "u_star_derived": u_star,
            "J_min": J_min,
            "target": target,
            "target_label": label,
            "error_pct": 100.0 * (u_star / target - 1.0),
            "u_grid": us.tolist(),
            "J_grid": [float(c) for c in curve],
        }
        print(f"{plant:20s}  argmin J(u) = {u_star:.4f}   "
              f"vs {label} -> {100 * (u_star / target - 1):+.2f}%")

    with open("d38_results_2026-08-29.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
