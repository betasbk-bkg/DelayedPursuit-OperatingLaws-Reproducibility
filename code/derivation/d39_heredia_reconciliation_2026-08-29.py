#!/usr/bin/env python3
"""
d39 -- Reconciliation with Heredia & Ollero (2007), the closest prior art.

  G. Heredia and A. Ollero, "Stability of autonomous vehicle path tracking with
  pure delays in the control loop," Advanced Robotics 21(1-2):23-50, 2007.
  doi:10.1163/156855307779293715

THEIR MODEL (their Eq. 11), a car-like vehicle with a FIRST-ORDER STEERING
ACTUATOR of time constant T:

    x_dot     = -V sin(theta)
    theta_dot =  V gamma
    gamma_dot = -(1/T) ( gamma - gamma_R(t - tau) )

Their pure pursuit law (their Eq. 29) linearises (their Eq. 31) to
    phi_x = 2/L^2,  phi_theta = -2/L,  phi_gamma = 0.
They non-dimensionalise by the STEERING TIME CONSTANT: x -> x/(V T),
gamma -> V T gamma, tau -> tau/T (their Eq. 12), giving a THIRD-order
quasi-polynomial (their Eq. 47).  Their delay-free condition is their Eq. (32):

    L / (V T) > 1.

OUR MODEL (d36/d38) is the same control law with NO steering actuator -- the
commanded curvature is applied directly, which is exactly what Nav2 RPP does on
a differential-drive plant in a kinematic simulator.  We non-dimensionalise by
the lookahead transit time L/V, giving a SECOND-order quasi-polynomial and the
single number u_c = 0.520494.

THE RELATIONSHIP.  Carrying their steering lag through OUR non-dimensionalisation
with  sigma = s L / V,  u = V tau / L,  kappa = V T / L  gives one family:

    sigma^2 (1 + kappa sigma) + 2 (1 + sigma) exp(-sigma u) = 0          (*)

    kappa -> 0  : sigma^2 + 2(1+sigma) exp(-sigma u) = 0  -> our u_c = 0.520494
    u     -> 0  : kappa sigma^3 + sigma^2 + 2 sigma + 2 = 0, Routh -> kappa < 1
                  which is exactly their Eq. (32).

So our result is the ZERO-STEERING-LAG EDGE of their family, and their condition
is the ZERO-DELAY EDGE of the same family.  Their non-dimensionalisation divides
by T and therefore cannot express the kappa = 0 case at all.  This script
verifies both edges numerically and maps u_c(kappa) between them.

Run:    python d39_heredia_reconciliation_2026-08-29.py
Writes: d39_results_2026-08-29.json
"""
import json

import numpy as np
from scipy.optimize import brentq


def crossing(kappa):
    """
    Critical delay margin u_c(kappa) from (*): put sigma = j w and solve the
    real and imaginary parts simultaneously.

      real:  -w^2 + 2[cos(wu) + w sin(wu)] = 0
      imag:  kappa w^3 ... (kappa enters the imaginary part via sigma^3)

    Written out with sigma = jw:
      sigma^2 (1 + kappa sigma) = -w^2 - j kappa w^3
      2(1+sigma) e^{-sigma u}   = 2[(cos - j sin)(1 + jw)]
                                = 2[(cos + w sin) + j (w cos - sin)]
    so
      Re: -w^2 + 2( cos(wu) + w sin(wu) ) = 0
      Im: -kappa w^3 + 2( w cos(wu) - sin(wu) ) = 0
    """
    def eqs(w):
        # solve Re = 0 for u given w, then evaluate Im
        # Re: 2(cos(wu) + w sin(wu)) = w^2  ->  R(u) = 0
        def R(u):
            return 2 * (np.cos(w * u) + w * np.sin(w * u)) - w * w
        lo, hi = 1e-9, np.pi / w if w > 0 else 1.0
        try:
            if R(lo) * R(hi) > 0:
                return None, None
            u = brentq(R, lo, hi, xtol=1e-14)
        except ValueError:
            return None, None
        im = -kappa * w ** 3 + 2 * (w * np.cos(w * u) - np.sin(w * u))
        return u, im

    # At kappa = 0 the Re equation is TANGENT in u at the solution (it is a
    # critical point), so a 1-D bracket fails.  Solve the pair simultaneously.
    from scipy.optimize import fsolve

    def F(p):
        w, u = p
        return [-w * w + 2 * (np.cos(w * u) + w * np.sin(w * u)),
                -kappa * w ** 3 + 2 * (w * np.cos(w * u) - np.sin(w * u))]

    guess = crossing.last if getattr(crossing, "last", None) else (2.197368, 0.520494)
    sol, info, ier, _ = fsolve(F, guess, full_output=True)
    if ier != 1 or sol[0] <= 0 or sol[1] <= 0:
        return None, None
    crossing.last = (sol[0], sol[1])
    return float(sol[1]), float(sol[0])


def main():
    out = {"prior_art": "Heredia & Ollero, Advanced Robotics 21(1-2):23-50, 2007, "
                        "doi:10.1163/156855307779293715"}

    # --- edge 1: kappa -> 0 must reproduce our closed form -------------------
    w_cf = np.sqrt(2 + 2 * np.sqrt(2))
    u_cf = float(np.arctan(w_cf) / w_cf)
    u0, w0 = crossing(0.0)
    out["edge_zero_steering_lag"] = {
        "closed_form_u_c": u_cf, "closed_form_omega": float(w_cf),
        "numeric_u_c": u0, "numeric_omega": w0,
        "error_pct": 100 * (u0 / u_cf - 1),
    }
    print(f"kappa -> 0   : u_c numeric {u0:.6f}  vs our closed form {u_cf:.6f}"
          f"  ({100*(u0/u_cf-1):+.4f}%)")

    # --- edge 2: u -> 0 must reproduce their Eq. (32), kappa < 1 -------------
    # kappa sigma^3 + sigma^2 + 2 sigma + 2 = 0 -> Routh: 1*2 > kappa*2 -> kappa<1
    roots_stable = {}
    for kap in (0.90, 0.99, 1.01, 1.10):
        r = np.roots([kap, 1.0, 2.0, 2.0])
        roots_stable[f"kappa={kap}"] = bool(np.all(r.real < 0))
    out["edge_zero_delay"] = {
        "their_condition": "L/(V T) > 1  i.e. kappa = V T / L < 1",
        "routh_check": roots_stable,
    }
    print("u -> 0       : Routh on kappa*s^3+s^2+2s+2  ->",
          {k: ("stable" if v else "unstable") for k, v in roots_stable.items()})

    # --- the family in between ----------------------------------------------
    fam = []
    for kap in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        u, w = crossing(kap)
        fam.append({"kappa": kap, "u_c": u, "omega": w})
        print(f"  kappa={kap:4.2f}   u_c = {u:.5f}" if u else f"  kappa={kap:4.2f}   (none)")
    out["family"] = fam

    # --- where their own vehicles sit ---------------------------------------
    # HMMWV: tau = 0.55 s, T = 1.3 s (their Sec. 5)
    out["their_vehicles"] = {
        "HMMWV": {"tau_s": 0.55, "T_s": 1.3,
                  "note": "their non-dimensional delay tau/T = 0.423; kappa = V T / L "
                          "is order 1 for their speeds, i.e. far from the kappa = 0 edge"}
    }

    with open("d39_results_2026-08-29.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()


def asymptote_in_their_axes():
    """
    In THEIR variables the two axes are  L_nd = L/(V T)  and  tau_nd = tau/T.
    Since  u = V tau / L = tau_nd / L_nd  and  kappa = V T / L = 1 / L_nd,
    their stability boundary is   tau_nd = L_nd * u_c(1 / L_nd).
    For large L_nd (long lookahead / large delay) this is a straight RAY whose
    slope is our constant:   L_crit / tau  ->  1 / u_c = 1.9213.
    """
    rows = []
    for L_nd in (2.0, 5.0, 10.0, 50.0, 200.0, 1000.0):
        u, _ = crossing(1.0 / L_nd)
        if u is None:
            continue
        tau_nd = L_nd * u
        rows.append({"L_nd": L_nd, "tau_nd": tau_nd, "slope_L_over_tau": L_nd / tau_nd})
    return rows


if __name__ == "__main__":
    print()
    print("their-axes asymptote  (L_nd = L/(VT), tau_nd = tau/T):")
    print(f"{'L_nd':>8} {'tau_nd':>9} {'L/tau':>9}")
    rows = asymptote_in_their_axes()
    for r in rows:
        print(f"{r['L_nd']:8.1f} {r['tau_nd']:9.3f} {r['slope_L_over_tau']:9.4f}")
    print(f"  -> limit 1/u_c = {1/0.520494:.4f}")
    d = json.load(open("d39_results_2026-08-29.json"))
    d["their_axes_asymptote"] = rows
    d["asymptotic_slope_limit"] = 1 / 0.520494
    json.dump(d, open("d39_results_2026-08-29.json", "w"), indent=2)
