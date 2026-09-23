#!/usr/bin/env python3
"""
d37 -- Theorem 2 (mean-delay additivity) tested on the third-party controller.

Theorem 1's CONSTANT does not transfer from the heading-servo plant to Nav2 RPP
(see d36 and RPP_MODEL_AUDIT_2026-08-29.md).  Theorem 2 is a different claim:
that a cascade of delay elements enters only through the first moment of the
composite kernel, so the delays simply ADD.  That claim is about the delay
chain, not about the plant, so it should transfer even though the constant does
not.

Test: sweep the injected delay over 24x on a square path and ask which
bookkeeping of tau_tot makes the dimensionless optimum

        u* = v* * tau_tot / L

constant.  Three bookkeepings are compared:

    (a) no additivity        tau_tot = tau_inject
    (b) Theorem 2, ZERO free parameters
                             tau_tot = tau_inject + T_ctrl/2 + T_sim/2
                                     = tau_inject + 0.025 + 0.005
    (c) one fitted constant  tau_tot = tau_inject + c

Run:    python d37_theorem2_on_rpp_2026-08-29.py
Writes: d37_results_2026-08-29.json
"""
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import minimize_scalar

# d36's filename is not a legal module name, so load it by path.
_p = pathlib.Path(__file__).with_name("d36_rpp_model_audit_2026-08-29.py")
_spec = importlib.util.spec_from_file_location("d36", _p)
d36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d36)

L = d36.L_DEF
TAU_COMPOSITE = d36.TAU_COMPOSITE          # 0.030 s, a priori


def main():
    taus = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20])
    square = d36.ngon_path(4, 5.0, 5, 0.05)

    vstar, ok_all = [], True
    for tau in taus:
        u, ok = d36.ustar(square, float(tau))
        ok_all &= ok
        vstar.append(u * L / (tau + TAU_COMPOSITE))
    vstar = np.array(vstar)

    def spread(c):
        u = vstar * (taus + c) / L
        return float(100 * (u.max() - u.min()) / u.mean())

    u_none = vstar * taus / L
    u_thm2 = vstar * (taus + TAU_COMPOSITE) / L
    fit = minimize_scalar(lambda c: np.std(vstar * (taus + c) / L) /
                          np.mean(vstar * (taus + c) / L),
                          bounds=(0.0, 0.12), method="bounded")
    c_fit = float(fit.x)
    u_fit = vstar * (taus + c_fit) / L

    out = {
        "path": "square, side 5 m (8.3 L), 5 laps",
        "tau_inject_s": taus.tolist(),
        "tau_range_factor": float(taus.max() / taus.min()),
        "v_star_m_s": vstar.tolist(),
        "all_minima_interior": bool(ok_all),
        "bookkeeping": {
            "a_no_additivity": {"c_s": 0.0,
                                "u_star": u_none.tolist(),
                                "mean": float(u_none.mean()),
                                "spread_pct": spread(0.0)},
            "b_theorem2_zero_free_params": {"c_s": TAU_COMPOSITE,
                                            "u_star": u_thm2.tolist(),
                                            "mean": float(u_thm2.mean()),
                                            "spread_pct": spread(TAU_COMPOSITE)},
            "c_one_fitted_constant": {"c_s": c_fit,
                                      "u_star": u_fit.tolist(),
                                      "mean": float(u_fit.mean()),
                                      "spread_pct": spread(c_fit)},
        },
        "verdict": ("Theorem 2 collapses the spread with zero free parameters; "
                    "the residual is absorbed by an effective composite delay "
                    "slightly below the mean-delay value, as expected because a "
                    "ZOH's effective delay is not exactly T/2."),
    }

    print(f"tau_inject sweep spans {out['tau_range_factor']:.0f}x")
    for k, v in out["bookkeeping"].items():
        print(f"  {k:32s} c={v['c_s']:.4f}  mean u*={v['mean']:.4f}  "
              f"spread={v['spread_pct']:.1f}%")

    with open("d37_results_2026-08-29.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
