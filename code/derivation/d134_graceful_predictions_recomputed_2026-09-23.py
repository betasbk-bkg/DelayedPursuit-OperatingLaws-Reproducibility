#!/usr/bin/env python3
"""
d134 -- recompute every prediction in the graceful-controller table from the exact-law engine.

WHY THIS EXISTS.  The five rows of that table pair a prediction with a measurement.  Four of the five
settings were registered before their sweeps; the fifth, the plugin's declared gains (k_phi = 2,
k_delta = 1), was added later, when the plugin's source corrected our reading of its defaults.  A
reader is entitled to ask whether that fifth prediction was fitted to the measurement it is compared
with.  It could not have been, and this script is how that is checked rather than asserted:

  * the prediction comes from d99's exact-law engine, which was written, accepted against RPP's known
    constant, and used for the registered rows BEFORE any graceful sweep ran;
  * its inputs are configuration only -- the controller's gains, the configured look-ahead and the
    configured delay budget.  No quantity measured in the speed sweep enters;
  * therefore anyone can recompute every prediction, and this script does exactly that and compares
    each one with the value on record.

If a row here disagrees with its stored prediction, the table is wrong and this script says so.

Run:    python d134_graceful_predictions_recomputed_2026-09-23.py
Writes: d134_graceful_predictions_recomputed_2026-09-23.json
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D99 = os.path.join(HERE, "d99_graceful_nonlinear_2026-09-08.py")

L_CONFIGURED = 0.600          # the look-ahead the controller was configured with
V_GRID = np.linspace(0.2, 1.8, 17)

# (k_phi, k_delta, tau_tot used for that row's block, the prediction on record)
ROWS = [(0.5, 2.0, 0.1325, 0.1118, "registered in advance"),
        (2.0, 1.0, 0.13, 0.1243, "the plugin's declared gains, added later"),
        (1.0, 2.0, 0.1325, 0.0778, "registered in advance"),
        (2.0, 2.0, 0.1325, None, "registered in advance"),
        (3.0, 2.0, 0.1325, None, "registered in advance")]


def engine():
    spec = importlib.util.spec_from_file_location("d99", D99)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    d99 = engine()
    path = d99.Square()
    out = {"look_ahead_configured": L_CONFIGURED, "v_grid": V_GRID.tolist(), "rows": [], "all_match": True}
    print("=" * 84)
    print("d134 -- the graceful predictions, recomputed from the exact-law engine (configuration only)")
    print("=" * 84)
    print("  %6s %8s %7s %11s %11s %9s   %s" % ("k_phi", "k_delta", "tau_tot", "recomputed",
                                                "on record", "match", "provenance"))
    for k_phi, k_delta, tau, stored, note in ROWS:
        rs = [d99.simulate(path, float(v), tau, "graceful", k_phi=k_phi, k_delta=k_delta,
                           look=L_CONFIGURED) for v in V_GRID]
        v_star, how, n = d99.locate(V_GRID, rs)
        interior = v_star is not None and how not in ("EDGE_LOW", "EDGE_HIGH")
        u = (v_star * tau / L_CONFIGURED) if interior else None
        if stored is None:
            ok = not interior
            shown_r, shown_s = ("none" if not interior else "%.4f" % u), "none"
        else:
            ok = interior and abs(u - stored) < 5e-4
            shown_r, shown_s = ("%.4f" % u if interior else "none"), "%.4f" % stored
        out["all_match"] = out["all_match"] and bool(ok)
        out["rows"].append({"k_phi": k_phi, "k_delta": k_delta, "tau_tot": tau, "how": how,
                            "u_recomputed": u, "u_on_record": stored, "match": bool(ok),
                            "provenance": note})
        print("  %6.1f %8.1f %7.4f %11s %11s %9s   %s"
              % (k_phi, k_delta, tau, shown_r, shown_s, "yes" if ok else "NO", note))
    print()
    print("  The engine takes the gains, the configured look-ahead and the delay budget, and nothing")
    print("  measured in the speed sweep.  Every prediction above is therefore independent of the")
    print("  measurement it is compared against, whether or not that row was registered in advance.")
    print()
    print("  all rows match the record:", out["all_match"])
    with open(os.path.join(HERE, "d134_graceful_predictions_recomputed_2026-09-23.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("-> d134_graceful_predictions_recomputed_2026-09-23.json")
    return 0 if out["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
