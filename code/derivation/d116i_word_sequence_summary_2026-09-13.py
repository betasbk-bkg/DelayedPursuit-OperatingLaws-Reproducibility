#!/usr/bin/env python3
"""
d116i -- the numbers the manuscript states about the word sequence W_n and about
the initial-condition dependence of the switch, in one JSON.

  word intervals and the condition at each end            d116b
  general death condition a = 2 sqrt(q)(1 - sqrt(q))       d116d  (15 deaths, n = 1..5)
  W_1 death quartic vs the zero of the grazing margin      recomputed and saved here
  W_n fixed points confirmed by integrator v3; census      d116
  first-section-state prediction; initial-condition splits d117

Usage:  python d116i_word_sequence_summary_2026-09-13.py
"""
import importlib.util
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))


def quartic(q, u):
    s, v = math.sqrt(q), math.sqrt(u)
    return (q + 2 * s * (1 - v) - (1 - u)) ** 2 - 4 * q * (q - 2 * s * v)


def main():
    out = {}
    bd = J("d116b_word_borders_2026-09-13.json")
    out["birth_by"] = dict(Counter(r["birth_by"] for r in bd if r.get("birth_by")))
    out["death_by"] = dict(Counter(r["death_by"] for r in bd if r.get("death_by")))
    out["border_u_range"] = [min(r["u"] for r in bd), max(r["u"] for r in bd)]
    out["intervals"] = {"%.2f" % u: [[r["n"], r.get("birth"), r.get("death")] for r in bd if abs(r["u"] - u) < 1e-9 and r["exists"]]
                        for u in (0.3, 0.5, 0.7)}
    dd = [r for r in J("d116d_general_death_condition_2026-09-13.json") if r.get("death")]
    out["general_death_n"] = len(dd)
    out["general_death_max_err"] = max(max(r["err_a"], r["err_S"]) for r in dd)
    out["general_death_orders"] = sorted({r["n"] for r in dd})
    # W_1 death quartic vs the grazing-margin zeros of d116d
    errs = []
    for r in dd:
        if r["n"] != 1:
            continue
        u, qd = r["u"], r["death"]
        root = brentq(quartic, qd - 0.05, qd + 0.05, args=(u,), xtol=1e-14)
        errs.append(abs(root - qd))
    out["W1_quartic_n"] = len(errs)
    out["W1_quartic_max_err"] = max(errs)
    s116 = J("d116_word_sequence_closure_2026-09-13.json")
    v3rows = s116["v3"]
    out["v3_confirmed"] = sum(1 for r in v3rows if r["ok"])
    out["v3_checked"] = len(v3rows)
    ms = [r["mult"] for r in v3rows if r["mult"] is not None]
    out["v3_mult_range"] = [min(ms), max(ms)]
    out["v3_orders"] = sorted({r["n"] for r in v3rows})
    cc = s116["census_compare"]
    out["census_words_n"] = len(cc)
    out["census_words_with_closed_form"] = sum(1 for r in cc if r["exists"])
    ic = J("d117_switch_initial_condition_2026-09-13.json")
    out["prediction_matches"] = sum(1 for r in ic["a"] if r["match"])
    out["prediction_n"] = len(ic["a"])
    out["prediction_mismatch_at"] = [(r["u"], r["q"]) for r in ic["a"] if not r["match"]]
    out["ic_inside"] = [r for r in ic["b"] if r["where"] == "inside band"]
    out["ic_outside_single_word"] = all(len(r["words"]) == 1 for r in ic["b"] if r["where"] == "outside band")
    out["ic_n_initial"] = sum(ic["b"][0]["words"].values())
    json.dump(out, open(os.path.join(HERE, "d116i_word_sequence_summary_2026-09-13.json"), "w"), indent=1, default=float)
    for k, v in out.items():
        print("%-32s %s" % (k, v))


if __name__ == "__main__":
    main()
