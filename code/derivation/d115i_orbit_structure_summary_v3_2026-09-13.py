#!/usr/bin/env python3
"""
d115i -- the numbers the manuscript states about the quantized loop's orbit
structure, from the v3 recomputation (d115), the basin shares (d114, d114b), the
integrator validation (d113b) and the flow's switch (d30), in one JSON.

Also measures the one quantity d115 left undefined: the slope of the section map
on the short-excursion side of the (+,-,+) birth border (d115 could not fit an
exponent there, which a locally constant map would explain).

Usage:  python d115i_orbit_structure_summary_v3_2026-09-13.py
"""
import importlib.util
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    v3 = _load("d113", "d113_integrator_v3_2026-09-13.py")
    S = J("d115_orbit_structure_v3_2026-09-13.json")
    out = {}

    # integrator validation
    val = J("d113b_v3_validation_domain_2026-09-13.json")
    rows = val["rows"]
    out["v3_validation_n"] = sum(1 for r in rows if r.get("brute") is not None and r.get("v3") is not None)
    out["v3_validation_still_bad"] = val["v3_still_bad"]

    # (+) orbit
    out["plus_mult_n"] = S["plus"]["mult_n"]
    out["plus_mult_max_err"] = S["plus"]["mult_max_err"]
    ends = [e for e in S["plus"]["ends"] if e.get("q_end") is not None]
    out["plus_end_n"] = len(ends)
    out["plus_end_max_err"] = max(abs(e["err"]) for e in ends)

    # (+,-,+) band
    tot = sum(v["stats"].get("fixed point", 0) for v in S["pmp_band"].values())
    stable = sum(v["stats"].get("stable", 0) for v in S["pmp_band"].values())
    unstable = sum(v["stats"].get("UNSTABLE", 0) for v in S["pmp_band"].values())
    not1d = sum(v["stats"].get("section not 1-D", 0) for v in S["pmp_band"].values())
    main_piece = sum(v["stats"].get("piece gap>th", 0) for v in S["pmp_band"].values())
    closed_n = sum(v["closed_n"] for v in S["pmp_band"].values())
    closed_max = max(v["closed_max_err"] for v in S["pmp_band"].values() if v["closed_max_err"] is not None)
    out.update({"pmp_fixed_points": tot, "pmp_stable": stable, "pmp_unstable": unstable,
                "pmp_section_not_1d": not1d, "pmp_main_piece": main_piece,
                "pmp_closed_n": closed_n, "pmp_closed_max_err": closed_max,
                "pmp_u_values": sorted(float(k) for k in S["pmp_band"].keys())})

    # census
    cen = S["census"]
    low = [c for c in cen if c["u"] <= 0.71 + 1e-9]
    out["census_low_u_all_fixed_points"] = all(set(c["census"].keys()) == {"p1"} for c in low)
    words = set()
    for c in cen:
        for w in c["words"]:
            words.add(w.split(":", 1)[1])
    out["census_words"] = sorted(words, key=len)
    hi = [c for c in cen if abs(c["u"] - 0.9) < 1e-9]
    frac = [c["census"].get("leaves 1-D section", 0) / 60.0 for c in hi]
    out["census_u09_leave_frac"] = [min(frac), max(frac)]
    out["census_any_aperiodic_or_cycle"] = any(k in ("aperiodic",) or (k.startswith("p") and k != "p1")
                                               for c in cen for k in c["census"])
    out["census_q_range"] = [min(c["q"] for c in cen), max(c["q"] for c in cen)]

    # basin shares at q = 3.14 (k = 8)
    b = J("d114_basin_shares_v3_2026-09-13.json")
    bb = J("d114b_flow_basin_membership_v3_2026-09-13.json")
    for r in b:
        if abs(r["u"] - 0.71) < 1e-9:
            sh = {x["word"]: x["share"] for x in r["branches"]}
            out["basin_u071_pmp"] = sh["-++"]
            out["basin_u071_plus"] = sh["+"]
    for r in bb:
        if abs(r["u"] - 0.71) < 1e-9:
            out["flow_crossings_in_pmp_basin"] = r["basin_votes"].get("-++", 0)
            out["flow_crossings_n"] = sum(r["basin_votes"].values())

    # basin crossing vs the flow's switch
    bc = []
    for r in S["basin_crossing"]:
        bc.append({"u": r["u"], "basin_crossing": r["basin_crossing"], "switch": r["switch"],
                   "diff": (r["basin_crossing"] - r["switch"]) if r["basin_crossing"] else None})
    out["basin_crossing"] = bc

    # borders
    bd = S["borders"]
    b1 = [x for x in bd if x["border"] == 1 and x["right"] is not None and x["right"] > 0.1]
    out["border1_inner_exp"] = [min(x["left"] for x in b1), max(x["left"] for x in b1)]
    out["border1_outer_exp"] = [min(x["right"] for x in b1), max(x["right"] for x in b1)]
    out["border1_outer_exp_u"] = [x["u"] for x in b1]
    b2 = [x for x in bd if x["border"] == 2]
    out["border2_jump_max"] = max(x["jump_1e-10"] for x in b2)
    out["border2_right_exp"] = [min(x["right"] for x in b2), max(x["right"] for x in b2)]
    # the undefined side: slope measured directly over finite steps
    slopes = []
    for x in b2:
        u, q, zs = x["u"], x["q"], x["z_s"]
        k = 2 * q / D
        P = lambda z: v3.orbit_v3(z, u, k, D, u / k)["z_next"]
        s_left = [(P(zs - h) - P(zs - 2 * h)) / h for h in (1e-6, 1e-5, 1e-4)]
        s_right = [(P(zs + 2 * h) - P(zs + h)) / h for h in (1e-6, 1e-5, 1e-4)]
        slopes.append({"u": u, "left": s_left, "right": s_right})
    out["border2_slopes"] = slopes

    json.dump(out, open(os.path.join(HERE, "d115i_orbit_structure_summary_v3_2026-09-13.json"), "w"), indent=1, default=float)
    for kk, vv in out.items():
        print("%-32s %s" % (kk, vv))


if __name__ == "__main__":
    main()
