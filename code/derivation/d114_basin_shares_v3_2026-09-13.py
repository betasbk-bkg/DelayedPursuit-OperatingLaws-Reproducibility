#!/usr/bin/env python3
"""
d114 -- the manuscript's basin shares, recomputed with integrator v3.

Sec. IV-F quotes, at kD/2 = 3.14 (k = 8) and u = 0.71, "the realised branch
holds 64-65% of the line against 6-7% for the other", from d29, which iterates
d28.orbit.  d113 found d28 wrong or undefined at a material fraction of section
states.  Here d29's own basin_map and d28's own scan are run unchanged, with
d28.orbit replaced by v3 (same return fields: z_next, period, word, fold,
min_gap), and the shares compared with the stored d29 results.  The flow's own
crossing state (the faithful flow, which uses no event integrator) is reused
from d29 as stored.

Usage:  python d114_basin_shares_v3_2026-09-13.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    t0 = time.time()
    v3 = _load("d113", "d113_integrator_v3_2026-09-13.py")
    d29 = _load("d29", "d29_basin_and_tongues_2026-08-29.py")
    stored = json.load(open(os.path.join(HERE, "d29_results_2026-08-29.json"), encoding="utf-8"))["basins"]

    def orbit_as_d28(z0, u, k, delta, th, max_ev=16):
        o = v3.orbit_v3(z0, u, k, delta, th)
        return None if o is None else {"z_next": o["z_next"], "period": o["period"], "word": o["word"],
                                       "fold": o["fold"], "min_gap": o["min_gap"]}

    d29.d28.orbit = orbit_as_d28        # d28.scan and d29.basin_map both look it up at call time
    out = []
    for u in (0.71, 0.90):
        fps = [f for f in d29.d28.scan(u, 8.0, D) if abs(f["mult"]) < 1.0]
        zs, lab = d29.basin_map(u, 8.0, D, fps)
        share = {j: float(np.mean(lab == j)) for j in range(len(fps))}
        old = [b for b in stored if abs(b["u"] - u) < 1e-9][0]
        zc_mean = old.get("z_flow_mean")
        dom = -1
        if zc_mean is not None and len(fps):
            idx = int(np.argmin(np.abs(zs - zc_mean)))
            dom = int(lab[idx])
        row = {"u": u, "branches": [{"word": f["word"], "mult": f["mult"], "z": f["z"], "share": share[j]}
                                    for j, f in enumerate(fps)],
               "unlabelled_share": float(np.mean(lab == -1)),
               "flow_mean_z_basin": dom,
               "stored": old}
        out.append(row)
        print("u=%.2f  v3: %s  unlabelled %.1f%%  | stored d28: %s"
              % (u, ["%s %+.3f %.1f%%" % (b["word"], b["mult"], 100 * b["share"]) for b in row["branches"]],
                 100 * row["unlabelled_share"],
                 ["%s %+.3f %.1f%%" % (b["word"], b["mult"], 100 * b["share"]) for b in old["branches"]]))
    json.dump(out, open(os.path.join(HERE, "d114_basin_shares_v3_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d114_basin_shares_v3_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
