#!/usr/bin/env python3
"""
d123 -- file records for the remaining numbers quoted in DECISION_RECORD_2026-09-14.md
that were first obtained inline.

(a) P4: at every short-excursion birth of W_2 and W_3 in d119c, which excursion is the
    short one (index of min_j (f_j - b_j - theta)).
(b) The W_2 short-excursion border curve of d121 at the d119c border points: |curve| at
    the point over |curve| 1e-3 away in q, in 60-digit arithmetic.
(c) Sensitivity of the Krawczyk unknowns near u = 0.94 (n = 3): |dx/du|, |dx/dq| by
    central differences of the float solution, at a few points inside the existence
    interval (the reason the proof domain stops at u = 0.90).
(d) The aggregate table of the Krawczyk proof from the checkpoints d120d_rows_*.jsonl
    (read in shared mode while runs are still appending).

Usage:  python d123_record_gaps_2026-09-14.py
"""
import importlib.util
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def part_a(m):
    rows = [r for r in json.load(open(os.path.join(HERE, "d119c_birth_border_2d_2026-09-14.json"), encoding="utf-8"))
            if r["status"] == "ok" and r["border"] == "short_exc" and r["n"] > 1]
    counts = {}
    for r in rows:
        u, q, n, S = r["u"], r["q_B"], r["n"], r["S_B"]
        c = 1 - u - q - 2 * q * S
        ev = m.rec_c(c, u, q, n)[1]
        th = u / (2 * q)
        j = int(np.argmin([f - b - th for b, f in ev])) + 1
        key = "n=%d, short excursion j=%d" % (n, j)
        counts[key] = counts.get(key, 0) + 1
    return {"borders": len(rows), "counts": counts, "all_last": all(k.endswith("j=%s" % k.split("n=")[1].split(",")[0]) for k in counts)}


def part_b():
    import mpmath as mp
    import sympy as sp
    mp.mp.dps = 60
    u, q = sp.symbols("u q")
    curve = sp.sympify(json.load(open(os.path.join(HERE, "d121_sympy_border_polynomials_2026-09-14.json"), encoding="utf-8"))["W2_short_border_curve"])
    f = sp.lambdify((u, q), curve, "mpmath")
    rows = [r for r in json.load(open(os.path.join(HERE, "d119c_birth_border_2d_2026-09-14.json"), encoding="utf-8"))
            if r["status"] == "ok" and r["n"] == 2 and r["border"] == "short_exc"]
    out = []
    for r in rows:
        uv, qv = mp.mpf(r["u"]), mp.mpf(r["q_B"])
        out.append({"u": r["u"], "q_B": r["q_B"], "ratio": float(abs(f(uv, qv)) / abs(f(uv, qv + mp.mpf("1e-3"))))})
    return {"points": len(out), "max_ratio": max(x["ratio"] for x in out), "min_ratio": min(x["ratio"] for x in out), "rows": out}


def part_c(Kr, B):
    n = 3
    out = []
    for u in (0.942, 0.944):
        tr, _ = B.P0.track(u, n)
        qb, qd = tr[-1][0], tr[0][0] - 0.01
        for rel in (0.15, 0.3):
            qc = qb + rel * (qd - qb)
            Sg = min(tr, key=lambda p: abs(p[0] - qc))[1]

            def sol(uu, qq):
                Ss, _ = B.newton_float(np.array([uu]), np.array([qq]), n, np.array([Sg]))
                bs, fs, c = Kr.crossings_float(np.array([uu]), np.array([qq]), n, Ss)
                return Kr.pack(bs, fs, c)[0]
            h = 1e-6
            dxdu = (sol(u + h, qc) - sol(u - h, qc)) / (2 * h)
            dxdq = (sol(u, qc + h) - sol(u, qc - h)) / (2 * h)
            out.append({"u": u, "q": qc, "rel_position": rel, "max_abs_dx_du": float(np.abs(dxdu).max()),
                        "max_abs_dx_dq": float(np.abs(dxdq).max())})
    return out


def part_d():
    spec = [("n13", 1), ("n13", 2), ("n13", 3), ("n4a", 4), ("n4b", 4), ("n4c", 4), ("n5a", 5), ("n5b", 5), ("n5c", 5)]
    out = []
    for tag, n in spec:
        f = os.path.join(HERE, "d120d_rows_%s_n%d_2026-09-14.jsonl" % (tag, n))
        rows = []
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) < 10:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        g = [r for r in rows if not r.get("no_track")]
        pa = sum(r["proved_area"] for r in g)
        fa = sum(r["failed_area"] for r in g)
        bins = [int(x) for x in np.sum([r["failed_rel_bins_lt0.02_lt0.1_mid_le0.98_gt0.98"] for r in g], 0)] if g else []
        out.append({"tag": tag, "n": n, "rows": len(g), "u_min": min((r["u0"] for r in g), default=None),
                    "u_max": max((r["u1"] for r in g), default=None), "proved_area": pa, "failed_area": fa,
                    "failed_fraction": fa / (pa + fa) if pa else None, "failed_position_bins": bins,
                    "fold_unchecked": sum(r["fold_unchecked"] for r in g)})
    tot_p = sum(x["proved_area"] for x in out)
    tot_f = sum(x["failed_area"] for x in out)
    return {"at": time.strftime("%Y-%m-%d %H:%M"), "rows": out, "total_proved_area": tot_p, "total_failed_area": tot_f,
            "total_failed_fraction": tot_f / (tot_p + tot_f)}


def main():
    m = _load("d118", "d118_word_multiplier_closed_form_2026-09-14.py")
    Kr = _load("d120d", "d120d_krawczyk_stability_proof_2026-09-14.py")
    B = Kr.B
    out = {}
    for name, fn in (("a_short_excursion_is_last", lambda: part_a(m)), ("b_W2_border_curve_check", part_b),
                     ("c_sensitivity_u094_n3", lambda: part_c(Kr, B)), ("d_proof_aggregate", part_d)):
        out[name] = fn()
        print("%s: %s" % (name, json.dumps(out[name])[:300]), flush=True)
    json.dump(out, open(os.path.join(HERE, "d123_record_gaps_2026-09-14.json"), "w"), indent=1)
    print("-> d123_record_gaps_2026-09-14.json")


if __name__ == "__main__":
    main()
