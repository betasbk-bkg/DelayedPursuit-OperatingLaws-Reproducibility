#!/usr/bin/env python3
"""
d119c -- birth type of W_n from the border solved as a two-dimensional root problem.

UNKNOWNS (q, S).  E(q, S) = F_n(c; u, q) - S with c = 1 - u - q - 2 q S (the word's
fixed-point residual, d116 recursion), and s_k(q, S) the slack of one ordering
constraint k in {short_exc (an excursion as short as theta), end (last window reaching
the period end)}.  The border is E = 0, s_k = 0, solved by a Newton-type solver from a
verified fixed point just above the d116b birth.

TYPE.  Along the branch E = 0, dq/dS = -E_S / E_q.  The physical side is where the
slack grows: ds/dS|branch = s_S + s_q dq/dS.  If q grows moving into the physical
side, the word is born at the border (border type), with multiplier P_B.  Otherwise
q has a minimum inside the physical side: a smooth fold, E = 0 and E_S = 0 (then
P' = 1), solved from the border point; the stable member leaves the fold on one side
and the unstable one runs to the border.

CHECKS.  P' at the fold must be 1; P_B < 1 must coincide with border type; the born
stable fixed point is run through the validated integrator v3 (word and multiplier).

Usage:  python d119c_birth_border_2d_2026-09-14.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d118", os.path.join(HERE, "d118_word_multiplier_closed_form_2026-09-14.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)
w = m.w


def E(q, S, u, n):
    r = w.recursion(S, u, q, n)
    return np.nan if r is None else r[0] - S


def slacks(q, S, u, n):
    c = 1 - u - q - 2 * q * S
    r = m.rec_c(c, u, q, n)
    if r is None:
        return None
    th = u / (2 * q)
    ev = r[1]
    out = {"short_exc": min(f - b for b, f in ev) - th, "end": 1 - ev[-1][1] - th, "b1": ev[0][0] - th}
    if n > 1:
        out["gap"] = min(ev[j + 1][0] - ev[j][1] for j in range(n - 1)) - th
    return out


def grad(fun, q, S, h=1e-7):
    return ((fun(q + h, S) - fun(q - h, S)) / (2 * h), (fun(q, S + h) - fun(q, S - h)) / (2 * h))


def solve_border(u, n, k, q0, S0):
    def sys_(x):
        q, S = x
        sl = slacks(q, S, u, n)
        return [E(q, S, u, n), (sl[k] if sl else np.nan)]
    x, info, ier, _ = fsolve(sys_, [q0, S0], full_output=True, xtol=1e-13)
    if ier != 1 or not np.all(np.isfinite(info["fvec"])) or max(abs(v) for v in info["fvec"]) > 1e-10:
        return None
    return float(x[0]), float(x[1])


def main():
    t0 = time.time()
    bd = json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))
    out = []
    for n in (1, 2, 3):
        rows = sorted((r["u"], r["birth"]) for r in bd if r["n"] == n and r["exists"] and r["birth"] is not None)
        us, qbs = [x[0] for x in rows], [x[1] for x in rows]
        for u in np.round(np.arange(0.20, 0.951, 0.025), 3):
            u = float(u)
            qg = float(np.interp(u, us, qbs))
            start = None
            for dq in (0.003, 0.01, 0.03, 0.06):
                sols = [s for s in w.solve_word(u, qg + dq, n) if s["pattern_ok"]]
                if sols:
                    start = (qg + dq, sols)
                    break
            if start is None:
                out.append({"u": u, "n": n, "status": "no start"})
                print("n=%d u=%.3f no start" % (n, u))
                continue
            cands = []
            for sol in start[1]:
                for k in ("short_exc", "end"):
                    r = solve_border(u, n, k, start[0], sol["S_n"])
                    if r is None:
                        continue
                    qB, SB = r
                    sl = slacks(qB, SB, u, n)
                    others = [v for kk, v in sl.items() if kk != k]
                    if min(others) < -1e-9:
                        continue
                    cands.append((abs(qB - qg), k, qB, SB))
            if not cands:
                out.append({"u": u, "n": n, "status": "no border solution"})
                print("n=%d u=%.3f no border solution" % (n, u))
                continue
            _, k, qB, SB = min(cands)
            Eq, ES = grad(lambda q, S: E(q, S, u, n), qB, SB)
            sq_, sS_ = grad(lambda q, S: slacks(q, S, u, n)[k], qB, SB)
            dqdS = -ES / Eq
            dsdS = sS_ + sq_ * dqdS
            inward = 1.0 if dsdS > 0 else -1.0
            dq_in = dqdS * inward
            PB = m.multiplier(u, qB, n, {"S_n": SB})[0]
            row = {"u": u, "n": n, "status": "ok", "border": k, "q_B": qB, "S_B": SB, "dq_dS": dqdS,
                   "dq_inward": dq_in, "P_B": PB, "type": "border" if dq_in > 0 else "fold"}
            if dq_in <= 0:
                def fold_sys(x):
                    q, S = x
                    return [E(q, S, u, n), grad(lambda qq, SS: E(qq, SS, u, n), q, S)[1]]
                xf, info, ier, _ = fsolve(fold_sys, [qB - 1e-4, SB + inward * 1e-4], full_output=True, xtol=1e-12)
                if ier == 1:
                    qF, SF = float(xf[0]), float(xf[1])
                    slF = slacks(qF, SF, u, n)
                    row.update({"q_F": qF, "S_F": SF, "fold_width": qB - qF,
                                "P_F": m.multiplier(u, qF, n, {"S_n": SF})[0],
                                "fold_slack_min": min(slF.values()) if slF else None})
            # v3 check of the stable member a little inside the existence interval
            qin = (row.get("q_F", qB)) + 0.01
            st = [s for s in w.solve_word(u, qin, n) if s["pattern_ok"]]
            if st:
                best = min(st, key=lambda s: abs(m.multiplier(u, qin, n, s)[0]))
                ok, mult, wd = w.v3_check(u, qin, n, best)
                row.update({"v3_ok_inside": ok, "v3_mult_inside": mult,
                            "closed_mult_inside": m.multiplier(u, qin, n, best)[0]})
            out.append(row)
            extra = (" | fold q_F=%.6f width %.2e P_F=%.6f fold-slack %.1e"
                     % (row["q_F"], row["fold_width"], row["P_F"], row["fold_slack_min"])) if "q_F" in row else ""
            print("n=%d u=%.3f %-6s at %-9s q_B=%.6f P_B=%.4f dq_in=%+.3f%s"
                  % (n, u, row["type"], k, qB, PB, dq_in, extra))
    ok = [r for r in out if r["status"] == "ok"]
    cons = all((r["P_B"] < 1) == (r["type"] == "border") for r in ok)
    folds = [r for r in ok if "P_F" in r]
    print("\n%d borders solved; P_B < 1 <=> border type: %s; folds with |P_F - 1| < 1e-6: %d of %d; v3 inside ok: %d of %d"
          % (len(ok), cons, sum(abs(r["P_F"] - 1) < 1e-6 for r in folds), len(folds),
             sum(1 for r in ok if r.get("v3_ok_inside")), sum(1 for r in ok if "v3_ok_inside" in r)))
    json.dump(out, open(os.path.join(HERE, "d119c_birth_border_2d_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d119c_birth_border_2d_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
