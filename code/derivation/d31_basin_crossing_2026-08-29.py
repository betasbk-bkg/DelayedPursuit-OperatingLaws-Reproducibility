# -*- coding: utf-8 -*-
"""
Derivation note -- D31: the transition curve as a basin crossing.

D30 established two things: the one-dimensional section stays valid across the
transition (event gaps remain wider than theta_d), and the measured boundary,
refined to 0.05 in k*Delta/2, is

    u = 0.30 -> 2.12,  0.50 -> 2.08,  0.71 -> 2.48,  0.90 -> 2.83

against Eq. (11.5)'s existence criterion 2.40 / 2.91 / 3.40 / 3.80.  But D30's
basin step found the "+-+" branch only sporadically, because it located branches
by bracketing sign changes of P(z) - z, and narrow basins slip between grid
points.

Fix: do not bracket at all.  Stable branches ARE the attractors of P, so iterate
P from a dense set of starting z and cluster where the iterates land.  That
yields the stable branches and their basin shares in one pass, which is exactly
what the crossing needs.

Transition predicted as the k*Delta/2 where share("+-+") overtakes share("+").

Outputs: d31_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
D = math.radians(45.0)
DEG = math.degrees(1.0)
US = (0.30, 0.50, 0.71, 0.90)


def attractors(u, k, delta, n=220, iters=160, tol=1e-9):
    """iterate P from a dense set of z and cluster the limits."""
    th = u/k
    zs = np.linspace(-0.4*delta, 1.8*delta, n)
    found = []          # list of dicts: z, word, m_s, count
    lost = 0
    for z0 in zs:
        z = z0
        last = None
        ok = False
        for _ in range(iters):
            o = d28.orbit(z, u, k, delta, th)
            if o is None:
                break
            if last is not None and abs(o["z_next"] - z) < tol:
                ok = True
                last = o
                break
            last = o
            z = o["z_next"]
        if not ok or last is None or last["fold"] is None:
            lost += 1
            continue
        w = last["fold"]
        while w < -delta/2:
            w += delta
        while w > delta/2:
            w -= delta
        for f in found:
            if abs(f["z"] - z) < 1e-4*delta and f["word"] == last["word"]:
                f["count"] += 1
                break
        else:
            found.append(dict(z=float(z), word=last["word"], m_s=float(w*DEG),
                              count=1, period_ratio=last["period"]/delta))
    for f in found:
        f["share"] = f["count"]/n
    return found, lost/n


def shares(found):
    sp = sum(f["share"] for f in found if f["word"] == "+")
    sm = sum(f["share"] for f in found if f["word"] in ("-++", "+-+"))
    return sp, sm


if __name__ == "__main__":
    out = {"scan": [], "crossing": []}
    MEAS = {0.30: 2.12, 0.50: 2.08, 0.71: 2.48, 0.90: 2.83}
    print("attractors of P and their basin shares")
    for u in US:
        print(f"\n  u = {u:.2f}   (measured transition at k*Delta/2 = {MEAS[u]})")
        print(f"    {'kD/2':>6} {'share(+)':>9} {'share(+-+)':>11} {'unresolved':>11} "
              f"{'winner':>8}")
        seq = []
        for kd2 in np.arange(1.90, 3.05, 0.05):
            k = 2*kd2/D
            found, lost = attractors(u, k, D)
            sp, sm = shares(found)
            win = "+" if sp > sm else ("+-+" if sm > sp else "-")
            seq.append(dict(kD2=float(kd2), sp=sp, sm=sm, lost=lost, win=win))
            out["scan"].append(dict(u=u, **seq[-1]))
            print(f"    {kd2:6.2f} {sp:9.3f} {sm:11.3f} {lost:11.3f} {win:>8}")
        cross = None
        for a, b in zip(seq[:-1], seq[1:]):
            if a["win"] == "+" and b["win"] == "+-+":
                da, db = a["sp"]-a["sm"], b["sp"]-b["sm"]
                cross = a["kD2"] + (b["kD2"]-a["kD2"])*da/(da-db) if da != db else b["kD2"]
                break
        lo, hi = 1.0, 12.0
        for _ in range(80):
            mm = 0.5*(lo+hi)
            if (-mm*D*D/8.0 - (1.0-u)**2/(2.0*mm) + u*D/2.0) > -D/2:
                lo = mm
            else:
                hi = mm
        eq = 0.5*(lo+hi)*D/2.0
        out["crossing"].append(dict(u=u, basin_crossing=cross, measured=MEAS[u], eq115=eq))
        print(f"    -> basin crossing {cross if cross else float('nan'):.2f}, "
              f"measured {MEAS[u]:.2f}, Eq.(11.5) {eq:.2f}")

    print("\nsummary")
    print(f"    {'u':>5} {'basin':>8} {'measured':>9} {'err':>7} {'Eq.(11.5)':>10} {'err':>7}")
    eb, ee = [], []
    for r in out["crossing"]:
        b = r["basin_crossing"]
        if b:
            eb.append(abs(b-r["measured"]))
        ee.append(abs(r["eq115"]-r["measured"]))
        print(f"    {r['u']:5.2f} {b if b else float('nan'):8.2f} {r['measured']:9.2f} "
              f"{(b-r['measured']) if b else float('nan'):+7.2f} {r['eq115']:10.2f} "
              f"{r['eq115']-r['measured']:+7.2f}")
    if eb:
        print(f"\n    mean |error| : basin crossing {np.mean(eb):.2f}  vs  "
              f"existence criterion {np.mean(ee):.2f}   (in k*Delta/2)")
    json.dump(out, open(os.path.join(HERE, "d31_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d31_results_2026-08-29.json")
