# -*- coding: utf-8 -*-
"""
Derivation note -- D30: the transition curve.

Sec. 17.3 left one quantitative item: where the flow switches from the word "+"
to "+-+".  The geometric criterion of Eq. (11.5) -- the "+" orbit ceasing to
exist -- overpredicts by 0.24-0.85 in k*Delta/2, and Sec. 17.2 says why: the two
words coexist for a while and the flow takes the branch with the larger basin.
So the transition should be the BASIN CROSSING, not the existence limit.

Two things have to be checked before that can be computed:

  (A) VALIDITY of the one-dimensional section.  The reduction assumes no earlier
      crossing still sits inside the delay window; near the transition the
      crossings crowd together, so the minimum event gap is compared with
      theta_d for every branch found.  If the gap closes, the section state is
      not z alone and the 1-D map cannot decide the transition.

  (B) A REFINED measured boundary.  The regime map of D29 used steps of ~0.2 in
      k*Delta/2; here the flow is scanned finely enough to bracket the switch.

Then, where (A) permits, basin shares are computed on a fine k grid and the
crossing share(+-+) = share(+) is located and compared with (B).

Outputs: d30_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
d29 = _load("d29", "d29_basin_and_tongues_2026-08-29.py")
D = math.radians(45.0)
DEG = math.degrees(1.0)
US = (0.30, 0.50, 0.71, 0.90)


def branches(u, k):
    fps = d28.scan(u, k, D)
    return [f for f in fps if abs(f["mult"]) < 1.0]


if __name__ == "__main__":
    out = {}

    print("[A] is the 1-D section valid near the transition?")
    print(f"    {'kD/2':>6} {'u':>5} {'#stable':>8} {'words':>14} {'gaps>theta_d':>13}")
    rowsA = []
    for kd2 in (2.16, 2.36, 2.55, 2.75):
        k = 2*kd2/D
        for u in US:
            b = branches(u, k)
            ws = ",".join(sorted(set(f["word"] for f in b))) or "-"
            gp = all(f["gap_ok"] for f in b) if b else None
            rowsA.append(dict(kD2=kd2, u=u, n=len(b), words=ws, gap_ok=gp))
            print(f"    {kd2:6.2f} {u:5.2f} {len(b):8d} {ws:>14} {str(gp):>13}")
    out["A_validity"] = rowsA

    print("\n[B] refined measured boundary (faithful flow)")
    print(f"    {'u':>5} {'last +':>8} {'first +-+':>10} {'midpoint':>9}")
    rowsB = []
    for u in US:
        lastp, firstm = None, None
        for kd2 in np.arange(1.90, 3.35, 0.05):
            k = 2*kd2/D
            w, nf, nb = d29.realised_word(u, k, D, T=600.0)
            if w == "+":
                lastp = kd2
            elif w == "+-+" and firstm is None:
                firstm = kd2
                break
        mid = 0.5*(lastp+firstm) if (lastp and firstm) else None
        rowsB.append(dict(u=u, last_plus=lastp, first_pmp=firstm, mid=mid))
        print(f"    {u:5.2f} {lastp if lastp else float('nan'):8.2f} "
              f"{firstm if firstm else float('nan'):10.2f} "
              f"{mid if mid else float('nan'):9.2f}")
    out["B_measured"] = rowsB

    print("\n[C] basin shares across the transition (where the section is valid)")
    print(f"    {'kD/2':>6} {'u':>5} {'share(+)':>9} {'share(+-+)':>11} {'winner':>8}")
    rowsC = []
    for u in US:
        for kd2 in np.arange(2.00, 3.30, 0.10):
            k = 2*kd2/D
            b = branches(u, k)
            if not b:
                continue
            zs, lab = d29.basin_map(u, k, D, b, n=160, iters=120)
            sh = {}
            for j, f in enumerate(b):
                sh[f["word"]] = sh.get(f["word"], 0.0) + float(np.mean(lab == j))
            sp, sm = sh.get("+", 0.0), sh.get("-++", 0.0) + sh.get("+-+", 0.0)
            if sp == 0 and sm == 0:
                continue
            win = "+" if sp > sm else ("+-+" if sm > sp else "tie")
            rowsC.append(dict(u=u, kD2=float(kd2), share_plus=sp, share_pmp=sm, winner=win))
            print(f"    {kd2:6.2f} {u:5.2f} {sp:9.3f} {sm:11.3f} {win:>8}")
    out["C_basins"] = rowsC

    print("\n[D] basin crossing vs measured transition")
    print(f"    {'u':>5} {'basin crossing':>15} {'measured mid':>13} {'Eq.(11.5)':>10}")
    rowsD = []
    for u in US:
        seq = sorted([r for r in rowsC if r["u"] == u], key=lambda r: r["kD2"])
        cross = None
        for a, bb in zip(seq[:-1], seq[1:]):
            if a["winner"] == "+" and bb["winner"] == "+-+":
                da = a["share_plus"] - a["share_pmp"]
                db = bb["share_plus"] - bb["share_pmp"]
                cross = a["kD2"] + (bb["kD2"]-a["kD2"])*da/(da-db)
                break
        mid = [r["mid"] for r in rowsB if r["u"] == u][0]
        lo, hi = 1.0, 12.0
        for _ in range(80):
            m = 0.5*(lo+hi)
            if (-m*D*D/8.0 - (1.0-u)**2/(2.0*m) + u*D/2.0) > -D/2:
                lo = m
            else:
                hi = m
        eq = 0.5*(lo+hi)*D/2.0
        rowsD.append(dict(u=u, basin_crossing=cross, measured=mid, eq115=eq))
        print(f"    {u:5.2f} {cross if cross else float('nan'):15.2f} "
              f"{mid if mid else float('nan'):13.2f} {eq:10.2f}")
    out["D_compare"] = rowsD
    good = [r for r in rowsD if r["basin_crossing"] and r["measured"]]
    if good:
        e = [abs(r["basin_crossing"]-r["measured"]) for r in good]
        e2 = [abs(r["eq115"]-r["measured"]) for r in good]
        print(f"\n    mean |error|: basin crossing {np.mean(e):.2f}, "
              f"Eq.(11.5) {np.mean(e2):.2f}  (in k*Delta/2)")

    json.dump(out, open(os.path.join(HERE, "d30_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d30_results_2026-08-29.json")
