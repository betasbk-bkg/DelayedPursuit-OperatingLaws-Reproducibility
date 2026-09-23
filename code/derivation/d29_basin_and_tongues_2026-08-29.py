# -*- coding: utf-8 -*-
"""
Derivation note -- D29: the two remaining limitations.

(2) BISTABILITY.  Beyond the window D28 finds two stable branches at some u --
    the "+" word and the "+-+" word -- and the flow realises "+-+".  Stability
    alone cannot say which; the basins can.  Here P is iterated from a grid of
    initial z to partition the line into basins, and the z the faithful flow
    actually carries at its crossings is located inside them.

(3) THE TRANSITION BAND.  Rather than "fix" the mode locking near k*Delta/2 =
    2.5, map it: run the faithful flow over a (k*Delta/2, u) grid, count forward
    and backward crossings, and label the realised word.  That turns the band
    into a regime map with a boundary, which is what a locking structure asks
    for.

Outputs: d29_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
DEG = math.degrees(1.0)
D = math.radians(45.0)


# ----------------------------------------------------------------- (2) basins
def basin_map(u, k, delta, fps, n=400, iters=200, tol=1e-7):
    th = u/k
    zs = np.linspace(-0.3*delta, 1.6*delta, n)
    lab = []
    for z0 in zs:
        z = z0
        ok = True
        for _ in range(iters):
            o = d28.orbit(z, u, k, delta, th)
            if o is None:
                ok = False
                break
            zn = o["z_next"]
            if abs(zn - z) < tol:
                z = zn
                break
            z = zn
        if not ok:
            lab.append(-1); continue
        hit = -1
        for j, f in enumerate(fps):
            if abs(z - f["z"]) < 1e-3*delta:
                hit = j; break
        lab.append(hit)
    return zs, np.array(lab)


def flow_crossing_z(u, k, delta, T=900.0, dt=2.0e-3, burn=0.5):
    """z carried by the faithful flow at its forward crossings."""
    mm, bb, Dl = d19.sawtooth(math.degrees(delta))
    nb = len(mm); step = (mm[-1]-mm[0])/(nb-1); m0 = mm[0]; bl = bb.tolist()
    th = u/k; nd = max(0, int(round(th/dt))); n = int(T/dt)
    W = [0.0]*(n+1); Z = [0.0]*(n+1)
    m = 0.0; w = 0.0; c0 = (0.5-u)/k; half = Dl/2
    prev = None; zc = []
    for i in range(n):
        mw = ((m+half) % Dl)-half
        x = (mw-m0)/step; j = int(x); j = 0 if j < 0 else (nb-2 if j > nb-2 else j)
        f = x-j
        bv = bl[j]*(1.0-f)+bl[j+1]*f
        z = c0-(W[i-nd] if i >= nd else 0.0)+bv
        zd = Z[i-nd] if i >= nd else 0.0
        w += dt*k*z; m += dt*(1.0-k*zd)
        W[i+1] = w; Z[i+1] = z
        if i > burn*n and prev is not None and mw-prev < -Dl/2:
            zc.append(z)
        prev = mw
    return np.array(zc)


# -------------------------------------------------------------- (3) tongues
def realised_word(u, k, delta, T=600.0, dt=2.0e-3, burn=0.5):
    mm, bb, Dl = d19.sawtooth(math.degrees(delta))
    nb = len(mm); step = (mm[-1]-mm[0])/(nb-1); m0 = mm[0]; bl = bb.tolist()
    th = u/k; nd = max(0, int(round(th/dt))); n = int(T/dt)
    W = [0.0]*(n+1); Z = [0.0]*(n+1)
    m = 0.0; w = 0.0; c0 = (0.5-u)/k; half = Dl/2
    prev = None; nf = nbk = 0
    for i in range(n):
        mw = ((m+half) % Dl)-half
        x = (mw-m0)/step; j = int(x); j = 0 if j < 0 else (nb-2 if j > nb-2 else j)
        f = x-j
        bv = bl[j]*(1.0-f)+bl[j+1]*f
        z = c0-(W[i-nd] if i >= nd else 0.0)+bv
        zd = Z[i-nd] if i >= nd else 0.0
        w += dt*k*z; m += dt*(1.0-k*zd)
        W[i+1] = w; Z[i+1] = z
        if i > burn*n and prev is not None:
            if mw-prev < -Dl/2: nf += 1
            elif mw-prev > Dl/2: nbk += 1
        prev = mw
    if nf == 0:
        return "?", nf, nbk
    r = nbk/max(nf-nbk, 1)
    return ("+" if nbk == 0 else ("+-+" if abs(r-1.0) < 0.15 else "other")), nf, nbk


if __name__ == "__main__":
    out = {}
    print("[2] basins where two branches are stable  (k = 8, Delta = 45 deg)")
    rows = []
    for u in (0.71, 0.90):
        fps = [f for f in d28.scan(u, 8.0, D) if abs(f["mult"]) < 1.0]
        zs, lab = basin_map(u, 8.0, D, fps)
        zc = flow_crossing_z(u, 8.0, D)
        share = {j: float(np.mean(lab == j)) for j in range(len(fps))}
        # which basin holds the flow's own crossing z?
        hits = []
        for zz in zc[:200]:
            idx = int(np.argmin(np.abs(zs - zz)))
            hits.append(lab[idx])
        vals, cnt = np.unique([h for h in hits if h >= 0], return_counts=True)
        dom = int(vals[np.argmax(cnt)]) if len(vals) else -1
        print(f"  u = {u:.2f}:  {len(fps)} stable branches")
        for j, f in enumerate(fps):
            print(f"     [{j}] word {f['word']:>5}  mult {f['mult']:+.3f}  "
                  f"m_s {f['m_s']:+7.2f}  basin share {100*share[j]:5.1f}%"
                  + ("   <-- flow lives here" if j == dom else ""))
        rows.append(dict(u=u, branches=[dict(word=f["word"], mult=f["mult"],
                                             m_s=f["m_s"], share=share[j])
                                        for j, f in enumerate(fps)],
                         flow_branch=dom,
                         flow_word=fps[dom]["word"] if dom >= 0 else None,
                         z_flow_mean=float(np.mean(zc)) if len(zc) else None))
    out["basins"] = rows

    print("\n[3] regime map: word realised by the faithful flow")
    ks = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
    us = [0.30, 0.50, 0.71, 0.90]
    print("    kD/2  " + "".join(f"{'u='+format(u,'.2f'):>10}" for u in us))
    grid = []
    for k in ks:
        cells = []
        for u in us:
            w, nf, nb = realised_word(u, k, D)
            cells.append(w)
            grid.append(dict(k=k, kD2=k*D/2, u=u, word=w, nf=nf, nb=nb))
        print(f"    {k*D/2:5.2f}  " + "".join(f"{c:>10}" for c in cells))
    out["regime_map"] = grid
    first = {}
    for u in us:
        rows_u = [g for g in grid if g["u"] == u and g["word"] != "+"]
        first[str(u)] = min((g["kD2"] for g in rows_u), default=None)
    print("    first non-'+' word at kD/2 = " +
          ", ".join(f"u={u}: {first[str(u)]}" for u in us))
    out["first_transition"] = first

    json.dump(out, open(os.path.join(HERE, "d29_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d29_results_2026-08-29.json")
