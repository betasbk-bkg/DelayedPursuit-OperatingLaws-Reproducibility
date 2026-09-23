# -*- coding: utf-8 -*-
"""
Derivation note -- D28: a robust return map, removing the spurious fixed
points of D27.

D27 produced up to fifteen "stable" fixed points at k = 5, where the faithful
flow has ZERO backward crossings -- so most of them were artefacts.  Two defects
in that implementation:

  (1) the cell frame was recomputed from m by comparison, so right after a
      crossing (m sits exactly ON the boundary) the frame sometimes failed to
      advance and the same boundary was detected again a moment later, inventing
      events and hence fake words;
  (2) roots were bracketed on a uniform z grid, which straddles the
      discontinuities of P and turns each jump into a false sign change.

Fixes: the cell index is carried explicitly and incremented at each event, so m
is always strictly inside its cell; and brackets are accepted only when the WORD
is identical at both ends (P is smooth only within a fixed symbol sequence), with
every surviving root additionally required to be continuous, |P(z+h) - P(z-h)|
small, and to have an O(1) multiplier.

Acceptance test: inside the window the map must return exactly ONE stable fixed
point, with word "+" and the edge of Eq. (12); beyond it, the "+-+" branch.

Outputs: d28_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0)


def orbit(z0, u, k, delta, th, max_ev=16):
    """Exact event-driven integration from a forward crossing.
    The cell index n is explicit: the current cell is [n*delta - delta/2,
    n*delta + delta/2], and it is updated at every event, so m is never
    ambiguous at a boundary."""
    n_cell = 0
    m = -delta/2.0            # sitting on the lower boundary of cell 0, moving up
    z = z0
    T = 0.0
    events = [(0.0, +1)]
    word = ""
    fold = None
    for _ in range(max_ev):
        cand = [te + th for te, _ in events if te + th > T + 1e-13]
        T_bp = min(cand) if cand else float("inf")
        J = sum(s for te, s in events if T - th < te <= T)
        z_ref = z + T                                  # z(T') = z_ref - T'
        A = (1.0 - u) - k*z_ref + k*delta*J            # m'(T') = A + k T'
        lo = n_cell*delta - delta/2.0
        hi = n_cell*delta + delta/2.0
        t_hit, which = None, 0
        for bnd, w in ((hi, +1), (lo, -1)):
            a2, b2 = k/2.0, A
            c2 = m - bnd - A*T - k*T*T/2.0
            d2 = b2*b2 - 4*a2*c2
            if d2 < 0:
                continue
            r = math.sqrt(d2)
            for root in ((-b2 - r)/(2*a2), (-b2 + r)/(2*a2)):
                if root > T + 1e-11 and (t_hit is None or root < t_hit):
                    t_hit, which = root, w
        Tn = min(t_hit, T_bp) if t_hit is not None else T_bp
        if not np.isfinite(Tn):
            return None
        tf = -A/k                                       # m' = 0
        if T < tf < Tn:
            mf = m + A*(tf - T) + k*(tf*tf - T*T)/2.0
            fold = mf if fold is None else min(fold, mf)
        m = m + A*(Tn - T) + k*(Tn*Tn - T*T)/2.0
        z -= (Tn - T)
        T = Tn
        if t_hit is not None and Tn == t_hit:
            z += which*delta
            n_cell += which                              # <-- explicit frame update
            events.append((T, which))
            word += "+" if which > 0 else "-"
            if which > 0 and n_cell == 1:
                gaps = [events[i+1][0]-events[i][0] for i in range(len(events)-1)]
                return dict(z_next=z, period=T, word=word, fold=fold,
                            min_gap=min(gaps) if gaps else float("inf"))
            if abs(n_cell) > 2:
                return None
    return None


def scan(u, k, delta, n=2000):
    th = u/k
    zs = np.linspace(-0.3*delta, 1.6*delta, n)
    res = [orbit(z, u, k, delta, th) for z in zs]
    out = []
    for i in range(n-1):
        a, b = res[i], res[i+1]
        if a is None or b is None:
            continue
        if a["word"] != b["word"]:                     # bracket straddles a jump
            continue
        fa, fb = a["z_next"]-zs[i], b["z_next"]-zs[i+1]
        if fa == 0 or np.sign(fa) == np.sign(fb):
            continue
        zl, zr, fl = zs[i], zs[i+1], fa
        for _ in range(90):
            zm = 0.5*(zl+zr)
            o = orbit(zm, u, k, delta, th)
            if o is None or o["word"] != a["word"]:
                break
            f = o["z_next"] - zm
            if np.sign(f) == np.sign(fl):
                zl, fl = zm, f
            else:
                zr = zm
        zst = 0.5*(zl+zr)
        o = orbit(zst, u, k, delta, th)
        if o is None or o["fold"] is None:
            continue
        # a genuine fixed point must actually satisfy P(z) = z; and then the
        # period is forced to be exactly Delta (z falls at unit rate and regains
        # Delta per net winding), which is an independent consistency check
        if abs(o["z_next"] - zst) > 1e-6*delta:
            continue
        if abs(o["period"]/delta - 1.0) > 1e-6:
            continue
        h = 1e-7
        op, om = orbit(zst+h, u, k, delta, th), orbit(zst-h, u, k, delta, th)
        if op is None or om is None or op["word"] != o["word"] or om["word"] != o["word"]:
            continue
        jump = abs(op["z_next"] - om["z_next"])
        if jump > 1e-3:                                 # discontinuous: not a root
            continue
        mult = (op["z_next"] - om["z_next"])/(2*h)
        w = o["fold"]
        while w < -delta/2:
            w += delta
        while w > delta/2:
            w -= delta
        cand = dict(z=zst, mult=float(mult), word=o["word"], m_s=w*DEG,
                    period_ratio=o["period"]/delta,
                    gap_ok=bool(o["min_gap"] > th))
        if not any(abs(c["z"]-zst) < 1e-4 for c in out):
            out.append(cand)
    return out


def flow_edge(u, k, delta_deg=45.0):
    M, _ = d10.flow(u, k=k, T=900.0, mgrid=d19.sawtooth(delta_deg))
    ctr, rho = d10.density(M, nbin=360)
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+1.5) & (rho > 0)
    if sel.sum() < 4:
        return float("nan")
    cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
    return float(-cf[1]/cf[0])


if __name__ == "__main__":
    D = math.radians(45.0)
    out = []
    print("robust return map: fixed points per (k, u)")
    for k in (5.0, 6.5, 8.0):
        print(f"\n  k = {k}  (k*Delta/2 = {k*D/2:.2f})")
        print(f"    {'u':>5} {'#stable':>8} {'word':>6} {'mult':>7} {'T/D':>5} "
              f"{'m_s map':>9} {'m_s flow':>9} {'diff':>7} {'eq(12)':>8}")
        for u in (0.30, 0.50, 0.71, 0.90):
            fps = scan(u, k, D)
            st = [f for f in fps if abs(f["mult"]) < 1.0]
            fl = flow_edge(u, k)
            eq = d19.m_s_closed(u, k, D)*DEG
            if not st:
                print(f"    {u:5.2f} {0:8d}   (none)")
                continue
            for f in st:
                print(f"    {u:5.2f} {len(st):8d} {f['word']:>6} {f['mult']:7.3f} "
                      f"{f['period_ratio']:5.2f} {f['m_s']:9.2f} {fl:9.2f} "
                      f"{f['m_s']-fl:+7.2f} {eq:8.2f}")
                out.append(dict(k=k, u=u, flow_edge=fl, eq12=eq, n_stable=len(st), **f))
    ok = [r for r in out if np.isfinite(r["flow_edge"])]
    if ok:
        d = [r["m_s"]-r["flow_edge"] for r in ok]
        print(f"\nall stable branches vs flow: n = {len(ok)}, "
              f"mean {np.mean(d):+.2f} deg, max |diff| {max(abs(x) for x in d):.2f} deg")
    json.dump(out, open(os.path.join(HERE, "d28_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d28_results_2026-08-29.json")
