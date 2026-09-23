# -*- coding: utf-8 -*-
"""
Derivation note -- D4: the command-grid descriptor (open problem (iv)),
and the error floor it produces.

Manuscript status: "(iv) a quantitative coherence descriptor for command-grid
alignment" is open; the closure report's Sec. 12(c) concluded that the second
geometric axis is "the coherence of the visited cell-offset set
{(m0 + k*alpha) mod Delta}", direction fixed but functional form undetermined.

Derivation offered here (PROPOSITION C, grid-lock offset)
--------------------------------------------------------
The vote mixture is a static nonlinearity measured open loop (D3):
    aggregate = ideal + b(ideal mod Delta).
On a straight segment whose direction sits m_geo off the nearest grid direction
the loop cannot simply command m_geo, because the mixture snaps: b(m) ~ -m for
|m| <~ 19 deg.  The loop therefore keeps pushing the aim angle until the
AGGREGATE points along the segment.  The fixed point is

        m* + b(m*) = m_geo                                       (lock equation)

and the aim angle must be held m* - m_geo off the segment, which costs a lateral
offset (the aim angle is the lateral error divided by the look-ahead):

        e_lock = L * (m* - m_geo) = -L * b(m*)     [rad]          (Proposition C)

so the floor of a polygon is the dwell-weighted RMS of the per-segment lock cost

        floor = sqrt( sum_j (l_j/P) * (L*b(m*_j))^2 )                     (*)

Two immediate consequences, both testable on published curves with no fitting:
  * floor decreases monotonically as m_geo moves from the cell centre to the cell
    boundary -- because the lock equation needs less overdrive.  This is the
    monotone floor sequence of Sec. 12(c), which the "hunting amplitude" story
    only reproduced semi-quantitatively.
  * grid-aligned segments (m_geo = 0) cost nothing, which is the square's
    +4.7% alignment bonus.

Also tested: whether the residual after (*) needs a coherence (exponential-sum)
term  Z = sum_j (l_j/P) exp(i 2*pi*m_j/Delta),  |Z| = 1 for grid-commensurate
polygons (n | n_dirs) and exactly 0 otherwise.

Ground truth: published rotation test (phase1_rotation_results_2026-07-18.json)
and square-rotation h-sweep (phase3_hB_tri_results_2026-07-19.json), both run
with the full 150-agent engine, random start phase.

Outputs: d4_results_2026-08-29.json
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = next(
    p for p in [os.path.join(HERE, "..", "..", "data"),
                os.path.join(HERE, "..", "closure_pack")]
    if os.path.isdir(p))
DELTA = 45.0
L = 2.0


def load_transfer():
    d = json.load(open(os.path.join(HERE, "d3_results_2026-08-29.json")))
    t = d["transfer"]
    return np.array(t["m"]), np.array(t["b"]), np.array(t["s"])


def dither_smooth(ms, b, sigma):
    """b smoothed by the loop's own in-cell jitter: b_bar = b * N(0, sigma^2),
    periodic in Delta (the quantiser characteristic is Delta-periodic)."""
    if sigma <= 1e-9:
        return b.copy()
    grid = np.linspace(-DELTA/2, DELTA/2, len(ms))
    per = np.concatenate([b, b, b])
    xs = np.concatenate([grid-DELTA, grid, grid+DELTA])
    out = np.empty_like(b)
    for i, m in enumerate(grid):
        w = np.exp(-0.5*((xs-m)/sigma)**2)
        out[i] = float(np.sum(w*per)/np.sum(w))
    return out


def lock_offset(m_geo, ms, b, refine=4001):
    """solve m* + b(m*) = m_geo on [0, Delta/2]; return b(m*) in degrees."""
    grid = np.linspace(0.0, DELTA/2, refine)
    g = grid + np.interp(grid, ms, b)
    sgn = np.sign(m_geo)
    tgt = abs(m_geo)
    # g is ~0 over most of the cell then rises steeply near the boundary
    idx = np.where(np.diff(np.sign(g - tgt)) != 0)[0]
    if len(idx) == 0:
        mstar = grid[-1]
    else:
        i = idx[-1]
        mstar = float(np.interp(tgt, [g[i], g[i+1]], [grid[i], grid[i+1]]))
    bstar = float(np.interp(mstar, ms, b))
    return sgn*mstar, sgn*bstar


def floor_pred(misaligns, lens, ms, b, s_res=0.0):
    w = np.array(lens, float); w = w/w.sum()
    e = []
    for m in misaligns:
        mm = ((m + DELTA/2) % DELTA) - DELTA/2
        _, bs = lock_offset(mm, ms, b)
        e.append(L*math.radians(abs(bs)))
    e = np.array(e)
    lock2 = float(np.sum(w*e**2))
    return float(np.sqrt(lock2 + (L*math.radians(s_res))**2)), e


def coherence(misaligns, lens):
    w = np.array(lens, float); w = w/w.sum()
    z = np.sum(w*np.exp(1j*2*math.pi*np.array(misaligns)/DELTA))
    return float(abs(z))


def sweep_sigma(rot, hs, ms, b0, s_res):
    """How much in-cell dither does the floor data imply, and is it consistent
    with the independently measured residual spread s(m)?"""
    cases = []
    for name, c in rot["cases"].items():
        n = len(c["grid_misalign_deg"])
        cases.append((c["grid_misalign_deg"], [c["circ"]/n]*n,
                      min(c["taus"]["433"]["rmses"])))
    for k, v in hs.items():
        cases.append(([float(k)]*4, [20.0]*4, min(v["taus"]["433"]["rmses"])))
    rows = []
    for sig in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        bb = dither_smooth(ms, b0, sig)
        errs = []
        for mis, lens, fo in cases:
            fp, _ = floor_pred(mis, lens, ms, bb, s_res)
            errs.append((fp-fo)/fo)
        errs = np.array(errs)
        rows.append(dict(sigma=sig, mean_abs_err=float(np.mean(np.abs(errs))),
                         bias=float(np.mean(errs))))
    return rows


if __name__ == "__main__":
    ms, b, s = load_transfer()
    print("transfer check:  b(5)=%.2f  b(15)=%.2f  b(19)=%.2f  b(21)=%.2f  b(22.4)=%.2f"
          % tuple(np.interp([5, 15, 19, 21, 22.4], ms, b)))
    print("lock equation g(m) = m + b(m):  g(15)=%.2f g(19)=%.2f g(21)=%.2f g(22.4)=%.2f\n"
          % tuple([x + np.interp(x, ms, b) for x in [15, 19, 21, 22.4]]))

    out = {"rotation": [], "hsweep": []}

    rot = json.load(open(os.path.join(PACK, "phase1_rotation_results_2026-07-18.json")))
    hs0 = json.load(open(os.path.join(PACK, "phase3_hB_tri_results_2026-07-19.json")))["h_sweep"]
    S_RES = float(np.mean(s))                       # measured residual spread, deg
    print(f"measured residual spread s: mean {S_RES:.2f} deg, "
          f"min {s.min():.2f}, max {s.max():.2f}  -> L*s = {L*math.radians(S_RES):.3f} m")
    sig_rows = sweep_sigma(rot, hs0, ms, b, S_RES)
    print("\nin-cell dither sweep (11 published floors):")
    for r in sig_rows:
        print(f"   sigma_m = {r['sigma']:4.1f} deg  mean|err| = {100*r['mean_abs_err']:5.1f}%  "
              f"bias = {100*r['bias']:+6.1f}%")
    best = min(sig_rows, key=lambda r: r["mean_abs_err"])
    SIG = best["sigma"]
    print(f"   -> best sigma_m = {SIG:.1f} deg")
    out["sigma_sweep"] = sig_rows
    out["sigma_used"] = SIG
    out["s_res"] = S_RES
    b = dither_smooth(ms, b, SIG)
    print("[A] rotation test -- floor = min RMSE over the speed grid, tau = 433 ms")
    print(f"{'case':13s} {'misalign':28s} {'|Z|':>5} {'floor_obs':>10} {'floor_pred':>11} {'err':>8}")
    for name, c in rot["cases"].items():
        mis = c["grid_misalign_deg"]
        n = len(mis)
        lens = [c["circ"]/n]*n
        fp, e = floor_pred(mis, lens, ms, b, S_RES)
        Z = coherence(mis, lens)
        fo = min(c["taus"]["433"]["rmses"])
        out["rotation"].append(dict(case=name, misalign=mis, Z=Z, floor_obs=fo,
                                    floor_pred=fp, rel=(fp-fo)/fo))
        print(f"{name:13s} {str([round(x,1) for x in mis]):28s} {Z:5.2f} "
              f"{fo:10.3f} {fp:11.3f} {100*(fp-fo)/fo:+7.1f}%")

    hs = hs0
    print("\n[B] square rotation sweep (all four sides coherent at m), tau = 433 ms")
    print(f"{'m (deg)':>8} {'floor_obs':>10} {'floor_pred':>11} {'m*':>7} {'err':>8}")
    rows = [(0.0, min(rot["cases"]["sq_rot0"]["taus"]["433"]["rmses"]))]
    for k, v in hs.items():
        rows.append((float(k), min(v["taus"]["433"]["rmses"])))
    rows.append((22.5, min(rot["cases"]["sq_rot22.5"]["taus"]["433"]["rmses"])))
    for m, fo in sorted(rows):
        fp, _ = floor_pred([m]*4, [20.0]*4, ms, b, S_RES)
        mstar, _ = lock_offset(m, ms, b)
        out["hsweep"].append(dict(m=m, floor_obs=fo, floor_pred=fp, m_star=mstar,
                                  rel=(fp-fo)/fo if fo else None))
        print(f"{m:8.3f} {fo:10.3f} {fp:11.3f} {mstar:7.2f} {100*(fp-fo)/max(fo,1e-9):+7.1f}%")

    print("\n[C] coherence |Z| for regular n-gons on an 8-direction grid")
    for n in [3, 4, 5, 6, 8, 12]:
        alpha = 360.0/n
        mis = [((k*alpha) % DELTA) for k in range(n)]
        print(f"  n={n:3d} alpha={alpha:6.1f}  offsets(mod 45)={[round(x,1) for x in mis]}  |Z|={coherence(mis,[1]*n):.3f}")

    json.dump(out, open("d4_results_2026-08-29.json", "w"), indent=2, default=float)
    print("\n-> d4_results_2026-08-29.json")
