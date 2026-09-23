# -*- coding: utf-8 -*-
"""
Derivation note -- D11: the caustic-edge law, and the coefficient c
(NEXT_c_and_mc plan, target 2 -- resolution).

D8 left m_s(u) = -m_c + c*u/k with c = 2.40 from a free line and c = 1.59 when
the line was anchored at the u = 0 edge.  D10 now supplies the whole curve from
ONE estimator (continuous-time delayed flow, closed-form b(m) from D9), so the
functional form can be read off instead of assumed.

What the D10 curve shows:
  * m_s(0) = -20.1 deg against the D9 closed form -m_c = -19.5 deg (0.6 deg);
    the intercept is the snap half-width, as predicted.
  * a SHOULDER for u <~ 0.25, where the edge stays pinned at -m_c;
  * a straight branch for u >~ 0.3 carrying the whole delay dependence.
The anchored estimate of D8 (c = 1.59) forced one line through both regimes,
which is why it disagreed with the free fit.  Restricted to the linear branch --
which is where every published optimum sits, u* in [0.655, 0.753] -- the two
estimates coincide.

Models compared on each dataset:
    M1  m_s = A + B u                     (linear branch; B -> c = B k/deg)
    M2  m_s = A + (2/k) u^2               first-order perturbation of the stall
    M4  m_s = A + u^2 (2-u)^2 / (2k)      the same to all orders in the fold map
The perturbative stall condition k z0 = 1 - 2u (equivalently (1-u)^2) is also
tested directly against the turning points recorded by D10.

Outputs: d11_results_2026-08-29.json
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5.0
DEG = math.degrees(1.0)
M_C = 22.5 - 3.0


def fits(U, MS, tag, k=K):
    out = {}
    c1 = np.polyfit(U, MS, 1)
    out["M1"] = dict(A=float(c1[1]), B=float(c1[0]), c=float(c1[0]*k/DEG),
                     rms=float(np.sqrt(np.mean((MS-np.polyval(c1, U))**2))))
    q2 = (2.0/k)*U**2*DEG
    A2 = float(np.mean(MS-q2))
    out["M2"] = dict(A=A2, rms=float(np.sqrt(np.mean((MS-A2-q2)**2))))
    q4 = (U**2*(2-U)**2/(2*k))*DEG
    A4 = float(np.mean(MS-q4))
    out["M4"] = dict(A=A4, rms=float(np.sqrt(np.mean((MS-A4-q4)**2))))
    print(f"  {tag} (n={len(U)}):")
    print(f"    M1 line      m_s = {out['M1']['A']:+7.2f} {out['M1']['B']:+7.2f}u"
          f"      rms {out['M1']['rms']:5.2f}   -> c = {out['M1']['c']:.2f}")
    print(f"    M2 2u^2/k    A   = {out['M2']['A']:+7.2f}"
          f"                      rms {out['M2']['rms']:5.2f}")
    print(f"    M4 fold map  A   = {out['M4']['A']:+7.2f}"
          f"                      rms {out['M4']['rms']:5.2f}")
    return out


if __name__ == "__main__":
    res = {"m_c": M_C, "k": K}
    d10 = json.load(open(os.path.join(HERE, "d10_results_2026-08-29.json")))
    rows = d10["rows"]

    print("[A] intercept: is m_s(0) the snap half-width?")
    z = [r for r in rows if r["u"] == 0.0][0]
    print(f"    flow  m_s(0) = {z['m_s']:+.2f} deg   (R^2 {z['r2']:.4f})")
    print(f"    D9    -m_c   = {-M_C:+.2f} deg   -> diff {z['m_s']+M_C:+.2f} deg")
    res["intercept"] = dict(m_s0=z["m_s"], minus_mc=-M_C, diff=z["m_s"]+M_C)

    print("\n[B] regime split: shoulder then linear branch")
    for r in rows:
        print(f"    u={r['u']:4.2f}  m_s={r['m_s']:+7.2f}  R^2={r['r2']:.4f}")
    print("\n[C] fits")
    lin = [r for r in rows if r["u"] >= 0.30]
    U = np.array([r["u"] for r in lin]); MS = np.array([r["m_s"] for r in lin])
    res["linear_branch"] = fits(U, MS, "D10 linear branch u >= 0.30")
    allr = [r for r in rows]
    Ua = np.array([r["u"] for r in allr]); MSa = np.array([r["m_s"] for r in allr])
    res["all_u"] = fits(Ua, MSa, "D10 all u (mixes both regimes)")

    c = res["linear_branch"]["M1"]["c"]
    print(f"\n  ==> c = {c:.2f} on the linear branch "
          f"(two-channel argument: 2; D8 bracket was [1.3, 2.4])")

    print("\n[D] continuous-time flow vs the ZOH-sampled surrogate (same quantity)")
    d8 = json.load(open(os.path.join(HERE, "d8_results_2026-08-29.json")))
    print(f"{'u':>6} {'flow':>8} {'surrogate':>10} {'diff':>7}")
    diffs = []
    for u in sorted(set(r["u"] for r in d8["rows"])):
        sur = [r["m_s"] for r in d8["rows"] if r["u"] == u and not math.isnan(r["m_s"])]
        fl = [r["m_s"] for r in rows if abs(r["u"]-u) < 1e-9]
        if sur and fl:
            d = fl[0]-np.mean(sur); diffs.append(abs(d))
            print(f"{u:6.2f} {fl[0]:8.2f} {np.mean(sur):10.2f} {d:+7.2f}")
    if diffs:
        print(f"    max |diff| = {max(diffs):.2f} deg  (acceptance criterion: 1.5 deg)")
        res["flow_vs_surrogate_max_diff"] = float(max(diffs))

    print("\n[E] stall condition at the turning points")
    print(f"{'u':>6} {'k z0 measured':>14} {'1 - 2u':>8} {'(1-u)^2':>9}")
    S = []
    for r in rows:
        if r.get("n_turns"):
            print(f"{r['u']:6.2f} {r['kz0_turn']:14.3f} {1-2*r['u']:8.3f} "
                  f"{(1-r['u'])**2:9.3f}")
            S.append((r["u"], r["kz0_turn"]))
    if len(S) >= 3:
        Us = np.array([s[0] for s in S]); Zs = np.array([s[1] for s in S])
        sel = Us >= 0.3
        cf = np.polyfit(Us[sel], Zs[sel], 1)
        print(f"    measured slope d(k z0)/du = {cf[0]:.2f}  (perturbation: -2)")
        res["stall_slope"] = float(cf[0])

    json.dump(res, open(os.path.join(HERE, "d11_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d11_results_2026-08-29.json")
