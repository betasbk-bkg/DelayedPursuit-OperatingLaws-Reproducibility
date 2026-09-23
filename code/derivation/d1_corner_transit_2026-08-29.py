# -*- coding: utf-8 -*-
"""
Derivation note -- D1: corner transit dynamics (open problem (i)).

Goal: replace Proposition 1's assumptions (ii) "the transit error is dominated by
its peak, attained at the symmetric straddle" and (iii) "Theorem 1's compensation
condition applies instantaneously with the current chord as effective look-ahead"
by an actual solution of the transit dynamics.

Three layers, each checked against the next:
  L0  exact 2D noise-free simulator (ground truth; same primitives as the engine:
      DT, ZOH period WIN, pure delay DELAY_F, first-order velocity lerp SMOOTH)
  L1  reduced linear-in-e transit model with the exact chord law psi(f), c(f)
      (delay chain kept explicitly: ZOH + pure delay + first-order lag)
  L2  same but with the delay chain collapsed to its mean delay tau_tot
      (Theorem 2 reduction -- tests whether mean-delay composition, derived for
      quasi-static bias, survives a non-quasi-static transit)

New analytic content tested here:
  LEMMA A (corner area law).  For a single isolated corner of exterior angle a,
      \int e(t)/c(f(t)) dt  =  (L*a/v) * (L/2 - v*tau_tot)
  i.e. the *signed area* of the corner error bump vanishes at v*tau_tot = L/2
  EXACTLY, independent of a.  (Proof in the report: integrate the linearized
  transit ODE over the whole transit; the commanded-turn ramp psi(f(t-tau)) is
  antisymmetric about the symmetric straddle f=1/2 -- Lemma 1 of the manuscript
  gives the chord, and psi(1-f) = a - psi(f) gives the antisymmetry -- so its
  area centroid sits exactly at the straddle time t = tau_tot - L/(2v), while
  the path-direction step sits at t = 0.)

  Consequence: the cos(a/2) in Proposition 1 is NOT a first-moment (bias-zero)
  effect -- the first moment is a-independent.  It must come from the RMS
  weighting, i.e. from the shape of the transit bump.  That is what this script
  measures, and what D2 turns into a closed form.

Objective actually minimised by the polygon experiments:
      RMSE^2 over a lap = n_corner * I2(v) / (P/v)  =  (n_corner/P) * v * I2(v)
  with I2(v) = \int e^2 dt over one transit.  So the corner optimum minimises
  v*I2(v), not I2(v).  (phase9's fixed-window RMS has the same v-weighting.)

Outputs: d1_results_2026-08-29.json
"""
import json, math, time
import numpy as np

# ---- engine primitives (simulation_main.py constants) ------------------------
DT       = 1.0/60.0
WIN      = 0.3           # ZOH / vote period
VI       = int(round(WIN/DT))
DELAY_F  = 26            # tau = 433 ms
SMOOTH   = 0.2
T_S      = DT/SMOOTH     # 1/12 s
L        = 2.0
TAU_PURE = DELAY_F*DT
TAU_TOT  = TAU_PURE + WIN/2 + T_S     # 0.6663 s


# ============================ L0: exact 2D, noise free =======================
def _seg_project(a, b, p):
    v = b - a
    t = float(np.clip((p-a) @ v / (v @ v), 0.0, 1.0))
    q = a + t*v
    return np.linalg.norm(p-q), t, float(np.linalg.norm(v)), q


def sim2d(alpha_deg, v, seglen=60.0, dt=DT, vi=VI, delay_f=DELAY_F,
          smooth=SMOOTH, Lv=L):
    """Exact noise-free transit of one corner.  Vehicle starts on segment 1,
    heading along it, and runs until it is seglen*0.9 past the corner.
    Returns t, signed lateral error e (positive = inside of the turn), arc s."""
    a = math.radians(alpha_deg)
    p0 = np.array([-seglen, 0.0]); pc = np.array([0.0, 0.0])
    p1 = np.array([seglen*math.cos(a), seglen*math.sin(a)])
    e1 = np.array([1.0, 0.0]); e2 = np.array([math.cos(a), math.sin(a)])
    l0 = seglen; l1 = seglen

    def arc_and_err(p):
        d0, t0, ln0, q0 = _seg_project(p0, pc, p)
        d1, t1, ln1, q1 = _seg_project(pc, p1, p)
        if d0 <= d1:
            s = t0*ln0 - l0            # arc measured from corner (negative before)
            tang = e1; q = q0; d = d0
        else:
            s = t1*ln1
            tang = e2; q = q1; d = d1
        # sign: positive to the left of travel direction = inside of a left turn
        cr = tang[0]*(p-q)[1] - tang[1]*(p-q)[0]
        return s, math.copysign(d, cr)

    def aim_point(pd):
        s, _ = arc_and_err(pd)
        sa = s + Lv
        if sa <= 0.0:                      # aim point still on segment 1
            return pc + max(sa, -l0)*e1
        return pc + min(sa, l1)*e2         # aim point past the corner

    pos = p0.copy(); vel = e1*v; cur = e1.copy()
    hist = [pos.copy()]
    nf = int((seglen*1.9)/(v*dt))
    T, E, S = [], [], []
    for f in range(nf):
        if f % vi == 0:
            pd = hist[max(0, len(hist)-1-delay_f)]
            dd = aim_point(pd) - pd
            n = np.linalg.norm(dd)
            if n > 1e-12:
                cur = dd/n
        vel = vel + smooth*(cur*v - vel)
        pos = pos + vel*dt
        hist.append(pos.copy())
        s, e = arc_and_err(pos)
        T.append((f+1)*dt); S.append(s); E.append(e)
    return np.array(T), np.array(E), np.array(S)


# ================== L1 / L2: reduced linear transit model =====================
def chord(alpha_rad, f):
    """chord direction psi(f) and length factor c(f) for straddle fraction f."""
    f = min(max(f, 0.0), 1.0)
    x = (1.0-f) + f*math.cos(alpha_rad)
    y = f*math.sin(alpha_rad)
    return math.atan2(y, x), math.hypot(x, y)


def theory(alpha_deg, v, seglen=60.0, dt=DT, mean_delay=False, Lv=L,
           tau_pure=TAU_PURE, win=WIN, t_s=T_S):
    """Linear-in-e transit model.  mean_delay=False -> explicit ZOH+delay+lag
    chain (L1);  True -> single pure delay tau_tot with no ZOH/lag (L2)."""
    a = math.radians(alpha_deg)
    nf = int((seglen*1.9)/(v*dt))
    vi = max(1, int(round(win/dt)))
    dly = int(round(tau_pure/dt))
    tau_tot = tau_pure + win/2.0 + t_s
    dly_m = int(round(tau_tot/dt))

    s = -seglen                    # arc from corner
    e = 0.0                        # lateral error
    th = 0.0                       # heading angle (rel. segment 1)
    eh = [0.0]*(max(dly, dly_m)+2) # history of e
    sh = [s]*(max(dly, dly_m)+2)
    cmd = 0.0
    T, E, S = [], [], []
    for f in range(nf):
        k = dly_m if mean_delay else dly
        e_d = eh[-1-k]; s_d = sh[-1-k]
        fr = (s_d + Lv)/Lv
        psi, c = chord(a, fr)
        cmd_now = psi - e_d/(Lv*max(c, 1e-6))
        if mean_delay:
            th = cmd_now                      # no ZOH, no lag: all in tau_tot
        else:
            if f % vi == 0:
                cmd = cmd_now
            th += (dt/t_s)*(cmd - th)         # first-order heading lag
        th_path = 0.0 if s < 0.0 else a
        e = e + dt*v*(th - th_path)
        s = s + dt*v
        eh.append(e); sh.append(s)
        T.append((f+1)*dt); E.append(e); S.append(s)
    return np.array(T), np.array(E), np.array(S)


# ============================ transit functionals =============================
def functionals(T, E, S, D=None):
    """I1 = \int e dt, I2 = \int e^2 dt over the transit window |s|<D (or all)."""
    dt = T[1]-T[0]
    m = np.ones_like(S, bool) if D is None else (np.abs(S) < D)
    return float(np.sum(E[m])*dt), float(np.sum(E[m]**2)*dt)


def ustar(model, alpha_deg, vgrid, D=None, **kw):
    """u* = v* tau_tot / L minimising  v * I2(v)  (= lap RMSE^2 up to a constant)."""
    J = []
    for v in vgrid:
        T, E, S = model(alpha_deg, v, **kw)
        _, I2 = functionals(T, E, S, D)
        J.append(v*I2)
    J = np.array(J); i = int(np.argmin(J))
    vs = vgrid[i]
    if 1 <= i <= len(vgrid)-2:                    # parabolic refine
        c = np.polyfit(vgrid[i-1:i+2], J[i-1:i+2], 2)
        if c[0] > 0:
            vv = -c[1]/(2*c[0])
            if vgrid[i-1] <= vv <= vgrid[i+1]:
                vs = vv
    return vs*TAU_TOT/L, J


# ================================ experiments =================================
def exp_area_law(alphas=(30, 60, 90, 120), vs=(0.6, 0.9, 1.2, 1.5, 1.8)):
    """LEMMA A: \int e/c dt should equal (L*a/v)(L/2 - v*tau_tot), and in
    particular cross zero at v*tau_tot = L/2 for every alpha."""
    out = {}
    for ad in alphas:
        a = math.radians(ad); row = []
        for v in vs:
            T, E, S = theory(ad, v, mean_delay=True)
            dt = T[1]-T[0]
            # weight by 1/c(f) exactly as the lemma states
            w = []
            tau_tot = TAU_TOT
            for t, s in zip(T, S):
                s_d = s - v*tau_tot
                _, c = chord(a, (s_d + L)/L)
                w.append(1.0/max(c, 1e-6))
            I1c = float(np.sum(E*np.array(w))*dt)
            pred = (L*a/v)*(L/2.0 - v*tau_tot)
            row.append(dict(v=v, I1c=I1c, pred=pred,
                            rel=(I1c-pred)/abs(pred) if abs(pred) > 1e-9 else None))
        # zero crossing of the measured area
        vv = np.linspace(0.7, 2.2, 61)
        A = []
        for v in vv:
            T, E, S = theory(ad, v, mean_delay=True)
            dt = T[1]-T[0]
            w = np.array([1.0/max(chord(a, (s - v*TAU_TOT + L)/L)[1], 1e-6) for s in S])
            A.append(float(np.sum(E*w)*dt))
        A = np.array(A); k = int(np.argmin(np.abs(A)))
        v0 = float(np.interp(0.0, [A[k+1], A[k]], [vv[k+1], vv[k]])) if 0 < k < len(vv)-1 else vv[k]
        out[ad] = dict(rows=row, v_zero=v0, u_zero=v0*TAU_TOT/L)
    return out


def exp_area_law_sim(alphas=(30, 60, 90, 120)):
    """Same zero-crossing, but in the exact 2D noise-free simulator."""
    out = {}
    vv = np.linspace(0.8, 2.4, 33)
    for ad in alphas:
        a = math.radians(ad); A = []
        for v in vv:
            T, E, S = sim2d(ad, v)
            dt = T[1]-T[0]
            w = np.array([1.0/max(chord(a, (s - v*TAU_TOT + L)/L)[1], 1e-6) for s in S])
            A.append(float(np.sum(E*w)*dt))
        A = np.array(A)
        sgn = np.where(np.diff(np.sign(A)) != 0)[0]
        v0 = float(np.interp(0.0, [A[sgn[0]], A[sgn[0]+1]], [vv[sgn[0]], vv[sgn[0]+1]])) if len(sgn) else float('nan')
        out[ad] = dict(v_zero=v0, u_zero=v0*TAU_TOT/L)
    return out


def exp_amplitude(alphas=(15, 30, 45, 60, 72, 90, 105, 120), D=None):
    """A(alpha) = u*/cos(alpha/2) for the three layers."""
    vg = np.linspace(0.5, 3.0, 26)
    out = {}
    for ad in alphas:
        ca = math.cos(math.radians(ad)/2.0)
        u0, _ = ustar(lambda A, v, **k: sim2d(A, v, **k), ad, vg, D)
        u1, _ = ustar(lambda A, v, **k: theory(A, v, mean_delay=False, **k), ad, vg, D)
        u2, _ = ustar(lambda A, v, **k: theory(A, v, mean_delay=True, **k), ad, vg, D)
        out[ad] = dict(cos=ca, u_sim=u0, u_L1=u1, u_L2=u2,
                       A_sim=u0/ca, A_L1=u1/ca, A_L2=u2/ca)
        print(f"  alpha={ad:5.1f}  A_sim={u0/ca:.4f}  A_L1={u1/ca:.4f}  A_L2={u2/ca:.4f}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    print(f"tau_tot = {TAU_TOT:.4f} s,  L/2 = {L/2}, u(L/2) = {L/2*TAU_TOT/L/TAU_TOT*TAU_TOT/L:.3f}")
    print("\n[1] LEMMA A -- corner area law (reduced model, mean delay)")
    A1 = exp_area_law()
    for ad, d in A1.items():
        print(f"  alpha={ad:4}: zero of \\int e/c at u = {d['u_zero']:.4f}   (theory 0.5)")
        for r in d['rows']:
            rel = f"{100*r['rel']:+.1f}%" if r['rel'] is not None else "  --"
            print(f"        v={r['v']:.2f}  measured={r['I1c']:+.4f}  predicted={r['pred']:+.4f}  {rel}")

    print("\n[2] LEMMA A in the exact 2D simulator")
    A2 = exp_area_law_sim()
    for ad, d in A2.items():
        print(f"  alpha={ad:4}: zero at u = {d['u_zero']:.4f}   (theory 0.5)")

    print("\n[3] amplitude A(alpha) = u*/cos(alpha/2), objective v*I2")
    A3 = exp_amplitude()

    json.dump(dict(tau_tot=TAU_TOT, area_law_theory=A1, area_law_sim=A2,
                   amplitude=A3),
              open("d1_results_2026-08-29.json", "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> d1_results_2026-08-29.json")
