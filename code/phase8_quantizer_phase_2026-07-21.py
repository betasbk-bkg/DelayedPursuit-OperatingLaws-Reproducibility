# -*- coding: utf-8 -*-
"""this study -- DECISIVE experiment: quantizer phase sweep (circle-map structure).
2026-07-21.

Control parameter  rho = v*W/(R*Delta)  = quantizer cells advanced per vote
window.  Circle-map / three-gap prediction:
  - rho rational p/q  -> ideal direction revisits q grid-phase offsets
    (grid locking, coherent sawtooth) -> LARGER residual quantization error,
    LOWER x_q (dither less effective).
  - rho irrational     -> Weyl equidistribution of the phase -> maximal
    self-dither -> HIGHER x_q.

Design that separates rho from the dither variable x = v*sqrt(tau):
  For each radius R we measure the full RMSE(v) curve at fixed tau=433ms and
  extract the low-speed dither-linearization scale x_q by fitting
  RMSE^2 = F^2 + Q0(R)^2 * exp(-2 (x/x_q)^2) + k * x^2   on the ascending branch,
  with Q0(R) = L*Delta/sqrt(12) fixed (geometry-independent steady offset).
  Radii are chosen so that at the SAME operating speed the phase rho differs:
  small R -> large rho (locking accessible), large R -> small rho.

Because rho = v*W/(R*Delta) and the optimum sits near v ~ x*/sqrt(tau), we scan
R across a decade and, at each R, read the local rho at the measured v*.  We
also do a fine rho-scan near a strong rational (rho=1/2, grid-locked) versus a
nearby irrational to expose an Arnold-tongue dip if present.

Engine: crowd_control_engine (unmodified; only DELAY_F patched).  Circle(R).
Pre-registered two-branch outcome (per session protocol):
  A) x_q varies systematically with rho, dipping at rationals -> dither regime
     closes to circle-map structure (constant becomes a computed rho-average).
  B) x_q flat in rho -> grid bonus is NOT circle-map; x_q stays one irreducible
     constant (honest ceiling).
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(HERE, "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from simulation_main import Circle

OUT = os.path.join(HERE, "..", "data", "phase8_quantizer_phase_results_2026-07-21.json")

MC, N, TROLL = 20, 150, 0.05
TAU_F, TAU_S = 26, 0.433
W, DELTA, L = 0.3, np.pi/4, 2.0
Q0 = L*DELTA/np.sqrt(12)

def sweep_rmse(R, speeds):
    orig = sm.DELAY_F; sm.DELAY_F = TAU_F
    traj = Circle(R=R)
    try:
        rm = []
        for v in speeds:
            r = sm.run_condition(traj, N, TROLL, MC, method='fixed',
                                 speed_override=v, seed_base=31, seed_offset=N)
            rm.append(r['rmse_mean'])
    finally:
        sm.DELAY_F = orig
    return np.array(rm)

def fit_xq(speeds, rm):
    """Fit ascending (dither) branch: RMSE^2 = F^2 + Q0^2 exp(-2(x/xq)^2) + k x^2.
    x = v*sqrt(tau). Q0 fixed. Return (xq, k, F, vstar)."""
    x = speeds*np.sqrt(TAU_S)
    y = rm**2
    def model(x, F, xq, k):
        return F**2 + Q0**2*np.exp(-2*(x/xq)**2) + k*x**2
    try:
        p0 = [0.02, 1.0, 0.1]
        popt, _ = curve_fit(model, x, y, p0=p0,
                            bounds=([0,0.2,0.0],[0.5,5.0,5.0]), maxfev=20000)
        F, xq, k = popt
    except Exception as e:
        return None
    vstar = float(speeds[np.argmin(rm)])
    return dict(F=float(F), xq=float(xq), k=float(k), vstar=vstar)

if __name__ == "__main__":
    t0 = time.time()
    out = {"tau_s": TAU_S, "W": W, "Delta": DELTA, "L": L, "Q0": Q0, "runs": {}}

    # radius decade: small R -> large rho at a given v
    R_VALUES = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 20.0]
    SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0]
    print("=== radius decade (fixed tau=433ms) ===", flush=True)
    for R in R_VALUES:
        rm = sweep_rmse(R, np.array(SPEEDS))
        fit = fit_xq(np.array(SPEEDS), rm)
        if fit is None:
            print(f"  R={R}: fit failed", flush=True); continue
        rho_at_vstar = fit['vstar']*W/(R*DELTA)
        out["runs"][f"R{R}"] = dict(R=R, rmses=rm.tolist(),
                                     rho_at_vstar=rho_at_vstar, **fit)
        print(f"  R={R:>5} v*={fit['vstar']:.3f} x_q={fit['xq']:.3f} "
              f"rho(v*)={rho_at_vstar:.3f}  RMSEmin={rm.min():.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved: {OUT}", flush=True)

    # quick structure test: is x_q monotone in rho, or dipping?
    rs = [(v['rho_at_vstar'], v['xq']) for v in out['runs'].values()]
    rs.sort()
    print("\nrho vs x_q (sorted by rho):")
    for rho, xq in rs:
        print(f"  rho={rho:.3f}  x_q={xq:.3f}")
