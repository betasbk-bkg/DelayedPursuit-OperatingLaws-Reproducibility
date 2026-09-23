"""
this study THEOREM verification -- 2026-07-19
Theorem 1 (sagitta cancellation): on a circle of radius R, linearized command
inward angle chi = L/(2R) - e_delayed/L - v*tau_eff/R, giving steady bias
    e_ss(v) = (L/R) * (L/2 - v*tau_eff),  tau_eff = tau + WIN/2 + DT/SMOOTH.
Zero-parameter prediction of the ENTIRE continuum RMSE(v) curve:
    RMSE(v) ~ |e_ss(v)|  (+ small noise floor, + invalid for u = v*tau/L -> pi/2).

Requires result JSONs from the closure pack in the same directory.
"""
import os
import json, os, numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
C = 0.2333; R = 10.0

p2 = json.load(open(os.path.join(HERE, "phase2_probes_results_2026-07-18.json")))
Ls = json.load(open(os.path.join(HERE, "phase2_L_sweep_results_2026-07-18.json")))

def check(speeds, rmses, tau_s, L, label):
    te = tau_s + C
    diffs = []
    print(f"--- {label} (tau={tau_s}, L={L}) ---")
    for v, rm in zip(speeds, rmses):
        pred = (L/R)*abs(L/2 - v*te)
        diffs.append(rm - pred)
        print(f"  v={v:<5} obs={rm:.4f}  pred={pred:.4f}  diff={rm-pred:+.4f}")
    print(f"  mean|diff| = {np.mean(np.abs(diffs))*1000:.1f} mm\n")

SP = p2['speeds']
for tau_s, key in [(0.433,'433'),(1.0,'1000')]:
    check(SP, p2['probeA']['360'][key]['rmses'], tau_s, 2.0, f"probeA n360 tau={key}ms")
for r in Ls['runs']:
    if r['name'] == 'circle_n360' and r['tau_ms'] == 433:
        sp = Ls['speed_grids'][str(r['L'])]
        check(sp, r['rmses'], 0.433, r['L'], f"L-sweep n360 L={r['L']}")

print("Slope check: |dRMSE/dv| pred = L*tau_eff/R = %.4f (L=2, tau=0.433)"
      % (2*0.6663/10))
print("Theorem-candidate 3: aligned-square u* = 0.362 vs 1/e = %.4f (1.6%%)"
      % (1/np.e))
