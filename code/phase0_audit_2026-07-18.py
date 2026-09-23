"""
this study Phase 0 numerical audit — 2026-07-18
Recomputes every quantitative claim in the companion documentation
directly from the raw JSON outputs in crowd_control_engine/ (the the reference engine repo).
Run from anywhere; DATA_DIR below is absolute.

Verification labels used in the printed report:
  [VERIFIED]   recomputed here and matches the documented claim
  [CORRECTED]  recomputed here and DIFFERS from the documented claim (flagged)
"""
import os
import json, os, numpy as np
from scipy import stats, optimize

DATA_DIR = os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine"))

def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)

uv = load("unified_vstar_results.json")
cf = load("corner_freq_results.json")
ta = load("theta_autocorr_results.json")
ea = load("expa_results.json")

print("="*72)
print("1. Unified model fit + per-trajectory residuals (Problem B anchor numbers)")
print("="*72)

results = uv["results"]
all_alpha, all_tau, all_vstar, all_traj = [], [], [], []
for tname, rd in results.items():
    alpha = rd["alpha"]
    for tau_ms, troll_data in rd["data"].items():
        v = troll_data["0.05"]["vstar"]
        all_alpha.append(alpha); all_tau.append(float(tau_ms))
        all_vstar.append(v); all_traj.append(tname)

all_alpha = np.array(all_alpha); all_tau = np.array(all_tau)
all_vstar = np.array(all_vstar); all_traj = np.array(all_traj)

def unified_model(X, A, gamma, beta):
    a, t = X
    return A * np.cos(np.radians(a/2))**gamma * t**(-beta)

popt, pcov = optimize.curve_fit(
    unified_model, (all_alpha, all_tau), all_vstar,
    p0=[200.0, 2.0, 0.5], bounds=([1.0,0.1,0.1],[10000.,8.,2.]))
A_fit, gamma_fit, beta_fit = popt
pred_all = unified_model((all_alpha, all_tau), *popt)
r2_all = 1 - np.sum((all_vstar-pred_all)**2)/np.sum((all_vstar-np.mean(all_vstar))**2)
print(f"Fit: v*={A_fit:.2f} * cos^{gamma_fit:.3f}(a/2) * tau^-{beta_fit:.3f}  (global R2={r2_all:.4f})")
print(f"  Companion-paper heuristic claimed p=1.688, A=1.347 (that A is x*=v*sqrt(tau) at tau in SECONDS, not this A_fit at tau in ms -- different units, not comparable directly)")
print()

print(f"{'traj':<10} {'alpha':>6} {'n_cond':>7} {'mean|resid%|':>14}  {'max|resid%|':>13}")
print("-"*56)
per_traj_resid = {}
for tname in results:
    mask = all_traj == tname
    resid_pct = np.abs(all_vstar[mask]-pred_all[mask])/all_vstar[mask]*100
    per_traj_resid[tname] = resid_pct
    print(f"{tname:<10} {results[tname]['alpha']:>6.1f} {mask.sum():>7d} {resid_pct.mean():>13.1f}% {resid_pct.max():>12.1f}%")

print("""
Documented claim: Circle 0%, Square 4.7%, Triangle 13.1%, Hexagon ~36% (C-ratio 1.06 vs 0.78)
-> compare printed mean|resid%| above per trajectory. Label result per row below.
""")

print("="*72)
print("2. x* = v*sqrt(tau) invariant (tau in seconds) -- circle/square only (6 tau pts)")
print("="*72)
for tname in ["circle","square"]:
    rd = results[tname]
    taus_ms = sorted(int(t) for t in rd["data"].keys())
    xstars = []
    for tms in taus_ms:
        v = rd["data"][str(tms)]["0.05"]["vstar"]
        xstars.append(v*np.sqrt(tms/1000.0))
    xstars = np.array(xstars)
    cv = xstars.std(ddof=0)/xstars.mean()
    print(f"{tname:<8}: x* values = {np.round(xstars,3)}  mean={xstars.mean():.3f}  CV={cv*100:.2f}%")

print("\nDocumented claim: mean x*=1.347, CV=6.4% (pooled across trajectories, tau in [100,1000]ms)")

print()
print("="*72)
print("3. RMSE_min tau-independence")
print("="*72)
for tname in ["circle","square"]:
    rd = results[tname]
    taus_ms = sorted(int(t) for t in rd["data"].keys())
    rmse_mins = []
    for tms in taus_ms:
        rmses = rd["data"][str(tms)]["0.05"]["rmses"]
        rmse_mins.append(min(rmses))
    rmse_mins = np.array(rmse_mins)
    cv = rmse_mins.std(ddof=0)/rmse_mins.mean()
    print(f"{tname:<8}: RMSE_min per tau = {np.round(rmse_mins,3)}  mean={rmse_mins.mean():.3f}  CV={cv*100:.2f}%")
print("\nDocumented claim: RMSE_min ~ 0.244 m, CV 1.7%, tau-independent")

print()
print("="*72)
print("4. corner_freq vs alpha collinearity (Problem B core claim, r=+0.94)")
print("="*72)
# natural corner_freq per the 4 canonical trajectories (radius=10 for polygons; circle=0 corners)
natural = {}
for tname, rd in results.items():
    alpha = rd["alpha"]
    natural[tname] = alpha
# circ from RegularPolygon(n, r=10): side = 2*r*sin(pi/n); circ = n*side
import math
def polygon_circ(n, r=10.0):
    return n*2*r*math.sin(math.pi/n)
nat_freq = {
    "circle":   0.0,                                   # no corners
    "square":   4/polygon_circ(4),
    "hexagon":  6/polygon_circ(6),
    "triangle": 3/polygon_circ(3),
}
alphas_n = np.array([natural[k] for k in nat_freq])
freqs_n  = np.array([nat_freq[k] for k in nat_freq])
r_nat, p_nat = stats.pearsonr(alphas_n, freqs_n)
print("Natural (radius=10) corner_freq per trajectory:")
for k in nat_freq:
    print(f"  {k:<10} alpha={natural[k]:>6.1f}  freq={nat_freq[k]:.4f} /m")
print(f"Pearson r(alpha, corner_freq) across the 4 canonical trajectories: r={r_nat:.3f}, p={p_nat:.4f}  (n=4, CAUTION: low power)")

print()
print("-- ScaledSquare h-sweep (corner angle FIXED at 90 deg, n=11700 runs): "
      "tests whether corner_freq has an effect INDEPENDENT of alpha --")
h_values = cf["h_values"]
freqs_h = [cf["corner_freqs"][str(h)] for h in h_values]
vstars_433 = [cf["results"][str(h)]["433"]["0.05"]["vstar"] for h in h_values]
r_h, p_h = stats.pearsonr(freqs_h, vstars_433)
print(f"r(corner_freq, v*) at alpha=90 fixed, tau=433ms: r={r_h:.3f}, p={p_h:.4f}  (n={len(h_values)})")
for h, f, v in zip(h_values, freqs_h, vstars_433):
    print(f"  h={h:>4}  freq={f:.4f}  v*={v:.3f}")

print("""
Interpretation: r_nat is the alpha~freq collinearity among the 4 real trajectories
(small n=4, so treat cautiously). r_h is corner_freq's effect on v* when alpha is
held fixed at 90 deg by construction -- this is the actual identifying experiment.
Companion-paper heuristic claimed r=+0.94 for the alpha~freq collinearity and independent-explanatory
power rejected. Check both numbers above against that claim.
""")

print("="*72)
print("5. theta_autocorr: sigma_theta N-independence + k_c + lag-13 secondary peak")
print("="*72)
sN = ta["sigma_vs_N"]
sig = np.array(sN["sigma_theta"])
print(f"sigma_theta(N) values: {np.round(sig,3)}  mean={sig.mean():.3f} std={sig.std(ddof=0):.3f} "
      f"(handoff: 8.60 +/- 0.15)")
print(f"power_law exponent = {sN['power_law']:.4f}  (handoff: N^-0.008)  R2={sN['R2']:.3f}")
acf = ta["autocorr"]
print(f"k_c = {acf['k_c']}  (handoff: k_c=3)")
peak_idx = int(np.argmax(acf['acf'][4:])) + 4  # secondary peak search after initial decay
print(f"ACF secondary peak search (index>=4): peak at lag={peak_idx}, value={acf['acf'][peak_idx]:.3f} "
      f"(handoff: lag k=13)")
print(f"independence flag = {acf['independence']}  (handoff: non-Markovian)")

quantization_floor = 22.5/np.sqrt(6)
print(f"\nQuantization-floor prediction 22.5/sqrt(6) = {quantization_floor:.3f} deg "
      f"vs observed sigma_theta mean = {sig.mean():.3f} deg "
      f"-> ratio = {sig.mean()/quantization_floor:.3f} ({(sig.mean()/quantization_floor-1)*100:+.1f}%)")

print()
print("="*72)
print("6. beta(n_dirs) resonance structure (Problem A, EXP-A)")
print("="*72)
for nd in ea["n_dirs_list"]:
    print(f"  n_dirs={nd:>4}  beta={ea['beta'][str(nd)]:.4f}")
print("Handoff: continuum limit beta~0.57, 8-direction beta~0.511 (~0.06 shift)")

print()
print("="*72)
print("7. Heuristic diffusion-floor anchor: sigma_pos ~ v*sigma_theta*sqrt(2*k_c*WIN*tau)")
print("="*72)
WIN = 0.3
k_c = acf['k_c']
sigma_theta_rad = np.radians(sig.mean())
# use circle v* at tau=433ms troll=5% as the representative v (per handoff wording)
v_ref = results["circle"]["data"]["433"]["0.05"]["vstar"]
tau_s = 0.433
sigma_pos_pred = v_ref * sigma_theta_rad * np.sqrt(2*k_c*WIN*tau_s)
rmse_min_circle_433 = min(results["circle"]["data"]["433"]["0.05"]["rmses"])
print(f"v_ref(circle,tau=433ms)={v_ref:.3f} m/s, sigma_theta={sig.mean():.3f} deg = {sigma_theta_rad:.4f} rad")
print(f"sigma_pos predicted = {sigma_pos_pred:.4f} m")
print(f"RMSE_min observed (circle, tau=433ms) = {rmse_min_circle_433:.4f} m")
print(f"ratio pred/obs = {sigma_pos_pred/rmse_min_circle_433:.3f} "
      f"({(sigma_pos_pred/rmse_min_circle_433-1)*100:+.1f}%)")
print("Companion-paper heuristic claimed: predicted 0.271 m vs observed RMSE_min 0.244 m (~11% over)")
